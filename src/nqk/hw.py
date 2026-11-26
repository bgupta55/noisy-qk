"""Hardware helpers: circuit builders (Emb-Trot), transpile with CZ-count assertion, Aer ideal/noise-model runs, IBM connection."""
import os, sys, json, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

NQ = 4


def slice_gates(qc, x, beta, L, inverse=False):
    sl = QuantumCircuit(qc.num_qubits)
    for i in range(qc.num_qubits):
        sl.rx(2 * beta * x[i] / L, i)
    for i in range(qc.num_qubits - 1):
        sl.rzz(2 * beta ** 2 * x[i] * x[i + 1] / L, i, i + 1)
    qc.compose(sl.inverse() if inverse else sl, inplace=True)
    qc.barrier()


def kernel_circuit(x, xp, beta, L, fold=0, N=NQ):
    """Compute U(x) (L slices), uncompute U(x')^dagger (L slices); each slice optionally followed by `fold` W^dag W identity pairs."""
    qc = QuantumCircuit(N)
    for _ in range(L):
        slice_gates(qc, x, beta, L)
        for _ in range(fold):
            slice_gates(qc, x, beta, L, inverse=True)
            slice_gates(qc, x, beta, L)
    for _ in range(L):
        slice_gates(qc, xp, beta, L, inverse=True)
        for _ in range(fold):
            slice_gates(qc, xp, beta, L)
            slice_gates(qc, xp, beta, L, inverse=True)
    qc.measure_all()
    return qc


def idle_kernel_circuit(x, xp, beta, delay_us, N=NQ):
    """HW4: L=1 kernel circuit with an idle delay between U(x) and U(x')^dagger (all qubits)."""
    qc = QuantumCircuit(N)
    slice_gates(qc, x, beta, 1)
    if delay_us > 0:
        qc.delay(int(delay_us * 1000), range(N), unit="ns")
        qc.barrier()
    slice_gates(qc, xp, beta, 1, inverse=True)
    qc.measure_all()
    return qc


def expected_cz(L, fold, N=NQ):
    per_slice = 2 * (N - 1)             # each RZZ -> 2 CZ
    return 2 * L * per_slice * (1 + 2 * fold)


def p0_from_counts(counts, shots, N=NQ):
    return counts.get("0" * N, 0) / shots


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    if "IBMQ_KEY" in os.environ and "IBMQ_CRN" in os.environ:
        return QiskitRuntimeService(channel="ibm_cloud", token=os.environ["IBMQ_KEY"], instance=os.environ["IBMQ_CRN"])
    # Round Y only (author-approved): fall back to the account saved by qiskit (name 'default-ibm-cloud'); the token is never read or printed here.
    return QiskitRuntimeService(name="default-ibm-cloud")


def best_chain(backend, N=NQ):
    """Lowest-error connected chain of N qubits from the backend target (cz error + readout error), greedy over all paths via DFS on coupling map."""
    t = backend.target
    cz = t["cz"]
    meas = t["measure"]
    err = {}
    for (a, b), props in cz.items():
        if props is not None and props.error is not None:
            err[(a, b)] = props.error
            err[(b, a)] = props.error
    adj = {}
    for (a, b) in err:
        adj.setdefault(a, set()).add(b)
    rd = {q[0]: (meas[q].error if meas[q] is not None and meas[q].error is not None else 0.05) for q in meas}
    best = None

    def dfs(path, cost):
        nonlocal best
        if len(path) == N:
            tot = cost + sum(rd[q] for q in path)
            if best is None or tot < best[0]:
                best = (tot, list(path))
            return
        for nb in adj.get(path[-1], ()):
            if nb not in path:
                dfs(path + [nb], cost + err[(path[-1], nb)])
    good = sorted(adj.keys(), key=lambda q: rd.get(q, 1))[:60]
    for s in good:
        dfs([s], 0.0)
    return best[1], best[0]


def snapshot(backend, layout):
    t = backend.target
    snap = dict(backend=backend.name, qubits=layout, time=time.strftime("%Y-%m-%d %H:%M:%S"), qubit_props={}, cz={})
    try:
        for q in layout:
            p = backend.qubit_properties(q)
            snap["qubit_props"][q] = dict(T1_us=p.t1 * 1e6 if p.t1 else None, T2_us=p.t2 * 1e6 if p.t2 else None,
                                          readout_err=t["measure"][(q,)].error)
        for a, b in zip(layout[:-1], layout[1:]):
            e = t["cz"].get((a, b)) or t["cz"].get((b, a))
            snap["cz"][f"{a}-{b}"] = e.error if e else None
    except Exception as ex:
        snap["error"] = str(ex)
    return snap
