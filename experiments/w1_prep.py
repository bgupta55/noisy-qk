"""W1 prep (NO QPU time): second qubit chain on ibm_marrakesh. Implements results/PRESPEC_marrakesh_chains.md sections 0-3.
Order: hash prespec -> read calibration -> choose chain B (and C, recorded only) -> write chain_choice.json -> build + transpile circuits -> CZ asserts -> Aer + idle-model predictions.
Writes results/hardware_w1/{prespec_hash.txt, chain_choice.json, prep_meta.json, isa_circuits.pkl}."""
import os, sys, json, pickle, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from nqk.hw import *
from nqk.channels import kraus
from nqk.kernel import apply_noise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "hardware_w1")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---- 0. hash of the prespec file, before anything else
ps = os.path.join(ROOT, "results", "PRESPEC_marrakesh_chains.md")
h = hashlib.sha256(open(ps, "rb").read()).hexdigest()
if not os.path.exists(os.path.join(OUT, "prespec_hash.txt")):
    open(os.path.join(OUT, "prespec_hash.txt"), "w").write(f"{h}  PRESPEC_marrakesh_chains.md\nutc {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\nfile mtime {time.ctime(os.path.getmtime(ps))}\n")
print("prespec sha256", h)

x4 = json.load(open(os.path.join(ROOT, "results", "hardware_x4", "prep_meta.json")))
y2 = json.load(open(os.path.join(ROOT, "results", "hardware_y2", "prep_meta.json")))
BACKEND, BETA = "ibm_marrakesh", x4["beta"]
Xall = np.array(x4["Xall"])
A = [52, 53, 54, 55]
LAYERS = {2: (1, 0), 6: (3, 0), 12: (2, 1), 20: (2, 2), 28: (2, 3), 36: (2, 4)}
SH_R, SH_A, SH_T, SH_REP = 1024, 2048, 1024, 1024
DELAYS, ECHO = [0, 100, 200], [10, 30, 60, 120, 240]
pairs3 = [tuple(p) for p in y2["pairs4"]]

# ---- 1. six inputs, farthest-point sampling among the 9 training inputs (no hardware data)
Xtr = Xall[:9]
d0 = np.linalg.norm(Xtr - Xtr.mean(0), axis=1)
chosen = [int(np.argmax(d0))]
while len(chosen) < 6:
    dm = [min(np.linalg.norm(Xtr[k] - Xtr[c]) for c in chosen) if k not in chosen else -1 for k in range(9)]
    chosen.append(int(np.argmax(dm)))
print("six inputs (indices into the 14 points):", chosen)
X6 = Xtr[chosen]

# ---- 2. calibration and chain choice
svc = get_service()
backend = svc.backend(BACKEND)
tg = backend.target
cz = {}
for (a, b), pr in tg["cz"].items():
    if pr is not None and pr.error is not None:
        e = pr.error
        cz[(a, b)] = e; cz[(b, a)] = e
rd = {q[0]: (tg["measure"][q].error if tg["measure"][q] is not None and tg["measure"][q].error is not None else 0.05) for q in tg["measure"]}
adj = {}
for (a, b) in cz:
    adj.setdefault(a, set()).add(b)
qp = {}
for q in range(backend.num_qubits):
    p = backend.qubit_properties(q)
    qp[q] = dict(T1_us=p.t1 * 1e6 if p.t1 else None, T2_us=p.t2 * 1e6 if p.t2 else None, readout_err=rd.get(q))
paths = set()
for s in adj:
    if s in A:
        continue
    stack = [[s]]
    while stack:
        pth = stack.pop()
        if len(pth) == 4:
            paths.add(tuple(pth) if pth[0] < pth[-1] else tuple(pth[::-1])); continue
        for nb in adj[pth[-1]]:
            if nb not in pth and nb not in A:
                stack.append(pth + [nb])
paths = sorted(paths)
score = lambda p: sum(cz[(a, b)] for a, b in zip(p[:-1], p[1:])) + sum(rd[q] for q in p)
ranked = sorted(paths, key=lambda p: (score(p), list(p)))
B = list(ranked[0])
okT2 = [p for p in paths if all(qp[q]["T2_us"] is not None for q in p) and min(qp[q]["T2_us"] for q in p) >= 100 and not (set(p) & (set(A) | set(B)))]
C = list(max(okT2, key=lambda p: (min(qp[q]["T2_us"] for q in p), [-x for x in p]))) if okT2 else None
snapA = snapshot(backend, A); snapB = snapshot(backend, B)
choice = dict(backend=BACKEND, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), A=A, B=B, C=C, n_candidate_paths=len(paths),
              score_A=score(tuple(A)), score_B=score(tuple(B)), top5_by_score=[dict(path=list(p), score=score(p)) for p in ranked[:5]],
              scoreC=score(tuple(C)) if C else None, minT2_C=min(qp[q]["T2_us"] for q in C) if C else None,
              qubit_props={str(q): qp[q] for q in sorted(set(A) | set(B) | set(C or []))},
              cz={f"{a}-{b}": cz.get((a, b)) for ch in (A, B, C or []) for a, b in zip(ch[:-1], ch[1:])},
              snapshot_A=snapA, snapshot_B=snapB, six_inputs=chosen, pairs3=pairs3)
json.dump(choice, open(os.path.join(OUT, "chain_choice.json"), "w"), indent=1, default=float)
print("A", A, "score", choice["score_A"], "| B", B, "score", choice["score_B"], "| C", C, "of", len(paths), "paths")
print("T2 A", [round(qp[q]["T2_us"], 1) for q in A], "T2 B", [round(qp[q]["T2_us"], 1) for q in B])
print("T1 A", [round(qp[q]["T1_us"], 1) for q in A], "T1 B", [round(qp[q]["T1_us"], 1) for q in B])

# ---- 3. circuits
chains = dict(A=A, B=B)
pms = {c: generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=chains[c]) for c in chains}


def exactK(L, fold, i, j):
    qc = kernel_circuit(X6[i], X6[j], BETA, L, fold); qc.remove_final_measurements()
    return float(abs(Statevector(qc).data[0]) ** 2)


Kex = {lay: np.array([[exactK(*LAYERS[lay], i, j) for j in range(6)] for i in range(6)]) for lay in LAYERS}
rep_entries = {}
for lay in (2, 12, 36):
    order = sorted(((Kex[lay][i, j], i, j) for i in range(6) for j in range(6)))
    rep_entries[lay] = [(order[r][1], order[r][2]) for r in np.round(np.linspace(0, 35, 12)).astype(int)]


def Uop(x):
    q = QuantumCircuit(NQ); slice_gates(q, x, BETA, 1); return Operator(q).data


Uall = [Uop(x) for x in Xall]


def idle_k(i, j, tau, t1, t2):
    g = [1 - np.exp(-tau / a) for a in t1]
    lp = [max(0.0, 1 - np.exp(-2 * tau * (1 / b - 1 / (2 * a)))) for a, b in zip(t1, t2)]
    rho = np.zeros((16, 16), complex); rho[0, 0] = 1
    rho = Uall[i] @ rho @ Uall[i].conj().T
    rho = apply_noise(rho[None], ("local_pq", [kraus("ad", gg) for gg in g[::-1]]), NQ)
    rho = apply_noise(rho, ("local_pq", [kraus("pd", ll) for ll in lp[::-1]]), NQ)[0]
    rho = Uall[j].conj().T @ rho @ Uall[j]
    return float(rho[0, 0].real)


def build_items(chain, kind):
    its = []
    if kind == "R":
        for lay, (L, f) in LAYERS.items():
            for i in range(6):
                for j in range(6):
                    its.append(dict(kind="R", chain=chain, layers=lay, i=i, j=j, shots=SH_R, qc=kernel_circuit(X6[i], X6[j], BETA, L, f), cz=expected_cz(L, f)))
    elif kind == "Rrep":
        for lay in (2, 12, 36):
            L, f = LAYERS[lay]
            for (i, j) in rep_entries[lay]:
                its.append(dict(kind="Rrep", chain=chain, layers=lay, i=i, j=j, shots=SH_REP, qc=kernel_circuit(X6[i], X6[j], BETA, L, f), cz=expected_cz(L, f)))
    elif kind == "A":
        for tau in DELAYS:
            for (i, j) in pairs3:
                for (a, b) in ((i, j), (j, i)):
                    its.append(dict(kind="A", chain=chain, tau=tau, i=a, j=b, shots=SH_A, qc=idle_kernel_circuit(Xall[a], Xall[b], BETA, tau), cz=12))
    elif kind == "T":
        for tot in ECHO:
            for prep in ("+", "-"):
                qc = QuantumCircuit(NQ, NQ)
                for q in range(NQ):
                    if prep == "-":
                        qc.x(q)
                    qc.h(q)
                qc.barrier(); qc.delay(int(tot * 500), range(NQ), unit="ns"); qc.barrier()
                for q in range(NQ):
                    qc.x(q)
                qc.barrier(); qc.delay(int(tot * 500), range(NQ), unit="ns"); qc.barrier()
                for q in range(NQ):
                    qc.h(q)
                qc.measure(range(NQ), range(NQ))
                its.append(dict(kind="T", chain=chain, tau=tot, prep=prep, shots=SH_T, qc=qc, cz=0))
    return its


JOBS = [("A-R", "A", "R"), ("B-R", "B", "R"), ("A-A", "A", "A"), ("B-A", "B", "A"), ("A-T", "A", "T"), ("B-T", "B", "T"), ("A-Rrep", "A", "Rrep")]
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(backend)
sim_n = AerSimulator(noise_model=nm)
isa_all, jobs_meta = {}, {}
for name, chain, kind in JOBS:
    its = build_items(chain, kind)
    isa = pms[chain].run([it["qc"] for it in its])
    for it, c in zip(its, isa):
        assert c.count_ops().get("cz", 0) == it["cz"], (name, dict(c.count_ops()))
    if kind in ("R", "Rrep", "A"):
        sh = 2048
        r = sim_n.run(isa, shots=sh, seed_simulator=11).result()
        pa = [r.get_counts(k).get("0" * NQ, 0) / sh for k in range(len(isa))]
    else:
        pa = None
    meta = [{k: v for k, v in it.items() if k != "qc"} for it in its]
    for m, it in zip(meta, its):
        if it["kind"] in ("R", "Rrep"):
            m["K_exact"] = float(Kex[it["layers"]][it["i"], it["j"]])
        if it["kind"] == "A":
            sn = snapA if chain == "A" else snapB
            t1 = [sn["qubit_props"][q]["T1_us"] for q in chains[chain]]; t2 = [sn["qubit_props"][q]["T2_us"] for q in chains[chain]]
            m["K_idle_cal"] = idle_k(it["i"], it["j"], float(it["tau"]), t1, t2)
            m["K_exact"] = float(abs((Uall[it["j"]].conj().T @ Uall[it["i"]])[0, 0]) ** 2)
    jobs_meta[name] = dict(chain=chain, kind=kind, layout=chains[chain], n=len(its), items=meta, aer_noisy_2048=pa, shots=[it["shots"] for it in its],
                           depth_max=max(c.depth() for c in isa))
    isa_all[name] = isa
    print(name, chain, kind, len(its), "circuits, CZ asserts OK, max depth", jobs_meta[name]["depth_max"], round(time.time() - t0, 1), "s", flush=True)

meta = dict(backend=BACKEND, beta=BETA, A=A, B=B, C=C, layers_map={str(k): v for k, v in LAYERS.items()}, six_inputs=chosen, X6=X6.tolist(), Xall=Xall.tolist(),
            Kexact={str(k): v.tolist() for k, v in Kex.items()}, rep_entries={str(k): v for k, v in rep_entries.items()}, pairs3=pairs3, delays=DELAYS, echo_delays=ECHO,
            shots=dict(R=SH_R, A=SH_A, T=SH_T, Rrep=SH_REP), job_order=[j[0] for j in JOBS], jobs=jobs_meta, snapshot_start=dict(A=snapA, B=snapB), prespec_sha256=h)
json.dump(meta, open(os.path.join(OUT, "prep_meta.json"), "w"), default=float)
pickle.dump(isa_all, open(os.path.join(OUT, "isa_circuits.pkl"), "wb"))
print("prep done", round(time.time() - t0, 1), "s; circuits", {k: len(v) for k, v in isa_all.items()})
