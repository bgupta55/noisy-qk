"""Hardware prep (no QPU): choose HW2 data (n_train=6, n_test=3, D1/Iris 1v2, Emb-Trot, N=4) by NOISELESS Aer/exact statevector criteria
only; choose beta by noiseless LOO; build all HW circuits for ibm_kingston; transpile at optimisation_level=0 with fixed layout;
assert CZ counts; run Aer ideal + Aer backend-noise-model on the same transpiled circuits; save everything for the run script."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from itertools import product
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from nqk.hw import *
from nqk.datasets import make_split
from nqk.krr import ridge
from nqk.metrics import auc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware")
os.makedirs(HW, exist_ok=True)
t0 = time.time()
BACKEND = "ibm_kingston"
NTR, NTE = 6, 3
LS = [1, 2, 3]
FOLDS_L2 = [0, 1, 2, 4]


def exact_kernel_trot(X, beta, L):
    """Noiseless exact kernel of the Trotter circuit (L slices; L-dependent because RX/RZZ do not commute)."""
    def state(x):
        qc = QuantumCircuit(NQ)
        for _ in range(L):
            slice_gates(qc, x, beta, L)
        return Statevector(qc).data
    P = np.array([state(x) for x in X])
    return np.abs(np.conj(P) @ P.T) ** 2


# ---- choose split (noiseless-only criterion) and beta
best = None
for sp in range(40):
    Xtr, Xte, ytr, yte = make_split("D1", NTR, sp, n_test_cap=NTE, scale="unit")
    if len(yte) < NTE:
        continue
    # keep exactly 3 test points with both classes present
    if len(set(yte[:NTE])) < 2 or len(set(ytr)) < 2:
        continue
    Xte3, yte3 = Xte[:NTE], yte[:NTE]
    for beta in (0.5, 0.75, 1.0, 1.5, 2.0):
        K = exact_kernel_trot(np.vstack([Xtr, Xte3]), beta, 2)
        # mean pairwise leave-one-out AUC over the 9 points (noiseless), centred ridge lam=0.1
        Xall = np.vstack([Xtr, Xte3]); yall = np.concatenate([ytr, yte3]).astype(float)
        sc = []
        for tst in range(NTR + NTE):
            tr = [i for i in range(NTR + NTE) if i != tst]
            h = ridge(K[np.ix_(tr, tr)], K[np.ix_([tst], tr)], yall[tr], 0.1, True)
            sc.append(float(h[0] * yall[tst]))
        # also train->test auc
        h = ridge(K[:NTR, :NTR], K[NTR:, :NTR], ytr.astype(float), 0.1, True)
        a = auc(h, yte3)
        offd = K[:NTR, :NTR][~np.eye(NTR, dtype=bool)]
        # require meaningful spread of kernel values (so retention can be measured) and perfect noiseless test AUC
        crit = (a, offd.std(), np.mean(np.array(sc) > 0))
        if a == 1.0 and offd.std() > 0.05:
            score = crit[2] + 0.1 * offd.std()
            if best is None or score > best[0]:
                best = (score, sp, beta, a, offd.std(), crit[2])
print("selected (noiseless-only criterion):", best)
_, SP, BETA, a0, sd, loo = best
Xtr, Xte, ytr, yte = make_split("D1", NTR, SP, n_test_cap=NTE, scale="unit")
Xte, yte = Xte[:NTE], yte[:NTE]
Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
P = NTR + NTE

# ---- backend and layout
svc = get_service()
backend = svc.backend(BACKEND)
layout, cost = best_chain(backend)
print("layout", layout, "cost", cost)
snap = snapshot(backend, layout)
pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=layout)

# ---- circuit list for HW2: pairs (i,j) i<j among all 9 points that are train-train (15) or test-train (18) ; one order only (i x-arm, j x'-arm)
pairs = [(i, j) for i in range(NTR) for j in range(i + 1, NTR)] + [(NTR + t, j) for t in range(NTE) for j in range(NTR)]
settings = [(1, 0), (2, 0), (3, 0), (2, 1), (2, 2), (2, 4)]
hw2 = []
for (L, f) in settings:
    for (i, j) in pairs:
        hw2.append(dict(L=L, fold=f, i=i, j=j, qc=kernel_circuit(Xall[i], Xall[j], BETA, L, f)))
print("HW2 circuits", len(hw2))
# reverse-order circuits for asymmetry test at L=2 fold 0 : (j,i)
hw2_rev = [dict(L=2, fold=0, i=j, j=i, qc=kernel_circuit(Xall[j], Xall[i], BETA, 2, 0)) for (i, j) in pairs[:15]]

# ---- HW4: idle delay sweep: 3 pairs x 2 orders x delays
DELAYS = [0, 100, 200]
# pick 3 pairs predicted (by T1/amplitude-damping model) to show largest asymmetry; selection uses only the noiseless/idle model, not hardware data
def idle_model_asym(i, j, tau_us, T1_us=250.0):
    g = 1 - np.exp(-tau_us / T1_us)
    from nqk.channels import kraus
    from nqk.kernel import noisy_kernels
    # exact-state density-matrix evolution with AD(g) on each qubit between U(x) and U(x')^dag
    qc1 = QuantumCircuit(NQ); slice_gates(qc1, Xall[i], BETA, 1)
    qc2 = QuantumCircuit(NQ); slice_gates(qc2, Xall[j], BETA, 1)
    from qiskit.quantum_info import Operator
    U1, U2 = Operator(qc1).data, Operator(qc2).data
    from nqk.kernel import apply_noise
    def kval(UA, UB):
        rho = np.zeros((16, 16), complex); rho[0, 0] = 1
        rho = UA @ rho @ UA.conj().T
        rho = apply_noise(rho[None], ("local", kraus("ad", g)), NQ)[0]
        rho = UB.conj().T @ rho @ UB
        return rho[0, 0].real
    return abs(kval(U1, U2) - kval(U2, U1))
cand = []
for i in range(P):
    for j in range(i + 1, P):
        cand.append((idle_model_asym(i, j, 200.0), i, j))
cand.sort(reverse=True)
hw4_pairs = [(i, j) for _, i, j in cand[:3]]
print("HW4 pairs (model-selected)", hw4_pairs, [round(c[0], 4) for c in cand[:3]])
hw4 = []
for tau in DELAYS:
    for (i, j) in hw4_pairs:
        for (a, b) in ((i, j), (j, i)):
            hw4.append(dict(tau=tau, i=a, j=b, qc=idle_kernel_circuit(Xall[a], Xall[b], BETA, tau)))

# ---- HW1 circuits: 1-qubit process tomography with idle delay, run on 4 qubits in parallel (same circuit pattern on each qubit)
def tomo_circuits(delay_us):
    out = []
    prep = {"0": [], "1": [("x",)], "+": [("h",)], "+i": [("h",), ("s",)]}
    meas = {"X": [("h",)], "Y": [("sdg",), ("h",)], "Z": []}
    for pk, pg in prep.items():
        for mk, mg in meas.items():
            qc = QuantumCircuit(NQ, NQ)
            for g in pg:
                for q in range(NQ):
                    getattr(qc, g[0])(q)
            qc.barrier()
            if delay_us > 0:
                qc.delay(int(delay_us * 1000), range(NQ), unit="ns")
            qc.barrier()
            for g in mg:
                for q in range(NQ):
                    getattr(qc, g[0])(q)
            qc.measure(range(NQ), range(NQ))
            out.append(dict(prep=pk, meas=mk, delay=delay_us, qc=qc))
    return out
hw1 = [c for d in (0, 100, 200) for c in tomo_circuits(d)]
print("HW1 circuits", len(hw1), "HW4", len(hw4), "HW2rev", len(hw2_rev))

def transpile_list(items, expected_fn=None):
    isa = pm.run([it["qc"] for it in items])
    out = []
    for it, c in zip(items, isa):
        ops = c.count_ops()
        cz = ops.get("cz", 0)
        if expected_fn is not None:
            assert cz == expected_fn(it), (it.get("L"), it.get("fold"), cz, expected_fn(it), dict(ops))
        out.append(c)
    return out
isa2 = transpile_list(hw2, lambda it: expected_cz(it["L"], it["fold"]))
isa2r = transpile_list(hw2_rev, lambda it: expected_cz(it["L"], it["fold"]))
isa4 = transpile_list(hw4, lambda it: 12)
isa1 = transpile_list(hw1)
print("transpile + CZ assertions OK:", time.time() - t0)
print("HW2 depth range:", min(c.depth() for c in isa2), max(c.depth() for c in isa2), "CZ range:", min(c.count_ops().get('cz', 0) for c in isa2), max(c.count_ops().get('cz', 0) for c in isa2))

# ---- Aer: ideal and backend-noise-model on the SAME transpiled circuits
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(backend)
sim_noisy = AerSimulator(noise_model=nm)
sim_ideal = AerSimulator()
SH = 1024
def run_aer(sim, circs, shots, seed=1):
    r = sim.run(circs, shots=shots, seed_simulator=seed).result()
    return [r.get_counts(k) for k in range(len(circs))]
def p0(counts, shots):
    return counts.get("0" * NQ, 0) / shots
aer = {}
aer["hw2_ideal"] = [p0(c, SH) for c in run_aer(sim_ideal, isa2, SH)]
aer["hw2_noisy"] = [p0(c, SH) for c in run_aer(sim_noisy, isa2, SH)]
aer["hw2_noisy_8192"] = [p0(c, 8192) for c in run_aer(sim_noisy, isa2, 8192, 5)]
aer["hw2rev_noisy_8192"] = [p0(c, 8192) for c in run_aer(sim_noisy, isa2r, 8192, 6)]
aer["hw4_ideal"] = [p0(c, 2048) for c in run_aer(sim_ideal, isa4, 2048)]
aer["hw4_noisy"] = [p0(c, 2048) for c in run_aer(sim_noisy, isa4, 2048)]
# exact noiseless values of the *circuits' ideal kernel* for HW2
def exact_pair(x, xp, L, f=0):
    qc = kernel_circuit(x, xp, BETA, L, f); qc.remove_final_measurements()
    return abs(Statevector(qc).data[0]) ** 2
aer["hw2_exact"] = [exact_pair(Xall[it["i"]], Xall[it["j"]], it["L"], it["fold"]) for it in hw2]
print("Aer ideal vs exact max dev (HW2):", np.max(np.abs(np.array(aer["hw2_ideal"]) - np.array(aer["hw2_exact"]))), " (shot noise ~ 0.015)")

meta = dict(backend=BACKEND, layout=layout, beta=BETA, split=SP, n_train=NTR, n_test=NTE, noiseless_test_auc_L2=a0,
            noiseless_offdiag_std=sd, snapshot=snap, settings=settings, delays=DELAYS, hw4_pairs=hw4_pairs, pairs=pairs,
            hw2_meta=[{k: v for k, v in it.items() if k != "qc"} for it in hw2], hw2rev_meta=[{k: v for k, v in it.items() if k != "qc"} for it in hw2_rev],
            hw4_meta=[{k: v for k, v in it.items() if k != "qc"} for it in hw4], hw1_meta=[{k: v for k, v in it.items() if k != "qc"} for it in hw1],
            Xall=Xall.tolist(), yall=yall.tolist(),
            cz_counts_hw2=[c.count_ops().get("cz", 0) for c in isa2], depth_hw2=[c.depth() for c in isa2],
            cz_counts_hw1=[c.count_ops().get("cz", 0) for c in isa1], aer=aer)
json.dump(meta, open(os.path.join(HW, "prep_meta.json"), "w"), indent=1, default=float)
pickle.dump(dict(isa2=isa2, isa2r=isa2r, isa4=isa4, isa1=isa1), open(os.path.join(HW, "isa_circuits.pkl"), "wb"))
print("prep done", time.time() - t0)
