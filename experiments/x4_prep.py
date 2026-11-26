"""X4 prep (NO QPU time): second Heron backend (ibm_marrakesh), n=14 points (9 train + 5 test), settings (L,fold) = (1,0),(2,0),(3,0),(2,1),(2,2),(2,3),(2,4).
Selection of split and beta by NOISELESS exact criteria only (same rule as hw_prep.py). Transpile at optimisation_level=0 with fixed layout, assert CZ counts,
run Aer ideal + Aer backend-noise-model on the same transpiled circuits. Writes results/hardware_x4/{prep_meta.json,isa_circuits.pkl}."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from nqk.hw import *
from nqk.datasets import make_split
from nqk.krr import ridge
from nqk.metrics import auc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware_x4")
os.makedirs(HW, exist_ok=True)
t0 = time.time()
BACKEND = "ibm_marrakesh"
NTR, NTE = 9, 5
SETTINGS = [(1, 0), (2, 0), (3, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
SHOTS = {s: 1024 for s in SETTINGS}
SHOTS[(2, 4)] = 4096


def exact_kernel_trot(X, beta, L):
    def state(x):
        qc = QuantumCircuit(NQ)
        for _ in range(L):
            slice_gates(qc, x, beta, L)
        return Statevector(qc).data
    P = np.array([state(x) for x in X])
    return np.abs(np.conj(P) @ P.T) ** 2


best = None
for sp in range(40):
    Xtr, Xte, ytr, yte = make_split("D1", NTR, sp, n_test_cap=NTE, scale="unit")
    if len(yte) < NTE or len(set(yte[:NTE])) < 2 or len(set(ytr)) < 2:
        continue
    Xte5, yte5 = Xte[:NTE], yte[:NTE]
    Xall_ = np.vstack([Xtr, Xte5]); yall_ = np.concatenate([ytr, yte5]).astype(float)
    for beta in (0.5, 0.75, 1.0, 1.5, 2.0):
        K = exact_kernel_trot(Xall_, beta, 2)
        sc = []
        for tst in range(NTR + NTE):
            tr = [i for i in range(NTR + NTE) if i != tst]
            h = ridge(K[np.ix_(tr, tr)], K[np.ix_([tst], tr)], yall_[tr], 0.1, True)
            sc.append(float(h[0] * yall_[tst]))
        h = ridge(K[:NTR, :NTR], K[NTR:, :NTR], ytr.astype(float), 0.1, True)
        a = auc(h, yte5)
        offd = K[:NTR, :NTR][~np.eye(NTR, dtype=bool)]
        if a == 1.0 and offd.std() > 0.05:
            score = np.mean(np.array(sc) > 0) + 0.1 * offd.std()
            if best is None or score > best[0]:
                best = (score, sp, beta, a, offd.std(), float(np.mean(np.array(sc) > 0)))
print("selected (noiseless-only):", best)
_, SP, BETA, a0, sd, loo = best
Xtr, Xte, ytr, yte = make_split("D1", NTR, SP, n_test_cap=NTE, scale="unit")
Xte, yte = Xte[:NTE], yte[:NTE]
Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
P = NTR + NTE

svc = get_service()
backend = svc.backend(BACKEND)
layout, cost = best_chain(backend)
print("layout", layout, "cost", cost)
snap = snapshot(backend, layout)
pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=layout)

pairs = [(i, j) for i in range(NTR) for j in range(i + 1, NTR)] + [(NTR + t, j) for t in range(NTE) for j in range(NTR)]
items = []
for (L, f) in SETTINGS:
    for (i, j) in pairs:
        items.append(dict(L=L, fold=f, i=i, j=j, qc=kernel_circuit(Xall[i], Xall[j], BETA, L, f)))
isa = pm.run([it["qc"] for it in items])
for it, c in zip(items, isa):
    assert c.count_ops().get("cz", 0) == expected_cz(it["L"], it["fold"]), (it["L"], it["fold"], dict(c.count_ops()))
print("circuits", len(items), "CZ assertions OK", round(time.time() - t0, 1), "s")

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(backend)
sim_n, sim_i = AerSimulator(noise_model=nm), AerSimulator()


def run_aer(sim, circs, shots, seed):
    r = sim.run(circs, shots=shots, seed_simulator=seed).result()
    return [r.get_counts(k).get("0" * NQ, 0) / shots for k in range(len(circs))]


def exact_pair(it):
    qc = kernel_circuit(Xall[it["i"]], Xall[it["j"]], BETA, it["L"], it["fold"]); qc.remove_final_measurements()
    return float(abs(Statevector(qc).data[0]) ** 2)


aer = dict(exact=[exact_pair(it) for it in items], noisy_8192=run_aer(sim_n, isa, 8192, 5), ideal_1024=run_aer(sim_i, isa, 1024, 1))
print("Aer ideal vs exact max dev", float(np.max(np.abs(np.array(aer["ideal_1024"]) - np.array(aer["exact"])))))
meta = dict(backend=BACKEND, layout=layout, beta=BETA, split=SP, n_train=NTR, n_test=NTE, noiseless_test_auc_L2=a0, noiseless_offdiag_std=sd, noiseless_loo_frac=loo,
            snapshot=snap, settings=SETTINGS, shots={f"{L},{f}": SHOTS[(L, f)] for (L, f) in SETTINGS}, pairs=pairs,
            meta=[{k: v for k, v in it.items() if k != "qc"} for it in items], Xall=Xall.tolist(), yall=yall.tolist(),
            cz_counts=[c.count_ops().get("cz", 0) for c in isa], depth=[c.depth() for c in isa], aer=aer,
            est_duration_us=[float(c.estimate_duration(backend.target, unit="s")) * 1e6 for c in isa[::len(pairs)]])
json.dump(meta, open(os.path.join(HW, "prep_meta.json"), "w"), indent=1, default=float)
pickle.dump(dict(isa=isa, n_pairs=len(pairs)), open(os.path.join(HW, "isa_circuits.pkl"), "wb"))
print("prep done", round(time.time() - t0, 1), "s; est durations (us)", [round(d, 1) for d in meta["est_duration_us"]])
