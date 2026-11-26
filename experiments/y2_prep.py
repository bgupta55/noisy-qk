"""Y2 + Y7a prep (NO QPU time), ibm_marrakesh, chain [52,53,54,55] (same as X4), X4 split (14 points) and beta.
Y2: HW4 repeat: idle_kernel_circuit (L=1, delay between U(x) and U(x')^dag), delays {0,100,200} us, both orders, 3 pairs chosen by the V1 rule (idle AD model, T1=250 us, tau=200 us,
    largest |K_ij - K_ji|) over all 91 pairs of the 14 points. Selection uses no hardware data. Predictions (a) per-qubit AD+PD from CALIBRATION T1,T2 read now; (b) Aer backend noise model.
Y7a: Hahn echo T2 on the 4 qubits in parallel: prep +/-, delay/2, X, delay/2, basis change, measure; total delays {10,30,60,120,240} us; 1024 shots.
Writes results/hardware_y2/{prep_meta.json,isa_circuits.pkl}. Pre-registered in results/PREREGISTRATION.md (ROUND Y)."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from nqk.hw import *
from nqk.channels import kraus
from nqk.kernel import apply_noise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "hardware_y2")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
x4 = json.load(open(os.path.join(ROOT, "results", "hardware_x4", "prep_meta.json")))
BACKEND, LAYOUT, BETA = x4["backend"], x4["layout"], x4["beta"]
Xall = np.array(x4["Xall"]); P = len(Xall)
DELAYS = [0, 100, 200]
ECHO = [10, 30, 60, 120, 240]
SH_Y2, SH_ECHO = 2048, 1024
svc = get_service()
backend = svc.backend(BACKEND)
snap = snapshot(backend, LAYOUT)
print("snapshot", json.dumps(snap["qubit_props"]))
T1 = [snap["qubit_props"][q]["T1_us"] for q in LAYOUT]
T2 = [snap["qubit_props"][q]["T2_us"] for q in LAYOUT]


def U_of(x):
    q = QuantumCircuit(NQ); slice_gates(q, x, BETA, 1); return Operator(q).data


Us = [U_of(x) for x in Xall]


def idle_k(i, j, tau, t1, t2):
    """K(i,j) = |<0|U_j^dag N(U_i |0>)|0>|^2 with per-qubit AD(gamma_q) then PD(lphi_q) during the idle gap tau (us)."""
    g = [1 - np.exp(-tau / a) for a in t1]
    lp = [max(0.0, 1 - np.exp(-2 * tau * (1 / b - 1 / (2 * a)))) for a, b in zip(t1, t2)]
    rho = np.zeros((16, 16), complex); rho[0, 0] = 1
    rho = Us[i] @ rho @ Us[i].conj().T
    # NB: Qiskit Operator is little-endian (qubit 0 = least significant) while apply_noise indexes qubit 0 as the most significant bit;
    # per-qubit lists are therefore reversed (checked: AD on list index 3 decays circuit qubit 0). Uniform lists (V1) are unaffected.
    rho = apply_noise(rho[None], ("local_pq", [kraus("ad", gg) for gg in g[::-1]]), NQ)
    rho = apply_noise(rho, ("local_pq", [kraus("pd", ll) for ll in lp[::-1]]), NQ)[0]
    rho = Us[j].conj().T @ rho @ Us[j]
    return float(rho[0, 0].real)


cand = []
for i in range(P):
    for j in range(i + 1, P):
        a = abs(idle_k(i, j, 200.0, [250.0] * 4, [250.0 * 2] * 4) - idle_k(j, i, 200.0, [250.0] * 4, [250.0 * 2] * 4))
        cand.append((a, i, j))
cand.sort(reverse=True)
pairs4 = [(i, j) for _, i, j in cand[:3]]
print("Y2 pairs (selection model, no HW data):", pairs4, [round(c[0], 4) for c in cand[:3]])

items = []
for tau in DELAYS:
    for (i, j) in pairs4:
        for (a, b) in ((i, j), (j, i)):
            items.append(dict(kind="hw4", tau=tau, i=a, j=b, qc=idle_kernel_circuit(Xall[a], Xall[b], BETA, tau)))
n4 = len(items)
for tot in ECHO:
    for prep in ("+", "-"):
        qc = QuantumCircuit(NQ, NQ)
        for q in range(NQ):
            if prep == "-":
                qc.x(q)
            qc.h(q)
        qc.barrier()
        qc.delay(int(tot * 500), range(NQ), unit="ns"); qc.barrier()
        for q in range(NQ):
            qc.x(q)
        qc.barrier()
        qc.delay(int(tot * 500), range(NQ), unit="ns"); qc.barrier()
        for q in range(NQ):
            qc.h(q)
        qc.measure(range(NQ), range(NQ))
        items.append(dict(kind="echo", tau=tot, prep=prep, qc=qc))
pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=LAYOUT)
isa = pm.run([it["qc"] for it in items])
for it, c in zip(items, isa):
    if it["kind"] == "hw4":
        assert c.count_ops().get("cz", 0) == 12, (it["tau"], dict(c.count_ops()))
    else:
        assert c.count_ops().get("cz", 0) == 0
print("circuits", len(items), "(hw4", n4, ", echo", len(items) - n4, ") CZ assertions OK", round(time.time() - t0, 1), "s")

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
nm = NoiseModel.from_backend(backend)
sim_n, sim_i = AerSimulator(noise_model=nm), AerSimulator()


def run_aer(sim, shots, seed, rng_idx):
    r = sim.run([isa[k] for k in rng_idx], shots=shots, seed_simulator=seed).result()
    return [r.get_counts(m) for m in range(len(rng_idx))]


idx4 = list(range(n4))
aer = dict(hw4_ideal=[c.get("0" * NQ, 0) / SH_Y2 for c in run_aer(sim_i, SH_Y2, 1, idx4)],
           hw4_noisy_8192=[c.get("0" * NQ, 0) / 8192 for c in run_aer(sim_n, 8192, 5, idx4)])
meta4 = [{k: v for k, v in it.items() if k != "qc"} for it in items]
pred = []
for m in meta4[:n4]:
    pred.append(dict(K_idle_cal=idle_k(m["i"], m["j"], float(m["tau"]), T1, T2), K_exact_ideal=None))
# exact noiseless K for reference
for m, p in zip(meta4[:n4], pred):
    p["K_exact_ideal"] = float(abs((Us[m["j"]].conj().T @ Us[m["i"]])[0, 0]) ** 2)
# est duration of the hw4 circuits (us) from the transpiled schedule (tau=0 ... 200)
est = [float(isa[k].estimate_duration(backend.target, unit="s")) * 1e6 for k in range(len(isa))]
meta = dict(backend=BACKEND, layout=LAYOUT, beta=BETA, snapshot=snap, T1_cal_us=T1, T2_cal_us=T2, delays=DELAYS, echo_delays=ECHO, pairs4=pairs4, selection_asym=[float(c[0]) for c in cand[:3]],
            shots_hw4=SH_Y2, shots_echo=SH_ECHO, n4=n4, n_echo=len(items) - n4, meta=meta4, pred_idle_cal=pred, aer=aer, est_duration_us=est, Xall=Xall.tolist(),
            source_x4_split=x4["split"], cz_counts=[c.count_ops().get("cz", 0) for c in isa])
json.dump(meta, open(os.path.join(OUT, "prep_meta.json"), "w"), indent=1, default=float)
pickle.dump(dict(isa=isa, n4=n4), open(os.path.join(OUT, "isa_circuits.pkl"), "wb"))
print("prep done", round(time.time() - t0, 1), "s; QPU estimate (V1 rate 16 s / 18 circuits x 2048 shots):", round(16 * (n4 / 18) + 0.3e-3 * (len(items) - n4) * SH_ECHO, 1), "s")
