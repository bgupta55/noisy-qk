"""Y2 + Y7a offline analysis (no QPU). Reads results/hardware_y2/{prep_meta.json,raw_y2.json}; compares with V1 HW4 (ibm_kingston, results/HW_analysis.json).
Pre-registered gates (PREREGISTRATION.md ROUND Y, Y2): (i) tau=0 |asym| <= 3 SE for each pair; (ii) mean |asym| increases 0<100<200 and every pair x tau>0 has the sign of prediction (a);
(iii) mean z = |asym|/SE at tau=200 >= 5. No gate on magnitude; RMS of (asym_hw - asym_pred)/SE reported. Y7a: Hahn-echo T2 per qubit. Writes results/Y2_Y7a_marrakesh.json."""
from common import *

t0 = time.time()
HY = os.path.join(RES, "hardware_y2")
meta = json.load(open(os.path.join(HY, "prep_meta.json")))
raw = json.load(open(os.path.join(HY, "raw_y2.json")))
n4, NQ = meta["n4"], 4
M = meta["meta"]
shots = np.array(raw["shots"], float)
rng = np.random.default_rng(2030)
counts = raw["counts"]
assert len(counts) == len(M)


def p0(c, s):
    return c.get("0" * NQ, 0) / s


# ------------------------------------------------------------------ Y2
p4 = np.array([p0(counts[k], shots[k]) for k in range(n4)])
pred_cal = np.array([p["K_idle_cal"] for p in meta["pred_idle_cal"]])
aer_n = np.array(meta["aer"]["hw4_noisy_8192"])
aer_i = np.array(meta["aer"]["hw4_ideal"])
exact = np.array([p["K_exact_ideal"] for p in meta["pred_idle_cal"]])
rows = []
for (i, j) in [tuple(p) for p in meta["pairs4"]]:
    for tau in meta["delays"]:
        k1 = next(k for k in range(n4) if M[k]["tau"] == tau and M[k]["i"] == i and M[k]["j"] == j)
        k2 = next(k for k in range(n4) if M[k]["tau"] == tau and M[k]["i"] == j and M[k]["j"] == i)
        a1, a2 = p4[k1], p4[k2]
        se = float(np.sqrt(a1 * (1 - a1) / shots[k1] + a2 * (1 - a2) / shots[k2]))
        rows.append(dict(pair=[i, j], tau_us=tau, K_ij=float(a1), K_ji=float(a2), asym_hw=float(a1 - a2), asym_se=se,
                         asym_pred_cal=float(pred_cal[k1] - pred_cal[k2]), asym_aer_noise=float(aer_n[k1] - aer_n[k2]), asym_exact=float(exact[k1] - exact[k2]),
                         K_ij_pred_cal=float(pred_cal[k1]), K_ji_pred_cal=float(pred_cal[k2])))
R = rows
tau0 = [r for r in R if r["tau_us"] == 0]
g1 = dict(max_abs_over_se=float(max(abs(r["asym_hw"]) / r["asym_se"] for r in tau0)), passed=bool(all(abs(r["asym_hw"]) <= 3 * r["asym_se"] for r in tau0)))
mean_abs = {t: float(np.mean([abs(r["asym_hw"]) for r in R if r["tau_us"] == t])) for t in meta["delays"]}
signs = [bool(np.sign(r["asym_hw"]) == np.sign(r["asym_pred_cal"])) for r in R if r["tau_us"] > 0]
g2 = dict(mean_abs_by_tau=mean_abs, increasing=bool(mean_abs[0] < mean_abs[100] < mean_abs[200]), n_sign_match=int(sum(signs)), n=len(signs), all_signs_match=bool(all(signs)))
g2["passed"] = bool(g2["increasing"] and g2["all_signs_match"])
z200 = float(np.mean([abs(r["asym_hw"]) / r["asym_se"] for r in R if r["tau_us"] == 200]))
g3 = dict(mean_z_tau200=z200, passed=bool(z200 >= 5))
res_cal = [(r["asym_hw"] - r["asym_pred_cal"]) / r["asym_se"] for r in R if r["tau_us"] > 0]
res_aer = [(r["asym_hw"] - r["asym_aer_noise"]) / r["asym_se"] for r in R if r["tau_us"] > 0]
mag = dict(rms_z_vs_pred_cal=float(np.sqrt(np.mean(np.square(res_cal)))), rms_z_vs_aer=float(np.sqrt(np.mean(np.square(res_aer)))),
           mean_ratio_hw_over_pred_cal=float(np.mean([r["asym_hw"] / r["asym_pred_cal"] for r in R if r["tau_us"] > 0])),
           mean_ratio_hw_over_aer=float(np.mean([r["asym_hw"] / r["asym_aer_noise"] for r in R if r["tau_us"] > 0])))
V1 = json.load(open(os.path.join(RES, "HW_analysis.json")))["results"]["HW4"]
v1 = dict(mean_abs_tau100=V1["mean_abs_asym_tau100"], mean_abs_tau200=V1["mean_abs_asym_tau200"], mean_z_tau200=V1["mean_z_tau200"], max_abs_tau0=V1["max_abs_asym_tau0"], signs_match_all=V1["sign_matches_idle_pred_all_tau_gt0"])
for r in R:
    print("HW4", r["pair"], r["tau_us"], "asym hw %.3f+-%.3f  pred_cal %.3f  aer %.3f  exact %.3f" % (r["asym_hw"], r["asym_se"], r["asym_pred_cal"], r["asym_aer_noise"], r["asym_exact"]))
print("gates", g1, g2, g3, mag, v1)

# ------------------------------------------------------------------ Y7a: Hahn echo
echo = [(M[k]["tau"], M[k]["prep"], k) for k in range(n4, len(M))]
taus = sorted(set(t for t, _, _ in echo))


def p1_qubit(c, q, s):
    n1 = sum(v for key, v in c.items() if key[::-1][q] == "1")
    return n1 / s


E = np.zeros((NQ, len(taus))); Ese = np.zeros((NQ, len(taus)))
for q in range(NQ):
    for a, t in enumerate(taus):
        kp = next(k for (tt, pp, k) in echo if tt == t and pp == "+"); km = next(k for (tt, pp, k) in echo if tt == t and pp == "-")
        pp_, pm_ = p1_qubit(counts[kp], q, shots[kp]), p1_qubit(counts[km], q, shots[km])
        E[q, a] = pm_ - pp_
        Ese[q, a] = np.sqrt(pp_ * (1 - pp_) / shots[kp] + pm_ * (1 - pm_) / shots[km])
tt = np.array(taus, float)
echo_rows = []
for q in range(NQ):
    ok = E[q] > 2 * Ese[q]
    def fit(Ev):
        m, c = np.polyfit(tt[ok], np.log(Ev[ok]), 1)
        return -1.0 / m, float(np.exp(c))
    if ok.sum() >= 3:
        T2e, A = fit(E[q])
        bs = []
        for _ in range(400):
            Eb = E[q] + rng.normal(0, Ese[q])
            if (Eb[ok] > 0).all():
                bs.append(fit(Eb)[0])
        sd = float(np.std(bs)) if len(bs) > 20 else None
    else:
        T2e, A, sd = None, None, None
    echo_rows.append(dict(qubit=meta["layout"][q], T2_echo_us=T2e, amplitude=A, T2_echo_se_boot=sd, n_points_used=int(ok.sum()), E=E[q].tolist(), E_se=Ese[q].tolist(),
                          T2_cal_us=meta["T2_cal_us"][q], T1_cal_us=meta["T1_cal_us"][q], E_over_se_min=float((E[q] / Ese[q]).min())))
    print("echo q", meta["layout"][q], "T2e", T2e, "+-", sd, "cal T2", meta["T2_cal_us"][q], "E", np.round(E[q], 3))

# ------------------------------------------------------------------ prediction (a') with echo T2 (T2e capped at 2*T1) vs calibration T2
import sys as _s
_s.path.insert(0, os.path.join(ROOT, "src"))
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from nqk.hw import slice_gates
from nqk.channels import kraus
from nqk.kernel import apply_noise
Xall = np.array(meta["Xall"])


def Uof(x):
    q = QuantumCircuit(NQ); slice_gates(q, x, meta["beta"], 1); return Operator(q).data


def idle_k(i, j, tau, t1, t2):
    g = [1 - np.exp(-tau / a) for a in t1]
    lp = [max(0.0, 1 - np.exp(-2 * tau * (1 / b - 1 / (2 * a)))) for a, b in zip(t1, t2)]
    rho = np.zeros((16, 16), complex); rho[0, 0] = 1
    Ui, Uj = Uof(Xall[i]), Uof(Xall[j])
    rho = Ui @ rho @ Ui.conj().T
    rho = apply_noise(rho[None], ("local_pq", [kraus("ad", x) for x in g[::-1]]), NQ)
    rho = apply_noise(rho, ("local_pq", [kraus("pd", x) for x in lp[::-1]]), NQ)[0]
    return float((Uj.conj().T @ rho @ Uj)[0, 0].real)


T2e_list = [r["T2_echo_us"] if r["T2_echo_us"] else r["T2_cal_us"] for r in echo_rows]
T2use = [min(t2, 2 * t1) for t2, t1 in zip(T2e_list, meta["T1_cal_us"])]
res_echo = []
for r in R:
    if r["tau_us"] == 0:
        continue
    i, j = r["pair"]
    pa = idle_k(i, j, float(r["tau_us"]), meta["T1_cal_us"], T2use) - idle_k(j, i, float(r["tau_us"]), meta["T1_cal_us"], T2use)
    r["asym_pred_echoT2"] = float(pa)
    res_echo.append((r["asym_hw"] - pa) / r["asym_se"])
mag["rms_z_vs_pred_echoT2"] = float(np.sqrt(np.mean(np.square(res_echo))))
mag["n_sign_match_echoT2"] = int(sum(np.sign(r["asym_hw"]) == np.sign(r["asym_pred_echoT2"]) for r in R if r["tau_us"] > 0))
print("rms z: cal %.2f echo %.2f aer %.2f" % (mag["rms_z_vs_pred_cal"], mag["rms_z_vs_pred_echoT2"], mag["rms_z_vs_aer"]))

save("Y2_Y7a_marrakesh", dict(source="results/hardware_y2/raw_y2.json", seed=2030), dict(
    job=raw["job_id"], backend=raw["backend"], layout=raw["layout"], qpu_seconds=raw["usage_seconds"], account_before=raw["account_usage_before"], account_after=raw["account_usage_after"], utc=raw["utc"],
    calibration=meta["snapshot"], pairs=meta["pairs4"], selection_asym=meta["selection_asym"], shots_hw4=meta["shots_hw4"], rows=R, gates=dict(i_tau0=g1, ii_monotone_and_sign=g2, iii_z=g3),
    magnitude=mag, v1_kingston=v1, echo=dict(delays_us=taus, shots=meta["shots_echo"], rows=echo_rows, T2_used_for_pred_echoT2=T2use)), t0)
