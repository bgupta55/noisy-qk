"""Offline analysis of the real-QPU data (HW1-HW5). Reads results/hardware/{prep_meta.json,raw_*.json,isa_circuits.pkl}. No QPU time used
(a read-only service connection is used only for gate durations). Writes results/HW_analysis.json."""
from common import *
import pickle
from nqk.krr import ridge, cv_lam, center_train, cnr_lambda
from nqk.metrics import auc
from nqk.channels import kraus
from nqk.hw import expected_cz, NQ

t0 = time.time()
HW = os.path.join(RES, "hardware")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
R = {k: json.load(open(os.path.join(HW, f"raw_{k}.json"))) for k in ("hw1", "hw2", "hw2r", "hw4")}
out = {}
rng = np.random.default_rng(2026)

# ================================================================ HW1: 1-qubit idle process tomography (4 qubits in parallel)
BLOCH_IN = {"0": (0, 0, 1), "1": (0, 0, -1), "+": (1, 0, 0), "+i": (0, 1, 0)}
MEAS = ["X", "Y", "Z"]


def marg_exp(counts, q, shots):
    n1 = sum(v for k, v in counts.items() if k[::-1][q] == "1")
    return 1 - 2 * n1 / shots


h1 = {}  # (delay, qubit) -> (T 3x3, t 3)
shots1 = R["hw1"]["shots"]
exp1 = {}
for it, c in zip(meta["hw1_meta"], R["hw1"]["counts"]):
    for q in range(NQ):
        exp1[(it["delay"], q, it["prep"], it["meas"])] = marg_exp(c, q, shots1)
for d in (0, 100, 200):
    for q in range(NQ):
        o = {p: np.array([exp1[(d, q, p, m)] for m in MEAS]) for p in BLOCH_IN}
        t = (o["0"] + o["1"]) / 2
        T = np.zeros((3, 3))
        T[:, 2] = (o["0"] - o["1"]) / 2
        T[:, 0] = o["+"] - t
        T[:, 1] = o["+i"] - t
        h1[(d, q)] = (T, t)
hw1_rows = []
for d in (100, 200):
    for q in range(NQ):
        T0, t0v = h1[(0, q)]
        Td, tdv = h1[(d, q)]
        TC = Td @ np.linalg.inv(T0)          # SPAM-removed idle linear part
        tC = tdv - TC @ t0v                  # SPAM-removed translation
        u = float(np.trace(TC.T @ TC) / 3)
        kap = float(np.linalg.svd(TC, compute_uv=False)[0])
        r = float((3 - np.trace(TC)) / 6)
        zz, xx, yy = TC[2, 2], TC[0, 0], TC[1, 1]
        T1 = -d / np.log(zz) if 0 < zz < 1 else np.nan
        dxy = abs(np.linalg.det(TC[:2, :2]))      # xy block may contain a rotation (detuning); |det| = exp(-2t/T2) is rotation-invariant
        T2 = -d / np.log(np.sqrt(dxy)) if 0 < dxy < 1 else np.nan
        rot = float(np.degrees(np.arctan2(TC[1, 0] - TC[0, 1], TC[0, 0] + TC[1, 1])))
        hw1_rows.append(dict(delay_us=d, qubit=meta["layout"][q], diag=[float(xx), float(yy), float(zz)], t_z=float(tC[2]), t_xy=[float(tC[0]), float(tC[1])],
                             u=u, kappa=kap, r=r, xy_rotation_deg=rot, T1_est_us=float(T1), T2_est_us=float(T2),
                             T1_cal_us=meta["snapshot"]["qubit_props"][str(meta["layout"][q])]["T1_us"],
                             T2_cal_us=meta["snapshot"]["qubit_props"][str(meta["layout"][q])]["T2_us"],
                             t_z_pred=float(1 - np.exp(-d / meta["snapshot"]["qubit_props"][str(meta["layout"][q])]["T1_us"])),
                             spam_T0_diag=[float(h1[(0, q)][0][i, i]) for i in range(3)]))
out["HW1"] = dict(rows=hw1_rows, shots=shots1, job=R["hw1"]["job_id"], qpu_s=R["hw1"]["usage_seconds"],
                  mean_kappa_200=float(np.mean([r["kappa"] for r in hw1_rows if r["delay_us"] == 200])),
                  mean_u_200=float(np.mean([r["u"] for r in hw1_rows if r["delay_us"] == 200])),
                  mean_tz_200=float(np.mean([r["t_z"] for r in hw1_rows if r["delay_us"] == 200])),
                  mean_T1_est_200=float(np.nanmean([r["T1_est_us"] for r in hw1_rows if r["delay_us"] == 200])),
                  mean_T2_est_200=float(np.nanmean([r["T2_est_us"] for r in hw1_rows if r["delay_us"] == 200])))
T1eff = out["HW1"]["mean_T1_est_200"]
T2eff = out["HW1"]["mean_T2_est_200"]
print("HW1: T1_est %.0f T2_est %.0f us, kappa200 %.3f u200 %.3f tz200 %.3f" % (T1eff, T2eff, out["HW1"]["mean_kappa_200"], out["HW1"]["mean_u_200"], out["HW1"]["mean_tz_200"]))

# ================================================================ HW2: kernels
pairs = [tuple(p) for p in meta["pairs"]]
settings = [tuple(s) for s in meta["settings"]]
NP = len(pairs)
shots2 = R["hw2"]["shots"]
C2 = R["hw2"]["counts"]
p_hw = np.array([c.get("0000", 0) for c in C2], float)          # counts of all-zero
ideal = np.array(meta["aer"]["hw2_exact"])
nzm = np.array(meta["aer"]["hw2_noisy_8192"])
layers = [2 * L * (1 + 2 * f) for L, f in settings]
ncz = [expected_cz(L, f) for L, f in settings]
NTR, NTE = meta["n_train"], meta["n_test"]
yall = np.array(meta["yall"], float)


def setting_slice(k):
    return slice(NP * k, NP * (k + 1))


def fit_aff(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


rows = []
for k, (L, f) in enumerate(settings):
    sl = setting_slice(k)
    ph = p_hw[sl] / shots2
    a, b = fit_aff(ideal[sl], ph)
    an, bn = fit_aff(ideal[sl], nzm[sl])
    # bootstrap over circuit-level resampling of counts (binomial) -> slope SE
    bs = [fit_aff(ideal[sl], rng.binomial(shots2, ph) / shots2)[0] for _ in range(200)]
    resid = ph - (a * ideal[sl] + b)
    rows.append(dict(L=L, fold=f, noise_factor=1 + 2 * f, noisy_layers=layers[k], cz=ncz[k], s_hw=a, s_hw_se=float(np.std(bs)), offset_hw=b,
                     s_aer_noise=an, offset_aer=bn, rmse_hw_vs_affine=float(np.sqrt((resid ** 2).mean())),
                     mean_abs_dev_hw_vs_ideal=float(np.abs(ph - ideal[sl]).mean()), mean_abs_dev_aer_vs_ideal=float(np.abs(nzm[sl] - ideal[sl]).mean())))
out["HW2_retention"] = rows
lay = np.array(layers, float)
s_hw = np.array([r["s_hw"] for r in rows])
s_nz = np.array([r["s_aer_noise"] for r in rows])
cz = np.array(ncz, float)


def fit_log(x, s):
    m, c = np.polyfit(x, np.log(s), 1)
    # slope through origin too
    m0 = float((x * np.log(s)).sum() / (x * x).sum())
    return float(m), float(c), m0


m_hw, c_hw, m0_hw = fit_log(lay, s_hw)
m_nz, c_nz, m0_nz = fit_log(lay, s_nz)
bs_m = []
for _ in range(200):
    sb = [fit_aff(ideal[setting_slice(k)], rng.binomial(shots2, p_hw[setting_slice(k)] / shots2 / 1.0 * 1.0) / shots2)[0] if False else
          fit_aff(ideal[setting_slice(k)], rng.binomial(shots2, p_hw[setting_slice(k)] / shots2) / shots2)[0] for k in range(len(settings))]
    bs_m.append(np.polyfit(lay, np.log(sb), 1)[0])
fits = dict(hw=dict(lnrho_per_layer=m_hw, intercept=c_hw, lnrho_through_origin=m0_hw, rho_eff=float(np.exp(m_hw)), rho_eff_origin=float(np.exp(m0_hw)), slope_se=float(np.std(bs_m))),
            aer_backend_noise=dict(lnrho_per_layer=m_nz, intercept=c_nz, lnrho_through_origin=m0_nz, rho_eff=float(np.exp(m_nz)), rho_eff_origin=float(np.exp(m0_nz))))
# only L=2 settings: noise-factor scan (pure noise-amplification, same ideal kernel)
sel = [i for i, (L, f) in enumerate(settings) if L == 2]
mL2 = np.polyfit(lay[sel], np.log(s_hw[sel]), 1)[0]
fits["hw_L2_only"] = dict(lnrho_per_layer=float(mL2), rho_eff=float(np.exp(mL2)), points=len(sel))
fits["aer_L2_only"] = dict(lnrho_per_layer=float(np.polyfit(lay[sel], np.log(s_nz[sel]), 1)[0]))
# (c) pure depolarizing per CZ: s = (1-q)^cz  (affine fit in cz)
mc = float((cz * np.log(s_hw)).sum() / (cz * cz).sum())
fits["depolarizing_per_cz_fit"] = dict(q_per_cz=float(1 - np.exp(mc)), pred_s=[float(np.exp(mc * c)) for c in cz], obs_s=s_hw.tolist(), max_abs_dev=float(np.max(np.abs(np.exp(mc * cz) - s_hw))))
out["HW2_fits"] = fits
print("HW2 s_hw", np.round(s_hw, 3), "s_aer", np.round(s_nz, 3), "rho_eff hw %.4f aer %.4f" % (np.exp(m_hw), np.exp(m_nz)))

# ---- (a) HW1-derived idle prediction: T1/T2-only Aer noise model with HW1-fitted T1,T2 on the SAME transpiled circuits
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from nqk.hw import get_service
isa = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))
backend = get_service().backend(meta["backend"])
tg = backend.target
lay_q = meta["layout"]
T2use = min(T2eff, 2 * T1eff) if np.isfinite(T2eff) else T1eff
nm = NoiseModel()
for q in lay_q:
    for g in ("sx", "x"):
        d = tg[g][(q,)].duration
        nm.add_quantum_error(thermal_relaxation_error(T1eff * 1e-6, T2use * 1e-6, d), g, [q])
for a_, b_ in zip(lay_q[:-1], lay_q[1:]):
    for (x_, y_) in ((a_, b_), (b_, a_)):
        if (x_, y_) in tg["cz"]:
            d = tg["cz"][(x_, y_)].duration
            e1 = thermal_relaxation_error(T1eff * 1e-6, T2use * 1e-6, d)
            nm.add_quantum_error(e1.tensor(e1), "cz", [x_, y_])
sim = AerSimulator(noise_model=nm)
res = sim.run(isa["isa2"], shots=8192, seed_simulator=11).result()
p_idle = np.array([res.get_counts(i).get("0000", 0) / 8192 for i in range(len(isa["isa2"]))])
s_idle = np.array([fit_aff(ideal[setting_slice(k)], p_idle[setting_slice(k)])[0] for k in range(len(settings))])
m_id = np.polyfit(lay, np.log(s_idle), 1)[0]
fits["idle_T1T2_only_prediction"] = dict(T1_us=T1eff, T2_us=T2use, s=s_idle.tolist(), lnrho_per_layer=float(m_id), rho_eff=float(np.exp(m_id)))
dur = [float(c.estimate_duration(tg, unit="s")) * 1e6 if hasattr(c, "estimate_duration") else None for c in isa["isa2"][::NP]]
fits["circuit_duration_us_per_setting"] = dur
print("idle-only s", np.round(s_idle, 3), "durations(us)", [round(d_, 1) if d_ else d_ for d_ in dur])

# ---- kernel matrices (symmetrised) per setting; diag from affine calibration
revc = R["hw2r"]["counts"]
rev_p = np.array([c.get("0000", 0) / R["hw2r"]["shots"] for c in revc])
rev_pair = [(it["i"], it["j"]) for it in meta["hw2rev_meta"]]
k22 = settings.index((2, 0))
fw = [p_hw[setting_slice(k22)][pairs.index((j, i))] / shots2 if (j, i) in pairs else None for (i, j) in rev_pair]  # forward values of the swapped circuits
fwd = np.array([p_hw[setting_slice(k22)][pairs.index((j, i))] / shots2 for (i, j) in rev_pair])  # K[j,i]; reverse circuit is K[i,j] with (i,j)=rev_pair entry
nz_rev = np.array(meta["aer"]["hw2rev_noisy_8192"])
nz_fwd = np.array([nzm[setting_slice(k22)][pairs.index((j, i))] for (i, j) in rev_pair])
se_diff = np.sqrt(fwd * (1 - fwd) / shots2 + rev_p * (1 - rev_p) / R["hw2r"]["shots"])
diff = rev_p - fwd
out["HW2_asymmetry_L2_fold0"] = dict(n=len(diff), mean_abs_diff_hw=float(np.abs(diff).mean()), mean_abs_diff_aer_noise=float(np.abs(nz_rev - nz_fwd).mean()),
                                      mean_shot_se_of_diff=float(se_diff.mean()), chi2=float(((diff / se_diff) ** 2).sum()), dof=len(diff),
                                      rms_z=float(np.sqrt(((diff / se_diff) ** 2).mean())), max_abs_diff_hw=float(np.abs(diff).max()),
                                      mean_signed_diff_rev_minus_fwd=float(diff.mean()), job_rev=R["hw2r"]["job_id"])


def kernel_mats(counts_vec, shots, k, a_diag=None):
    sl = setting_slice(k)
    v = counts_vec[sl] / shots
    K = np.zeros((NTR + NTE, NTR))
    Ktr = np.zeros((NTR, NTR))
    for (i, j), val in zip(pairs, v):
        if i < NTR and j < NTR:
            Ktr[i, j] = Ktr[j, i] = val
        else:
            K[i - NTR + NTR, j] = val
    return Ktr, K[NTR:, :]


def full_kernel(counts_vec, shots, k):
    Ktr, Kte = kernel_mats(counts_vec, shots, k)
    a, b = fit_aff(ideal[setting_slice(k)], counts_vec[setting_slice(k)] / shots)
    np.fill_diagonal(Ktr, a * 1.0 + b)
    return Ktr, Kte


def pooled_heldout(Ktr, Kte, ytr_, lam_fn):
    """6 leave-one-out predictions (train block) + 3 test predictions; returns pooled AUC, test AUC, LOO AUC."""
    n = NTR
    ytr = yall[:n]
    yte = yall[n:]
    hs, ys = [], []
    loo_h, loo_y = [], []
    for t in range(n):
        tr = [i for i in range(n) if i != t]
        lam = lam_fn(Ktr[np.ix_(tr, tr)], ytr[tr], len(tr))
        h = ridge(Ktr[np.ix_(tr, tr)], Ktr[np.ix_([t], tr)], ytr[tr], lam, lam_fn.center)
        loo_h.append(h[0]); loo_y.append(ytr[t])
    lam = lam_fn(Ktr, ytr, n)
    ht = ridge(Ktr, Kte, ytr, lam, lam_fn.center)
    pooled = auc(np.concatenate([loo_h, ht]), np.concatenate([loo_y, yte]))
    return float(pooled), float(auc(ht, yte)), float(auc(np.array(loo_h), np.array(loo_y)))


class FixedLam:
    center = False

    def __init__(self, lam, center=False):
        self.lam = lam
        self.center = center

    def __call__(self, Ktr, y, n):
        return self.lam


class CVLam:
    def __init__(self, center):
        self.center = center

    def __call__(self, Ktr, y, n):
        return cv_lam(Ktr, y, self.center, folds=min(3, n))


class CNRLam:
    center = True

    def __init__(self, Rtr, S):
        self.Rtr, self.S = Rtr, S

    def __call__(self, Ktr, y, n):
        # reference restricted to the same index set via shape match
        idx = self.idx if hasattr(self, "idx") else None
        Rtr = self.Rtr if Ktr.shape == self.Rtr.shape else self.Rtr[:Ktr.shape[0], :Ktr.shape[0]]
        return cnr_lambda(Ktr, Rtr, n, self.S, 0.1, 1.0)[0]


def cnr_pooled(Ktr, Kte, Rtr, S):
    n = NTR
    ytr = yall[:n]
    yte = yall[n:]
    loo_h, loo_y = [], []
    for t in range(n):
        tr = [i for i in range(n) if i != t]
        lam = cnr_lambda(Ktr[np.ix_(tr, tr)], Rtr[np.ix_(tr, tr)], len(tr), S, 0.1, 1.0)[0]
        h = ridge(Ktr[np.ix_(tr, tr)], Ktr[np.ix_([t], tr)], ytr[tr], lam, True)
        loo_h.append(h[0]); loo_y.append(ytr[t])
    lam = cnr_lambda(Ktr, Rtr, n, S, 0.1, 1.0)[0]
    ht = ridge(Ktr, Kte, ytr, lam, True)
    return float(auc(np.concatenate([loo_h, ht]), np.concatenate([loo_y, yte]))), float(auc(ht, yte)), float(auc(np.array(loo_h), np.array(loo_y))), float(lam)


# noiseless reference AUC (exact kernel, same 9 points) and per-setting hardware AUCs
ideal_vec = ideal
kk = {}
rowsA = []
ref_k = settings.index((2, 0))
for k, (L, f) in enumerate(settings):
    Ktr, Kte = full_kernel(p_hw, shots2, k)
    Ktr_i, Kte_i = full_kernel(ideal * shots2, shots2, k)
    Ktr_i = Ktr_i.copy(); np.fill_diagonal(Ktr_i, 1.0)
    Rtr, _ = full_kernel(p_hw, shots2, ref_k if L == 2 else settings.index((L, 0)))
    row = dict(L=L, fold=f, noise_factor=1 + 2 * f)
    row["ideal_fixed"] = pooled_heldout(Ktr_i, Kte_i, None, FixedLam(0.1, False))
    row["hw_fixed"] = pooled_heldout(Ktr, Kte, None, FixedLam(0.1, False))
    row["hw_fixed_centred"] = pooled_heldout(Ktr, Kte, None, FixedLam(0.1, True))
    row["hw_cv_unc"] = pooled_heldout(Ktr, Kte, None, CVLam(False))
    row["hw_cv_cen"] = pooled_heldout(Ktr, Kte, None, CVLam(True))
    row["hw_cnr"] = cnr_pooled(Ktr, Kte, Rtr, shots2)
    a_ratio = float(np.trace(center_train(Ktr)) / np.trace(center_train(Rtr)))
    row["a_L_hw"] = a_ratio
    rowsA.append(row)
    kk[k] = (Ktr, Kte)
out["HW2_auc_(pooled,test3,loo6)"] = rowsA
for r in rowsA:
    print("setting", r["L"], r["fold"], "pooled AUC ideal %.2f fixed %.2f cen %.2f cv_unc %.2f cv_cen %.2f cnr %.2f" % (
        r["ideal_fixed"][0], r["hw_fixed"][0], r["hw_fixed_centred"][0], r["hw_cv_unc"][0], r["hw_cv_cen"][0], r["hw_cnr"][0]))

# ---- centering removes the offset: std of centred kernel vs ideal; offset of mean
sc = []
for k, (L, f) in enumerate(settings):
    Ktr, _ = kk[k]
    Ki, _ = full_kernel(ideal * shots2, shots2, k)
    off = ~np.eye(NTR, dtype=bool)
    sc.append(dict(L=L, fold=f, s_c=float(center_train(Ktr)[off].std() / center_train(Ki)[off].std()), mean_offdiag_hw=float(Ktr[off].mean()), mean_offdiag_ideal=float(Ki[off].mean()),
                   eff_diag_calibrated=float(Ktr[0, 0])))
out["HW2_centred_signal_retention"] = sc

# ================================================================ HW3 summary (predict-then-verify)
out["HW3"] = dict(
    lnrho_per_noisy_layer=dict(measured_hw=m_hw, backend_noise_model_aer=m_nz, hw1_idle_T1T2_only=float(m_id), depolarizing_per_cz_fit=float(mc * (ncz[1] / layers[1]))),
    rho_eff=dict(measured_hw=float(np.exp(m_hw)), backend_noise_model_aer=float(np.exp(m_nz)), hw1_idle_T1T2_only=float(np.exp(m_id))),
    idle_only_too_optimistic=bool(np.exp(m_id) > np.exp(m_hw)),
    s_by_setting=dict(settings=settings, hw=s_hw.tolist(), aer_backend=s_nz.tolist(), idle=s_idle.tolist(), dep_per_cz=fits["depolarizing_per_cz_fit"]["pred_s"]))

# ================================================================ HW4: idle-delay sweep
shots4 = R["hw4"]["shots"]
p4 = np.array([c.get("0000", 0) / shots4 for c in R["hw4"]["counts"]])
m4 = meta["hw4_meta"]
ai, an4 = np.array(meta["aer"]["hw4_ideal"]), np.array(meta["aer"]["hw4_noisy"])
Xall = np.array(meta["Xall"])
# idle-model prediction using HW1-fitted (T1,T2): AD(gamma) then PD(lambda_phi) per qubit on the idle gap
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from nqk.hw import slice_gates
from nqk.kernel import apply_noise


def idle_pred(i, j, tau):
    g = 1 - np.exp(-tau / T1eff)
    lphi = max(0.0, 1 - np.exp(-2 * tau * (1 / T2use - 1 / (2 * T1eff))))
    def U(x):
        q = QuantumCircuit(NQ); slice_gates(q, x, meta["beta"], 1); return Operator(q).data
    UA, UB = U(Xall[i]), U(Xall[j])
    rho = np.zeros((16, 16), complex); rho[0, 0] = 1
    rho = UA @ rho @ UA.conj().T
    rho = apply_noise(rho[None], ("local", kraus("ad", g)), NQ)
    rho = apply_noise(rho, ("local", kraus("pd", lphi)), NQ)[0]
    rho = UB.conj().T @ rho @ UB
    return float(rho[0, 0].real)


h4 = []
for pair in meta["hw4_pairs"]:
    i, j = pair
    for tau in meta["delays"]:
        idx1 = next(k for k, m in enumerate(m4) if m["tau"] == tau and m["i"] == i and m["j"] == j)
        idx2 = next(k for k, m in enumerate(m4) if m["tau"] == tau and m["i"] == j and m["j"] == i)
        a1, a2 = p4[idx1], p4[idx2]
        se = float(np.sqrt(a1 * (1 - a1) / shots4 + a2 * (1 - a2) / shots4))
        h4.append(dict(pair=[i, j], tau_us=tau, K_ij=float(a1), K_ji=float(a2), asym_hw=float(a1 - a2), asym_se=se,
                       asym_aer_noise=float(an4[idx1] - an4[idx2]), asym_aer_ideal=float(ai[idx1] - ai[idx2]),
                       asym_idle_pred=float(idle_pred(i, j, tau) - idle_pred(j, i, tau)),
                       K_ij_idle_pred=idle_pred(i, j, tau), K_ji_idle_pred=idle_pred(j, i, tau)))
out["HW4"] = dict(rows=h4, job=R["hw4"]["job_id"], qpu_s=R["hw4"]["usage_seconds"], shots=shots4,
                  max_abs_asym_tau0=float(max(abs(r["asym_hw"]) for r in h4 if r["tau_us"] == 0)),
                  mean_abs_asym_tau100=float(np.mean([abs(r["asym_hw"]) for r in h4 if r["tau_us"] == 100])),
                  mean_abs_asym_tau200=float(np.mean([abs(r["asym_hw"]) for r in h4 if r["tau_us"] == 200])),
                  mean_z_tau200=float(np.mean([abs(r["asym_hw"]) / r["asym_se"] for r in h4 if r["tau_us"] == 200])),
                  sign_matches_idle_pred_all_tau_gt0=bool(all(np.sign(r["asym_hw"]) == np.sign(r["asym_idle_pred"]) for r in h4 if r["tau_us"] > 0)))
for r in h4:
    print("HW4", r["pair"], r["tau_us"], "asym hw %.3f±%.3f  aer %.3f  idle %.3f" % (r["asym_hw"], r["asym_se"], r["asym_aer_noise"], r["asym_idle_pred"]))

# ================================================================ HW5: shot subsampling of HW2 counts (L=2; noise factors 1,3,9 -> fold 0,1,4)
SUBS = [32, 64, 128, 256, 512, 1024]
h5 = []
for f in (0, 1, 4):
    k = settings.index((2, f))
    k0 = settings.index((2, 0))
    Ktr_full, Kte_full = kk[k]
    cnt = p_hw[setting_slice(k)].astype(int)
    cnt0 = p_hw[setting_slice(k0)].astype(int)
    for Sp in SUBS:
        res_ = {m: [] for m in ("cnr", "fixed", "cv_cen", "cv_unc")}
        for rep in range(200):
            def sub(c):
                return rng.hypergeometric(c, shots2 - c, Sp) / Sp
            vals = np.array([sub(c) for c in cnt]); vals0 = np.array([sub(c) for c in cnt0])
            def build(v, kk_):
                Ktr = np.zeros((NTR, NTR)); Kte = np.zeros((NTE, NTR))
                for (i, j), val in zip(pairs, v):
                    if i < NTR and j < NTR:
                        Ktr[i, j] = Ktr[j, i] = val
                    else:
                        Kte[i - NTR, j] = val
                a, b = fit_aff(ideal[setting_slice(kk_)], v)
                np.fill_diagonal(Ktr, a + b)
                return Ktr, Kte
            Ktr, Kte = build(vals, k); Rtr, _ = build(vals0, k0)
            res_["cnr"].append(cnr_pooled(Ktr, Kte, Rtr, Sp)[0])
            res_["fixed"].append(pooled_heldout(Ktr, Kte, None, FixedLam(0.1, False))[0])
            res_["cv_cen"].append(pooled_heldout(Ktr, Kte, None, CVLam(True))[0])
            res_["cv_unc"].append(pooled_heldout(Ktr, Kte, None, CVLam(False))[0])
        h5.append(dict(noise_factor=1 + 2 * f, S=Sp, **{m + "_mean": float(np.mean(v)) for m, v in res_.items()}, **{m + "_sd": float(np.std(v)) for m, v in res_.items()}))
out["HW5"] = dict(rows=h5, reps=200)
for r in h5:
    print("HW5 nf", r["noise_factor"], "S", r["S"], "cnr %.3f fixed %.3f cv_cen %.3f cv_unc %.3f" % (r["cnr_mean"], r["fixed_mean"], r["cv_cen_mean"], r["cv_unc_mean"]))

out["qpu_accounting"] = dict(jobs={k: dict(job_id=R[k]["job_id"], n_circuits=R[k]["n_circuits"], shots=R[k]["shots"], qpu_seconds=R[k]["usage_seconds"], utc=R[k]["utc"]) for k in R},
                             backend=meta["backend"], layout=meta["layout"], calibration=meta["snapshot"],
                             noiseless_test_auc_L2=meta["noiseless_test_auc_L2"], beta=meta["beta"], split=meta["split"])
save("HW_analysis", dict(source="results/hardware/raw_*.json", layout=meta["layout"]), out, t0)
