"""X1: E6 re-run with a fitted rho_eff from depths {1,4,16} (OLS of ln a_L on L) instead of the V1 rate (OLS over L<=128).
Pre-registered in results/PREREGISTRATION.md (ROUND X). Same data pipeline, splits and shot seeds as e6_e7.py (E6 part only).
Rule: if mean |err| for AD stays above ~10 layers, the AD viability claim is dropped.
Variants reported side by side (c is refitted on D1/dep for each variant and frozen elsewhere):
  v1        : rate = -OLS slope of ln a_L, L in [1,128]   (what V1 actually used)
  fit3      : rate = -OLS slope of ln a_L, L in {1,4,16}  (X1, pre-registered)
  spectral  : rate = k * (-ln lambda_2) of the single-layer noise map, k=2 unital / 1 non-unital (the 'envelope' diagnosis)
Descriptive diagnostics (NOT part of the pre-registered claim): per-channel in-sample c and the S-slope of L_emp vs the prediction."""
import zlib
from common import *
from joblib import Parallel, delayed
import pandas as pd
from nqk.datasets import make_split
from nqk.embeddings import SliceFamily
from nqk.pipeline import embS_kernel
from nqk.krr import ridge, cv_lam, cnr_lambda, shot_sample, center_train
from nqk.metrics import auc, L_shot
from nqk.channels import kraus, UNITAL
from nqk.kernel import _apply_kraus_qubit
from scipy.optimize import minimize_scalar

t0 = time.time()
N, n, GAMMA, BETA = 4, 30, 0.02, 2.0
LS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
SS = [64, 256, 1024, 4096, 16384]
NSPLIT = 10
DS = ["D1", "D2", "D3", "D4"]
CHS = ["dep", "ad"]
cfg = dict(N=N, n=n, gamma=GAMMA, beta=BETA, L=LS, S=SS, splits=NSPLIT, datasets=DS, channels=CHS, rate_depths_fit3=[1, 4, 16])


def run(dname, ch, sp):
    Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=100, scale="unit")
    fam = SliceFamily(np.vstack([Xtr, Xte]), BETA, N)
    Kn, _ = embS_kernel(fam, None, 0, 1)
    h = ridge(Kn[:n, :n], Kn[n:, :n], ytr.astype(float), cv_lam(Kn[:n, :n], ytr.astype(float), True), True)
    a_noiseless = auc(h, yte)
    Kref, _ = embS_kernel(fam, ch, GAMMA, 1)
    s0 = float(np.linalg.norm(center_train(Kn[:n, :n]), 2))
    rows = []
    for L in LS:
        Ks, _ = embS_kernel(fam, ch, GAMMA, L)
        aL = float(np.trace(center_train(Ks[:n, :n])) / np.trace(center_train(Kref[:n, :n])))
        for S in SS:
            rng = np.random.default_rng(zlib.crc32(repr((dname, ch, sp, L, S)).encode()) % (2 ** 31))
            Ktr_h, Kte_h = shot_sample(Ks[:n, :n], Ks[n:, :n], S, rng)
            Rtr_h, _ = shot_sample(Kref[:n, :n], Kref[n:, :n], S, rng)
            lam, _ = cnr_lambda(Ktr_h, Rtr_h, n, S, 0.1, 1.0)
            hh = ridge(Ktr_h, Kte_h, ytr.astype(float), lam, True)
            rows.append(dict(ds=dname, ch=ch, split=sp, L=L, S=S, auc=auc(hh, yte), aL=aL, auc_noiseless=a_noiseless, s0=s0))
    return rows


out = Parallel(n_jobs=10)(delayed(run)(d, c, s) for d in DS for c in CHS for s in range(NSPLIT))
df = pd.DataFrame([r for o in out for r in o])
curve = df.groupby(["ds", "ch", "S", "L"]).agg(auc=("auc", "mean"), aL=("aL", "mean"), a0=("auc_noiseless", "mean"), s0=("s0", "mean")).reset_index()


def l_emp(c):
    Ls_, A = c.L.values.astype(float), c.auc.values
    thr = 0.5 + 0.5 * (c.a0.iloc[0] - 0.5)
    if A[0] < thr:
        return 0.0
    for i in range(1, len(A)):
        if A[i] < thr:
            return float(Ls_[i - 1] + (A[i - 1] - thr) / (A[i - 1] - A[i]) * (Ls_[i] - Ls_[i - 1]))
    return float("inf")


def ols_rate(c, Lset):
    m = c.L.isin(Lset).values
    x, y = c.L.values[m].astype(float), np.log(np.maximum(c.aL.values[m], 1e-12))
    return float(-np.polyfit(x, y, 1)[0])


# spectral envelope of the single-layer noise map (local channel on N qubits): subdominant |eigenvalue|, k=2 unital / 1 non-unital
def noise_lambda2(ch):
    kr = kraus(ch, GAMMA)
    D = 2 ** N
    # superoperator on vec(rho): apply noise to each basis operator
    cols = []
    for a in range(D * D):
        R = np.zeros((1, D, D), complex)
        R[0, a // D, a % D] = 1
        for q in range(N):
            R = _apply_kraus_qubit(R, kr, q, N)
        cols.append(R[0].reshape(-1))
    S = np.array(cols).T
    w = np.linalg.eigvals(S)
    w = w[np.abs(w - 1) > 1e-7]
    return float(np.max(np.abs(w)))


SPEC = {}
for ch in CHS:
    l2 = noise_lambda2(ch)
    k = 2 if UNITAL[ch] else 1
    SPEC[ch] = dict(lambda2=l2, k=k, rate=float(-k * np.log(l2)))
print("spectral:", SPEC)

rows = []
for (d, ch, S), g in curve.groupby(["ds", "ch", "S"]):
    ca = g.sort_values("L")
    rows.append(dict(ds=d, ch=ch, S=S, L_emp=l_emp(ca), rate_v1=ols_rate(ca, [l for l in LS if l <= 128]), rate_fit3=ols_rate(ca, [1, 4, 16]),
                     rate_spec=SPEC[ch]["rate"], s0=float(ca.s0.iloc[0]), a0=float(ca.a0.iloc[0])))
T = pd.DataFrame(rows)
# bootstrap CI of L_emp over splits (same as E6)
bs = {}
for (d, ch, S), g in df.groupby(["ds", "ch", "S"]):
    sp_ids = sorted(g.split.unique())
    rng = np.random.default_rng(0)
    vals = []
    for b in range(300):
        pick = rng.choice(sp_ids, len(sp_ids))
        gg = pd.concat([g[g.split == s] for s in pick]).groupby("L").agg(auc=("auc", "mean"), a0=("auc_noiseless", "mean")).reset_index()
        vals.append(l_emp(gg))
    v = np.array(vals)
    v = v[np.isfinite(v)]
    bs[(d, ch, S)] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)
T["L_emp_lo"] = [bs[(r.ds, r.ch, r.S)][0] for r in T.itertuples()]
T["L_emp_hi"] = [bs[(r.ds, r.ch, r.S)][1] for r in T.itertuples()]


def fit_c(rate_col, mask):
    t = T[mask & np.isfinite(T.L_emp)]

    def loss(logc):
        c = np.exp(logc)
        return float(np.sum([(L_shot(r.s0, r.S, n, getattr(r, rate_col), c) - r.L_emp) ** 2 for r in t.itertuples()]))
    return float(np.exp(minimize_scalar(loss, bounds=(-8, 4), method="bounded").x))


summary = {}
for var, col in (("v1", "rate_v1"), ("fit3", "rate_fit3"), ("spectral", "rate_spec")):
    c_fit = fit_c(col, (T.ds == "D1") & (T.ch == "dep"))
    T[f"L_pred_{var}"] = [L_shot(r.s0, r.S, n, getattr(r, col), c_fit) for r in T.itertuples()]
    T[f"err_{var}"] = T[f"L_pred_{var}"] - T.L_emp
    fin = np.isfinite(T.L_emp)
    d1 = T[(T.ds == "D1") & (T.ch == "dep") & fin]
    ho = T[(T.ds != "D1") & (T.ch == "dep") & fin]
    ad = T[(T.ch == "ad") & fin]
    summary[var] = dict(c_fit_D1_dep=c_fit,
                        mean_abs_err_D1_dep_fit=float(d1[f"err_{var}"].abs().mean()),
                        mean_abs_err_heldout_D2_D4_dep=float(ho[f"err_{var}"].abs().mean()),
                        mean_signed_err_heldout_D2_D4_dep=float(ho[f"err_{var}"].mean()),
                        mean_abs_err_AD_all=float(ad[f"err_{var}"].abs().mean()),
                        mean_signed_err_AD_all=float(ad[f"err_{var}"].mean()),
                        AD_by_dataset={d: float(g[f"err_{var}"].abs().mean()) for d, g in ad.groupby("ds")},
                        n_AD_rows_finite=int(len(ad)))
# descriptive diagnostics (not pre-registered): the S-dependence and a per-channel in-sample c
diag = {}
for (d, ch), g in T[np.isfinite(T.L_emp)].groupby(["ds", "ch"]):
    slope_emp = float(np.polyfit(np.log(g.S.values), g.L_emp.values, 1)[0])
    slope_pred = float(np.polyfit(np.log(g.S.values), g.L_pred_fit3.values, 1)[0])
    diag[f"{d}_{ch}"] = dict(Lemp_per_lnS=slope_emp, Lpred_fit3_per_lnS=slope_pred, rate_fit3=float(g.rate_fit3.iloc[0]), rate_v1=float(g.rate_v1.iloc[0]))
c_ad = fit_c("rate_fit3", (T.ch == "ad"))
ad_ins = T[(T.ch == "ad") & np.isfinite(T.L_emp)].copy()
ad_ins["e"] = [L_shot(r.s0, r.S, n, r.rate_fit3, c_ad) - r.L_emp for r in ad_ins.itertuples()]
summary["descriptive_AD_in_sample_c"] = dict(c=c_ad, mean_abs_err_in_sample=float(ad_ins.e.abs().mean()), note="in-sample refit on AD itself; NOT a held-out result")
# reproduction check vs V1 table
v1 = json.load(open(os.path.join(RES, "E6_E7_viability_indistinguishability.json")))["results"]["E6"]["table"]
v1d = {(r["ds"], r["ch"], r["S"]): r["L_emp"] for r in v1}
mx = max(abs(v1d[(r.ds, r.ch, r.S)] - r.L_emp) for r in T.itertuples() if np.isfinite(r.L_emp) and v1d[(r.ds, r.ch, r.S)] is not None)
summary["reproduction_max_abs_diff_Lemp_vs_V1"] = float(mx)
AD_RULE = summary["fit3"]["mean_abs_err_AD_all"] > 10.0
summary["preregistered_rule"] = dict(threshold_layers=10.0, AD_error_fit3=summary["fit3"]["mean_abs_err_AD_all"], drop_AD_viability_claim=bool(AD_RULE))
pd.set_option("display.width", 250)
print(T.round(3).to_string())
print(json.dumps(summary, indent=1))
print("diag", json.dumps(diag, indent=1))
save("X1_E6_fitted_rho", cfg, dict(table=T.to_dict("records"), curves=curve.to_dict("records"), summary=summary, descriptive=diag, spectral=SPEC), t0)
