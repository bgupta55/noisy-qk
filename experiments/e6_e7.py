import zlib
"""E6 (shot-limited viability depth, Thm 7b / Alg 10) and E7 (shot-limited indistinguishability, Thm 7a / Alg 14).
E6: Emb-S, N=4, n=30, gamma=0.02, lam0=0.1, CNR(c=1 floor-constant c fitted below). S in {64,256,1024,4096,16384}.
    L_emp = first (linearly interpolated) crossing of the split-averaged CNR AUC below 0.5+0.5*(AUC_noiseless-0.5).
    Prediction: L_shot = ln(s0 sqrt(S/n)/c)/rate, rate = -d ln a_L / dL fitted on the exact kernel, s0 = ||Kc_ref||_op.
    c is fitted ONCE on D1 (dep) and frozen for D2-D4 and for AD.
E7: exact noisy kernel vectors of two opposite-class test points (closest pair in sup-norm), MC Bayes error vs Alg 14 bound."""
from common import *
from joblib import Parallel, delayed
from nqk.datasets import make_split
from nqk.embeddings import SliceFamily
from nqk.pipeline import embS_kernel
from nqk.krr import ridge, cv_lam, cnr_lambda, shot_sample, center_train
from nqk.metrics import auc, L_shot, tv_bound
from scipy.optimize import minimize_scalar

t0 = time.time()
N, n, GAMMA, BETA = 4, 30, 0.02, 2.0
LS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
SS = [64, 256, 1024, 4096, 16384]
NSPLIT = 10
DS = ["D1", "D2", "D3", "D4"]
CHS = ["dep", "ad"]
cfg = dict(N=N, n=n, gamma=GAMMA, beta=BETA, L=LS, S=SS, splits=NSPLIT, datasets=DS, channels=CHS)


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
            lam, a_est = cnr_lambda(Ktr_h, Rtr_h, n, S, 0.1, 1.0)
            hh = ridge(Ktr_h, Kte_h, ytr.astype(float), lam, True)
            rows.append(dict(ds=dname, ch=ch, split=sp, L=L, S=S, auc=auc(hh, yte), aL=aL, auc_noiseless=a_noiseless, s0=s0))
    return rows


out = Parallel(n_jobs=10)(delayed(run)(d, c, s) for d in DS for c in CHS for s in range(NSPLIT))
import pandas as pd
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


def rate_fit(c):
    m = c.L.values <= 128
    x, y = c.L.values[m].astype(float), np.log(np.maximum(c.aL.values[m], 1e-12))
    x, y = x[x >= 1], y[x >= 1]
    return float(-np.polyfit(x, y, 1)[0])


rows = []
for (d, ch, S), g in curve.groupby(["ds", "ch", "S"]):
    ca = g.sort_values("L")
    rate = rate_fit(ca[ca.S == S] if False else ca)
    rows.append(dict(ds=d, ch=ch, S=S, L_emp=l_emp(ca), rate=rate, s0=float(ca.s0.iloc[0]), a0=float(ca.a0.iloc[0])))
T = pd.DataFrame(rows)
# bootstrap L_emp over splits
bs = {}
for (d, ch, S), g in df.groupby(["ds", "ch", "S"]):
    sp_ids = sorted(g.split.unique())
    rng = np.random.default_rng(0)
    vals = []
    for b in range(300):
        pick = rng.choice(sp_ids, len(sp_ids))
        gg = pd.concat([g[g.split == s] for s in pick]).groupby("L").agg(auc=("auc", "mean"), a0=("auc_noiseless", "mean")).reset_index()
        vals.append(l_emp(gg))
    v = np.array(vals); v = v[np.isfinite(v)]
    bs[(d, ch, S)] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)
T["L_emp_lo"] = [bs[(r.ds, r.ch, r.S)][0] for r in T.itertuples()]
T["L_emp_hi"] = [bs[(r.ds, r.ch, r.S)][1] for r in T.itertuples()]


# fit c on D1/dep only
def loss(logc):
    c = np.exp(logc)
    t = T[(T.ds == "D1") & (T.ch == "dep") & np.isfinite(T.L_emp)]
    return float(np.sum([(L_shot(r.s0, r.S, n, r.rate, c) - r.L_emp) ** 2 for r in t.itertuples()]))
opt = minimize_scalar(loss, bounds=(-6, 3), method="bounded")
c_fit = float(np.exp(opt.x))
T["L_pred"] = [L_shot(r.s0, r.S, n, r.rate, c_fit) for r in T.itertuples()]
T["err_layers"] = T.L_pred - T.L_emp
# log-slope of L_emp in S (expected ~ 0.5/rate * ... small)
sl = {}
for (d, ch), g in T[np.isfinite(T.L_emp)].groupby(["ds", "ch"]):
    sl[f"{d}_{ch}"] = float(np.polyfit(np.log(g.S.values), g.L_emp.values, 1)[0])
test = T[(T.ds != "D1") & (T.ch == "dep") & np.isfinite(T.L_emp)]
summary = dict(c_fit_D1_dep=c_fit, mean_abs_err_layers_D1_fit=float(T[(T.ds == "D1") & (T.ch == "dep") & np.isfinite(T.L_emp)].err_layers.abs().mean()),
               mean_abs_err_layers_heldout_D2_D4_dep=float(test.err_layers.abs().mean()),
               mean_err_layers_heldout_D2_D4_dep=float(test.err_layers.mean()),
               mean_abs_err_layers_AD=float(T[(T.ch == "ad") & np.isfinite(T.L_emp)].err_layers.abs().mean()),
               L_emp_per_lnS_slope=sl)
res6 = dict(table=T.to_dict("records"), curves=curve.to_dict("records"), summary=summary)
print(T.round(2).to_string()); print(summary)

# ======================================================= E7
N, n = 4, 30
rows7 = []
rng = np.random.default_rng(2026)
for dname in ("D1", "D2"):
    for sp in range(3):
        Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=100, scale="unit")
        fam = SliceFamily(np.vstack([Xtr, Xte]), BETA, N)
        for L in (64, 128, 192):
            Ks, Kraw = embS_kernel(fam, "dep", GAMMA, L)
            V = Kraw[n:, :n]                      # test x train kernel vectors (rows)
            pos, neg = np.where(yte > 0)[0], np.where(yte < 0)[0]
            best = None
            for i in pos:
                for j in neg:
                    d_ = np.abs(V[i] - V[j]).max()
                    if best is None or d_ < best[0]:
                        best = (d_, i, j)
            d_, i, j = best
            q0, q1 = np.clip(V[i], 1e-9, 1 - 1e-9), np.clip(V[j], 1e-9, 1 - 1e-9)
            eps = d_ / 2
            v = float(np.min(q0 * (1 - q0)))
            for S in (16, 64, 256, 1024, 4096):
                M = 20000
                r0 = rng.binomial(S, q0, (M, n)); r1 = rng.binomial(S, q1, (M, n))
                w = np.log(q1 / q0); w2 = np.log((1 - q1) / (1 - q0))
                def llr(r):
                    return (r * w + (S - r) * w2).sum(1)
                l0, l1 = llr(r0), llr(r1)
                # Bayes error with equal priors: decide H1 iff llr>0
                bayes = 0.5 * (np.mean(l0 > 0) + 0.5 * np.mean(l0 == 0)) + 0.5 * (np.mean(l1 < 0) + 0.5 * np.mean(l1 == 0))
                # exact TV by MC (importance: TV = 0.5 E_0 |LR - 1|)
                tv_mc = 0.5 * np.mean(np.abs(np.exp(np.clip(l0, -50, 50)) - 1))
                tvb = tv_bound(eps, n, S, v)
                rows7.append(dict(ds=dname, split=sp, L=L, S=S, eps=float(eps), v=v, n=n, tv_bound=tvb, tv_mc=float(tv_mc), bayes_mc=float(bayes),
                                  bayes_lower_bound=0.5 * (1 - tvb), bound_holds=bool(bayes >= 0.5 * (1 - tvb) - 3 / np.sqrt(M)),
                                  slack=float(bayes - 0.5 * (1 - tvb))))
B = pd.DataFrame(rows7)
sum7 = dict(frac_bound_holds=float(B.bound_holds.mean()), n_rows=len(B), mean_slack=float(B.slack.mean()),
            nontrivial_rows=int((B.tv_bound < 0.9).sum()), tv_ratio_bound_over_mc_median=float((B.tv_bound / B.tv_mc.clip(lower=1e-6))[B.tv_bound < 0.9].median()))
print(B.groupby(["L", "S"])[["eps", "tv_bound", "tv_mc", "bayes_mc", "bayes_lower_bound"]].mean().round(4).to_string()); print(sum7)
save("E6_E7_viability_indistinguishability", cfg, dict(E6=res6, E7=dict(rows=B.to_dict("records"), summary=sum7)), t0)
