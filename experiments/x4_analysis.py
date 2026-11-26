"""X4 offline analysis (no QPU). Reads results/hardware_x4/{prep_meta.json,raw_x4_*.json}; compares with V1 (results/HW_analysis.json). Writes results/X4_HW2_repeat.json.
Pre-registered (PREREGISTRATION.md, ROUND X, X4):
  * rho_eff from OLS of ln s on noisy layers; early (2..12) vs late (20..36) local rates with parametric-bootstrap CI;
    'geometric' supported iff the CI of (late rate - early rate) contains 0, else reported as non-constant.
  * pooled AUC (9 LOO + 5 test) for fixed lambda=0.1, CV centred/uncentred, CNR c=1 and c=2, at nf 9 (4096 shots) and at 1024-shot hypergeometric subsamples,
    paired parametric bootstrap (200 reps). RULE: report whichever result appears."""
from common import *
from joblib import Parallel, delayed
from nqk.krr import ridge, cv_lam, center_train, cnr_lambda
from nqk.metrics import auc
from nqk.hw import expected_cz

t0 = time.time()
HX = os.path.join(RES, "hardware_x4")
meta = json.load(open(os.path.join(HX, "prep_meta.json")))
raws = [json.load(open(os.path.join(HX, f"raw_x4_{k}.json"))) for k in range(3)]
settings = [tuple(s) for s in meta["settings"]]
pairs = [tuple(p) for p in meta["pairs"]]
NP, NTR, NTE = len(pairs), meta["n_train"], meta["n_test"]
yall = np.array(meta["yall"], float)
exact = np.array(meta["aer"]["exact"])
aer_noisy = np.array(meta["aer"]["noisy_8192"])
layers = np.array([2 * L * (1 + 2 * f) for L, f in settings], float)
shots = {s: meta["shots"][f"{s[0]},{s[1]}"] for s in settings}
# assemble counts of all-zero per circuit, in the order of the prep meta (setting-major)
cnt0 = []
for g in raws:
    for c in g["counts"]:
        cnt0.append(c.get("0000", 0))
cnt0 = np.array(cnt0, float)
assert len(cnt0) == NP * len(settings), (len(cnt0), NP, len(settings))
shot_vec = np.concatenate([[shots[s]] * NP for s in settings]).astype(float)
for g in raws:
    assert g["layout"] == meta["layout"] and g["backend"] == meta["backend"]
rng = np.random.default_rng(2027)


def sl(k):
    return slice(NP * k, NP * (k + 1))


def fit_aff(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def retention(cvec):
    """affine retention s_k per setting from counts vector"""
    return np.array([fit_aff(exact[sl(k)], cvec[sl(k)] / shot_vec[sl(k)])[0] for k in range(len(settings))])


s_hw = retention(cnt0)
s_aer = np.array([fit_aff(exact[sl(k)], aer_noisy[sl(k)])[0] for k in range(len(settings))])


def rates(s, lay):
    early = np.isin(lay, [2, 4, 6, 12])
    late = np.isin(lay, [20, 28, 36])
    all_ = np.polyfit(lay, np.log(s), 1)[0]
    e = np.polyfit(lay[early], np.log(s[early]), 1)[0]
    l = np.polyfit(lay[late], np.log(s[late]), 1)[0]
    return dict(all=float(all_), early=float(e), late=float(l), late_minus_early=float(l - e))


R_hw = rates(s_hw, layers)
R_aer = rates(s_aer, layers)
# parametric bootstrap on counts
B = 1000
bs = []
for _ in range(B):
    cb = rng.binomial(shot_vec.astype(int), np.clip(cnt0 / shot_vec, 0, 1)).astype(float)
    bs.append(rates(retention(cb), layers))


def ci(key):
    v = np.array([b[key] for b in bs])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std())]


R_ci = {k: ci(k) for k in R_hw}
ret_rows = []
for k, (L, f) in enumerate(settings):
    a, b = fit_aff(exact[sl(k)], cnt0[sl(k)] / shot_vec[sl(k)])
    ret_rows.append(dict(L=L, fold=f, noise_factor=1 + 2 * f, noisy_layers=int(layers[k]), cz=expected_cz(L, f), shots=int(shots[(L, f)]), s_hw=float(a), offset=float(b),
                         s_aer_noise_model=float(s_aer[k]), est_duration_us=meta["est_duration_us"][k],
                         s_hw_se_boot=float(np.std([retention_b for retention_b in [0.0]])) if False else None))
# per-setting SE from the same bootstrap would require storing s; recompute cheaply
sb = np.array([retention(rng.binomial(shot_vec.astype(int), np.clip(cnt0 / shot_vec, 0, 1)).astype(float)) for _ in range(400)])
for k, r in enumerate(ret_rows):
    r["s_hw_se_boot"] = float(sb[:, k].std())
    # model-free diagnostics (do not rely on the affine model): mean all-zero probability, mean over the high-fidelity pairs (exact > 0.9), affine misfit
    pk, ek = cnt0[sl(k)] / shot_vec[sl(k)], exact[sl(k)]
    a_, b_ = fit_aff(ek, pk)
    r["corr_exact_vs_hw"] = float(np.corrcoef(ek, pk)[0, 1])
    r["rmse_affine_misfit"] = float(np.sqrt(((pk - (a_ * ek + b_)) ** 2).mean()))
    r["shot_noise_rmse_expected"] = float(np.sqrt((pk * (1 - pk) / shot_vec[sl(k)]).mean()))
    r["mean_p_all"] = float(pk.mean())
    r["mean_p_exact_gt_0.9"] = float(pk[ek > 0.9].mean()) if (ek > 0.9).any() else None
    r["n_exact_gt_0.9"] = int((ek > 0.9).sum())
rho_all = float(np.exp(R_hw["all"]))
geometric = bool(R_ci["late_minus_early"][0] <= 0 <= R_ci["late_minus_early"][1])
# V1 comparison
V1 = json.load(open(os.path.join(RES, "HW_analysis.json")))["results"]
v1_ret = V1["HW2_retention"]
v1_lay = np.array([r["noisy_layers"] for r in v1_ret], float)
v1_s = np.array([r["s_hw"] for r in v1_ret])
V1_rates = rates(v1_s, v1_lay) if False else None
# V1 has layers 2,4,6,12,20,36; early = {2,4,6,12}, late = {20,36}
e1 = np.polyfit(v1_lay[v1_lay <= 12], np.log(v1_s[v1_lay <= 12]), 1)[0]
l1 = np.polyfit(v1_lay[v1_lay >= 20], np.log(v1_s[v1_lay >= 20]), 1)[0]
print("X4 s_hw", np.round(s_hw, 3), "layers", layers.astype(int))
print("rates", R_hw, "CI", R_ci, "geometric", geometric)

# ------------------------------------------------------------------ AUC machinery
def build(cvec, k):
    """symmetric train block (diag from affine a+b) and test x train block from counts vector at setting k"""
    v = cvec[sl(k)] / shot_vec[sl(k)]
    Ktr = np.zeros((NTR, NTR)); Kte = np.zeros((NTE, NTR))
    for (i, j), val in zip(pairs, v):
        if i < NTR and j < NTR:
            Ktr[i, j] = Ktr[j, i] = val
        else:
            Kte[i - NTR, j] = val
    a, b = fit_aff(exact[sl(k)], v)
    np.fill_diagonal(Ktr, a + b)
    return Ktr, Kte


def pooled(Ktr, Kte, lam_fn, center):
    ytr, yte = yall[:NTR], yall[NTR:]
    lh, ly = [], []
    for t in range(NTR):
        tr = [i for i in range(NTR) if i != t]
        lam = lam_fn(Ktr[np.ix_(tr, tr)], ytr[tr], tr)
        h = ridge(Ktr[np.ix_(tr, tr)], Ktr[np.ix_([t], tr)], ytr[tr], lam, center)
        lh.append(h[0]); ly.append(ytr[t])
    lam = lam_fn(Ktr, ytr, list(range(NTR)))
    ht = ridge(Ktr, Kte, ytr, lam, center)
    return float(auc(np.concatenate([lh, ht]), np.concatenate([ly, yte]))), float(auc(ht, yte)), float(auc(np.array(lh), np.array(ly)))


def all_methods(Ktr, Kte, Rtr, S):
    out = {}
    out["fixed"] = pooled(Ktr, Kte, lambda K, y, idx: 0.1, False)[0]
    out["fixed_centred"] = pooled(Ktr, Kte, lambda K, y, idx: 0.1, True)[0]
    out["cv_unc"] = pooled(Ktr, Kte, lambda K, y, idx: cv_lam(K, y, False, folds=min(3, len(y))), False)[0]
    out["cv_cen"] = pooled(Ktr, Kte, lambda K, y, idx: cv_lam(K, y, True, folds=min(3, len(y))), True)[0]
    for c, nm in ((1.0, "cnr_c1"), (2.0, "cnr_c2")):
        out[nm] = pooled(Ktr, Kte, lambda K, y, idx, c=c: cnr_lambda(K, Rtr[np.ix_(idx, idx)], len(y), S, 0.1, c)[0], True)[0]
    return out


ref_k = settings.index((2, 0))


def auc_for(cvec, k, S):
    L, f = settings[k]
    kr = ref_k if L == 2 else settings.index((L, 0))
    Ktr, Kte = build(cvec, k)
    Rtr, _ = build(cvec, kr)
    return all_methods(Ktr, Kte, Rtr, S)


# point estimates per setting
auc_rows = []
ideal_fixed = []
for k, (L, f) in enumerate(settings):
    r = auc_for(cnt0, k, shots[(L, f)])
    Ki = np.zeros((NTR, NTR)); Kti = np.zeros((NTE, NTR))
    for (i, j), val in zip(pairs, exact[sl(k)]):
        if i < NTR and j < NTR:
            Ki[i, j] = Ki[j, i] = val
        else:
            Kti[i - NTR, j] = val
    np.fill_diagonal(Ki, 1.0)
    r["ideal_fixed"] = pooled(Ki, Kti, lambda K, y, idx: 0.1, False)[0]
    r.update(L=L, fold=f, noise_factor=1 + 2 * f)
    auc_rows.append(r)
    print("AUC", L, f, {m: round(v, 2) for m, v in r.items() if m not in ("L", "fold", "noise_factor")})

# parametric bootstrap of AUC at nf 9 (4096 shots) with its reference (2,0): paired differences
k9 = settings.index((2, 4))
p_hat = np.clip(cnt0 / shot_vec, 0, 1)


def boot_one(seed):
    r = np.random.default_rng(seed)
    cb = cnt0.copy()
    for k in (k9, ref_k):
        cb[sl(k)] = r.binomial(int(shots[settings[k]]), p_hat[sl(k)])
    return auc_for(cb, k9, shots[(2, 4)])


NB = 200
boot9 = Parallel(n_jobs=6)(delayed(boot_one)(1000 + i) for i in range(NB))
METH = ["fixed", "fixed_centred", "cv_unc", "cv_cen", "cnr_c1", "cnr_c2"]
bt = {m: np.array([b[m] for b in boot9]) for m in METH}
boot_summary = {m: dict(mean=float(bt[m].mean()), sd=float(bt[m].std()), ci=[float(np.percentile(bt[m], 2.5)), float(np.percentile(bt[m], 97.5))]) for m in METH}
best_cv = np.maximum(bt["cv_unc"], bt["cv_cen"])
paired9 = {}
for nm in ("cnr_c1", "cnr_c2"):
    for rn, rv in (("fixed", bt["fixed"]), ("best_cv", best_cv), ("cv_cen", bt["cv_cen"])):
        d = bt[nm] - rv
        paired9[f"{nm}-{rn}"] = dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_pos=float((d > 0).mean()))
print("boot nf9", boot_summary)
print("paired", {k: (round(v['mean'], 3), [round(x, 3) for x in v['ci']]) for k, v in paired9.items()})

# hypergeometric subsamples to 1024 (and 256) shots from the 4096-shot nf9 circuits (reference (2,0) subsampled from its own 1024 shots)
def sub_one(seed, Sp, f):
    r = np.random.default_rng(seed)
    k = settings.index((2, f))
    cb = cnt0.copy()
    for kk_ in {k, ref_k}:
        c = cnt0[sl(kk_)].astype(int); n_ = int(shots[settings[kk_]])
        sp_ = min(Sp, n_)
        cb[sl(kk_)] = r.hypergeometric(c, n_ - c, sp_) / sp_ * shot_vec[sl(kk_)]    # rescale to the shot_vec scale
    return auc_for(cb, k, Sp)


sub_rows = []
for f in (0, 1, 4):
    for Sp in (256, 1024):
        res = Parallel(n_jobs=6)(delayed(sub_one)(5000 + i, Sp, f) for i in range(200))
        row = dict(noise_factor=1 + 2 * f, S=Sp, reps=200)
        for m in METH:
            v = np.array([x[m] for x in res]); row[m + "_mean"] = float(v.mean()); row[m + "_sd"] = float(v.std())
        sub_rows.append(row)
        print("SUB nf", 1 + 2 * f, "S", Sp, {m: round(row[m + "_mean"], 3) for m in METH})

# ------------------------------------------------------------------ V1 comparison numbers
v1_hw5 = [r for r in V1["HW5"]["rows"] if r["noise_factor"] == 9 and r["S"] == 1024]
v1_pooled9 = [r for r in V1["HW2_auc_(pooled,test3,loo6)"] if r["fold"] == 4]
out = dict(
    backend=meta["backend"], layout=meta["layout"], n_train=NTR, n_test=NTE, n_pairs_per_setting=NP, settings=settings, shots=[int(shots[s]) for s in settings],
    jobs=[dict(group=g["group"], job_id=g["job_id"], utc=g["utc"], qpu_seconds=g["usage_seconds"], account_usage_before=g["account_usage_before"], account_usage_after=g["account_usage_after"],
               n_circuits=g["n_circuits"], wall_seconds=g["wall_seconds"]) for g in raws],
    qpu_seconds_total=int(sum(g["usage_seconds"] for g in raws)),
    calibration_snapshot=meta["snapshot"], noiseless_test_auc_L2=meta["noiseless_test_auc_L2"], beta=meta["beta"], split=meta["split"],
    retention=ret_rows,
    rho_eff_all_layers=rho_all, rho_eff_early=float(np.exp(R_hw["early"])), rho_eff_late=float(np.exp(R_hw["late"])),
    rates_ln_per_layer=R_hw, rates_bootstrap_ci_and_sd=R_ci, bootstrap_reps=B,
    aer_backend_noise_model=dict(rates_ln_per_layer=R_aer, rho_eff_all=float(np.exp(R_aer["all"])), s=s_aer.tolist()),
    geometric_verdict=dict(rule="CI of (late - early) contains 0", ci=R_ci["late_minus_early"][:2], contains_zero=geometric,
                           statement="constant rate NOT rejected" if geometric else "rate NOT constant (late differs from early)"),
    v1_ibm_kingston=dict(layers=v1_lay.tolist(), s=v1_s.tolist(), rho_eff_all=float(V1["HW2_fits"]["hw"]["rho_eff"]),
                         rho_early=float(np.exp(e1)), rho_late=float(np.exp(l1)), late_minus_early_ln=float(l1 - e1),
                         pooled_auc_nf9=v1_pooled9[0] if v1_pooled9 else None, hw5_nf9_S1024=v1_hw5[0] if v1_hw5 else None, n_points=9),
    auc_per_setting=auc_rows,
    nf9_4096_bootstrap=dict(reps=NB, summary=boot_summary, paired=paired9),
    subsample_hypergeometric=sub_rows,
)
save("X4_HW2_repeat", dict(source="results/hardware_x4/raw_x4_*.json", bootstrap_reps=B, auc_boot_reps=NB), out, t0)
