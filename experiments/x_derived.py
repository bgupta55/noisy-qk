"""Derived numbers quoted in results_v2.md (so that none is typed by hand). Reads results/X*.json, HW_analysis.json; writes results/X_derived.json."""
from common import *
t0 = time.time()
J = lambda f: json.load(open(os.path.join(RES, f + ".json")))["results"]
out = {}

# ---------------- X4: noise under-prediction by the backend model, marrakesh vs kingston
x4 = J("X4_HW2_repeat")
rt = x4["retention"]
out["X4_ln_ratio_hw_over_aer"] = [dict(noisy_layers=r["noisy_layers"], s_hw=r["s_hw"], s_aer=r["s_aer_noise_model"], ratio=float(np.log(r["s_hw"]) / np.log(r["s_aer_noise_model"]))) for r in rt]
v1 = J("HW_analysis")["HW2_retention"]
out["V1_ln_ratio_hw_over_aer"] = [dict(noisy_layers=r["noisy_layers"], s_hw=r["s_hw"], s_aer=r["s_aer_noise"], ratio=float(np.log(r["s_hw"]) / np.log(r["s_aer_noise"]))) for r in v1]
s = np.array([r["s_hw"] for r in rt]); lay = np.array([r["noisy_layers"] for r in rt])
out["X4_monotone_violations"] = [dict(from_layers=int(lay[i]), to_layers=int(lay[i + 1]), s_from=float(s[i]), s_to=float(s[i + 1]),
                                      increase=float(s[i + 1] - s[i]), z=float((s[i + 1] - s[i]) / np.hypot(rt[i]["s_hw_se_boot"], rt[i + 1]["s_hw_se_boot"])))
                                 for i in range(len(s) - 1) if s[i + 1] > s[i]]
# local rates between consecutive settings (ln s per layer)
out["X4_local_rate_between_consecutive"] = [dict(layers=[int(lay[i]), int(lay[i + 1])], rate=float(np.log(s[i + 1] / s[i]) / (lay[i + 1] - lay[i]))) for i in range(len(s) - 1)]
v1s = np.array([r["s_hw"] for r in v1]); v1l = np.array([r["noisy_layers"] for r in v1])
out["V1_local_rate_between_consecutive"] = [dict(layers=[int(v1l[i]), int(v1l[i + 1])], rate=float(np.log(v1s[i + 1] / v1s[i]) / (v1l[i + 1] - v1l[i]))) for i in range(len(v1s) - 1)]
# affine-model adequacy: misfit / shot noise
out["X4_misfit_over_shot_noise"] = [dict(noisy_layers=r["noisy_layers"], ratio=float(r["rmse_affine_misfit"] / r["shot_noise_rmse_expected"]), corr=r["corr_exact_vs_hw"]) for r in rt]
# fit restricted to the 5 points with layers <= 20 (before the plateau) and rate of consecutive settings up to 20
m = lay <= 20
out["X4_rate_layers_le_20"] = dict(ln_per_layer=float(np.polyfit(lay[m], np.log(s[m]), 1)[0]), rho=float(np.exp(np.polyfit(lay[m], np.log(s[m]), 1)[0])))

# ---------------- X8: Wang read-off vs ours
x8 = J("X8_wang_sensitivity")
W = {24: 0.07, 28: 0.28, 32: 0.47}
Lw, ew = list(W), list(W.values())
wang_cross = {str(t): float(np.interp(t, ew, Lw)) for t in (0.10, 0.25, 0.40)}
base = [r for r in x8["summary"] if r["scale"] == "unit" and r["pairs"] == "all_ordered_incl_diag"][0]
out["X8"] = dict(wang_readoff_interp_crossing=wang_cross,
                 ours_crossing_mean5={t: base[f"L_cross_{t}_mean5"] for t in ("0.1", "0.25", "0.4")},
                 diff_ours_minus_wang={t: base[f"L_cross_{t}_mean5"] - wang_cross[t] for t in ("0.1", "0.25", "0.4")},
                 ours_seed_range_at_0p25=[base["L_cross_0.25_per_seed_min"], base["L_cross_0.25_per_seed_max"]], ours_seed_sd_at_0p25=base["L_cross_0.25_per_seed_sd"],
                 standard_error_of_5seed_mean_at_0p25=base["L_cross_0.25_per_seed_sd"] / np.sqrt(5),
                 wang_within_our_seed_range=bool(base["L_cross_0.25_per_seed_min"] <= wang_cross["0.25"] <= base["L_cross_0.25_per_seed_max"]),
                 mean5_vs_mean8_err_at_28=[base["err_tr_at_24_28_32_mean5"][1], base["err_tr_at_24_28_32_mean8"][1]],
                 variants=[dict(scale=r["scale"], pairs=r["pairs"], L025=r["L_cross_0.25_mean5"], L010=r["L_cross_0.1_mean5"], L040=r["L_cross_0.4_mean5"], rms=r["rms_to_wang_readoff_mean5"],
                                err_24_28_32=r["err_tr_at_24_28_32_mean5"], offdiag_mean=r["offdiag_mean"], offdiag_std=r["offdiag_std"]) for r in x8["summary"]],
                 diag_term_invariance_max_diff=float(max(abs(a - b) for r1 in x8["summary"] if r1["pairs"] == "all_ordered_incl_diag"
                                                          for r2 in x8["summary"] if r2["pairs"] == "i_ne_j_ordered" and r2["scale"] == r1["scale"]
                                                          for a, b in zip(r1["err_tr_mean5"], r2["err_tr_mean5"]))))

# ---------------- X2: counts without the trivial gdep rows (global depolarising is a single-mode case)
rows = [r for r in J("X2_E1_modal_weights")["rows"] if r["ch"] != "gdep"]
X2 = {}
for p in ("spectral", "weighted", "heavy", "heavy_pair", "env"):
    X2[p] = dict(within2se=int(sum(r["within2se_" + p] for r in rows)), n=len(rows), median_abs_err=float(np.median([r["abs_err_" + p] for r in rows])),
                 mean_abs_err=float(np.mean([r["abs_err_" + p] for r in rows])),
                 obs_steeper_than_pred=int(sum(r["obs_slope_recomputed"] < r["pred_" + p] - 1e-6 for r in rows)),
                 obs_shallower_than_pred=int(sum(r["obs_slope_recomputed"] > r["pred_" + p] + 1e-6 for r in rows)))
    for ch in ("dep", "bf", "pd", "ad"):
        X2[p][f"within2se_{ch}"] = int(sum(r["within2se_" + p] for r in rows if r["ch"] == ch))
out["X2_excluding_gdep"] = X2
out["X2_sample"] = dict(n_rows_total=len(J("X2_E1_modal_weights")["rows"]), n_rows_excl_gdep=len(rows),
                        max_recon_err=float(max(r["recon_err"] for r in J("X2_E1_modal_weights")["rows"])))

# ---------------- X4 exploratory (POST-HOC, not pre-registered): centred-signal retention s_c as in V1 (std of centred off-diagonal, hw / ideal)
from nqk.krr import center_train
HX = os.path.join(RES, "hardware_x4")
meta = json.load(open(os.path.join(HX, "prep_meta.json")))
raws = [json.load(open(os.path.join(HX, f"raw_x4_{k}.json"))) for k in range(3)]
cnt = np.array([c.get("0000", 0) for g in raws for c in g["counts"]], float)
settings = [tuple(x) for x in meta["settings"]]; pairs = [tuple(p) for p in meta["pairs"]]; NP = len(pairs); NTR = meta["n_train"]
shv = np.concatenate([[meta["shots"][f"{a},{b}"]] * NP for a, b in settings]).astype(float)
ex = np.array(meta["aer"]["exact"])
def blk(vec, k):
    K = np.zeros((NTR, NTR))
    for (i, j), v in zip(pairs, vec[NP * k:NP * (k + 1)]):
        if i < NTR and j < NTR:
            K[i, j] = K[j, i] = v
    return K
off = ~np.eye(NTR, dtype=bool)
sc = []
for k, (L, f) in enumerate(settings):
    ph = cnt[NP * k:NP * (k + 1)] / shv[NP * k:NP * (k + 1)]
    a, b = np.polyfit(ex[NP * k:NP * (k + 1)], ph, 1)
    Kh = blk(cnt / shv, k); np.fill_diagonal(Kh, a + b)
    Ki = blk(ex, k); np.fill_diagonal(Ki, 1.0)
    sc.append(dict(L=L, fold=f, noisy_layers=2 * L * (1 + 2 * f), s_c=float(center_train(Kh)[off].std() / center_train(Ki)[off].std())))
out["X4_posthoc_centred_retention"] = sc
sca = np.array([r["s_c"] for r in sc]); ll = np.array([r["noisy_layers"] for r in sc], float)
out["X4_posthoc_centred_rate"] = dict(all=float(np.polyfit(ll, np.log(sca), 1)[0]), early=float(np.polyfit(ll[ll <= 12], np.log(sca[ll <= 12]), 1)[0]), late=float(np.polyfit(ll[ll >= 20], np.log(sca[ll >= 20]), 1)[0]))

save("X_derived", dict(source="X2, X4, X8 JSON + HW_analysis.json"), out, t0)
print(json.dumps(out["X8"]["diff_ours_minus_wang"]), out["X8"]["wang_within_our_seed_range"], out["X8"]["diag_term_invariance_max_diff"])
print([ (d["noisy_layers"], round(d["ratio"], 2)) for d in out["X4_ln_ratio_hw_over_aer"]]); print([ (d["noisy_layers"], round(d["ratio"], 2)) for d in out["V1_ln_ratio_hw_over_aer"]])
print(out["X4_monotone_violations"]); print(out["X4_rate_layers_le_20"]); print(out["X4_misfit_over_shot_noise"])
print(out["X4_local_rate_between_consecutive"]); print(out["V1_local_rate_between_consecutive"])
print({p: (v["within2se"], round(v["median_abs_err"], 4), v["obs_steeper_than_pred"]) for p, v in X2.items()})

print(out["X4_posthoc_centred_retention"]); print(out["X4_posthoc_centred_rate"])
