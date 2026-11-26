"""Y3: S-exponent. If the viability threshold depends on S through the CNR floor, L_emp(S) is the depth at which the retained signal a_L equals
a floor f(S); the sqrt(S) form of Thm 7b says a*(S) := a_L(L_emp(S)) ~ S^(-alpha) with alpha = 0.5. Fit alpha from the existing E6/X1 data.
Pre-registered in results/PREREGISTRATION.md (ROUND Y, Y3). Reads results/X1_E6_fitted_rho.json only (same data as E6). Writes results/Y3_S_exponent.json."""
from common import *
from scipy import stats

t0 = time.time()
X1 = json.load(open(os.path.join(RES, "X1_E6_fitted_rho.json")))["results"]
T, C = X1["table"], X1["curves"]
SS = [64, 256, 1024, 4096, 16384]
rng = np.random.default_rng(2029)
curves = {}
for r in C:
    curves.setdefault((r["ds"], r["ch"], r["S"]), []).append((r["L"], r["aL"]))
for k in curves:
    curves[k].sort()
# aL(L) is the split-averaged exact a_L; it does not depend on S (verified below)
for ds in ("D1", "D2", "D3", "D4"):
    for ch in ("dep", "ad"):
        base = np.array([a for _, a in curves[(ds, ch, SS[0])]])
        for S in SS[1:]:
            assert np.allclose(base, np.array([a for _, a in curves[(ds, ch, S)]]))


def a_at(ds, ch, L):
    Ls = np.array([l for l, _ in curves[(ds, ch, SS[0])]], float)
    a = np.array([x for _, x in curves[(ds, ch, SS[0])]])
    return float(np.exp(np.interp(L, Ls, np.log(a))))


def alpha_fit(S, y):
    m, c = np.polyfit(np.log(S), y, 1)
    return float(m)


rows = {}
for ds in ("D1", "D2", "D3", "D4"):
    for ch in ("dep", "ad"):
        t = sorted([r for r in T if r["ds"] == ds and r["ch"] == ch], key=lambda r: r["S"])
        S = np.array([r["S"] for r in t], float)
        Le = np.array([r["L_emp"] for r in t], float)
        sd = np.array([(r["L_emp_hi"] - r["L_emp_lo"]) / 3.92 if r["L_emp_hi"] is not None and r["L_emp_lo"] is not None else np.nan for r in t])
        ast = np.array([a_at(ds, ch, L) for L in Le])
        lnA = np.log(ast)
        al_all = -alpha_fit(S, lnA)
        al_low = -alpha_fit(S[:3], lnA[:3])
        sd_f = np.where(np.isfinite(sd), sd, np.nanmean(sd))
        draws = []
        for _ in range(2000):
            Ld = np.clip(rng.normal(Le, sd_f), 1.0, 192.0)
            draws.append(-alpha_fit(S, np.log([a_at(ds, ch, L) for L in Ld])))
        draws = np.array(draws)
        slope_L = alpha_fit(S, Le)               # dL_emp/d ln S
        # local decay rate of a_L at the observed crossing depths (central difference in ln a over a +-25% window), mean over S
        loc = []
        for L in Le:
            lo, hi = max(1.0, 0.75 * L), min(192.0, 1.25 * L)
            loc.append(-(np.log(a_at(ds, ch, hi)) - np.log(a_at(ds, ch, lo))) / (hi - lo))
        rows[f"{ds}_{ch}"] = dict(ds=ds, ch=ch, S=S.tolist(), L_emp=Le.tolist(), L_emp_sd=sd_f.tolist(), a_star=ast.tolist(), alpha=al_all, alpha_S_le_1024=al_low,
                                  alpha_draw_sd=float(draws.std()), alpha_draw_ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                                  dLemp_dlnS=slope_L, local_rate_at_crossing=[float(x) for x in loc], rate_v1=float(t[0]["rate_v1"]), rate_fit3=float(t[0]["rate_fit3"]),
                                  alpha_rate_v1=float(t[0]["rate_v1"] * slope_L), alpha_rate_fit3=float(t[0]["rate_fit3"] * slope_L),
                                  local_alpha_between_S=[float(-(lnA[i + 1] - lnA[i]) / np.log(S[i + 1] / S[i])) for i in range(len(S) - 1)])
        print(ds, ch, "alpha %.3f (S<=1024: %.3f) draw CI [%.3f, %.3f]" % (al_all, al_low, *rows[f'{ds}_{ch}']["alpha_draw_ci"]),
              "dL/dlnS %.2f" % slope_L, "local", np.round(rows[f"{ds}_{ch}"]["local_alpha_between_S"], 2), "L_emp", np.round(Le, 1))

pooled = {}
for ch in ("dep", "ad"):
    al = np.array([rows[f"{d}_{ch}"]["alpha"] for d in ("D1", "D2", "D3", "D4")])
    al_low = np.array([rows[f"{d}_{ch}"]["alpha_S_le_1024"] for d in ("D1", "D2", "D3", "D4")])
    dsd = np.array([rows[f"{d}_{ch}"]["alpha_draw_sd"] for d in ("D1", "D2", "D3", "D4")])
    tq = stats.t.ppf(0.975, 3)
    across = float(al.std(ddof=1) * tq / 2)
    within = float(np.sqrt((dsd ** 2).mean()) * 1.96 / 2)
    half = float(np.sqrt(across ** 2 + within ** 2))
    pooled[ch] = dict(alpha_mean=float(al.mean()), alpha_by_dataset=al.tolist(), ci_halfwidth=half, ci=[float(al.mean() - half), float(al.mean() + half)],
                      across_dataset_halfwidth=across, within_draw_halfwidth=within, alpha_mean_S_le_1024=float(al_low.mean()),
                      contains_0p5=bool(al.mean() - half <= 0.5 <= al.mean() + half), contains_1=bool(al.mean() - half <= 1.0 <= al.mean() + half),
                      dLemp_dlnS_mean=float(np.mean([rows[f"{d}_{ch}"]["dLemp_dlnS"] for d in ("D1", "D2", "D3", "D4")])),
                      local_rate_at_crossing_mean=float(np.mean([np.mean(rows[f"{d}_{ch}"]["local_rate_at_crossing"]) for d in ("D1", "D2", "D3", "D4")])),
                      alpha_rate_v1_mean=float(np.mean([rows[f"{d}_{ch}"]["alpha_rate_v1"] for d in ("D1", "D2", "D3", "D4")])),
                      alpha_rate_fit3_mean=float(np.mean([rows[f"{d}_{ch}"]["alpha_rate_fit3"] for d in ("D1", "D2", "D3", "D4")])),
                      local_alpha_mean=np.mean([rows[f"{d}_{ch}"]["local_alpha_between_S"] for d in ("D1", "D2", "D3", "D4")], axis=0).tolist())
    print("POOLED", ch, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pooled[ch].items() if k not in ("alpha_by_dataset",)})
save("Y3_S_exponent", dict(S=SS, draws=2000, seed=2029, sqrtS_prediction=0.5, source="X1_E6_fitted_rho.json"), dict(rows=rows, pooled=pooled), t0)
