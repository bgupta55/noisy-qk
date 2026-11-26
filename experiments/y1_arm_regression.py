"""Y1: offline regression of hardware p(0000) on the exact kernel K plus per-input terms (b_j for the x'-arm, c_i for the x-arm).
Pre-registered in results/PREREGISTRATION.md (ROUND Y, Y1). Reads existing raw counts only (no QPU). Writes results/Y1_arm_regression.json.
Data: ibm_marrakesh (X4, n=14, 81 pairs/setting, 7 settings) and ibm_kingston (V1 HW2, n=9, 33 pairs/setting, 6 settings). Settings are fitted separately."""
from common import *

t0 = time.time()
rng = np.random.default_rng(2028)


def load_marrakesh():
    HX = os.path.join(RES, "hardware_x4")
    meta = json.load(open(os.path.join(HX, "prep_meta.json")))
    raws = [json.load(open(os.path.join(HX, f"raw_x4_{k}.json"))) for k in range(3)]
    cnt = np.array([c.get("0000", 0) for g in raws for c in g["counts"]], float)
    settings = [tuple(s) for s in meta["settings"]]
    pairs = [tuple(p) for p in meta["pairs"]]
    shots = np.concatenate([[meta["shots"][f"{s[0]},{s[1]}"]] * len(pairs) for s in settings]).astype(float)
    return dict(name="ibm_marrakesh", settings=settings, pairs=pairs, cnt=cnt, shots=shots, exact=np.array(meta["aer"]["exact"]), NTR=meta["n_train"])


def load_kingston():
    HW = os.path.join(RES, "hardware")
    meta = json.load(open(os.path.join(HW, "prep_meta.json")))
    raw = json.load(open(os.path.join(HW, "raw_hw2.json")))
    cnt = np.array([c.get("0000", 0) for c in raw["counts"]], float)
    settings = [tuple(s) for s in meta["settings"]]
    pairs = [tuple(p) for p in meta["pairs"]]
    shots = np.full(len(cnt), float(raw["shots"]))
    return dict(name="ibm_kingston", settings=settings, pairs=pairs, cnt=cnt, shots=shots, exact=np.array(meta["aer"]["hw2_exact"]), NTR=meta["n_train"])


def design(model, K, ii, jj, nI, nJ):
    n = len(K)
    cols = []
    if model in ("M0", "M1", "M2", "M3"):
        cols.append(K[:, None])
    cols.append(np.ones((n, 1)))
    if model in ("M1", "M3", "M4"):
        cols.append(np.eye(nJ)[jj])
    if model in ("M2", "M3", "M5"):
        cols.append(np.eye(nI)[ii])
    return np.hstack(cols)


MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]


def fit(model, K, p, ii, jj, nI, nJ):
    X = design(model, K, ii, jj, nI, nJ)
    beta = np.linalg.lstsq(X, p, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(X))
    res = p - X @ beta
    a = float(beta[0]) if model in ("M0", "M1", "M2", "M3") else None
    return X, beta, res, rank, a


def loo_rmse(model, K, p, ii, jj, nI, nJ):
    n = len(p)
    e = np.zeros(n)
    for t in range(n):
        m = np.arange(n) != t
        X = design(model, K, ii, jj, nI, nJ)
        beta = np.linalg.lstsq(X[m], p[m], rcond=None)[0]
        e[t] = p[t] - X[t] @ beta
    return float(np.sqrt((e ** 2).mean()))


def adj_r2(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rss = float(((y - X @ beta) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    n, r = len(y), int(np.linalg.matrix_rank(X))
    if tss <= 0 or n - r <= 0:
        return None
    return float(1 - (rss / (n - r)) / (tss / (n - 1)))


def analyse(D):
    S, pairs, NTR = D["settings"], D["pairs"], D["NTR"]
    NP = len(pairs)
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    nI, nJ = int(ii.max()) + 1, int(jj.max()) + 1
    rows = []
    for k, (L, f) in enumerate(S):
        sl = slice(NP * k, NP * (k + 1))
        p = D["cnt"][sl] / D["shots"][sl]
        K = D["exact"][sl]
        sh = D["shots"][sl]
        shot_rmse = float(np.sqrt((p * (1 - p) / sh).mean()))
        r = dict(L=L, fold=f, noise_factor=1 + 2 * f, noisy_layers=2 * L * (1 + 2 * f), n=NP, shots=int(sh[0]), shot_rmse=shot_rmse, mean_p=float(p.mean()), models={})
        for m in MODELS:
            X, beta, res, rank, a = fit(m, K, p, ii, jj, nI, nJ)
            dof = NP - rank
            r["models"][m] = dict(rank=rank, dof=int(dof), rmse_raw=float(np.sqrt((res ** 2).mean())),
                                  misfit_ratio=float(np.sqrt((res ** 2).sum() / dof) / shot_rmse) if dof > 0 else None,
                                  loo_rmse=loo_rmse(m, K, p, ii, jj, nI, nJ), a=a)
        # exact-K reference: how much of the exact kernel's own variance is explained by the j factor / i factor alone
        Xj = design("M4", K, ii, jj, nI, nJ); Xi = design("M5", K, ii, jj, nI, nJ)
        r["adjR2"] = dict(M4_p_on_j=adj_r2(p, Xj), M5_p_on_i=adj_r2(p, Xi), exactK_on_j=adj_r2(K, Xj), exactK_on_i=adj_r2(K, Xi))
        bj = fit("M1", K, p, ii, jj, nI, nJ)[1][2:]
        r["bj_sd"] = float(np.std(bj[np.unique(jj)])); r["bj_range"] = float(np.ptp(bj[np.unique(jj)]))
        rows.append(r)
    return rows, ii, jj, nI, nJ


def rates_boot(D, ii, jj, nI, nJ, B=400):
    S, pairs = D["settings"], D["pairs"]
    NP = len(pairs)
    lay = np.array([2 * L * (1 + 2 * f) for L, f in S], float)
    early = np.isin(lay, [2, 4, 6, 12]); late = np.isin(lay, [20, 28, 36])
    ph = np.clip(D["cnt"] / D["shots"], 0, 1)

    def slopes(cnt):
        out = {m: [] for m in ("M0", "M1", "M3")}
        for k in range(len(S)):
            sl = slice(NP * k, NP * (k + 1))
            p = cnt[sl] / D["shots"][sl]; K = D["exact"][sl]
            for m in out:
                a = fit(m, K, p, ii, jj, nI, nJ)[4]
                out[m].append(a)
        r = {}
        for m, a in out.items():
            a = np.array(a)
            if np.any(a <= 0):
                r[m] = dict(all=np.nan, early=np.nan, late=np.nan, d=np.nan, a=a.tolist())
                continue
            ly = np.log(a)
            e = np.polyfit(lay[early], ly[early], 1)[0]; l = np.polyfit(lay[late], ly[late], 1)[0]
            r[m] = dict(all=float(np.polyfit(lay, ly, 1)[0]), early=float(e), late=float(l), d=float(l - e), a=a.tolist())
        return r
    pt = slopes(D["cnt"])
    bs = []
    for _ in range(B):
        cb = rng.binomial(D["shots"].astype(int), ph).astype(float)
        bs.append(slopes(cb))
    ci = {}
    for m in pt:
        v = np.array([b[m]["d"] for b in bs], float)
        v = v[np.isfinite(v)]
        ci[m] = dict(point=pt[m], d_ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else None, n_valid=int(len(v)),
                     geometric_supported=bool(np.percentile(v, 2.5) <= 0 <= np.percentile(v, 97.5)) if len(v) > 20 else None)
    return ci


OUT = {}
for D in (load_marrakesh(), load_kingston()):
    rows, ii, jj, nI, nJ = analyse(D)
    OUT[D["name"]] = dict(rows=rows, rates=rates_boot(D, ii, jj, nI, nJ))
    print(D["name"])
    for r in rows:
        print(r["L"], r["fold"], r["noisy_layers"], "mis M0 %.2f M1 %.2f M2 %.2f M3 %.2f M4 %.2f M5 %.2f" % tuple(r["models"][m]["misfit_ratio"] or np.nan for m in MODELS),
              "loo M0 %.3f M1 %.3f M3 %.3f" % tuple(r["models"][m]["loo_rmse"] for m in ("M0", "M1", "M3")), "adjR2 j %.2f exactK_j %.2f | i %.2f exactK_i %.2f" % (
                  r["adjR2"]["M4_p_on_j"] or np.nan, r["adjR2"]["exactK_on_j"] or np.nan, r["adjR2"]["M5_p_on_i"] or np.nan, r["adjR2"]["exactK_on_i"] or np.nan))

# ------------------------------------------------------------------ pre-registered rules (marrakesh)
Mq = {(r["L"], r["fold"]): r for r in OUT["ibm_marrakesh"]["rows"]}
chk = {}
for key in ((2, 3), (2, 4)):
    r = Mq[key]["models"]
    chk[f"{key}"] = dict(m_M1=r["M1"]["misfit_ratio"], loo_M1=r["M1"]["loo_rmse"], loo_M0=r["M0"]["loo_rmse"], m_M2=r["M2"]["misfit_ratio"],
                         c_misfit_le_2=bool(r["M1"]["misfit_ratio"] <= 2.0), c_loo_le_0p8=bool(r["M1"]["loo_rmse"] <= 0.8 * r["M0"]["loo_rmse"]), c_beats_M2=bool(r["M1"]["misfit_ratio"] < r["M2"]["misfit_ratio"]))
    chk[f"{key}"]["all_three"] = bool(chk[f"{key}"]["c_misfit_le_2"] and chk[f"{key}"]["c_loo_le_0p8"] and chk[f"{key}"]["c_beats_M2"])
n_ok = sum(v["all_three"] for v in chk.values())
R1 = dict(per_setting=chk, n_settings_pass=int(n_ok), verdict="explained" if n_ok == 2 else ("partial" if n_ok == 1 else "not explained"))
a24, a20 = Mq[(2, 4)]["adjR2"], Mq[(2, 0)]["adjR2"]
dj = lambda a: None if a["M4_p_on_j"] is None else a["M4_p_on_j"] - a["exactK_on_j"]
di = lambda a: None if a["M5_p_on_i"] is None else a["M5_p_on_i"] - a["exactK_on_i"]
R2 = dict(excess_j_at_nf9=dj(a24), excess_j_at_nf1=dj(a20), excess_i_at_nf9=di(a24), excess_i_at_nf1=di(a20),
          j_excess_larger_at_nf9=bool(dj(a24) > dj(a20)), i_excess_smaller_at_nf9=bool(di(a24) < di(a20)),
          read_only_if_R1_not_not_explained=R1["verdict"] != "not explained")
R2["both"] = bool(R2["j_excess_larger_at_nf9"] and R2["i_excess_smaller_at_nf9"])
OUT["rules"] = dict(R1=R1, R2=R2)
print(json.dumps(OUT["rules"], indent=1))
for nm in ("ibm_marrakesh", "ibm_kingston"):
    for m, v in OUT[nm]["rates"].items():
        print(nm, m, {k: round(x, 4) for k, x in v["point"].items() if k != "a"}, "CI(late-early)", v["d_ci"], "geometric", v["geometric_supported"])
save("Y1_arm_regression", dict(models=MODELS, bootstrap_reps=400, seed=2028, preregistered="PREREGISTRATION.md ROUND Y / Y1"), OUT, t0)
