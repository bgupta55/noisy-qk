"""X3: E5 rerun with c=2 (frozen from D1/E12) for CNR; all other settings identical to e5_central.py (same splits, same shot seeds).
Check: fixed / cv_unc / cv_cen / cnr(c=1) reproduce the V1 E5_raw.csv.gz exactly. Report CNR(c=2) vs best CV and vs fixed per dataset (D1 = where c was chosen,
D2-D4 = held-out), in the V1 256-cell format."""
import zlib
from common import *
from joblib import Parallel, delayed
from nqk.datasets import make_split
from nqk.embeddings import SliceFamily
from nqk.pipeline import embS_kernel, evaluate_methods
import pandas as pd

t0 = time.time()
N, n, GAMMA, BETA = 4, 30, 0.02, 2.0
LS = [1, 4, 16, 64]
SS = [256, 1024, 4096, np.inf]
CHS = ["dep", "ad", "pd", "bf"]
DSETS = ["D1", "D2", "D4", "D3"]
NSPLIT = 20
C_FROZEN = 2.0
cfg = dict(N=N, n_train=n, gamma=GAMMA, beta=BETA, L=LS, S=[float(s) if np.isfinite(s) else "inf" for s in SS], channels=CHS, datasets=DSETS,
           splits=NSPLIT, lam0=0.1, c=C_FROZEN, c_source="D1/E12 (mean AUC 0.888 at c=2)", cv_grid="1e-8..1e2", folds=5)


def run(dname, sp):
    Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=100, scale="unit")
    Xall = np.vstack([Xtr, Xte])
    fam = SliceFamily(Xall, BETA, N)
    rows = []
    for ch in CHS:
        Kref, _ = embS_kernel(fam, ch, GAMMA, 1)
        for L in LS:
            Ks, _ = embS_kernel(fam, ch, GAMMA, L)
            for S in SS:
                seed = zlib.crc32(repr((dname, sp, ch, L, float(S))).encode()) % (2 ** 31)
                # same seed as E5: both c values see the same shot realisation
                res = evaluate_methods(Ks, Kref, n, ytr, yte, S, np.random.default_rng(abs(seed)), c=C_FROZEN)
                for m, v in res.items():
                    if m == "a_L_est":
                        continue
                    rows.append(dict(ds=dname, split=sp, ch=ch, L=L, S=float(S) if np.isfinite(S) else -1.0,
                                     method=("cnr_c2" if m == "cnr" else m), auc=v["auc"], lam=v["lam"]))
    return rows


jobs = [(d, s) for d in DSETS for s in range(NSPLIT)]
out = Parallel(n_jobs=6)(delayed(run)(*j) for j in jobs)
new = pd.DataFrame([r for o in out for r in o])
old = pd.read_csv(os.path.join(RES, "E5_raw.csv.gz"))
key = ["ds", "split", "ch", "L", "S"]

# ---- reproduction check of V1
chk = new[new.method.isin(["fixed", "fixed_centered", "cv_unc", "cv_cen"])].merge(old[["ds", "split", "ch", "L", "S", "method", "auc"]], on=key + ["method"], suffixes=("_x", "_v1"))
repro = dict(n_compared=int(len(chk)), max_abs_auc_diff=float((chk.auc_x - chk.auc_v1).abs().max()))
print("reproduction:", repro)

c2 = new[new.method == "cnr_c2"].rename(columns={"auc": "auc_c2"})[key + ["auc_c2"]]
base = old.pivot_table(index=key, columns="method", values="auc").reset_index()
M = base.merge(c2, on=key)
M["best_cv"] = M[["cv_unc", "cv_cen"]].max(axis=1)
assert len(M) == len(base)


def paired(df, a, b):
    d = (df[a] - df[b]).values
    mu, lo, hi = bootstrap_ci(d, reps=4000)
    return mu, lo, hi


def cell_table(sub, cnr_col):
    res = dict(vs_best_cv=dict(better=0, worse=0, tie=0, diffs_better=[], diffs_worse=[]),
               vs_fixed=dict(better=0, worse=0, tie=0, diffs_better=[], diffs_worse=[]))
    cells = []
    for (d, ch, L, S), g in sub.groupby(["ds", "ch", "L", "S"]):
        for nm, ref in (("vs_best_cv", "best_cv"), ("vs_fixed", "fixed")):
            mu, lo, hi = paired(g, cnr_col, ref)
            tag = "better" if lo > 0 else ("worse" if hi < 0 else "tie")
            res[nm][tag] += 1
            if tag == "better":
                res[nm]["diffs_better"].append(mu)
            if tag == "worse":
                res[nm]["diffs_worse"].append(mu)
            cells.append(dict(ds=d, ch=ch, L=int(L), S=float(S), cmp=nm, diff=mu, lo=lo, hi=hi, tag=tag))
    for nm in res:
        db, dw = res[nm].pop("diffs_better"), res[nm].pop("diffs_worse")
        res[nm]["median_gain_when_better"] = float(np.median(db)) if db else None
        res[nm]["max_gain"] = float(np.max(db)) if db else None
        res[nm]["median_loss_when_worse"] = float(np.median(dw)) if dw else None
        res[nm]["max_loss"] = float(np.min(dw)) if dw else None
    return res, cells


OUT = dict(reproduction_check=repro, groups={})
cells_all = {}
for gname, dsl in (("all_256", DSETS), ("D1_tuning_set", ["D1"]), ("D2_D4_heldout", ["D2", "D3", "D4"]), ("D2", ["D2"]), ("D3", ["D3"]), ("D4", ["D4"])):
    sub = M[M.ds.isin(dsl)]
    r1, c1 = cell_table(sub, "cnr")
    r2, c2_ = cell_table(sub, "auc_c2")
    OUT["groups"][gname] = dict(n_cells=int(sub.groupby(["ds", "ch", "L", "S"]).ngroups), c1=r1, c2=r2,
                                mean_auc=dict(cnr_c1=float(sub.cnr.mean()), cnr_c2=float(sub.auc_c2.mean()), best_cv=float(sub.best_cv.mean()),
                                              cv_cen=float(sub.cv_cen.mean()), cv_unc=float(sub.cv_unc.mean()), fixed=float(sub.fixed.mean())))
    if gname == "all_256":
        cells_all = dict(c1=c1, c2=c2_)
OUT["cells"] = cells_all
# by S and L for the headline cells (D1, L=64)
head = []
for (d, ch, S), g in M[(M.L == 64)].groupby(["ds", "ch", "S"]):
    head.append(dict(ds=d, ch=ch, S=float(S), cnr_c1=float(g.cnr.mean()), cnr_c2=float(g.auc_c2.mean()), cv_best=float(g.best_cv.mean()), cv_cen=float(g.cv_cen.mean()), fixed=float(g.fixed.mean())))
OUT["L64_means"] = head
# by shot level
byS = []
for S, g in M.groupby("S"):
    byS.append(dict(S=float(S), cnr_c1=float(g.cnr.mean()), cnr_c2=float(g.auc_c2.mean()), best_cv=float(g.best_cv.mean()), fixed=float(g.fixed.mean())))
OUT["by_S"] = byS
save("X3_E5_c2", cfg, OUT, t0)
for g, v in OUT["groups"].items():
    print(g, v["n_cells"], "c1:", {k: v["c1"]["vs_best_cv"][k] for k in ("better", "worse", "tie")}, "c2:", {k: v["c2"]["vs_best_cv"][k] for k in ("better", "worse", "tie")},
          {k: round(x, 4) for k, x in v["mean_auc"].items()})
