import zlib
"""E5 (central): ranking survives in exact arithmetic; resolution (shots) decides.
Emb-S (beta=2, x in [0,1]), N=4, n_train=30, gamma=0.02/slice, lam0=0.1, L in {1,4,16,64}, S in {256,1024,4096,inf},
methods: fixed (uncentred), fixed_centered, cv_unc, cv_cen, cnr (c=1; reference = same channel at L=1 with same S).
Datasets D1, D2, D4 (D3 added as extra), 20 paired splits, same shot realisation for all methods."""
from common import *
from joblib import Parallel, delayed
from nqk.datasets import make_split
from nqk.embeddings import SliceFamily
from nqk.pipeline import embS_kernel, evaluate_methods
from nqk.krr import ridge, cv_lam, center_train
from nqk.metrics import auc, dispersion

t0 = time.time()
N, n, GAMMA, BETA = 4, 30, 0.02, 2.0
LS = [1, 4, 16, 64]
SS = [256, 1024, 4096, np.inf]
CHS = ["dep", "ad", "pd", "bf"]
DSETS = ["D1", "D2", "D4", "D3"]
NSPLIT = 20
cfg = dict(N=N, n_train=n, gamma=GAMMA, beta=BETA, L=LS, S=[float(s) if np.isfinite(s) else "inf" for s in SS], channels=CHS, datasets=DSETS,
           splits=NSPLIT, lam0=0.1, c=1.0, cv_grid="1e-8..1e2", folds=5)


def run(dname, sp):
    Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=100, scale="unit")
    P = len(ytr) + len(yte)
    Xall = np.vstack([Xtr, Xte])
    fam = SliceFamily(Xall, BETA, N)
    rows = []
    # noiseless exact reference (L-independent)
    Kn, _ = embS_kernel(fam, None, 0, 1)
    ytr_f = ytr.astype(float)
    lc = cv_lam(Kn[:n, :n], ytr_f, True)
    h = ridge(Kn[:n, :n], Kn[n:, :n], ytr_f, lc, True)
    noiseless = dict(auc=auc(h, yte), lam=lc)
    for ch in CHS:
        Kref, _ = embS_kernel(fam, ch, GAMMA, 1)
        for L in LS:
            Ks, Kraw = embS_kernel(fam, ch, GAMMA, L)
            asym = float(np.abs(Kraw - Kraw.T).max())
            a_exact = float(np.trace(center_train(Ks[:n, :n])) / np.trace(center_train(Kref[:n, :n])))
            for S in SS:
                seed = zlib.crc32(repr((dname, sp, ch, L, float(S))).encode()) % (2 ** 31)
                rng = np.random.default_rng(abs(seed))
                res = evaluate_methods(Ks, Kref, n, ytr, yte, S, rng)
                for m, v in res.items():
                    if m == "a_L_est":
                        continue
                    rows.append(dict(ds=dname, split=sp, ch=ch, L=L, S=float(S) if np.isfinite(S) else -1.0, method=m, a_exact=a_exact,
                                     a_est=res["a_L_est"], asym=asym, auc_noiseless=noiseless["auc"], **v))
    return rows


jobs = [(d, s) for d in DSETS for s in range(NSPLIT)]
out = Parallel(n_jobs=10)(delayed(run)(*j) for j in jobs)
rows = [r for o in out for r in o]
import pandas as pd
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(RES, "E5_raw.parquet")) if False else df.to_csv(os.path.join(RES, "E5_raw.csv.gz"), index=False)
# aggregate with bootstrap CI
agg = []
for (d, ch, L, S, m), g in df.groupby(["ds", "ch", "L", "S", "method"]):
    mu, lo, hi = bootstrap_ci(g.auc.values, reps=4000)
    agg.append(dict(ds=d, ch=ch, L=int(L), S=S, method=m, auc=mu, lo=lo, hi=hi, acc=float(g.acc.mean()), brier=float(g.brier.mean()),
                    ece=float(g.ece.mean()), disp=float(g.disp.mean()), spread=float(g.spread.mean()), a_exact=float(g.a_exact.mean()),
                    auc_noiseless=float(g.auc_noiseless.mean()), n=len(g)))
# paired differences CNR - CV(best-of cen/unc per split) and CNR - fixed
pairs = []
for (d, ch, L, S), g in df.groupby(["ds", "ch", "L", "S"]):
    piv = g.pivot(index="split", columns="method", values="auc")
    best_cv = piv[["cv_unc", "cv_cen"]].max(axis=1)
    for name, ref in (("cnr-cv_cen", piv["cv_cen"]), ("cnr-cv_unc", piv["cv_unc"]), ("cnr-best_cv", best_cv), ("cnr-fixed", piv["fixed"]), ("cnr-fixed_centered", piv["fixed_centered"])):
        diff = (piv["cnr"] - ref).values
        mu, lo, hi = bootstrap_ci(diff, reps=4000)
        pairs.append(dict(ds=d, ch=ch, L=int(L), S=S, cmp=name, diff=mu, lo=lo, hi=hi, cnr_better=bool(lo > 0), cnr_worse=bool(hi < 0)))
save("E5_ranking_vs_resolution", cfg, dict(aggregate=agg, paired=pairs), t0)
A = pd.DataFrame(agg)
pd.set_option("display.width", 250)
for d in ("D1",):
    for S in (256.0, 1024.0, 4096.0, -1.0):
        print("=== ", d, "S=", S)
        t = A[(A.ds == d) & (A.S == S)].pivot_table(index=["ch", "L"], columns="method", values="auc").round(3)
        print(t.to_string())
