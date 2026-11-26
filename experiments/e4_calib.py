"""E4: calibration (Prop 6). Exact signed expression vs measured ECE/dispersion.
D2 (imbalanced, 37% positive) and D1 (balanced); channels dep (local) and ad; lambda in {0.1,1,10}; N in {4,6,8}.
Layer: Emb-ZZ-like fixed layer replaced by Emb-R (angle H/8 with N-chain) so any N is available; depth L grid; local noise via tensor method."""
from common import *
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import ridge
from nqk.metrics import *
from nqk.datasets import make_split
t0 = time.time()
n, nte = 25, 50
Ls = [1, 4, 16, 64, 256]
rows = []
for dname in ("D2", "D1"):
    for N in (4, 6):
        D = 2 ** N
        for sp in range(3):
            Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=nte, scale="unit", n_feat=4)
            ytr = ytr.astype(float); yte = yte.astype(float)
            # features: pad/tile to N dims
            reps = int(np.ceil(N / Xtr.shape[1]))
            Xa = np.hstack([np.vstack([Xtr, Xte])] * reps)[:, :N]
            Vs = np.array(emb_R_layers(Xa, 2.0, N, 8)) if N <= 6 else np.array(emb_R_layers(Xa, 2.0, N, 8))
            for ch in ("dep", "ad"):
                noise = ("local", kraus(ch, 0.05))
                Ks = noisy_kernels(Vs, noise, N, Ls)
                for L in Ls:
                    K = Ks[L]
                    Ktr, Kte = K[:n, :n], K[n:, :n]
                    for lam in (0.1, 1.0, 10.0):
                        h = np.clip(ridge(Ktr, Kte, ytr, lam, False), -1, 1)
                        pi_hat, pi = float((ytr > 0).mean()), float((yte > 0).mean())
                        # collapsed-constant targets: for dep c=1/D ; for AD the limit is x'-dependent: use empirical mean of K entries as c
                        c = 1 / D if ch == "dep" else float(Ktr.mean())
                        hb = c * ytr.sum() / (lam + c * n)
                        ece_pred = ece_exact(pi_hat, pi, n, D, lam, c)
                        rows.append(dict(dataset=dname, N=N, split=sp, ch=ch, L=L, lam=lam, pi_hat=pi_hat, pi=pi,
                                         ece=ece(h, yte), ece_width=ece(h, yte, mode="width"), disp=dispersion(h), spread=spread(h),
                                         mean_h=float(h.mean()), hbar=float(hb), ece_exact_pred=float(ece_pred),
                                         brier=brier(h, yte), auc=auc(h, yte), ece_const_target=float(abs((1 + hb) / 2 - pi))))
    print(dname, "done", time.time() - t0)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "E4_raw.csv.gz"), index=False)
# summary on collapsed regime: (a) dep, L=256
sm = []
for (dname, N, lam), g in df[(df.ch == "dep") & (df.L == 256)].groupby(["dataset", "N", "lam"]):
    sm.append(dict(dataset=dname, N=N, lam=lam, ece=float(g.ece.mean()), ece_width=float(g.ece_width.mean()), ece_const_target=float(g.ece_const_target.mean()), ece_exact=float(g.ece_exact_pred.mean()), abs_diff_width=float((g.ece_width - g.ece_exact_pred).abs().mean()), abs_diff_const=float((g.ece_const_target - g.ece_exact_pred).abs().mean()),
                   abs_diff=float((g.ece - g.ece_exact_pred).abs().mean()), disp=float(g.disp.mean()), spread=float(g.spread.mean()), auc=float(g.auc.mean())))
sm_ad = []
for (dname, N, lam), g in df[(df.ch == "ad") & (df.L == 256)].groupby(["dataset", "N", "lam"]):
    sm_ad.append(dict(dataset=dname, N=N, lam=lam, ece=float(g.ece.mean()), ece_width=float(g.ece_width.mean()), ece_const_target=float(g.ece_const_target.mean()), disp=float(g.disp.mean()), spread=float(g.spread.mean()), auc=float(g.auc.mean())))
agg = df.groupby(["dataset", "ch", "N", "lam", "L"])[["ece", "ece_width", "ece_const_target", "ece_exact_pred", "disp", "spread", "auc", "brier"]].mean().reset_index()
save("E4_calibration", dict(n=n, n_test=nte, Ls=Ls, gamma=0.05), dict(summary_dep_L256=sm, summary_ad_L256=sm_ad, curves=agg.to_dict("records")), t0)
print(pd.DataFrame(sm).round(4).to_string())
print(pd.DataFrame(sm_ad).round(4).to_string())
