"""Z2: E4 at N = 8. Same protocol as e4_calib.py (N only changes). One job = (dataset, split, channel).
Usage: python3 z2_e4_n8.py D1 0 dep   -> results/z2_parts/E4N8_D1_sp0_dep.csv"""
from common import *
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import ridge
from nqk.metrics import *
from nqk.datasets import make_split
import pandas as pd
dname, sp, ch = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.time()
N = 8; D = 2 ** N; n, nte = 25, 50; Ls = [1, 4, 16, 64, 256]
Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=nte, scale="unit", n_feat=4)
ytr = ytr.astype(float); yte = yte.astype(float)
reps = int(np.ceil(N / Xtr.shape[1]))
Xa = np.hstack([np.vstack([Xtr, Xte])] * reps)[:, :N]
Vs = np.array(emb_R_layers(Xa, 2.0, N, 8))
Ks = noisy_kernels(Vs, ("local", kraus(ch, 0.05)), N, Ls)
rows = []
for L in Ls:
    K = Ks[L]; Ktr, Kte = K[:n, :n], K[n:, :n]
    for lam in (0.1, 1.0, 10.0):
        h = np.clip(ridge(Ktr, Kte, ytr, lam, False), -1, 1)
        pi_hat, pi = float((ytr > 0).mean()), float((yte > 0).mean())
        c = 1 / D if ch == "dep" else float(Ktr.mean())
        hb = c * ytr.sum() / (lam + c * n)
        rows.append(dict(dataset=dname, N=N, split=sp, ch=ch, L=L, lam=lam, pi_hat=pi_hat, pi=pi,
                         ece=ece(h, yte), ece_width=ece(h, yte, mode="width"), disp=dispersion(h), spread=spread(h),
                         mean_h=float(h.mean()), hbar=float(hb), ece_exact_pred=float(ece_exact(pi_hat, pi, n, D, lam, c)),
                         brier=brier(h, yte), auc=auc(h, yte), ece_const_target=float(abs((1 + hb) / 2 - pi))))
out = os.path.join(RES, "z2_parts"); os.makedirs(out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out, f"E4N8_{dname}_sp{sp}_{ch}.csv"), index=False)
print(dname, sp, ch, "done", round(time.time() - t0, 1), "s", flush=True)
