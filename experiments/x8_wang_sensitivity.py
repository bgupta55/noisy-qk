"""X8: sensitivity of the E0 training-error transition to the choices Wang et al. (arXiv:2401.17526, Sec. IV) leave unspecified.
Sweep: (feature scale after PCA) x (pair-sum convention in U_Z) x seeds 0..7 (seeds 0-4 = E0).
Everything else as E0: n=500/500, p~=0.1 per layer on 2L layers, lambda=0.5, uncentred ridge, h clipped to [-1,1], train error = frequency of sign(h)!=y.
Wang reference values are READ OFF their Fig. 3b (reference/wang_p7_fig3.png) by eye and are approximate (+-0.03)."""
from common import *
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from nqk.embeddings import bits_table
from nqk.krr import ridge

t0 = time.time()
N, n, nte, pt, lam = 10, 500, 500, 0.1, 0.5
D = 2 ** N
LS = list(range(0, 61))
SEEDS = list(range(8))
SCALES = ["unit", "pi", "std", "raw"]
PAIRS = ["all_ordered_incl_diag", "i_lt_j", "i_ne_j_ordered"]
WANG = {24: 0.07, 28: 0.28, 32: 0.47}   # read off Fig. 3b, approximate
cfg = dict(N=N, n=n, n_test=nte, p_tilde=pt, lam=lam, L=[LS[0], LS[-1]], seeds=SEEDS, scales=SCALES, pair_conventions=PAIRS, wang_readoff_fig3b=WANG)


def data(seed, scale):
    z = np.load(os.path.join(ROOT, "reference", "fashion_binary.npz"))
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n], idx[n:n + nte]
    Xtr, Xte = X[tr] / 255.0, X[te] / 255.0
    p = PCA(N, random_state=0).fit(Xtr)
    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    if scale in ("unit", "pi"):
        hi = 1.0 if scale == "unit" else np.pi
        mm = MinMaxScaler((0, hi)).fit(Xtr)
        Xtr, Xte = np.clip(mm.transform(Xtr), 0, hi), np.clip(mm.transform(Xte), 0, hi)
    elif scale == "std":
        mu, sd = Xtr.mean(0), Xtr.std(0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xtr, Xte, y[tr], y[te]


def states(Xm, pairs):
    m = Xm.shape[0]
    zs = 1 - 2 * bits_table(N)
    lin = Xm @ zs
    s = lin ** 2                              # (sum_i x_i z_i)^2 = sum_{i,j ordered} x_i x_j z_i z_j
    diag = (Xm ** 2).sum(1)[:, None]          # i=j terms (z_i^2 = 1)
    if pairs == "all_ordered_incl_diag":
        quad = s
    elif pairs == "i_ne_j_ordered":
        quad = s - diag
    elif pairs == "i_lt_j":
        quad = (s - diag) / 2
    UZ = np.exp(1j * (lin + quad))

    def had(psi):
        psi = psi.reshape((m,) + (2,) * N)
        for ax in range(1, N + 1):
            a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([(a0 + a1), (a0 - a1)], axis=ax) / np.sqrt(2)
        return psi.reshape(m, D)
    psi = UZ * (np.ones((m, D), complex) / np.sqrt(D))
    return UZ * had(psi)


def run(seed, scale, pairs):
    Xtr, Xte, ytr, yte = data(seed, scale)
    P = states(np.vstack([Xtr, Xte]), pairs)
    K = np.abs(np.conj(P) @ P.T) ** 2
    Ktr, Kte = K[:n, :n], K[n:, :n]
    off = Ktr[~np.eye(n, dtype=bool)]
    et, ee = [], []
    for L in LS:
        a = (1 - pt) ** (2 * L)
        Kt = a * Ktr + (1 - a) / D
        Ke = a * Kte + (1 - a) / D
        h_tr = np.clip(ridge(Kt, Kt, ytr, lam, False), -1, 1)
        h_te = np.clip(ridge(Kt, Ke, ytr, lam, False), -1, 1)
        et.append(float(np.mean(np.where(h_tr >= 0, 1, -1) != ytr)))
        ee.append(float(np.mean(np.where(h_te >= 0, 1, -1) != yte)))
    return dict(seed=seed, scale=scale, pairs=pairs, err_tr=et, err_te=ee, offdiag_mean=float(off.mean()), offdiag_std=float(off.std()))


jobs = [(s, sc, pr) for sc in SCALES for pr in PAIRS for s in SEEDS]
runs = Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs)


def first(arr, thr):
    for L, v in zip(LS, arr):
        if v >= thr:
            return L
    return None


def cross(arr, thr):
    """linearly interpolated L where arr first reaches thr"""
    for i in range(1, len(arr)):
        if arr[i] >= thr > arr[i - 1]:
            return LS[i - 1] + (thr - arr[i - 1]) / (arr[i] - arr[i - 1])
    return None


THR = [0.05, 0.10, 0.25, 0.40]
summary = []
for sc in SCALES:
    for pr in PAIRS:
        R = [r for r in runs if r["scale"] == sc and r["pairs"] == pr]
        E = np.array([r["err_tr"] for r in R])              # seeds x L
        m5 = E[:5].mean(0)                                   # E0-comparable (seeds 0-4)
        m8 = E.mean(0)
        row = dict(scale=sc, pairs=pr, offdiag_mean=float(np.mean([r["offdiag_mean"] for r in R])), offdiag_std=float(np.mean([r["offdiag_std"] for r in R])))
        for thr in THR:
            row[f"L_first_{thr}_mean5"] = first(m5, thr)
            row[f"L_cross_{thr}_mean5"] = cross(m5, thr)
            per = [cross(e, thr) for e in E]
            per_ok = [p for p in per if p is not None]
            row[f"L_cross_{thr}_per_seed_min"] = float(min(per_ok)) if per_ok else None
            row[f"L_cross_{thr}_per_seed_max"] = float(max(per_ok)) if per_ok else None
            row[f"L_cross_{thr}_per_seed_sd"] = float(np.std(per_ok, ddof=1)) if len(per_ok) > 1 else None
        row["err_tr_at_24_28_32_mean5"] = [float(m5[24]), float(m5[28]), float(m5[32])]
        row["err_tr_at_24_28_32_mean8"] = [float(m8[24]), float(m8[28]), float(m8[32])]
        row["rms_to_wang_readoff_mean5"] = float(np.sqrt(np.mean([(m5[L] - v) ** 2 for L, v in WANG.items()])))
        row["err_tr_max_seedspread_at_28"] = [float(E[:, 28].min()), float(E[:, 28].max())]
        row["err_tr_mean5"] = [float(x) for x in m5]
        summary.append(row)

# reproduction check against E0 for the baseline variant
E0 = json.load(open(os.path.join(RES, "E0_wang_reproduction.json")))["results"]["aggregate"]
base = next(s for s in summary if s["scale"] == "unit" and s["pairs"] == "all_ordered_incl_diag")
repro = float(max(abs(a["err_tr"] - b) for a, b in zip(E0, base["err_tr_mean5"])))
print("reproduction vs E0 (max |diff| of 5-seed mean train error):", repro)

# per-seed spread of the L(err>=0.25) crossing in the baseline
res = dict(reproduction_max_abs_diff_vs_E0=repro, summary=summary,
           wang_text_prediction=float(np.log(500) / np.log((1 - pt) ** -2)),
           seeds_used=SEEDS)
save("X8_wang_sensitivity", cfg, res, t0)
pd_cols = ["scale", "pairs", "L_cross_0.05_mean5", "L_cross_0.1_mean5", "L_cross_0.25_mean5", "L_cross_0.4_mean5", "L_cross_0.25_per_seed_sd", "rms_to_wang_readoff_mean5"]
import pandas as pd
print(pd.DataFrame(summary)[pd_cols].round(2).to_string())
