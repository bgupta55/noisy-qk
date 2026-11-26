"""Kernel ridge regression utilities (Alg 4-7): symmetrise/PSD, ridge, centering, CV, CNR, shot model."""
import numpy as np


def symmetrise_psd(K, clip=True):
    Ks = (K + K.T) / 2
    if not clip:
        return Ks
    w, Q = np.linalg.eigh(Ks)
    return (Q * np.maximum(w, 0)) @ Q.T


def audit(K):
    Ks = (K + K.T) / 2
    w, Q = np.linalg.eigh(Ks)
    wc = np.maximum(w, 0)
    Kp = (Q * wc) @ Q.T
    pr = (wc.sum() ** 2) / max((wc ** 2).sum(), 1e-300)
    return dict(asym=float(np.abs(K - K.T).max()), min_eig=float(w.min()),
                mass_removed=float(np.linalg.norm(Kp - Ks)), eff_rank=float(pr))


def ridge(Ktr, Kte, y, lam, center):
    n = len(y)
    if center:
        mk = Ktr.mean(0)
        mm = Ktr.mean()
        Kc = Ktr - mk[None] - mk[:, None] + mm
        Kec = Kte - Kte.mean(1, keepdims=True) - mk[None] + mm
        return Kec @ np.linalg.solve(Kc + lam * np.eye(n), y - y.mean()) + y.mean()
    return Kte @ np.linalg.solve(Ktr + lam * np.eye(n), y)


def center_train(Ktr):
    mk = Ktr.mean(0)
    return Ktr - mk[None] - mk[:, None] + Ktr.mean()


LAM_GRID = 10.0 ** np.arange(-8, 3)


def cv_lam(Ktr, y, center, grid=LAM_GRID, folds=5, seed=0):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_of = np.empty(n, int)
    fold_of[idx] = np.arange(n) % folds
    err = np.zeros(len(grid))
    for f in range(folds):
        tr = fold_of != f
        te = ~tr
        if te.sum() == 0 or tr.sum() < 2:
            continue
        for gi, lam in enumerate(grid):
            try:
                h = ridge(Ktr[np.ix_(tr, tr)], Ktr[np.ix_(te, tr)], y[tr], lam, center)
            except np.linalg.LinAlgError:
                err[gi] = np.inf
                continue
            err[gi] += np.mean((np.clip(h, -1, 1) - y[te]) ** 2)
    return float(grid[int(np.argmin(err))])


def cnr_lambda(Ktr, Ktr_ref, n, S, lam0=0.1, c=1.0):
    """CNR (Alg 5): lambda' = max(lam0 * a_L, c*sqrt(n/S)), a_L = tr(Kc)/tr(Kc_ref)."""
    a = np.trace(center_train(Ktr)) / max(np.trace(center_train(Ktr_ref)), 1e-300)
    floor = 0.0 if (S is None or np.isinf(S)) else c * np.sqrt(n / S)
    return max(lam0 * a, floor), float(a)


def shot_sample(Ktr, Kte, S, rng):
    """Alg 7: binomial shot noise; train block drawn on the upper triangle (incl. diagonal) and mirrored."""
    if S is None or np.isinf(S):
        return Ktr, Kte
    n = Ktr.shape[0]
    iu = np.triu_indices(n)
    Kh = np.zeros_like(Ktr)
    Kh[iu] = rng.binomial(int(S), np.clip(Ktr[iu], 0, 1)) / S
    Kh = Kh + Kh.T - np.diag(np.diag(Kh))
    Kth = rng.binomial(int(S), np.clip(Kte, 0, 1)) / S
    return Kh, Kth
