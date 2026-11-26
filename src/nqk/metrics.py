"""Metrics (Alg 8) and theory helpers (Alg 9, 10, 14, Wang bound)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def to_prob(h):
    return (np.clip(h, -1, 1) + 1) / 2


def auc(h, y):
    y01 = (y > 0).astype(int)
    if len(np.unique(y01)) < 2:
        return np.nan
    return float(roc_auc_score(y01, h))


def accuracy(h, y):
    return float(np.mean(np.where(h >= 0, 1, -1) == y))


def brier(h, y):
    return float(np.mean((to_prob(h) - (y > 0)) ** 2))


def ece(h, y, M=10, mode="mass"):
    p = to_prob(h)
    y01 = (y > 0).astype(float)
    n = len(p)
    if mode == "mass":
        bins = np.array_split(np.argsort(p, kind="stable"), M)
    else:
        edges = np.linspace(0, 1, M + 1)
        b = np.clip(np.digitize(p, edges[1:-1]), 0, M - 1)
        bins = [np.where(b == k)[0] for k in range(M)]
    e = 0.0
    for ix in bins:
        if len(ix):
            e += len(ix) / n * abs(p[ix].mean() - y01[ix].mean())
    return float(e)


def dispersion(h):
    p = to_prob(h)
    return float(np.mean(np.abs(p - p.mean())))


def spread(h):
    return float(np.mean(np.abs(h - h.mean())))


def hbar(y, n, D, lam, c=None):
    c = 1.0 / D if c is None else c
    return c * y.sum() / (lam + c * n)


def ece_exact(pi_hat, pi, n, D, lam, c=None):
    """Prop 6: |(pi_hat - pi) c n/(lam + c n) - (2 pi - 1) lam / (2 (lam + c n))|."""
    c = 1.0 / D if c is None else c
    return abs((pi_hat - pi) * c * n / (lam + c * n) - (2 * pi - 1) * lam / (2 * (lam + c * n)))


def wang_bound(n, D, lam, p_tilde, L, delta=0.05):
    """Wang et al. Thm III.1 (as printed): f(z)=(z+8 sqrt(z/lam))/(1-z), z=(n/lam)(1-p)(1+1/D)... returns total, f, t2, t3."""
    p = 1 - (1 - p_tilde) ** (2 * L)
    z = n / lam * (1 - p) * (1 + 1 / D)
    f = np.inf if z >= 1 else (z + 8 * np.sqrt(z / lam)) / (1 - z)
    t2 = 8 * np.sqrt(D * n) / (D * lam + n)
    t3 = 6 * np.sqrt(np.log(4 / delta) / (2 * n))
    return f + t2 + t3, f, t2, t3


def fit_rate(Ls, delta, window, floor=1e-11):
    Ls = np.asarray(Ls, float)
    m = (Ls >= window[0]) & (Ls <= window[1]) & (delta > floor)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum()), np.nan, np.nan
    x, yv = Ls[m], np.log(delta[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = (resid ** 2).sum() / dof
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0]))
    A2 = np.vstack([x, np.ones_like(x), np.log(x)]).T
    c2 = np.linalg.lstsq(A2, yv, rcond=None)[0]
    return float(coef[0]), float(coef[1]), int(m.sum()), float(c2[2]), se


def L_shot(s0, S, n, rate_per_layer, c=1.0):
    arg = s0 * np.sqrt(S / n) / c
    if arg <= 1 or rate_per_layer <= 0:
        return 0.0
    return float(np.log(arg) / rate_per_layer)


def tv_bound(eps, n, S, v):
    return float(min(1.0, eps * np.sqrt(2 * n * S / v)))
