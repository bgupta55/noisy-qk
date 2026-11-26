"""Datasets D1-D9 (Sec 2.1) and split utilities. All preprocessing is fitted on the training split only."""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _binary(Xd, yd, a, b):
    m = (yd == a) | (yd == b)
    return Xd[m], np.where(yd[m] == b, 1, -1)


def load_dataset(name, seed=0):
    """Return (X, y, n_qubits_dim, use_pca)."""
    if name == "D1":
        d = load_iris()
        X, y = _binary(d.data, d.target, 1, 2)
        return X, y, 4, False
    if name == "D2":
        d = load_breast_cancer()
        # malignant(0) is the minority (~37%); label +1 = malignant
        return d.data, np.where(d.target == 0, 1, -1), 4, True
    if name == "D3":
        d = load_wine()
        X, y = _binary(d.data, d.target, 0, 1)
        return X, y, 4, True
    if name == "D4":
        d = load_digits()
        X, y = _binary(d.data, d.target, 3, 8)
        return X, y, 4, True
    if name == "D6":
        rng = np.random.default_rng(1234)
        X = rng.uniform(0, 1, (200, 4))
        y = np.sign(np.sin(X[:, 0] * X[:, 1] * 6) + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(200) - 0.15)
        y[y == 0] = 1
        return X, y.astype(int), 4, False
    if name == "D7":
        X, y = make_moons(200, noise=0.15, random_state=0)
        return X, np.where(y == 1, 1, -1), 2, False
    if name == "D5":
        # ad-hoc style 2-qubit set (Havlicek-type): label by sign of parity-phase function, gap 0.3
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 2 * np.pi, (400, 2))
        f = np.sin(X[:, 0]) * np.sin(X[:, 1]) + 0.0
        keep = np.abs(f) > 0.25
        X, f = X[keep], f[keep]
        return X[:200], np.where(f[:200] > 0, 1, -1), 2, False
    raise ValueError(name)


def make_split(name, n_train, seed, n_test_cap=100, scale="unit", n_feat=None):
    X, y, dim, use_pca = load_dataset(name)
    nf = dim if n_feat is None else n_feat
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n_train, stratify=y, random_state=seed)
    if len(yte) > n_test_cap:
        Xte, _, yte, _ = train_test_split(Xte, yte, train_size=n_test_cap, stratify=yte, random_state=seed)
    ss = StandardScaler().fit(Xtr)
    Xtr, Xte = ss.transform(Xtr), ss.transform(Xte)
    if use_pca:
        p = PCA(nf, random_state=0).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    hi = 1.0 if scale == "unit" else np.pi
    mm = MinMaxScaler((0, hi)).fit(Xtr)
    Xtr = np.clip(mm.transform(Xtr), 0, hi)
    Xte = np.clip(mm.transform(Xte), 0, hi)
    return Xtr, Xte, ytr, yte


def fashion_shirt_dress(n_train, n_test, seed, N=10):
    """Wang et al. Sec IV: Fashion-MNIST shirts (6) vs dresses (3); PCA to N dims (fitted on train); scaled to [0,1]."""
    import os
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "reference", "fashion_binary.npz")
    cache = os.path.abspath(cache)
    if not os.path.exists(cache):
        from sklearn.datasets import fetch_openml
        d = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        t = d.target.astype(int)
        m = (t == 6) | (t == 3)
        np.savez_compressed(cache, X=d.data[m].astype(np.float32), y=np.where(t[m] == 6, 1, -1))
    z = np.load(cache)
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
    Xtr, Xte = X[tr] / 255.0, X[te] / 255.0
    p = PCA(N, random_state=0).fit(Xtr)
    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    mm = MinMaxScaler((0, 1)).fit(Xtr)
    return np.clip(mm.transform(Xtr), 0, 1), np.clip(mm.transform(Xte), 0, 1), y[tr], y[te]
