"""T1-T14 (developer guide Sec 4). Run: pytest -q tests/"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from scipy import stats
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import *
from nqk.metrics import *

G = 0.05


def test_T1_u_kappa():
    exp_u = dict(dep=0.9025, bf=0.873333, pd=0.966667, ad=0.934167)
    exp_k2 = dict(dep=0.9025, bf=1.0, pd=1.0, ad=0.95)
    for ch in CHANNELS:
        s = unit_stats(kraus(ch, G))
        assert abs(s["u"] - exp_u[ch]) < 1e-6, ch
        assert abs(s["kappa2"] - exp_k2[ch]) < 1e-9, ch


def test_T2_composition():
    M = ptm(kraus("pd", G))
    T = (M @ M)[1:, 1:]
    u = np.trace(T.T @ T) / 3
    u1 = unit_stats(kraus("pd", G))["u"]
    assert abs(u - 0.935) < 1e-9
    assert abs(u - (u1 ** 2 + 2 * G ** 2 / 9)) < 1e-12
    assert abs(u1 ** 2 - 0.934444) < 1e-6


def test_T3_uN():
    exp = dict(dep=0.7370, bf=0.6695, pd=0.9033)
    for ch, v in exp.items():
        kr = kraus(ch, G)
        u = unit_stats(kr)["u"]
        assert abs(u_N_numeric(kr, 4) - v) < 1e-4
        for N in range(1, 9):
            assert abs(u_N_numeric(kr, N) - u_N_unital(u, N)) < 1e-12 if N <= 5 else True
    kr = kraus("ad", G)
    assert abs(u_N_numeric(kr, 4) - 0.8181) < 1e-4
    assert abs(u_N_unital(unit_stats(kr)["u"], 4) - 0.8159) < 1e-4
    assert abs(u_N_numeric(kr, 4) - u_N_unital(unit_stats(kr)["u"], 4)) > 1e-3   # non-unital: formula fails


def _haar(rng, M, D):
    Z = (rng.standard_normal((M, D, D)) + 1j * rng.standard_normal((M, D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R, axis1=1, axis2=2)
    return Q * (d / np.abs(d))[:, None, :]


def haar_purity(ch, g, N, L, draws, seed, chunk=20000):
    rng = np.random.default_rng(seed)
    kr = kraus(ch, g)
    D = 2 ** N
    vals = []
    for s in range(0, draws, chunk):
        M = min(chunk, draws - s)
        R = np.zeros((M, D, D), complex)
        R[:, 0, 0] = 1
        for _ in range(L):
            U = _haar(rng, M, D)
            R = U @ R @ np.conj(np.transpose(U, (0, 2, 1)))
            R = apply_noise(R, ("local", kr), N)
        vals.append(np.real(np.einsum("mab,mba->m", R, R)) - 1 / D)
    v = np.concatenate(vals)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


@pytest.mark.parametrize("ch,g,N,L", [("dep", 0.1, 2, 3), ("bf", 0.1, 2, 3), ("bf", 0.1, 3, 2), ("pd", 0.2, 2, 2)])
def test_T4_haar_law(ch, g, N, L):
    m, se = haar_purity(ch, g, N, L, 200000, seed=11)
    u_N = u_N_numeric(kraus(ch, g), N)
    pred = (2 ** N - 1) / 2 ** N * u_N ** L
    assert abs(m - pred) < 4 * se, (m, se, pred)


def test_T5_equal_infidelity():
    r = 0.01
    exp = dict(dep=0.0200, bf=0.0150, pd=0.0591, ad=0.0299)
    for ch in CHANNELS:
        g = gamma_for_infidelity(ch, r)
        assert abs(g - exp[ch]) < 6e-5, (ch, g)
        s = unit_stats(kraus(ch, g))
        assert abs(s["r"] - r) < 1e-9
        assert abs(s["u"] - (1 - 4 * r)) < 1e-3


def test_T6_hierarchy_crossing():
    for g in (0.05, 0.2, 0.4, 0.6, 0.9):
        d = unit_stats(kraus("dep", g))["u"] - unit_stats(kraus("bf", g))["u"]
        assert abs(d - g * (2 - 5 * g) / 3) < 1e-12
    assert unit_stats(kraus("dep", 0.39))["u"] > unit_stats(kraus("bf", 0.39))["u"]
    assert unit_stats(kraus("dep", 0.41))["u"] < unit_stats(kraus("bf", 0.41))["u"]


def test_T7_global_dep_affine_and_CNR():
    N, L, pt = 4, 5, 0.1
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (14, N))
    Vs = zz_layers(X, N)
    Kn = noisy_kernels(Vs, None, N, [L])[L]
    Kg = noisy_kernels(Vs, ("global", pt), N, [L])[L]
    a = (1 - pt) ** (2 * L)
    D = 2 ** N
    assert abs(Kg - (a * Kn + (1 - a) / D)).max() < 1e-12
    y = np.where(rng.standard_normal(14) > 0, 1.0, -1.0)
    tr, te = slice(0, 9), slice(9, 14)
    lam0 = 0.1
    h0 = ridge(Kn[tr, tr], Kn[te, tr], y[tr], lam0, True)
    h1 = ridge(Kg[tr, tr], Kg[te, tr], y[tr], a * lam0, True)
    assert abs(h0 - h1).max() < 1e-9


def test_T8_ece_exact():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n, N = 25, 4
        D = 2 ** N
        lam = rng.choice([0.1, 1, 10])
        y = np.where(rng.random(n) < 0.7, 1.0, -1.0)
        ytest = np.where(rng.random(100) < 0.7, 1.0, -1.0)
        c = 1 / D
        hb = hbar(y, n, D, lam)
        # collapsed predictor from the closed form: K = c J  => h = c 1^T (cJ + lam I)^-1 y
        K = c * np.ones((n, n))
        h = (c * np.ones(n)) @ np.linalg.solve(K + lam * np.eye(n), y)
        assert abs(h - hb) < 1e-12
        pi_hat, pi = (y > 0).mean(), (ytest > 0).mean()
        direct = abs((1 + hb) / 2 - pi)
        assert abs(direct - ece_exact(pi_hat, pi, n, D, lam)) < 1e-12


def _layers_for(ch, g, N, X):
    return zz_layers(X, N), ("local", kraus(ch, g))


def test_T9_limits():
    N, L = 4, 800
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (10, N))
    Vs = zz_layers(X, N)
    K = noisy_kernels(Vs, ("local", kraus("dep", G)), N, [L])[L]
    assert abs(K - 1 / 16).max() < 1e-9
    noise = ("local", kraus("ad", G))
    K = noisy_kernels(Vs, noise, N, [L])[L]
    assert K.std(axis=0).max() < 1e-7       # i.e. independent of x (rows)
    assert K.std(axis=1).max() > 5e-3       # varies with x' (columns)
    Nsup = noise_superop(noise, N)
    for j in range(10):
        S, Sp = dense_layer_maps(Vs[j], Nsup)
        Kinf = limit_and_rate(Sp)[0]
        assert abs(K[0, j] - Kinf) < 1e-7
    assert 0.05 < K.min() and K.max() < 0.2


def test_T10_symmetry_audit():
    """CORRECTION to guide G3/T10: only PD is exactly symmetric (Pi0 is PD-invariant). Dep/BF kernels are
    asymmetric at O(1e-2) for small L>=2 (noise acts on the other side of Pi0 under circuit reversal) and the
    asymmetry decays to 0 with L; AD asymmetry persists (non-unital limit depends on x' only)."""
    N = 4
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (10, N))
    Vs = zz_layers(X, N)
    a = lambda ch, L: np.abs((lambda K: K - K.T)(noisy_kernels(Vs, ("local", kraus(ch, G)), N, [L])[L])).max()
    for L in (1, 2, 6, 40):
        assert a("pd", L) < 1e-12
    assert a("dep", 1) < 1e-12 and a("bf", 1) < 1e-12
    for ch in ("dep", "bf"):
        assert a(ch, 2) > 1e-4
        assert a(ch, 160) < 1e-10
    assert a("ad", 160) > 0.05


def test_T11_idle_dephasing_counterexample():
    def ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], complex)
    pd = kraus("pd", 0.1)
    for x, xp in [(0.3, 1.1), (1.0, 2.0), (2.5, 0.4)]:
        rho = ry(x) @ np.diag([1, 0]).astype(complex) @ ry(x).conj().T
        for _ in range(1500):
            rho = sum(E @ rho @ E.conj().T for E in pd)
        rho = ry(xp).conj().T @ rho @ ry(xp)
        k = rho[0, 0].real
        assert abs(k - (1 + np.cos(x) * np.cos(xp)) / 2) < 1e-9
        assert abs(k - 0.5) > 1e-3 or abs(np.cos(x) * np.cos(xp)) < 1e-3


def test_T12_zz_layer_matches_qiskit_and_embS_L_independent():
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Operator
    N = 3
    rng = np.random.default_rng(2)
    x = rng.uniform(0, np.pi, N)
    fm = ZZFeatureMap(N, reps=1, entanglement="full")
    qc = fm.assign_parameters(x)
    U = Operator(qc).data
    # qiskit qubit i = bit i (LSB); my qubit i = MSB side -> reverse feature order
    V = zz_layer(x[::-1], N, "full")
    ph = np.vdot(U.reshape(-1), V.reshape(-1))
    ph /= abs(ph)
    assert abs(U - ph * V).max() < 1e-10
    # Emb-S: noiseless total unitary independent of L
    xs = rng.uniform(0, 1, (3, 4))
    fam = SliceFamily(xs, 2.0, 4)
    ref = None
    for L in (1, 3, 8):
        tot = [np.linalg.matrix_power(v, L) for v in fam.slices(L)]
        if ref is None:
            ref = tot
        assert max(abs(a - b).max() for a, b in zip(tot, ref)) < 1e-10


def test_T13_rate_AD_matches_spectral():
    from nqk.metrics import fit_rate
    N = 4
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (10, N))
    Vs = zz_layers(X, N)
    noise = ("local", kraus("ad", G))
    Ls = list(range(1, 241))
    Ks = noisy_kernels(Vs, noise, N, Ls)
    delta = np.array([np.abs(Ks[L] - Ks[240]).max() for L in Ls[:-1]])
    slope, _, m, c, se = fit_rate(Ls[:-1], delta, (30, 80))
    Nsup = noise_superop(noise, N)
    rho = 0
    for V in Vs:
        S, Sp = dense_layer_maps(V, Nsup)
        rho = max(rho, limit_and_rate(S)[2], readout_sector_radius(Sp))
    pred = np.log(rho)
    assert abs(slope - pred) < max(0.02, 3 * se), (slope, pred, se)


def test_T14_shot_variance():
    rng = np.random.default_rng(0)
    S = 512
    for K in (0.05, 0.3, 0.7):
        draws = rng.binomial(S, K, 200000) / S
        var = draws.var(ddof=1)
        assert abs(var - K * (1 - K) / S) / (K * (1 - K) / S) < 0.02
        # chi-square on counts
        cnt = np.bincount(rng.binomial(S, K, 100000), minlength=S + 1)
        pk = stats.binom.pmf(np.arange(S + 1), S, K) * 100000
        m = pk > 5
        chi2 = ((cnt[m] - pk[m]) ** 2 / pk[m]).sum()
        assert stats.chi2.sf(chi2, m.sum() - 1) > 1e-4
