"""Channel algebra (Alg 1, Alg 11): Kraus, PTM, unitarity, kappa, u_N, equal-infidelity solver."""
import numpy as np
from scipy.optimize import brentq

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
PAULIS = [I2, X, Y, Z]
CHANNELS = ["dep", "ad", "pd", "bf"]
UNITAL = {"dep": True, "bf": True, "pd": True, "ad": False}


def kron_all(ops):
    o = ops[0]
    for a in ops[1:]:
        o = np.kron(o, a)
    return o


def kraus(ch, g):
    if ch == "dep":
        return [np.sqrt(1 - 3 * g / 4) * I2] + [np.sqrt(g / 4) * P for P in (X, Y, Z)]
    if ch == "bf":
        return [np.sqrt(1 - g) * I2, np.sqrt(g) * X]
    if ch == "pd":
        return [np.diag([1, np.sqrt(1 - g)]).astype(complex), np.diag([0, np.sqrt(g)]).astype(complex)]
    if ch == "ad":
        return [np.array([[1, 0], [0, np.sqrt(1 - g)]], complex), np.array([[0, np.sqrt(g)], [0, 0]], complex)]
    raise ValueError(ch)


def ptm(kr):
    M = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            M[a, b] = np.real(np.trace(PAULIS[a] @ sum(E @ PAULIS[b] @ E.conj().T for E in kr)) / 2)
    return M


def unit_stats(kr):
    M = ptm(kr)
    T = M[1:, 1:]
    t = M[1:, 0]
    u = np.trace(T.T @ T) / 3
    kappa = np.linalg.svd(T, compute_uv=False)[0]
    r = (3 - np.trace(T)) / 6
    return dict(u=float(u), kappa=float(kappa), kappa2=float(kappa ** 2), r=float(r), t=t, T=T)


def u_N_unital(u, N):
    return ((1 + 3 * u) ** N - 1) / (4 ** N - 1)


def u_N_numeric(kr, N):
    M = ptm(kr)
    MN = kron_all([M] * N)
    TN = MN[1:, 1:]
    return float(np.trace(TN.T @ TN) / (4 ** N - 1))


def gamma_for_infidelity(ch, r):
    if ch == "dep":
        return 2 * r
    if ch == "bf":
        return 1.5 * r
    if ch == "pd":
        return 1 - (1 - 3 * r) ** 2
    if ch == "ad":
        return brentq(lambda g: (2 + g - 2 * np.sqrt(1 - g)) / 6 - r, 1e-12, 0.9)
    raise ValueError(ch)


def single_qubit_superop(kr):
    """4x4 matrix S1 with vec_rowmajor(E rho E^dag) = S1 vec(rho)."""
    return sum(np.kron(E, E.conj()) for E in kr)


def local_noise_superop(kr, N):
    """Dense 4^N x 4^N superoperator of kr applied on each of N qubits (row-major vec)."""
    D = 2 ** N
    S = np.eye(D * D, dtype=complex)
    for q in range(N):
        def E(K):
            return kron_all([K if i == q else I2 for i in range(N)])
        S = sum(np.kron(E(K), E(K).conj()) for K in kr) @ S
    return S


def global_dep_superop(p, N):
    D = 2 ** N
    S = (1 - p) * np.eye(D * D, dtype=complex)
    v = np.eye(D).reshape(-1) / D
    S += p * np.outer(v, np.eye(D).reshape(-1))
    return S
