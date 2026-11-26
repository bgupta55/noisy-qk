"""Embeddings (Sec 2.3): Emb-S (exact slicing), Emb-R (repeated fixed layer), Emb-ZZ (Qiskit ZZFeatureMap layer)."""
import numpy as np
from itertools import combinations
from .channels import X, Z, I2, kron_all


def _op(single, i, N):
    return kron_all([single if k == i else I2 for k in range(N)])


def ham_S(x, beta, N):
    """H(x) = beta sum x_i X_i + beta^2 sum x_i x_{i+1} Z_i Z_{i+1} (open chain)."""
    D = 2 ** N
    H = np.zeros((D, D), complex)
    for i in range(N):
        H += beta * x[i] * _op(X, i, N)
    for i in range(N - 1):
        H += beta ** 2 * x[i] * x[i + 1] * (_op(Z, i, N) @ _op(Z, i + 1, N))
    return H


class SliceFamily:
    """exp(-i H(x) / L) for any L from one eigendecomposition per point."""

    def __init__(self, xs, beta, N):
        self.N = N
        self.eig = []
        for x in xs:
            w, V = np.linalg.eigh(ham_S(x, beta, N))
            self.eig.append((w, V))

    def slices(self, L):
        out = []
        for w, V in self.eig:
            out.append((V * np.exp(-1j * w / L)) @ V.conj().T)
        return out


def emb_R_layers(xs, beta, N, angle_div=8):
    """Fixed layer exp(-i H(x)/angle_div) repeated L times."""
    return SliceFamily(xs, beta, N).slices(angle_div)


def bits_table(N):
    idx = np.arange(2 ** N)
    return np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)])  # qubit 0 = MSB (matches kron order)


def zz_layer(x, N, entanglement="full"):
    """One ZZFeatureMap repetition: H on all, then P(2 x_i) and for pairs CX P(2 (pi-xi)(pi-xj)) CX.
    Returns D(x) @ H^{(x)N} (acts on |psi> as H first, then diagonal phases)."""
    b = bits_table(N)
    ph = np.zeros(2 ** N)
    for i in range(N):
        ph += 2 * x[i] * b[i]
    if entanglement == "full":
        pairs = list(combinations(range(N), 2))
    elif entanglement == "linear":
        pairs = [(i, i + 1) for i in range(N - 1)]
    else:
        raise ValueError(entanglement)
    for i, j in pairs:
        ph += 2 * (np.pi - x[i]) * (np.pi - x[j]) * (b[i] ^ b[j])
    Hn = kron_all([np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)] * N)
    return np.exp(1j * ph)[:, None] * Hn


def zz_layers(xs, N, entanglement="full"):
    return [zz_layer(x, N, entanglement) for x in xs]


def iqp_states(Xmat, N):
    """Wang et al. Eq (25)-(26): |phi> = U_Z H U_Z H |0>, U_Z = exp(i(sum_i x_i Z_i + sum_{i,j} x_i x_j Z_i Z_j))."""
    m = Xmat.shape[0]
    D = 2 ** N
    b = bits_table(N)
    zs = 1 - 2 * b
    lin = Xmat @ zs
    quad = (Xmat @ zs) ** 2
    UZ = np.exp(1j * (lin + quad))

    def had(psi):
        psi = psi.reshape((m,) + (2,) * N)
        for ax in range(1, N + 1):
            a0 = np.take(psi, 0, axis=ax)
            a1 = np.take(psi, 1, axis=ax)
            psi = np.stack([(a0 + a1), (a0 - a1)], axis=ax) / np.sqrt(2)
        return psi.reshape(m, D)

    psi = np.ones((m, D), complex) / np.sqrt(D)    # H|0>
    psi = UZ * psi                                  # U_Z
    psi = had(psi)                                  # H
    psi = UZ * psi                                  # U_Z
    return psi
