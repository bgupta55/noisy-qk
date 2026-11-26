"""Noisy kernel, Model B (compute-uncompute), Alg 2/3.

Convention: K[i,j] = Tr[Pi0 * Phi'_j^L Phi_i^L (Pi0)], i = x-arm (compute), j = x'-arm (uncompute).
Forward arm : rho_i^{(l+1)} = Noise( V_i rho V_i^dag )
Backward arm: O_j^{(l+1)}   = V_j Noise^dag(O) V_j^dag      (Hilbert-Schmidt adjoint of Noise o Ad(V_j^dag))
K[i,j] = Re Tr[O_j rho_i].
Noise specs: None | ("local", kraus_list) | ("local_pq", [kraus_q0, kraus_q1, ...]) | ("global", p)
"""
import numpy as np
from .channels import I2, kron_all, local_noise_superop, global_dep_superop


def _apply_kraus_qubit(R, kr, q, N):
    P = R.shape[0]
    A, B = 2 ** q, 2 ** (N - q - 1)
    R7 = R.reshape(P, A, 2, B, A, 2, B)
    out = np.zeros_like(R7)
    for K in kr:
        out += np.einsum("xi,paibcjd,yj->paxbcyd", K, R7, K.conj(), optimize=True)
    return out.reshape(R.shape)


def apply_noise(R, noise, N, dual=False):
    if noise is None:
        return R
    kind, par = noise
    if kind == "global":
        p = par
        D = R.shape[-1]
        tr = np.trace(R, axis1=-2, axis2=-1)
        return (1 - p) * R + p * tr[:, None, None] * np.eye(D)[None] / D
    if kind == "local":
        krs = [par] * N
    elif kind == "local_pq":
        krs = par
    else:
        raise ValueError(kind)
    for q in range(N):
        kr = [K.conj().T for K in krs[q]] if dual else krs[q]
        R = _apply_kraus_qubit(R, kr, q, N)
    return R


def _pi0(P, D):
    R = np.zeros((P, D, D), complex)
    R[:, 0, 0] = 1
    return R


def evolve(Vs, noise, N, snaps, backward=False):
    """Repeat the (fixed) layer V_p L times; return {L: array (P,D,D)} for L in snaps (L>=0)."""
    Vs = np.asarray(Vs)
    P, D = Vs.shape[0], Vs.shape[1]
    Vh = np.conj(np.transpose(Vs, (0, 2, 1)))
    R = _pi0(P, D)
    snaps = sorted(set(snaps))
    out = {}
    Lmax = max(snaps)
    if 0 in snaps:
        out[0] = R.copy()
    for l in range(1, Lmax + 1):
        if not backward:
            R = Vs @ R @ Vh
            R = apply_noise(R, noise, N)
        else:
            R = apply_noise(R, noise, N, dual=True)
            R = Vs @ R @ Vh
        if l in snaps:
            out[l] = R.copy()
    return out


def kernel_from_arms(Rf, Ob):
    """K[i,j] = Re Tr[O_j rho_i] = Re sum_ab O_j[a,b] rho_i[b,a]."""
    P = Rf.shape[0]
    Rflat = Rf.reshape(P, -1)
    Oflat_T = np.transpose(Ob, (0, 2, 1)).reshape(Ob.shape[0], -1)
    return np.real(Rflat @ Oflat_T.T)


def noisy_kernels(Vs, noise, N, Ls, Vs_back=None):
    """{L: K (P x P)} for a fixed layer family. K[i,j]: i x-arm (forward), j x'-arm (backward)."""
    Vb = Vs if Vs_back is None else Vs_back
    f = evolve(Vs, noise, N, Ls, backward=False)
    b = evolve(Vb, noise, N, Ls, backward=True)
    return {L: kernel_from_arms(f[L], b[L]) for L in Ls}


def noiseless_kernel(Vtot):
    psi = np.asarray(Vtot)[:, :, 0]
    G = np.conj(psi) @ psi.T
    return np.abs(G) ** 2


# ------------------------------------------------ dense reference (small N) and spectra
def ad_superop(V):
    return np.kron(V, V.conj())


def dense_layer_maps(V, Nsup):
    return Nsup @ ad_superop(V), Nsup @ ad_superop(V.conj().T)


def dense_kernel(Vs, Nsup, L, N):
    D = 2 ** N
    a0 = np.zeros(D * D, complex)
    a0[0] = 1
    A, B = [], []
    for V in Vs:
        S, Sp = dense_layer_maps(V, Nsup)
        A.append(np.linalg.matrix_power(S, L) @ a0)
        B.append(np.linalg.matrix_power(Sp.T, L) @ a0)
    return np.real(np.array(A) @ np.array(B).T)


def noise_superop(noise, N):
    kind, par = noise
    if kind == "global":
        return global_dep_superop(par, N)
    return local_noise_superop(par, N)


def limit_and_rate(Sp, tol_fix=1e-7, tol_ov=1e-9):
    """Alg 3. Returns (K_inf, rho2_all, rho2_sector, n_unit, n_unit_sector).
    K_inf = sigma_00 of the unit-eigenvalue eigenvector of Sp (normalised to trace 1).
    rho2_sector keeps only modes with non-negligible overlap with Pi0 (symmetry sector aware)."""
    w, V = np.linalg.eig(Sp)
    i = np.argmin(abs(w - 1))
    s = V[:, i]
    D = int(round(np.sqrt(len(s))))
    s = s.reshape(D, D)
    s = s / np.trace(s)
    Kinf = float(np.real(s[0, 0]))
    e0 = np.zeros(len(w), complex)
    e0[0] = 1
    c = np.linalg.lstsq(V, e0, rcond=None)[0]
    isfix = abs(w - 1) < tol_fix
    rho_all = float(np.max(abs(w[~isfix]))) if (~isfix).any() else 0.0
    sector = abs(c) > tol_ov
    j = np.where(isfix & sector)[0]
    rem = sector & ~isfix
    rho_sec = float(np.max(abs(w[rem]))) if rem.any() else 0.0
    return Kinf, rho_all, rho_sec, int(isfix.sum()), int(len(j))


def readout_sector_radius(S, tol_ov=1e-9):
    """Subdominant radius over modes of S with non-negligible readout overlap (e0^T v_k)."""
    w, V = np.linalg.eig(S)
    d = V[0, :]
    isfix = abs(w - 1) < 1e-7
    rem = (abs(d) > tol_ov) & ~isfix
    return float(np.max(abs(w[rem]))) if rem.any() else 0.0


def arm_radius(M, tol_ov=1e-9):
    """Sector-aware subdominant spectral radius of one arm: modes of M that (a) are not eigenvalue 1 and
    (b) have non-negligible component of e0 (= vec Pi0) in the eigen-expansion e0 = sum_k c_k v_k.
    Forward arm: M = S (layer map); backward arm: M = Sp^T."""
    w, V = np.linalg.eig(M)
    e0 = np.zeros(len(w), complex)
    e0[0] = 1
    c = np.linalg.lstsq(V, e0, rcond=None)[0]
    isfix = abs(w - 1) < 1e-7
    rem = (abs(c) > tol_ov) & ~isfix
    return float(np.max(abs(w[rem]))) if rem.any() else 0.0


def dense_kernel_limit(Vs, Nsup, L=8192):
    """Reference limit kernel by exact dense matrix powers (binary exponentiation); valid with symmetry sectors."""
    return dense_kernel(Vs, Nsup, L, int(round(np.log2(Vs[0].shape[0]))))


def arm_radius_weighted(M, tol_w=1e-6):
    """Contribution-weighted subdominant radius (refinement logged as DEVIATION D2): keep modes k whose weight
    |(e0^T v_k)(c_k)| exceeds tol_w, c = V^{-1} e0, i.e. modes that couple the initial state to the readout."""
    w, V = np.linalg.eig(M)
    e0 = np.zeros(len(w), complex)
    e0[0] = 1
    c = np.linalg.lstsq(V, e0, rcond=None)[0]
    wt = np.abs(V[0, :] * c)
    isfix = abs(w - 1) < 1e-7
    rem = (wt > tol_w) & ~isfix
    return float(np.max(abs(w[rem]))) if rem.any() else 0.0
