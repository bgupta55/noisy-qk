"""E3a (exact global-dep identity, Sherman-Morrison, CNR), E3b (two-term collapse bound, measured vs guaranteed depth),
E9a (idle dephasing non-ergodic counterexample), E9b (H layer makes PD/BF layer map primitive), E9c (Emb-S/R + BF parity sector)."""
from common import *
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import ridge, cv_lam
from nqk.metrics import auc, spread, fit_rate, hbar
from nqk.datasets import make_split
t0 = time.time()
res = {}

# ---------------------------------------------------------------- E3a
rows = []
for N in (4, 6):
    D = 2 ** N
    rng = np.random.default_rng(N)
    X = rng.uniform(0, np.pi, (20, N))
    Vs = zz_layers(X, N)
    y = np.where(rng.random(20) < 0.5, 1.0, -1.0)
    n = 14
    for pt in (0.02, 0.1):
        for L in (1, 3, 8, 20):
            Kn = noisy_kernels(Vs, None, N, [L])[L]
            Kn = noiseless_kernel([np.linalg.matrix_power(V, L) for V in Vs]) if False else noisy_kernels(Vs, None, N, [L])[L]
            Kg = noisy_kernels(Vs, ("global", pt), N, [L])[L]
            a = (1 - pt) ** (2 * L)
            dev_aff = float(np.abs(Kg - (a * Kn + (1 - a) / D)).max())
            lam = 0.1
            A, k = Kg[:n, :n], Kg[n:, :n]
            h_direct = k @ np.linalg.solve(A + lam * np.eye(n), y[:n])
            # Sherman-Morrison: K~ = a K + b J ; (aK + lam I + b 11^T)^-1
            b = (1 - a) / D
            M = a * Kn[:n, :n] + lam * np.eye(n)
            Mi = np.linalg.inv(M)
            one = np.ones(n)
            Mi_sm = Mi - b * np.outer(Mi @ one, one @ Mi) / (1 + b * one @ Mi @ one)
            kt = a * Kn[n:, :n] + b
            h_sm = kt @ Mi_sm @ y[:n]
            h_cnr = ridge(Kg[:n, :n], Kg[n:, :n], y[:n], a * lam, True)
            h_ref = ridge(Kn[:n, :n], Kn[n:, :n], y[:n], lam, True)
            rows.append(dict(N=N, p_tilde=pt, L=L, a=a, dev_affine=dev_aff, dev_SM_vs_direct=float(np.abs(h_sm - h_direct).max()),
                             dev_CNR_vs_noiseless=float(np.abs(h_cnr - h_ref).max())))
res["E3a"] = rows
res["E3a_summary"] = dict(max_dev_affine=max(r["dev_affine"] for r in rows), max_dev_SM=max(r["dev_SM_vs_direct"] for r in rows),
                          max_dev_CNR=max(r["dev_CNR_vs_noiseless"] for r in rows))
print("E3a", res["E3a_summary"], time.time() - t0)

# ---------------------------------------------------------------- E3b
N, D, n, nte, lam = 4, 16, 25, 50, 0.1
Ls = list(range(0, 401, 2))
rows = []
for gamma in (0.02, 0.05):
    for sp in range(5):
        Xtr, Xte, ytr, yte = make_split("D1", n, sp, n_test_cap=nte, scale="pi")
        ytr = ytr.astype(float)
        Xall = np.vstack([Xtr, Xte])
        Vs = zz_layers(Xall, N)
        Ks = noisy_kernels(Vs, ("local", kraus("dep", gamma)), N, [max(L, 1) for L in Ls])
        hb = hbar(ytr, n, D, lam)
        for L in Ls:
            K = Ks[max(L, 1)]
            Ktr, Kte = K[:n, :n], K[n:, :n]
            h = ridge(Ktr, Kte, ytr, lam, False)
            eps = max(np.abs(Ktr - 1 / D).max(), np.abs(Kte - 1 / D).max())
            bound = eps * (n / lam + (n / lam) ** 2)
            rows.append(dict(gamma=gamma, split=sp, L=L, mean_abs=float(np.mean(np.abs(np.clip(h, -1, 1) - hb))), spread=spread(h), eps=float(eps), bound=float(bound)))
import pandas as pd
df = pd.DataFrame(rows)
g = df.groupby(["gamma", "L"])[["mean_abs", "spread", "eps", "bound"]].mean().reset_index()
summ = []
for gamma in (0.02, 0.05):
    s = g[g.gamma == gamma]
    Lm = s[s.spread < 0.05].L.min()
    Lb = s[s.bound < 0.05].L.min()
    Lm2 = s[s.mean_abs < 0.05].L.min()
    summ.append(dict(gamma=gamma, L_measured_spread_lt_0p05=None if np.isnan(Lm) else int(Lm), L_measured_gap_lt_0p05=None if np.isnan(Lm2) else int(Lm2),
                     L_guaranteed_bound_lt_0p05=None if np.isnan(Lb) else int(Lb),
                     eps_star=float(0.05 / (n / lam + (n / lam) ** 2)), lam2_over_n2=float(lam ** 2 / n ** 2)))
res["E3b"] = dict(curves=g.to_dict("records"), summary=summ)
print("E3b", summ, time.time() - t0)

# ---------------------------------------------------------------- E9a
def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)
rows = []
for g_ in (0.05, 0.2):
    pd_ = kraus("pd", g_)
    for (x, xp) in [(0.3, 1.1), (1.0, 2.0), (2.5, 0.4), (1.5, 1.5)]:
        rho = ry(x) @ np.diag([1, 0]).astype(complex) @ ry(x).conj().T
        traj = {}
        for l in range(0, 401):
            if l in (0, 1, 5, 20, 100, 400):
                r2 = ry(xp).conj().T @ rho @ ry(xp)
                traj[l] = float(r2[0, 0].real)
            rho = sum(E @ rho @ E.conj().T for E in pd_)
        rows.append(dict(gamma=g_, x=x, xp=xp, traj=traj, limit_pred=(1 + np.cos(x) * np.cos(xp)) / 2, K_L400=traj[400],
                         dist_from_half=abs(traj[400] - 0.5)))
res["E9a"] = rows
print("E9a done", time.time() - t0)

# ---------------------------------------------------------------- E9b
N = 4
rng = np.random.default_rng(3)
X = rng.uniform(0, np.pi, (10, N))
def diag_only_layer(x):
    L = zz_layer(x, N, "full")
    Hn = kron_all([np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)] * N)
    return L @ Hn.conj().T     # = diagonal phases only (H removed)
rows = []
for ch in ("pd", "bf", "dep", "ad"):
    for tag, Vs in (("ZZ(H+phases)", zz_layers(X, N)), ("diag-only(no H)", np.array([diag_only_layer(x) for x in X]))):
        noise = ("local", kraus(ch, 0.05))
        Nsup = noise_superop(noise, N)
        info = []
        for V in Vs:
            S, Sp = dense_layer_maps(V, Nsup)
            Kinf, rho_all, rho_sec, nunit, nunit_sec = limit_and_rate(Sp)
            info.append((Kinf, rho_all, rho_sec, nunit, nunit_sec))
        info = np.array(info)
        K = dense_kernel_limit(Vs, Nsup, 16384)
        rows.append(dict(channel=ch, layer=tag, rho2_all_max=float(info[:, 1].max()), n_unit_eigs_max=int(info[:, 3].max()),
                         n_unit_eigs_in_sector_max=int(info[:, 4].max()), K16384_min=float(K.min()), K16384_max=float(K.max()),
                         K16384_dev_from_1_over_D=float(np.abs(K - 1 / 16).max())))
res["E9b"] = rows
print("E9b done", time.time() - t0)

# ---------------------------------------------------------------- E9c : Emb-S slice (angle 1/8) + BF; parity sector
rng = np.random.default_rng(4)
Xs = rng.uniform(0, 1, (10, N))
Vs = emb_R_layers(Xs, 2.0, N, 8)
rows = []
Xp = kron_all([np.array([[0, 1], [1, 0]], complex)] * N)
for ch in ("bf", "dep"):
    noise = ("local", kraus(ch, 0.05))
    Nsup = noise_superop(noise, N)
    allr, secr, wr, nun, nunsec = [], [], [], [], []
    comm = []
    for V in Vs:
        S, Sp = dense_layer_maps(V, Nsup)
        _, ra, rs, nu, nus = limit_and_rate(Sp)
        allr.append(ra); secr.append(rs); nun.append(nu); nunsec.append(nus)
        wr.append(max(arm_radius_weighted(S), arm_radius_weighted(Sp.T)))
        comm.append(float(np.abs(Xp @ V - V @ Xp).max()))
    Ls = list(range(1, 241))
    Ks = noisy_kernels(Vs, noise, N, Ls)
    Kref = dense_kernel_limit(Vs, Nsup, 8192)
    d = np.array([np.abs(Ks[L] - Kref).max() for L in Ls])
    sl = fit_rate(Ls, d, (30, 80), 1e-13)[0]
    rows.append(dict(channel=ch, max_commutator_V_with_Xparity=max(comm), n_unit_eigs_max=int(max(nun)), n_unit_eigs_in_Pi0_sector_max=int(max(nunsec)),
                     rho2_ignoring_sector=float(max(allr)), rho2_weighted_sector=float(max(wr)), measured_slope=sl,
                     weighted_prediction_2ln_rho=float(2 * np.log(max(wr)))))
res["E9c"] = rows
print("E9c", rows, time.time() - t0)
save("E3_E9_identity_collapse_counterexamples", dict(note="E3a,E3b,E9a,E9b,E9c"), res, t0)
