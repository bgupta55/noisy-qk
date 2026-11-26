"""E0: reproduce Wang et al. (arXiv:2401.17526) Sec. IV + overlay exact affine identity (Thm 4) + Gap-2 style check.
Setting (from the paper): Fashion-MNIST shirt vs dress, PCA10, N=10, IQP embedding (Eq. 25-26), n=500 train / 500 test,
global depolarizing p~=0.1 after each of 2L layers (p = 1-(1-p~)^(2L)), lambda=0.5, h clipped to [-1,1], train/test error = misclassification frequency."""
from common import *
from nqk.embeddings import iqp_states
from nqk.datasets import fashion_shirt_dress
from nqk.krr import ridge
from nqk.metrics import auc, wang_bound, hbar, spread
from nqk.kernel import apply_noise, kernel_from_arms

t0 = time.time()
N, n, nte, pt, lam = 10, 500, 500, 0.1, 0.5
D = 2 ** N
Ls = list(range(0, 61))
SEEDS = [0, 1, 2, 3, 4]
cfg = dict(N=N, n=n, n_test=nte, p_tilde=pt, lam=lam, L=Ls, seeds=SEEDS, embedding="IQP Eq.25-26 (U_Z with sum_{i,j} x_i x_j Z_i Z_j over ordered pairs)")

# ---- direct (non-affine) cross-check on a tiny instance: global depolarizing applied layerwise equals affine formula
rng = np.random.default_rng(0)
Xs = rng.uniform(0, 1, (6, 4))
psi = iqp_states(Xs, 4)
Vs = np.array([np.eye(16, dtype=complex)] * 6)   # not used: check via explicit superoperator on states
from nqk.channels import global_dep_superop
Nsup = global_dep_superop(0.1, 4)
L_chk = 5
rho = np.array([np.outer(p, p.conj()) for p in psi])
K0 = np.abs(np.conj(psi) @ psi.T) ** 2
cross_ok = None
# explicit: rho_i -> (1-p)^... ; forward arm noise 1..L, backward arm noise adjoint (global dep is self-adjoint, trace-1 on O)
a = (1 - 0.1) ** (2 * L_chk)
Kaff = a * K0 + (1 - a) / 16
# explicit layerwise: apply 2L global-dep maps to |phi_i><phi_i| then overlap with |phi_j><phi_j|
Kexp = np.zeros((6, 6))
for i in range(6):
    r = rho[i].copy()
    for _ in range(2 * L_chk):
        r = (1 - 0.1) * r + 0.1 * np.trace(r) * np.eye(16) / 16
    for j in range(6):
        Kexp[i, j] = np.real(np.trace(rho[j] @ r))
cross_dev = float(np.abs(Kexp - Kaff).max())

res = dict(cross_check_affine_vs_explicit_N4_L5=cross_dev, runs={})
for seed in SEEDS:
    Xtr, Xte, ytr, yte = fashion_shirt_dress(n, nte, seed, N)
    P = iqp_states(np.vstack([Xtr, Xte]), N)
    G = np.conj(P) @ P.T
    Kall = np.abs(G) ** 2                       # noiseless fidelity kernel
    Ktr, Kte = Kall[:n, :n], Kall[n:, :n]
    out = dict(noiseless_offdiag_mean=float(Ktr[~np.eye(n, dtype=bool)].mean()), noiseless_offdiag_std=float(Ktr[~np.eye(n, dtype=bool)].std()))
    rows = []
    for L in Ls:
        a = (1 - pt) ** (2 * L)
        p = 1 - a
        Kt_tr = a * Ktr + (1 - a) / D
        Kt_te = a * Kte + (1 - a) / D
        # Wang setting: uncentered ridge, fixed lambda
        h_te = np.clip(ridge(Kt_tr, Kt_te, ytr, lam, False), -1, 1)
        h_tr = np.clip(ridge(Kt_tr, Kt_tr, ytr, lam, False), -1, 1)
        hb = hbar(ytr, n, D, lam)
        # sign convention: error = frequency of sign(h)!=y (ties at 0 -> +1)
        err_tr = float(np.mean(np.where(h_tr >= 0, 1, -1) != ytr))
        err_te = float(np.mean(np.where(h_te >= 0, 1, -1) != yte))
        gap = float(np.mean(np.abs(h_te - hb)))
        # Thm 4(ii): centred ridge with lambda' = a*lambda0 reproduces noiseless centred predictor exactly
        h_c_noisy = ridge(Kt_tr, Kt_te, ytr, a * lam, True)
        h_c_ref = ridge(Ktr, Kte, ytr, lam, True)
        rows.append(dict(L=L, p=p, a=a, err_tr=err_tr, err_te=err_te, gap=gap, hbar=hb,
                         frac_pos=float(np.mean(h_te > 0)), spread=spread(h_te),
                         auc_uncentered_fixed=auc(h_te, yte), auc_centered_rescaled=auc(h_c_noisy, yte),
                         auc_noiseless_centered=auc(h_c_ref, yte),
                         thm4_dev=float(np.abs(h_c_noisy - h_c_ref).max()),
                         wang_bound=wang_bound(n, D, lam, pt, L)[0]))
    out["rows"] = rows
    res["runs"][seed] = out
    print("seed", seed, "done", time.time() - t0)

# aggregate
agg = []
for i, L in enumerate(Ls):
    r = [res["runs"][s]["rows"][i] for s in SEEDS]
    agg.append(dict(L=L, a=r[0]["a"], gap=np.mean([x["gap"] for x in r]), err_tr=np.mean([x["err_tr"] for x in r]),
                    err_te=np.mean([x["err_te"] for x in r]), frac_pos=np.mean([x["frac_pos"] for x in r]),
                    spread=np.mean([x["spread"] for x in r]),
                    auc_unc=np.mean([x["auc_uncentered_fixed"] for x in r]),
                    auc_cr=np.mean([x["auc_centered_rescaled"] for x in r]),
                    auc_ref=np.mean([x["auc_noiseless_centered"] for x in r]),
                    thm4_dev=max(x["thm4_dev"] for x in r), wang_bound=r[0]["wang_bound"]))
res["aggregate"] = agg
# collapse depth: first L where mean |h~-hbar| < 0.01 ; spread<0.05 ; training error transition (first L with err_tr >= 0.4)
def first(cond):
    for x in agg:
        if cond(x):
            return x["L"]
    return None
res["summary"] = dict(
    L_gap_lt_0p01=first(lambda x: x["gap"] < 0.01),
    L_gap_lt_0p05=first(lambda x: x["gap"] < 0.05),
    L_spread_lt_0p05=first(lambda x: x["spread"] < 0.05),
    L_all_test_pos_or_neg=first(lambda x: x["frac_pos"] in (0.0, 1.0)),
    L_train_err_ge_0p25=first(lambda x: x["err_tr"] >= 0.25),
    L_train_err_ge_0p4=first(lambda x: x["err_tr"] >= 0.4),
    wang_predicted_depth=float(np.log(500) / np.log((1 - pt) ** -2)),
    wang_bound_first_L_lt_1=first(lambda x: x["wang_bound"] < 1.0),
    max_thm4_dev=max(x["thm4_dev"] for x in agg),
    max_abs_auc_diff_centered_rescaled_vs_noiseless=max(abs(x["auc_cr"] - x["auc_ref"]) for x in agg),
)
save("E0_wang_reproduction", cfg, res, t0)
print(json.dumps(res["summary"], indent=1))
for x in agg[::5]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
