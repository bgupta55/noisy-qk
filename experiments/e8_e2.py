"""E8 (equal infidelity table), E2a (Thm 1 bound), E2b (limit values/asymmetry), E2c (AD consequence), E2d (Haar law)."""
from common import *
from joblib import Parallel, delayed
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import ridge, cv_lam
from nqk.metrics import auc
from nqk.datasets import make_split
import importlib.util
t0 = time.time()
N, D = 4, 16
res = {}

# ------------------------------------------------------------------ E8
rows = []
rng = np.random.default_rng(100)
Xpool = rng.uniform(0, np.pi, (20, N))
Vpool = zz_layers(Xpool, N)
for r in (0.002, 0.005, 0.01, 0.02):
    for ch in CHANNELS:
        g = gamma_for_infidelity(ch, r)
        kr = kraus(ch, g)
        s = unit_stats(kr)
        noise = ("local", kr)
        Nsup = noise_superop(noise, N)
        rho = 0
        for V in Vpool[:10]:
            S, Sp = dense_layer_maps(V, Nsup)
            rho = max(rho, arm_radius_weighted(S), arm_radius_weighted(Sp.T))
        Kinf = np.array([limit_and_rate(dense_layer_maps(V, Nsup)[1])[0] for V in Vpool])
        Ls = list(range(1, 241))
        Ks = noisy_kernels(Vpool[:10], noise, N, Ls)
        Kref = dense_kernel_limit(Vpool[:10], Nsup, 8192)
        d = np.array([np.abs(Ks[L] - Kref).max() for L in Ls])
        sl = fit_rate(Ls, d, (30, 80), 1e-13)[0] if False else None
        from nqk.metrics import fit_rate
        sl = fit_rate(Ls, d, (30, 80), 1e-13)[0]
        rows.append(dict(r=r, ch=ch, gamma=g, u=s["u"], u_1_minus_4r=1 - 4 * r, u4=u_N_numeric(kr, 4), kappa2=s["kappa2"],
                         rho2_weighted=rho, spectral_slope=(2 if ch != "ad" else 1) * np.log(rho), measured_slope=sl,
                         Kinf_min=float(Kinf.min()), Kinf_max=float(Kinf.max())))
res["E8"] = rows
print("E8 done", time.time() - t0)

# ------------------------------------------------------------------ E2a  (Thm 1)
a_rows = []
# adversarial: N=1 idle layer (V=I), dep: K-1/2 = 0.5 kappa^{2L}  (tight)
for g in (0.05, 0.1):
    kr = kraus("dep", g); kap = unit_stats(kr)["kappa"]
    for L in (1, 2, 5, 10, 30):
        K = noisy_kernels(np.array([np.eye(2, dtype=complex)]), ("local", kr), 1, [L])[L][0, 0]
        a_rows.append(dict(kind="N1_idle_dep", gamma=g, L=L, dev=float(K - 0.5), bound=float(0.5 * kap ** (2 * L)),
                           ratio=float((K - 0.5) / (0.5 * kap ** (2 * L)))))
# general: N=4 ZZ layers, dep (kappa<1), every pair, L grid
rng = np.random.default_rng(5)
Vs = zz_layers(rng.uniform(0, np.pi, (25, N)), N)
for g in (0.02, 0.05, 0.1):
    kr = kraus("dep", g); kap = unit_stats(kr)["kappa"]
    Ls = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    Ks = noisy_kernels(Vs, ("local", kr), N, Ls)
    for L in Ls:
        dev = np.abs(Ks[L] - 1 / D).max()
        bd = (1 - 1 / D) * kap ** (2 * L)
        a_rows.append(dict(kind="N4_ZZ_dep", gamma=g, L=L, dev=float(dev), bound=float(bd), ratio=float(dev / bd), holds=bool(dev <= bd + 1e-14)))
# Emb-S slices (L-dependent layers) dep
fam = SliceFamily(rng.uniform(0, 1, (25, N)), 2.0, N)
for g in (0.02,):
    kr = kraus("dep", g); kap = unit_stats(kr)["kappa"]
    for L in (1, 4, 16, 64):
        K = noisy_kernels(np.array(fam.slices(L)), ("local", kr), N, [L])[L]
        dev = np.abs(K - 1 / D).max(); bd = (1 - 1 / D) * kap ** (2 * L)
        a_rows.append(dict(kind="N4_EmbS_dep", gamma=g, L=L, dev=float(dev), bound=float(bd), ratio=float(dev / bd), holds=bool(dev <= bd + 1e-14)))
# BF / PD : kappa = 1 -> vacuous bound; report the measured decay of max|K-1/D|
for ch in ("bf", "pd"):
    kr = kraus(ch, 0.05)
    Ls = [1, 10, 30, 100, 300]
    Ks = noisy_kernels(Vs, ("local", kr), N, Ls)
    for L in Ls:
        a_rows.append(dict(kind=f"N4_ZZ_{ch}_vacuous", gamma=0.05, L=L, dev=float(np.abs(Ks[L] - 1 / D).max()), bound=float(1 - 1 / D),
                           ratio=None, holds=True))
res["E2a"] = a_rows
res["E2a_summary"] = dict(all_dep_hold=bool(all(r.get("holds", True) for r in a_rows if r["kind"] in ("N4_ZZ_dep", "N4_EmbS_dep"))),
                          N1_tight_max_abs_ratio_minus_1=float(max(abs(r["ratio"] - 1) for r in a_rows if r["kind"] == "N1_idle_dep")),
                          N4_ZZ_dep_min_ratio=float(min(r["ratio"] for r in a_rows if r["kind"] == "N4_ZZ_dep")),
                          N4_ZZ_dep_max_ratio=float(max(r["ratio"] for r in a_rows if r["kind"] == "N4_ZZ_dep")))
print("E2a done", res["E2a_summary"], time.time() - t0)

# ------------------------------------------------------------------ E2b
b_rows = []
for emb in ("ZZ", "R"):
    rng = np.random.default_rng(11)
    if emb == "ZZ":
        Vs = zz_layers(rng.uniform(0, np.pi, (25, N)), N)
    else:
        Vs = emb_R_layers(rng.uniform(0, 1, (25, N)), 2.0, N, 8)
    for ch in CHANNELS:
        g = 0.05
        noise = ("local", kraus(ch, g))
        Nsup = noise_superop(noise, N)
        Kref = dense_kernel_limit(Vs, Nsup, 16384)
        Kinf = np.array([limit_and_rate(dense_layer_maps(V, Nsup)[1])[0] for V in Vs])
        nunit = [limit_and_rate(dense_layer_maps(V, Nsup)[1])[3] for V in Vs]
        dev = np.abs(Kref - Kinf[None, :]).max()
        b_rows.append(dict(emb=emb, ch=ch, gamma=g, max_dev_K16384_vs_Kinf=float(dev), Kinf_min=float(Kinf.min()), Kinf_max=float(Kinf.max()),
                           col_std_max=float(Kref.std(axis=0).max()), row_std_mean=float(Kref.std(axis=1).mean()),
                           asym_limit=float(np.abs(Kref - Kref.T).max()), max_n_unit_eigs=int(max(nunit)),
                           dev_from_1_over_D=float(np.abs(Kref - 1 / D).max())))
res["E2b"] = b_rows
print("E2b done", time.time() - t0)

# ------------------------------------------------------------------ E2c (AD limit kernel is rank<=2 after symmetrisation)
c_rows = []
for dname in ("D1", "D2", "D3", "D4"):
    aucs_lim, aucs_g, aucs_ref, ranks = [], [], [], []
    for sp in range(10):
        Xtr, Xte, ytr, yte = make_split(dname, 30, sp, scale="pi")
        Xall = np.vstack([Xtr, Xte])
        Vs_ = zz_layers(Xall, N)
        Nsup = noise_superop(("local", kraus("ad", 0.05)), N)
        g = np.array([limit_and_rate(dense_layer_maps(V, Nsup)[1])[0] for V in Vs_])
        Kl = np.tile(g[None, :], (len(g), 1))                # K_inf[i,j] = g(x_j)
        Ks = (Kl + Kl.T) / 2
        n = 30
        sv = np.linalg.svd(Ks[:n, :n], compute_uv=False)
        ranks.append(int((sv > 1e-10 * sv[0]).sum()))
        # centred-ridge on symmetrised limit kernel (expected constant) and uncentred ridge (function of g)
        h_unc = ridge(Ks[:n, :n], Ks[n:, :n], ytr, 1e-3, False)
        h_cen = ridge(Ks[:n, :n], Ks[n:, :n], ytr, 1e-3, True)
        aucs_lim.append(auc(h_unc, yte))
        aucs_g.append(max(auc(g[n:], yte), 1 - auc(g[n:], yte)))     # oracle-oriented AUC of g alone (upper bound for any monotone map)
        # reference: depth-1 noiseless ZZ kernel with CV ridge
        Kn = noiseless_kernel(Vs_)
        lam = cv_lam(Kn[:n, :n], ytr.astype(float), True)
        aucs_ref.append(auc(ridge(Kn[:n, :n], Kn[n:, :n], ytr.astype(float), lam, True), yte))
    c_rows.append(dict(dataset=dname, rank_max=int(max(ranks)), auc_limit_uncentred=float(np.nanmean(aucs_lim)),
                       auc_g_oriented=float(np.nanmean(aucs_g)), auc_noiseless_ref=float(np.nanmean(aucs_ref))))
res["E2c"] = c_rows
print("E2c done", c_rows, time.time() - t0)

# ------------------------------------------------------------------ E2d (Haar average law)
spec = importlib.util.spec_from_file_location("tt", os.path.join(os.path.dirname(__file__), "..", "tests", "test_T1_T14.py"))
tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)
cells = [(ch, g, Nq, L) for ch, g in (("dep", 0.1), ("bf", 0.1), ("pd", 0.2)) for Nq in (2, 3, 4) for L in (1, 2, 3)]
def cell(ch, g, Nq, L):
    m, se = tt.haar_purity(ch, g, Nq, L, 200000 if Nq < 4 else 100000, seed=1000 + Nq * 10 + L)
    u_N = u_N_numeric(kraus(ch, g), Nq)
    pred = (2 ** Nq - 1) / 2 ** Nq * u_N ** L
    return dict(ch=ch, gamma=g, N=Nq, L=L, draws=200000 if Nq < 4 else 100000, mean=float(m), se=float(se), pred=float(pred),
                z=float((m - pred) / se), rel_dev=float((m - pred) / pred))
d_rows = Parallel(n_jobs=10)(delayed(cell)(*c) for c in cells)
zs = np.array([r["z"] for r in d_rows])
res["E2d"] = d_rows
res["E2d_summary"] = dict(n_cells=len(d_rows), chi2=float((zs ** 2).sum()), max_abs_z=float(np.abs(zs).max()), mean_z=float(zs.mean()),
                         max_abs_rel_dev=float(max(abs(r["rel_dev"]) for r in d_rows)))
from scipy import stats
res["E2d_summary"]["chi2_pvalue"] = float(stats.chi2.sf(res["E2d_summary"]["chi2"], len(zs)))
print("E2d", res["E2d_summary"], time.time() - t0)
save("E2_E8_limits_bounds_haar", dict(N=N, note="E8,E2a-d"), res, t0)
