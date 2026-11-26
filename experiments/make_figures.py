"""Figures from results/*.json only. Colours fixed: dep blue, AD orange, PD green, BF red."""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
C = dict(dep="#1f77b4", ad="#ff7f0e", pd="#2ca02c", bf="#d62728", gdep="#17becf")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
J = lambda f: json.load(open(os.path.join(RES, f + ".json")))["results"]

# Fig 2: E0 Wang reproduction
e0 = pd.DataFrame(J("E0_wang_reproduction")["aggregate"])
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].semilogy(e0.L, e0.gap, label="gap |h-mean|"); ax[0].semilogy(e0.L, e0.a, "--", label="a_L=(1-p~)^{2L}")
ax[0].axvline(29.5, c="k", ls=":", label="Wang depth 29.5"); ax[0].set_xlabel("L"); ax[0].legend(); ax[0].set_title("prediction collapse")
ax[1].plot(e0.L, e0.err_tr, label="train err"); ax[1].plot(e0.L, e0.err_te, label="test err"); ax[1].axvline(24, c="k", ls=":"); ax[1].set_xlabel("L"); ax[1].legend(); ax[1].set_title("errors (Wang reports transition ~24)")
ax[2].plot(e0.L, e0.auc_unc, label="uncentred fixed-λ"); ax[2].plot(e0.L, e0.auc_cr, label="centred+rescaled λ"); ax[2].plot(e0.L, e0.auc_ref, "k:", label="noiseless"); ax[2].set_xlabel("L"); ax[2].legend(); ax[2].set_title("AUC: ranking survives (Thm 4)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig02_E0_wang.png")); plt.close()

# Fig 4: E1 rates
e1 = pd.DataFrame(J("E1_rates")["rows"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, emb in zip(ax, ("ZZ", "R")):
    d = e1[e1.emb == emb]
    for ch in ("dep", "ad", "pd", "bf"):
        q = d[d.ch == ch]
        a.errorbar(q.pred_spectral, q.obs_slope, yerr=2 * q.se, fmt="o", c=C[ch], label=ch)
    lim = [-0.24, 0]; a.plot(lim, lim, "k--", lw=0.8); a.set_xlabel("spectral prediction (slope of ln δ)"); a.set_title("Emb-" + emb)
ax[0].set_ylabel("observed slope"); ax[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig04_E1_rates.png")); plt.close()

# Fig 3: E2b limits + E8
e2 = J("E2_E8_limits_bounds_haar")
b = pd.DataFrame(e2["E2b"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
xt, xl = [], []
for i, emb in enumerate(("ZZ", "R")):
    g = b[b.emb == emb]
    for j, (_, r) in enumerate(g.iterrows()):
        ax[0].plot([i * 5 + j] * 2, [r.Kinf_min, r.Kinf_max], c=C[r.ch], lw=6, solid_capstyle="butt")
    xt.append(i * 5 + 1.5); xl.append("Emb-" + emb)
ax[0].axhline(1 / 16, c="k", ls=":"); ax[0].set_xticks(xt); ax[0].set_xticklabels(xl); ax[0].set_ylabel("range of K_∞ (1/D dotted)"); ax[0].set_title("limit kernels (dep, ad, pd, bf)")
e8 = pd.DataFrame(e2["E8"])
for ch in ("dep", "ad", "pd", "bf"):
    q = e8[e8.ch == ch]; ax[1].plot(q.r, -q.measured_slope, "o-", c=C[ch], label=ch)
ax[1].set_xlabel("per-qubit infidelity r"); ax[1].set_ylabel("measured |slope|"); ax[1].legend(); ax[1].set_title("equal infidelity: unital faster than AD")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig03_E2_E8_limits.png")); plt.close()

# Fig 6: E3b
c3 = pd.DataFrame(J("E3_E9_identity_collapse_counterexamples")["E3b"]["curves"])
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for g, d in c3.groupby("gamma"):
    ax.semilogy(d.L, d.spread, label=f"measured spread γ={g}"); ax.semilogy(d.L, d.bound, "--", label=f"bound γ={g}")
ax.axhline(0.05, c="k", ls=":"); ax.set_xlabel("L"); ax.legend(fontsize=6); ax.set_title("E3b: bound loose (measured ≪ guaranteed)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig06_E3b.png")); plt.close()

# Fig 7: E4
e4 = pd.DataFrame(J("E4_calibration")["curves"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for ds, ls in (("D1", "-"), ("D2", "--")):
    d = e4[(e4.dataset == ds) & (e4.ch == "dep") & (e4.N == 4) & (e4.lam == 1.0)]
    ax[0].semilogx(d.L, d.ece_width, ls, c=C["dep"], label=f"{ds} ECE(width-bin)"); ax[0].semilogx(d.L, d.ece_exact_pred, ls, c="k", lw=0.8, label=f"{ds} exact (Prop 6)")
    ax[1].semilogx(d.L, d.disp, ls, c=C["dep"], label=f"{ds} dispersion")
ax[0].legend(fontsize=6); ax[1].legend(fontsize=6); ax[0].set_xlabel("L"); ax[1].set_xlabel("L"); ax[0].set_title("dep, N=4, λ=1: ECE→exact constant"); ax[1].set_title("dispersion→0 (calibrated and useless)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig07_E4_calibration.png")); plt.close()

# Fig 8: E5
e5 = pd.DataFrame(J("E5_ranking_vs_resolution")["aggregate"])
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
for a, S in zip(ax, (256.0, 1024.0, 4096.0, -1.0)):
    for ch in ("dep", "ad", "pd", "bf"):
        for m, ls in (("cnr", "-"), ("cv_cen", "--"), ("fixed", ":")):
            d = e5[(e5.ds == "D1") & (e5.ch == ch) & (e5.S == S) & (e5.method == m)].sort_values("L")
            a.plot(d.L, d.auc, ls, c=C[ch], marker="o", ms=2.5, label=f"{ch}-{m}" if S == 256.0 else None)
    a.set_xscale("log", base=2); a.set_title("S=∞ (exact)" if S < 0 else f"S={int(S)}"); a.set_xlabel("L")
ax[0].set_ylabel("AUC (D1)"); ax[0].legend(fontsize=5, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig08_E5_ranking_vs_resolution.png")); plt.close()

# Fig 9: E6
t6 = pd.DataFrame(J("E6_E7_viability_indistinguishability")["E6"]["table"])
fig, ax = plt.subplots(figsize=(4.8, 3.6))
for (ds, ch), g in t6.groupby(["ds", "ch"]):
    ax.errorbar(g.L_pred, g.L_emp, yerr=[g.L_emp - g.L_emp_lo, g.L_emp_hi - g.L_emp], fmt="o" if ch == "dep" else "s", ms=3, c=C[ch], alpha=0.7, label=ch if ds == "D1" else None)
ax.plot([0, 260], [0, 260], "k--", lw=0.8); ax.set_xlabel("predicted viability depth (Alg 10, c frozen from D1)"); ax.set_ylabel("empirical depth"); ax.legend(); ax.set_title("E6")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig09_E6_viability.png")); plt.close()

# Fig 10: hardware
h = J("HW_analysis")
fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
r1 = pd.DataFrame(h["HW1"]["rows"])
for d, mk in ((100, "o"), (200, "s")):
    q = r1[r1.delay_us == d]; ax[0].scatter(q.t_z, q.kappa, marker=mk, label=f"{d} µs")
ax[0].set_xlabel("translation t_z"); ax[0].set_ylabel("κ (σ_max of T)"); ax[0].legend(); ax[0].set_title("HW1: idle channel (4 qubits)")
r2 = pd.DataFrame(h["HW2_retention"]); lay = r2.noisy_layers.values
ax[1].semilogy(lay, r2.s_hw, "ko-", label="ibm_kingston"); ax[1].semilogy(lay, r2.s_aer_noise, "bs--", label="Aer backend noise"); ax[1].semilogy(lay, h["HW3"]["s_by_setting"]["idle"], "g^:", label="HW1 idle-only")
ax[1].set_xlabel("noisy layers 2L(1+2k)"); ax[1].set_ylabel("signal retention s"); ax[1].legend(fontsize=6); ax[1].set_title("HW2/HW3")
r4 = pd.DataFrame(h["HW4"]["rows"])
for pr, g in r4.groupby(r4.pair.astype(str)):
    ax[2].errorbar(g.tau_us, g.asym_hw, yerr=g.asym_se, fmt="o-", label=f"HW {pr}"); ax[2].plot(g.tau_us, g.asym_idle_pred, "k:", lw=0.8)
ax[2].set_xlabel("idle delay (µs)"); ax[2].set_ylabel("K(x,x')-K(x',x)"); ax[2].legend(fontsize=6); ax[2].set_title("HW4 (dotted: HW1 T1,T2 prediction)")
r5 = pd.DataFrame(h["HW5"]["rows"])
for nf, ls in ((1, "-"), (3, "--"), (9, ":")):
    q = r5[r5.noise_factor == nf]; ax[3].semilogx(q.S, q.cnr_mean, ls, c="k", label=f"CNR nf={nf}"); ax[3].semilogx(q.S, q.cv_cen_mean, ls, c="tab:purple", label=f"CV nf={nf}")
ax[3].set_xlabel("shots S'"); ax[3].set_ylabel("pooled AUC"); ax[3].legend(fontsize=5); ax[3].set_title("HW5 (small n: indicative only)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig10_hardware.png")); plt.close()

# Fig S: E10 / E12
o = J("E10_E13_N_mixtures_fairness_audit")
e10 = pd.DataFrame(o["E10"]["rows"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for (emb, ch), g in e10.groupby(["emb", "ch"]):
    ax[0].plot(g.N, g.slope, "o-" if emb == "S" else "s--", c=C[ch], label=f"{'Emb-R(small angle)' if emb=='S' else 'Emb-ZZ'} {ch}")
ax[0].set_xlabel("N"); ax[0].set_ylabel("fitted slope of ln δ"); ax[0].legend(fontsize=6); ax[0].set_title("E10 N-dependence")
e12 = pd.DataFrame(o["E12"]["rows"]); e12 = e12[e12.c == 1.0]
for ch in ("dep", "ad", "pd"):
    g = e12[(e12.ch == ch) & (e12.n == 30) & (e12.beta == 2.0)]
    ax[1].errorbar(g.S, g.d_vs_cv, yerr=[g.d_vs_cv - g.d_vs_cv_lo, g.d_vs_cv_hi - g.d_vs_cv], fmt="o", c=C[ch], label=f"{ch} CNR-CV")
ax[1].axhline(0, c="k"); ax[1].set_xscale("log"); ax[1].set_xlabel("S"); ax[1].legend(); ax[1].set_title("E12 (n=30, β=2, L=64): paired AUC diff")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "figS_E10_E12.png")); plt.close()
print(sorted(os.listdir(FIG)))

# ======================================================================== ROUND X figures (from results/X*.json only)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Fig 1: schematic (kernel -> contraction -> limit -> what survives); purely graphical, no numbers
fig, ax = plt.subplots(figsize=(12, 2.8)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
boxes = [(0.05, "noisy kernel\nK_L(x,x') = <0|B_L(x')^T A_L(x)|0>\nforward arm A, backward arm B"),
         (3.05, "layer maps contract\nrate rho_eff per layer\n(spectral rate = envelope)"),
         (6.05, "limit kernel K_inf\nstate-independent (unital)\nx'-dependent (non-unital)"),
         (9.05, "what survives\nranking: exact, centred + rescaled\nresolution (shots S) sets viability")]
for x0, t in boxes:
    ax.add_patch(FancyBboxPatch((x0, 0.8), 2.85, 1.5, boxstyle="round,pad=0.04", fc="#f7f8fa", ec="#3b82d4"))
    ax.text(x0 + 1.425, 1.55, t, ha="center", va="center", fontsize=7)
for x0 in (2.95, 5.95, 8.95):
    ax.add_patch(FancyArrowPatch((x0 - 0.02, 1.55), (x0 + 0.12, 1.55), arrowstyle="-|>", mutation_scale=10, color="k"))
ax.text(6, 0.3, "depth L: signal a_L ~ rho_eff^L decays; shot noise ~ 1/sqrt(S) sets the depth at which ranking is no longer resolvable", ha="center", fontsize=7.5, color="#57606a")
plt.savefig(os.path.join(FIG, "fig01_schematic.png"), bbox_inches="tight"); plt.close()

# Fig 9b: X1 fitted rho_eff
x1 = J("X1_E6_fitted_rho")
t1 = pd.DataFrame(x1["table"])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for a, ch in zip(ax, ("dep", "ad")):
    g = t1[t1.ch == ch]
    for ds, m in zip(("D1", "D2", "D3", "D4"), "osv^"):
        h = g[g.ds == ds]
        a.plot(h.L_emp, h.L_pred_v1, m, c="#7c5cd8", mfc="none", label=f"{ds} V1 rate" if ch == "dep" else None)
        a.plot(h.L_emp, h.L_pred_fit3, m, c=C[ch], label=f"{ds} fitted rho_eff (1,4,16)" if ch == "dep" else None)
    lim = [0, max(g.L_emp.max(), g.L_pred_fit3.max()) * 1.05]
    a.plot(lim, lim, "k:"); a.plot(lim, np.array(lim) + 10, "k--", lw=0.6); a.plot(lim, np.array(lim) - 10, "k--", lw=0.6)
    a.set_xlabel("empirical viability depth"); a.set_ylabel("predicted"); a.set_title(f"{ch.upper()}: dashed = +-10 layers")
ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig09b_X1_fitted_rho.png")); plt.close()

# Fig 11: X4 second-backend repeat
x4 = J("X4_HW2_repeat")
r4 = pd.DataFrame(x4["retention"]); v1 = x4["v1_ibm_kingston"]
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax[0].errorbar(r4.noisy_layers, r4.s_hw, yerr=2 * r4.s_hw_se_boot, fmt="o-", c="#1f77b4", label="ibm_marrakesh (X4, n=14)")
ax[0].plot(v1["layers"], v1["s"], "s--", c="#ff7f0e", label="ibm_kingston (V1, n=9)")
ax[0].plot(r4.noisy_layers, r4.s_aer_noise_model, "^:", c="#2ca02c", label="marrakesh Aer noise model")
ax[0].set_yscale("log"); ax[0].set_xlabel("noisy layers"); ax[0].set_ylabel("retention s"); ax[0].legend(fontsize=6); ax[0].set_title("retention vs noisy layers")
ax[1].plot(r4.noisy_layers, r4.corr_exact_vs_hw, "o-"); ax[1].plot(r4.noisy_layers, r4.rmse_affine_misfit, "s-", label="affine misfit RMSE")
ax[1].plot(r4.noisy_layers, r4.shot_noise_rmse_expected, "^:", label="shot-noise RMSE")
ax[1].set_xlabel("noisy layers"); ax[1].legend(fontsize=6); ax[1].set_title("corr(exact, hw) and misfit")
A4 = pd.DataFrame(x4["auc_per_setting"]); A4 = A4[A4.L == 2]
for m, c_ in (("fixed", "k"), ("cv_cen", "#d62728"), ("cnr_c1", "#1f77b4"), ("cnr_c2", "#17becf")):
    ax[2].plot(A4.noise_factor, A4[m], "o-", c=c_, label=m)
ax[2].set_xlabel("noise factor (L=2 settings only)"); ax[2].set_title("pooled AUC (9 LOO + 5 test), n=14"); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig11_X4_hw2_repeat.png")); plt.close()

# Fig S-X8: Wang sensitivity
x8 = J("X8_wang_sensitivity")["summary"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for s in x8:
    if s["pairs"] == "i_ne_j_ordered":
        continue
    ax[0].plot(range(61), s["err_tr_mean5"], label=f'{s["scale"]}/{s["pairs"].split("_")[0] if s["pairs"].startswith("i_lt") else "ordered"}', lw=1.2)
ax[0].plot([24, 28, 32], [0.07, 0.28, 0.47], "k*", ms=9, label="Wang Fig 3b (read off, +-0.03)")
ax[0].set_xlabel("L"); ax[0].set_ylabel("training error (5-seed mean)"); ax[0].legend(fontsize=5.5); ax[0].set_title("E0 under unspecified choices")
sub = [s for s in x8 if s["pairs"] != "i_ne_j_ordered"]
ax[1].bar(range(len(sub)), [s["L_cross_0.25_mean5"] for s in sub], yerr=[s["L_cross_0.25_per_seed_sd"] for s in sub], color="#3b82d4", capsize=2)
ax[1].axhline(27.4, c="k", ls=":"); ax[1].set_xticks(range(len(sub))); ax[1].set_xticklabels([f'{s["scale"]}\n{"lt" if "lt" in s["pairs"] else "all"}' for s in sub], fontsize=6)
ax[1].set_ylabel("L where train err reaches 0.25"); ax[1].set_title("dotted: Wang interpolated; bars: seed SD")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "figS_X8_wang_sensitivity.png")); plt.close()
