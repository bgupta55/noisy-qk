# PREREGISTRATION (written before any experiment script was run, 23 Nov 2026)

Source of grids and gates: `REVISION_23NOV/rev_v1/developer_guide_final.md` Sec. 5-6. This file freezes the choices
that the guide left open. Any deviation found later is logged in `results/DEVIATIONS.md` (never silently changed).

## Global
* N=4 unless stated; seeds: split seed = split index (0..19); kernel noise: per-slice/per-layer channel after every layer.
* Statistics: paired design across channels/depths; bootstrap 95% CI (10^4 resamples) over splits; 20 splits for headline
  experiments (E5), 10 for sweeps. Holm correction is applied only to the family of E1 acceptance tests.
* Datasets: D1 Iris 1v2, D2 BreastCancer PCA4 (positive = malignant, 37%), D3 Wine 0v1 PCA4, D4 Digits 3v8 PCA4.
  Preprocessing fitted on training split only. Wang reproduction: Fashion-MNIST shirt(6) vs dress(3), PCA10, [0,1].

## Frozen choices
* E0: N=10, n=500 train, 500 test, p~=0.1, lambda=0.5, global depolarizing, p=1-(1-p~)^(2L), IQP embedding (Wang Eq. 25-26);
  reproduction tolerance: collapse depth within +-15% of Wang's reported L~30 and train-error transition near L=24.
* E1: Emb-ZZ and Emb-R (angle H(x)/8, beta=2), N=4, gamma in {0.02,0.05,0.10}, L=1..240, five channels, 5 subsets of 10 points
  from a fixed pool of 25. Fit window L in [30,80]. Reference = K^(240). Acceptance: (i) observed slope <= spectral prediction + 2 SE
  for every channel (SE = across-subset SE); (ii) AD: |observed - prediction| < 2 SE; (iii) trichotomy/mu-based predictions
  rejected where wrong; (iv) |prefactor c| small (<0.5 reported, not gated).
* E2a gate: |K-1/D| <= (1-1/D) kappa^(2L) for all pairs (dep). E2b: |K^(2000) - K_inf| < 1e-6. E2d: >=1e5 Haar draws, |z|<4.
* E3a gate: max abs deviation <= 1e-8. E3b: measured depth = first L with spread<0.05; guaranteed depth = first L with bound<0.05.
* E5: Emb-S (beta=2, x in [0,1]), n_train=30, gamma=0.02 per slice, lambda0=0.1, L in {1,4,16,64}, S in {256,1024,4096,inf},
  methods fixed-lambda (uncentered), nested CV uncentered / centered (grid 10^-8..10^2, 5-fold), CNR (c=1, a_L from trace ratio with
  shot-variance bias correction, reference = same channel at L=1), 20 splits. Report every dataset where CNR is not better than CV.
* E6: S in {64,256,1024,4096,16384}; L_emp = first crossing of CNR AUC below 0.5+0.5*(AUC_noiseless-0.5); fit c on D1 only.
* E7: L in {64,128,192} dep; S in {16,64,256,1024,4096}; Bayes error by Monte-Carlo likelihood ratio (20000 draws per hypothesis).
* E8: r in {0.002,0.005,0.01,0.02}.
* Hardware: backend with lowest pending jobs among Heron devices on the day; opt level 0, no DD, no mitigation; CZ-count assertion;
  HW1 delays {0,100,200} us x 12 tomography circuits, 4 qubits in parallel, 1024 shots; HW2 L=2 folds {0,1,2,4} plus L in {1,3} fold 0,
  n_train=6, n_test=3 (split chosen by best NOISELESS Aer AUC among 20 candidate seeds, no hardware data used), beta by noiseless LOO;
  HW4 delays {0,100,200} us, 3 pairs (chosen by the idle-noise model's predicted asymmetry, not by hardware data) x 2 orders.
  QPU budget: <=100 s of the 554 s remaining Open-plan allowance.

## Addendum (added before final E1 run, logged for transparency)
First E1 draft run used a shifted window [10,30] for gamma=0.10; this contradicted the pre-registered [30,80] and was discarded.
The reported E1 numbers use [30,80] for all gamma; the late window [80,160] is an additional diagnostic only.

## DEVIATION D2 (logged after seeing E1 first-pass results)
The pre-registered sector-aware radius (overlap threshold 1e-9 on the Pi0 expansion coefficient) returned rho ~ 0.9999 for Emb-R + BF (a nearly conserved mode that couples
negligibly to the kernel), making gate (i) vacuous there. A contribution-weighted radius (|(e0^T v_k) c_k| > 1e-6) is added as a *refinement*; both columns are reported,
the pre-registered gate results are not overwritten. Gate (ii) (AD tightness) FAILED under the pre-registered radius in the first pass and is reported as failed.

### Deviation D3 (E4): N=8 dropped (dense Model-B kernel for 75 points costs ~100 min/config on this machine); N in {4,6}, 3 splits, n_test=50.

### Deviations D4-D6 (hardware / execution, logged after the QPU runs)
* D4: HW1 used 512 shots (not 1024) to stay inside the 100 s QPU cap; HW2 split search covered 40 candidate seeds (not 20), still by noiseless criteria only;
  HW4 used 3 delays {0,100,200} us (guide: 5) for the same reason. HW2 reverse-order check: 15 circuits, 1024 shots (this was not pre-registered).
* D5: E5 and E6/E7 shot-RNG seeds changed from Python hash() (process-randomised) to zlib.crc32 after the first pass; E5, E6/E7 were rerun, so
  all reported E5/E6/E7 numbers come from the deterministic rerun and differ slightly from first-pass numbers quoted in earlier notes.
* D6: HW5 at S'=1024 is the single realised data set (no resampling possible); S'<1024 are 200 hypergeometric subsamples. 9 points only: indicative.

---
# ROUND X (revision round 2) PRE-REGISTRATION, written 23 Nov 2026 BEFORE any X-script was run
Source of tasks: the "Work that remains before resubmission" table (X1-X8). Rules quoted from that table are binding.
Deviations from this section are logged under "X-deviations" at the end of this file, never edited in place.

* **X1 (E6, fitted rho_eff).** rho_eff (rate per layer) = -slope of ln a_L fitted by ordinary least squares on L in {1,4,16}
  (split-averaged exact a_L, per dataset x channel; a_1=1 by construction). Frozen c fitted on D1/dep only (same loss as E6),
  applied unchanged to D2-D4 dep and to D1-D4 AD. Also reported for comparison: (a) the V1 E6 rate (OLS of ln a_L on L<=128);
  (b) the noise-map spectral envelope (k*(-ln lambda_2) with k=2 unital, 1 non-unital). RULE: if mean |err| for AD stays above ~10
  layers, the AD viability claim is dropped.
* **X2 (E1 modal weights).** Exact eigen-expansion of both arms; K_ij(L)=sum_km w^ij_km (lam_k mu_m)^L, w=c_k d_m (v_k.u_m).
  Heaviest mode = argmax |w||lam mu|^L at L_c=55 over (i,j) in each of the 5 E1 subsets, excluding the unit x unit term.
  Prediction A = ln|lam mu| of that mode; prediction B = window slope (L in [30,80]) of the envelope max_ij max_km |w||lam mu|^L.
  Expansion validity gate: reconstruction error of K at L=30 and 80 below 1e-6, else the row is flagged. REPORTED AS AN EXPLANATION, NOT A THEOREM;
  no pass/fail gate; we report how many rows have |obs-pred|<2 SE (floor 1e-3) next to the V1 spectral/weighted predictors.
* **X3 (E5, c=2).** c=2 is taken from D1/E12 (mean AUC 0.888 at c=2; c=4 only +0.002). Nothing is tuned on D2-D4. E5 rerun with identical
  splits/shot seeds; CV/fixed numbers must reproduce V1 exactly (check). RULE: c is stated as a hyper-parameter; no claim above CV.
* **X4 (HW2 repeat).** Backend: ibm_marrakesh (fewest pending jobs among the two Heron devices other than ibm_kingston when decided;
  ibm_fez had 7 pending). Data: D1 Iris 1v2, n=14 (9 train + 5 test), split/beta chosen by the SAME noiseless-only rule as V1 (noiseless test AUC=1.0,
  offdiag std>0.05, best LOO fraction), 40 candidate seeds. Settings (L,fold): (1,0),(2,0),(3,0),(2,1),(2,2),(2,3),(2,4); noisy layers 2,4,6,12,20,28,36.
  Shots 1024 except (2,4) at 4096. Pairs: 36 train-train + 45 test-train per setting = 81. Transpile opt 0, fixed layout, CZ assertions, no DD/twirling/mitigation.
  QPU cap: <=260 s of the 465 s remaining at start; each job guarded by usage check (stop if remaining-after-job < 150 s).
  Analysis (fixed now): rho_eff from OLS of ln s on noisy layers; early (2..12 layers) vs late (20..36) local rates with parametric-bootstrap CI;
  "geometric" is supported if the CI of (late rate - early rate) contains 0, otherwise reported as non-constant. Pooled AUC (9 LOO + 5 test) for fixed lambda=0.1,
  CV (centred, uncentred), CNR with c=1 (V1 comparable) and c=2 (X3), at nf 9 with 4096 shots and with 1024-shot hypergeometric subsamples; paired parametric bootstrap
  (200 reps). RULE: report whichever result appears.
* **X5.** HW1 report limited to kappa, t_z, T1 (u and T2 dropped). No Hahn-echo/Ramsey run (T2 is not used by any claim); QPU budget kept for X4.
* **X6.** E4 AD AUC at L=256 removed from the report; N in {4,6} kept; N=8 stated as not run.
* **X7.** Fig 1 schematic from script; environment.yml pinned from the installed versions; deposit package (code, results JSON, raw counts, manifest with SHA-256).
  A DOI can only be minted by uploading to Zenodo (needs the author's account); it is NOT claimed here.
* **X8.** Sensitivity sweep of E0 over the choices Wang et al. leave unspecified (feature scale after PCA, pair-sum convention); report which choice moves
  the training-error transition and by how much, and the seed-to-seed spread. Transition defined at several thresholds (0.05, 0.10, 0.25, 0.40).

---
## X-deviations (ROUND X). Appended after the runs; the ROUND X section above is unchanged.
* **XD1 (X4, backend and chain both differ from V1).** ibm_marrakesh chain [52,53,54,55] has much worse T2 than the V1 chain on ibm_kingston (qubit 52: 30 us, 53: 20 us vs 52-252 us on kingston). A difference between the two repeats is therefore a difference of backend AND chain AND calibration day, not of backend alone. Pre-registered selection rule (lowest queue) was followed; the chain was chosen by the same calibration-only rule as V1.
* **XD2 (X4, QPU usage).** Planned cap 260 s; used 242 s (jobs 69 + 73 + 100 s). All three groups passed the 150 s guard, so nothing was skipped. Account: 135 -> 377 of 600 s used, 223 s left.
* **XD3 (X4, affine retention model).** The pre-registered analysis fits s per setting by an affine map of hardware p(0000) on the exact kernel. At noise factors 7 and 9 that model fits poorly (corr 0.71 / 0.74; misfit RMSE 6x and 12x the shot-noise RMSE). The pre-registered rate rule was applied as written (it uses s). Centred-signal retention (the V1 s_c) is added in x_derived.py and labelled POST-HOC; it is not used for any verdict.
* **XD4 (X2, counts).** The script summary counts all 30 rows. 6 of them are global depolarising, where every mode has the same |lambda mu| and every predictor is exact to 1e-11, so they cannot discriminate. The report quotes 24 non-gdep rows (x_derived.py), and the 30-row counts from the script next to them. Pre-registered definitions are unchanged.
* **XD5 (X3, c and D1).** c=2 is read from E12 (D1, L=64, 30 cells). D1 is therefore in-sample for c=2; D2-D4 is the held-out evaluation. Whether the E12 splits overlap the E5 D1 splits was not checked.
* **XD6 (X8, data).** Wang's Fig 3b values (0.07, 0.28, 0.47 at L=24, 28, 32) were read off the figure by eye, +-0.03. Seeds 0-7 were used for the per-seed spread; the 5-seed mean (seeds 0-4) reproduces E0 exactly (max difference 0.0). The pair-sum convention "i != j, ordered" is mathematically identical to "all ordered incl. i=j" (the i=j terms are the global phase sum_i x_i^2); confirmed numerically (difference 0.0).
* **XD7 (X5, X6).** Reporting-only; x5_x6_report.py reads existing JSON and stores the dropped E4 AD AUCs under `dropped_for_audit` for audit only.
* **XD8 (X7).** environment.yml was written from the installed versions but not built from scratch (no conda on this machine). No DOI exists; deposit manifest `DEPOSIT_MANIFEST.json` (SHA-256) is the ingredient for a Zenodo upload.

---
# ROUND Y (revision round 3) PRE-REGISTRATION, written 23 Nov 2026 BEFORE any Y-script was run
Source of tasks: the "Work still open" table (Y1-Y7). Rules quoted from that table are binding. Deviations are logged under "Y-deviations" at the end of this file, never edited in place.
Nothing below was looked at before writing: for Y1 only the X4 summary numbers (retention, misfit ratios 5.8x / 11.8x) were known.

* **Y1 (offline regression on existing hardware counts; ibm_marrakesh X4 data and ibm_kingston V1 HW2 data).** Per setting k separately (never pooled over settings),
  p = (all-zero counts)/shots, K = exact noiseless kernel of the same circuit, i = x-arm index (first), j = x'-arm index (second). Ordinary least squares (minimum-norm lstsq), models:
  M0 a K + b (the X4 model) | M1 a K + b + b_j | M2 a K + b + c_i | M3 a K + b + b_j + c_i | M4 b + b_j (K-free, x'-only) | M5 b + c_i (K-free, x-only).
  Metrics: (m) dof-corrected misfit ratio sqrt(RSS/(n-rank)) / sqrt(mean p(1-p)/S) (same shot-noise denominator as the X4 diagnostic); (loo) leave-one-pair-out RMSE with a full refit;
  adjusted R^2 of M4 and M5 next to the same quantity computed on the exact K ("noiseless reference").
  RULE R1 ("b_j explains the misfit"): at BOTH marrakesh settings (2,3) and (2,4): m(M1) <= 2.0 AND loo(M1) <= 0.8 loo(M0) AND m(M1) < m(M2). Both -> explained; one -> partial; none -> not explained.
  RULE R2 (hardware test of Thm 2, only read if R1 is not 'not explained'): [adjR2(M4) - adjR2(exact K, j-factor)] is larger at (2,4) than at (2,0) AND [adjR2(M5) - adjR2(exact K, i-factor)] is smaller at (2,4) than at (2,0).
  RULE R3 (geometric decay of the K-slope): a_k from M1 and M3 per setting, early layers {2,4,6,12} vs late {20,28,36}, parametric bootstrap (400 reps, binomial resampling of counts) for CI of (late rate - early rate); 'geometric' supported iff the CI contains 0 (X4 definition).
  Kingston: the same, with the caveat n = 33 pairs per setting (M3 has about 14 free parameters); layers 2,4,6,12,20,36 (early {2,4,6,12}, late {20,36}). RULE: report whichever result appears.
* **Y2 (HW4 repeat on the marrakesh chain).** Same chain [52,53,54,55], same X4 split (14 points) and beta=0.75. Circuits idle_kernel_circuit (L=1, delay between U(x) and U(x')^dag), delays {0,100,200} us, both orders,
  2048 shots, optimisation_level 0, fixed layout, CZ count asserted (12), no DD/twirling/mitigation. Pairs: V1 rule (idle AD model, T1=250 us, tau=200 us, largest |K_ij-K_ji|) over the 91 pairs of the 14 points, top 3; selection uses no hardware data.
  Predictions fixed before the run: (a) per-qubit AD(gamma_q) then PD(lambda_q) from the calibration T1,T2 read at prep time; (a') same with Hahn-echo T2 from Y7a and calibration T1; (b) Aer backend noise model on the same transpiled circuits.
  Gates as in V1 (C4): (i) tau=0: |asym| <= 3 SE for each pair; (ii) mean |asym| increases 0 < 100 < 200 and every pair x tau>0 has the sign of prediction (a); (iii) mean z = |asym|/SE at tau=200 >= 5.
  No gate on magnitude; RMS of (asym_hw - asym_pred)/SE is reported. QPU: 18 circuits, about 16 s (V1 measured 16 s); guard: stop if account remaining after the job would be < 150 s. RULE: report as measured.
* **Y3 (S-exponent).** Using the X1 JSON (exact a_L(L) curves and L_emp(S) per dataset x channel, S in {64,256,1024,4096,16384}): a*(S) = a_L evaluated at L = L_emp(S) by log-linear interpolation in L; alpha = -d ln a* / d ln S (OLS over the 5 S).
  Uncertainty: L_emp(S) redrawn 2000 times as Normal(L_emp, sd=(hi-lo)/3.92) from the E6 bootstrap interval; per-channel pooled alpha = mean over D1-D4 with t-based CI (3 dof) combined with the draw spread. Sensitivity: S <= 1024 only (3 points).
  Secondary: alpha_rate = rate * (dL_emp/d ln S) for the V1 and fit3 rates. The predicted exponent of the sqrt(S) form is alpha = 0.5.
  RULE: if the pooled CI of alpha for dep excludes 0.5 the sqrt(S) form is stated to be wrong and the observed alpha is stated; if alpha is approximately 1 (CI contains 1), this is stated as such; AD is stated separately.
* **Y4 (split overlap and the one-cell difference).** (a) Reconstruct the train/test index sets of E12 D1 (n in {20,30,40}, sp 0-9, beta in {1,2,3}) and of E5 D1 (n=30, sp 0-19) by calling the same train_test_split with the same seeds on np.arange(100); report identical splits, train-set Jaccard, and the number of E5 D1 cells that share data with the c-selection cells (D1, L=64, S in {256,1024}, ch in {dep,ad,pd}).
  (b) Rerun the X3 c=2 evaluation on the D1 cells with (L=64, S in {256,1024}, ch in {dep,ad,pd}) restricted to splits 10-19 (not used by E12) and report c=2 vs best CV / fixed there; also D1 excluding those 6 cells. Statement for the paper depends only on D2-D4 (different data).
  (c) Locate the one-cell difference between V1 (c=1 vs fixed 125/61/70) and X3 (125/62/69) by comparing the E5 JSON 'paired' table with the X3 cell table, and by recomputing both from E5_raw.csv.gz.
* **Y5.** Needs the author's Zenodo account. Not done; instructions only. **Y6.** No conda here. Substitute (labelled as such): a clean `python -m venv` with the pinned versions from environment.yml, pytest run inside it. Does not verify conda-forge resolution.
* **Y7.** (a) Hahn-echo T2 on the marrakesh chain (prep + and -, X pi-pulse at the centre, total delay {10,30,60,120,240} us, 4 qubits in parallel, 1024 shots, no DD): E = P1(prep -) - P1(prep +); ln E = ln A - tau/T2e (OLS), per qubit, binomial-bootstrap SE.
  Reported next to the calibration T2; no gate. About 5 s QPU. (b) N=8 for E4 with a tensor method: NOT attempted (needs a new simulator); (c) emailing Wang et al.: NOT done (needs the author).

## Y-deviations (ROUND Y). Appended after the runs; the ROUND Y section above is unchanged.
* **YD0 (before any QPU use).** IBMQ_KEY / IBMQ_CRN are not set in this shell session, so y2_prep.py (needs a read-only service connection) and y2_run.py were not run at the time of writing this line. Whether they ran is stated below.
* **YD1 (credentials, supersedes YD0).** The saved account in ~/.qiskit/qiskit-ibm.json was used for Y2 and Y7a with the author's approval (option C). `get_service()` in src/nqk/hw.py now falls back to `QiskitRuntimeService(name="default-ibm-cloud")` when IBMQ_KEY / IBMQ_CRN are absent. The token was never read or printed. One job (28 circuits: 18 HW4 at 2048 shots, 10 echo at 1024 shots), 19 s QPU; account 377 -> 396 s used, 204 s left.
* **YD2 (endianness in y2_prep.py).** apply_noise indexes qubit 0 as the most significant bit; Qiskit Operator is little-endian. The per-qubit noise lists in y2_prep.py are reversed accordingly (checked on |0001>). The V1 idle_pred used uniform lists and is unaffected.
* **YD3 (Y1 rank / dof).** M3 has 22 parameters for 81 pairs on marrakesh and 14 for 33 pairs on kingston; the dof-corrected misfit uses n - rank. M3 is nearly saturated and is read only as a ceiling. The x'-arm design is unbalanced (j = 8 occurs 13 times, j = 0 five times); not analysed further.
* **YD4 (Y4 tolerance).** A knife-edge tolerance of 1e-12 on paired differences was added AFTER finding that the V1/X3 one-cell difference is a 1e-17 rounding artefact. It is a diagnostic; the pre-registered tallies (no tolerance) are reported first.
* **YD5 (Y7a fit).** The pre-registered fit uses every delay with E > 2 SE (3-4 points per qubit). A post-hoc 3-point fit (delays <= 60 us) is in Y_derived.json and is used for no verdict. Qubit 54 has E(240) > E(120), read as artefact or revival and not interpreted.
* **YD6 (Y6).** No conda here. The check was a uv venv (Python 3.14) with the pinned versions from environment.yml. It does not verify conda-forge resolution.
* **YD7 (Y3 wording).** "alpha near 1" was ambiguous. The rule was applied literally: dep CI excludes 0.5 (so the sqrt(S) form is stated as not exact) and excludes 1. alpha_rate with the fit3 rate (0.089) is implied alpha 1.07, which is an artefact of that rate (local rate at the crossing: 0.048, implied alpha 0.57). This corrects the X1 remark "predicted 5.5 vs observed 10-14".

## ROUND Z (24 Nov 2026). Written before any Z script ran.
Source of tasks: the author's message after results_v3.md: "Y6: run conda env create -f environment.yml once" and "Not run: Y7b (N = 8 for E4)". Output file: results_v5.md.
* **Z1 (Y6, conda build).** micromamba 2.10.0 (downloaded to /tmp, no install into the system) creates the environment from environment.yml exactly as written, prefix /tmp/nqk_conda. Checks: (i) the solve succeeds with no edit to environment.yml; (ii) the imported versions equal the pins; (iii) pytest gives 17 passed; (iv) y1_arm_regression, y3_s_exponent, y4_split_overlap and x1_e6_fitted_rho re-run in that environment match the committed JSON (excluding provenance and wall time) to max |difference| <= 1e-9. RULE: if any check fails, state which and do not claim the conda build is verified.
* **Z2 (Y7b, E4 at N = 8).** Same protocol as E4 for N = 4, 6 (experiments/e4_calib.py): Emb-R layer (beta = 2, angle H/8), datasets D1 and D2, n = 25 train, n_test = 50, splits 0-2, channels dep and ad with gamma = 0.05, L in {1, 4, 16, 64, 256}, lambda in {0.1, 1, 10}, the same feature padding (reps = ceil(N/4)), the same functions noisy_kernels / ridge / ece / ece_exact. Only N changes. The code path is the tensor (density-matrix) Model-B kernel already used for N <= 6; no new simulator is written. Jobs are split by (dataset, split, channel) and run as separate processes; the aggregation script only concatenates and averages.
  GATE Z2-a (dep, Prop 6): at L = 256 the width-binned ECE equals the exact Prop 6 expression with |difference| <= 1e-8 in every one of the 6 (dataset, lambda) cells (N = 4, 6 had max 2.0e-13).
  GATE Z2-b (collapse): dispersion of the dep predictions at L = 256 is <= 1e-4 in every cell.
  AD: no gate. Report width-binned ECE range and maximum dispersion only. AD AUC is NOT reported (X6 rule). Dep AUC is NOT reported (float-level residuals, V1 section 4.5).
  RULE: if a gate fails, report it as failed and do not extend the claim to N = 8. If the run cannot finish (time), report the cells that did finish and say so; no cell is dropped silently.
  Cost estimate before running: 75 points, 4 layers, both arms: 16 s (dep) and 9 s (ad) on this machine, so about 17 min (dep) and 10 min (ad) per (dataset, split) at 256 layers in one process.

## Z-deviations (ROUND Z). Appended after the runs; the ROUND Z section above is unchanged.
* **ZD1 (Z1, tool).** conda and mamba are not installed. micromamba 2.10.0 was downloaded from micro.mamba.pm to /tmp and used with `micromamba create -y -f environment.yml -p /tmp/nqk_conda`. It reads the same environment.yml and solves against conda-forge, so it checks the conda-forge resolution; it is not the `conda` executable itself. environment.yml was not edited. The conda-forge solve gave Python 3.14.10 (the committed results were produced with 3.14.5; the pin is `python=3.14`). python-dateutil 4.0.0 came from conda-forge and the pip block then replaced it with 3.0.0 (pip warning, no failure). The pip block took qiskit, qiskit-aer and qiskit-ibm-runtime from PyPI.
* **ZD2 (Z1, the four re-runs).** The re-runs overwrote results/Y1, Y3, Y4 and X1 JSON in place (the scripts write there). The conda outputs were copied to results/z1_conda_outputs/ and the original committed files were restored from a backup taken before the re-run. z1_conda_check.py compares the two sets.
* **ZD3 (Z2, wall time).** The cost estimate in ROUND Z (about 17 min per dep job) was wrong by a factor of about 20. 12 single-thread jobs ran in parallel; AD jobs took 2000 s and dep jobs took 21 700 s (6.0 h) each. The estimate had been for one multithreaded process timed on an idle machine. No setting was changed to speed it up; all 12 jobs completed (no missing cell).
* **ZD4 (Z2, first launch lost).** The first launch used shell `&` inside a tool call and the 12 processes were killed when the call returned (no output, no CSV). They were relaunched with the tool's background mode and the same script. No result of the first launch exists.
* **ZD5 (Z2, control added after the run).** z2_control_n4.py (not pre-registered) runs the N = 8 script with N = 4 for D1/D2, split 0, dep/ad and compares with the committed E4_raw.csv.gz. Result: every column agrees to about 1e-14 except the dep columns `auc` and `ece` (quantile-binned), which differ in the conda environment (max 0.091 and 0.037 at L = 256) and agree exactly (0.0) in the system Python 3.14.5 environment that produced E4_raw. Cause: the dep predictions are constant to ~1e-13 at L = 256, so rank-based and quantile-binned quantities are decided by float residuals that depend on the BLAS build (V1 section 4.5 already said this). These two quantities are not reported at N = 8, as pre-registered. Width-binned ECE and dispersion are not affected (agree to 1e-14). The BLAS attribution is untested.
