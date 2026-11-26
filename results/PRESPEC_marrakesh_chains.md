# PRESPEC_marrakesh_chains (ROUND W), written 26 Nov 2026 BEFORE any W-script ran and BEFORE any job was submitted

Source: `additional_expr.md` (sections 1-6). This file fixes every choice the source leaves open. Anything changed after the first job goes into the dated deviations log at the end; this section is never edited in place. SHA-256 and UTC time of this file are written to `results/hardware_w1/prespec_hash.txt` before `w1_prep.py` is run.

## 0. Fixed settings (same as X4/Y2 runs)
* Backend `ibm_marrakesh`. Kernel: `kernel_circuit(x, x', beta=0.75, L, fold)` of `src/nqk/hw.py`, N = 4, beta = 0.75 (X4 value). `SamplerV2` in job mode (Open Plan has no sessions), `optimization_level=0`, fixed `initial_layout`, barriers between layers, no DD, no twirling, no mitigation, CZ count asserted equal to `expected_cz(L, fold)` (12 for the idle circuits, 0 for echo).
* "Layers" -> (L, fold): 2 -> (1,0); 6 -> (3,0); 12 -> (2,1); 20 -> (2,2); 28 -> (2,3); 36 -> (2,4) (noisy layers = 2 L (1+2 fold), as in Table 3).
* Shots: R 1024, A 2048, T 1024, repeat 1024.
* Chain A = [52,53,54,55].

## 1. Inputs (fixed by a rule that uses no hardware data)
The earlier marrakesh run used 14 points; the first 9 are the training inputs (`results/hardware_x4/prep_meta.json`, `Xall[:9]`). Six of these 9 are chosen by farthest-point sampling in Euclidean distance in the 4-d feature space: the first point is the one farthest from the mean of the 9; each next point maximises the minimum distance to the points already chosen; ties -> smaller index. The six indices are written to `chain_choice.json` and `prep_meta.json`. The same six on every chain.
Grid: all 36 ordered pairs (i, j) of the six, i = forward half (x), j = backward half (x'), diagonal included.

## 2. Chain choice (script `w1_prep.py`, run before any circuit is built)
All connected 4-qubit paths of the coupling map of the CZ edges that have a calibrated error, not using any qubit of A. score = sum of the 3 CZ errors + sum of the 4 readout errors. B = lowest score (tie -> lexicographically smaller list; a path and its reverse are the same path, stored with the smaller end first). C = highest min(T2) with min T2 >= 100 us, disjoint from A and B (only recorded, see section 7). Written to `chain_choice.json` with the calibration.

## 3. Circuits per chain
* R: depths [2,6,12,20,28,36] x 36 ordered pairs = 216 circuits, 1024 shots. p[chain,L,i,j] = counts("0000")/shots. K_exact from noiseless simulation of the same builder.
* A: the same 3 pairs as the Y2 run (`results/hardware_y2/prep_meta.json`, `pairs4`, indices into the 14 points), `idle_kernel_circuit`, delays 0, 100, 200 us, both orders = 18 circuits, 2048 shots.
* T: Hahn echo as in Y7a (4 qubits in parallel, prep + and -, X in the centre, total delays 10, 30, 60, 120, 240 us), 10 circuits, 1024 shots.
* Repeat (chain A only, last job): layers 2, 12, 36, 12 entries of the 36 per layer (the entries at ranks round(linspace(0,35,12)) after sorting the 36 entries by exact K, ties by (i,j)), 36 circuits, 1024 shots.
* Q: calibration snapshot (T1, T2, readout error, CZ error) of every chain at the start (read in `w1_prep.py`, before the circuits are built) and at the end of the session.

## 4. Job order and budget rule (Open Plan)
Order: A-R, B-R, A-A, B-A, A-T, B-T, A-R-repeat. Account allowance is 600 s per rolling period; 204 s remained when this file was written. Priority (low number = more important): A-R 0, B-R 0, repeat 1, B-A 2, A-A 3, B-T 4, A-T 5. Before each job: run it only if remaining - estimated_cost(job) - sum(estimated cost of all pending jobs that are of higher priority and later in the order) >= 3 s. The cost per shot is updated from the measured usage of the first job. A skipped job is reported as skipped. Chain C and run O are not run (the source says to drop them first); this is stated as a deviation from the full design, not as a result.

## 5. Analysis (script `w1_analysis.py`)
Retention: per chain and L, OLS p = s K + o over the 36 entries (diagonal included). SE of s by entry bootstrap (resample the 36 entries with replacement, same resampled indices at all depths, 2000 reps, seed 3101). Early slope = OLS slope of s against L (linear, not log) at L in {2,6,12,20}; late slope = at L in {20,28,36}. rise = late - early; 95% interval = percentile interval of the same bootstrap. Misfit ratio = sqrt(RSS/(n-2)) / sqrt(mean p(1-p)/S), as in X4.
Decomposition at L in {28, 36}: M0 p = aK + o; M1 = M0 + b_j; M2 = M0 + c_i (dummy columns, minimum-norm lstsq). R^2 = 1 - RSS/TSS on the 36 entries (adjusted R^2 also reported). gain_b = R2(M1) - R2(M0); gain_c = R2(M2) - R2(M0); ratio = gain_b / max(gain_c, 1e-3). b_j is the M1 coefficient centred to mean zero. Asymmetry P_ij - P_ji (i < j, 15 pairs) with SE sqrt(p1(1-p1)/S + p2(1-p2)/S).
Idle asymmetry: asym = p(x,x',tau) - p(x',x,tau) for the 3 pairs; prediction by the per-qubit AD+PD idle model of `y2_prep.py` (including its endianness correction) with T1, T2 of THIS chain read at the start. Also Aer noise-model prediction. Ratio = mean over the 6 (pair, tau in {100,200}) rows of asym_hw / asym_pred (the Y2 definition); the through-origin regression ratio sum(hw*pred)/sum(pred^2) is reported next to it.
Echo: E = P1(prep -) - P1(prep +) per qubit; ln E = ln A - t/T2e over points with E > 2 SE (needs >= 3 points); reported against the calibration T2 (factor max(T2e/T2cal, T2cal/T2e)).
Earlier chain-A retention for D1: `results/hardware_x4` data, per setting, 81 pairs, same OLS and entry bootstrap; sensitivity: restricted to the pairs of the six chosen inputs that exist there.

## 6. Decision rules (as in the source, with the definitions above)
| ID | Pass criterion |
|---|---|
| D1 | |s_today(A) - s_earlier(A)| <= 3 combined SE at layers 2, 12 and 36 AND |s_today(A, the same 12 entries as the repeat) - s_repeat(A)| <= 3 combined SE at layers 2, 12, 36. Combined SE = sqrt(SE1^2 + SE2^2). |
| D2 | the paired-bootstrap 95% interval of s(B) - s(A) at layer 36 excludes zero |
| D3 | the 95% interval of the rise excludes zero on both chains (sign reported) |
| D4 | ratio = gain_b / max(gain_c, 1e-3) >= 2 at layers 28 and 36 on both chains |
| D5 | Pearson correlation of b_j between A and B > 0.7 at both layers 28 and 36 |
| D6 | on chain B, sign(asym_hw) = sign(asym_pred) in all 6 (pair, tau>0) rows AND mean ratio in [0.6, 1.1] |

Results to expect (source, section 5) are used only as comparison; there is no gate on them. Whatever appears is reported. No claim of hardware proof of any theorem, no claim of advantage, no claim about devices or chains not measured.

## 7. Known limits fixed in advance
* Six inputs: b_j and c_i have 5 free parameters each beyond the offset; the decomposition is indicative, no significance test is claimed.
* Chain C and run O: not run for budget reasons (section 4).
* Aer prediction is computed from the calibration read in `w1_prep.py` before the first job (source section 4a).

---
## W-deviations (ROUND W). Appended after the runs; the section above is unchanged.
* **WD1 (26 Nov 2026, before the first job).** Open Plan has no sessions: the seven jobs were submitted back to back from one script (`w1_run.py`) in job mode, job ids recorded. Allowance left at start: 204 s; 19 s left at the end. All seven jobs ran; the budget rule skipped nothing.
* **WD2 (chain C and run O not run).** Section 4 of this file fixed that they are dropped. Chain C was only selected and recorded ([78,89,88,87], min T2 128.7 us); no circuit was run on it. Run O (dynamical decoupling) was not run. Both remain open.
* **WD3 (usage reporting).** `job.usage()` returned 0 for the two echo jobs; the account delta (6 s each) is used. Total account delta 64+64+16+16+6+6+13 = 185 s.
* **WD4 (rise definition).** This file fixed the rise as the difference of linear slopes of s against L (early L<=20, late L>=20). The earlier X4/Y1 analyses used slopes of ln s. The two are different statistics; results below use the linear one as fixed here. Post-hoc direct differences of s (layer 36 minus 20, 28 minus 20, 36 minus 28) are in `W1_posthoc.json` and are used for no verdict.
* **WD5 (time stamps).** `snapshot()` writes local time (UTC+5:30) in its `time` field; UTC times are in the raw job files. The prespec hash was written at 2026-11-26 03:08:41 UTC, the first job ran at 03:11:12 UTC.
* **WD6 (earlier-day comparison).** D1 compares today's chain A with X4 (23 Nov) using all 81 X4 pairs per layer (same OLS and entry bootstrap). A comparison restricted to the 15 oriented pairs shared with the six inputs is in `W1_supplement.json` (sensitivity only).
