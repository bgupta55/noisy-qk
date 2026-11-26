#!/bin/bash
# Reproduce everything. Hardware steps need IBMQ_KEY and IBMQ_CRN in the environment (never stored in files).
set -e; cd "$(dirname "$0")/experiments"
python3 -m pytest -q ../tests -W ignore
for f in e0_wang e1_rates e8_e2 e3_e9 e4_calib e5_central e6_e7; do python3 -u $f.py; done
python3 -u hw_prep.py                       # no QPU time: selection, transpile, CZ assertions, Aer ideal/noise
for s in hw4 hw1 hw2 hw2r; do python3 -u hw_run.py $s; done   # ~89 s QPU on ibm_kingston
python3 -u hw_analysis.py; python3 -u e10_e13.py
# ---- Round X (23 Nov 2026); pre-registered in results/PREREGISTRATION.md (ROUND X)
python3 -u x1_e6_fitted_rho.py; python3 -u x2_e1_modal_weights.py; python3 -u x3_e5_c2.py; python3 -u x8_wang_sensitivity.py
python3 -u x4_prep.py                        # no QPU time; ibm_marrakesh
python3 -u x4_run.py                         # ~242 s QPU (3 jobs); needs IBMQ_KEY / IBMQ_CRN
python3 -u x4_analysis.py; python3 -u x5_x6_report.py; python3 -u x_derived.py
# ---- Round Y; pre-registered in results/PREREGISTRATION.md (ROUND Y)
python3 -u y1_arm_regression.py; python3 -u y3_s_exponent.py; python3 -u y4_split_overlap.py
python3 -u y2_prep.py                        # no QPU time; ibm_marrakesh
python3 -u y2_run.py                         # ~19 s QPU; saved account or IBMQ_KEY / IBMQ_CRN
python3 -u y2_analysis.py; python3 -u y_derived.py
# ---- Round Z; pre-registered in results/PREREGISTRATION.md (ROUND Z)
# z1: needs micromamba/conda; outputs of the four re-runs are in results/z1_conda_outputs/ (see PREREGISTRATION ZD2)
python3 -u z1_conda_check.py                 # run with the conda python
# z2: 12 independent jobs (about 0.6 h AD, 6 h dep each, single thread, 12 in parallel); then aggregate
for ds in D1 D2; do for sp in 0 1 2; do for ch in dep ad; do OMP_NUM_THREADS=1 python3 -u z2_e4_n8.py $ds $sp $ch & done; done; done; wait
python3 -u z2_analysis.py; python3 -u z2_control_n4.py
python3 make_figures.py; python3 make_manifest.py
