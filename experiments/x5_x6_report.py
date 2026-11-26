"""X5 + X6: reporting-only tasks. No new simulation, no QPU time. Everything is read from existing results/*.json and written to results/X5_X6_reported.json.
X5: HW1 reported as kappa, t_z, T1 only (u and T2 dropped; T2 from 1-qubit tomography is not reliable, no Ramsey / Hahn-echo run was made).
X6: E4 amplitude-damping AUC at L=256 dropped from the report (the dropped values are stored under 'dropped_for_audit' only); N in {4,6} kept; N=8 was not run."""
from common import *
t0 = time.time()
H = json.load(open(os.path.join(RES, "HW_analysis.json")))["results"]["HW1"]
rows = H["rows"]
x5 = dict(job=H["job"], shots=H["shots"], qpu_seconds=H["qpu_s"], reported_fields=["kappa", "t_z", "T1_est_us", "T1_cal_us", "t_z_pred"], dropped_fields=["u", "T2_est_us"], per_row=[])
for r in rows:
    x5["per_row"].append({k: r[k] for k in ("delay_us", "qubit", "kappa", "t_z", "T1_est_us", "T1_cal_us", "t_z_pred")})
for d in (100, 200):
    s = [r for r in rows if r["delay_us"] == d]
    x5[f"mean_{d}us"] = dict(kappa=float(np.mean([r["kappa"] for r in s])), kappa_range=[min(r["kappa"] for r in s), max(r["kappa"] for r in s)],
                             t_z=float(np.mean([r["t_z"] for r in s])), t_z_range=[min(r["t_z"] for r in s), max(r["t_z"] for r in s)],
                             t_z_pred_from_calibrated_T1=float(np.mean([r["t_z_pred"] for r in s])),
                             T1_est_us=float(np.mean([r["T1_est_us"] for r in s])), T1_est_range=[min(r["T1_est_us"] for r in s), max(r["T1_est_us"] for r in s)],
                             T1_cal_us_mean=float(np.mean([r["T1_cal_us"] for r in s])))
E4 = json.load(open(os.path.join(RES, "E4_calibration.json")))
cfg4 = E4["config"]
ad = E4["results"]["summary_ad_L256"]
dep = E4["results"]["summary_dep_L256"]
x6 = dict(config=cfg4, N_values_in_E4=sorted({r["N"] for r in ad}), N8_run=bool(8 in {r["N"] for r in ad}),
          n_AD_cells=len(ad),
          retained_AD_fields=["ece", "ece_width", "ece_const_target", "disp", "spread"],
          AD_retained=[{k: r[k] for k in ("dataset", "N", "lam", "ece", "ece_width", "ece_const_target", "disp", "spread")} for r in ad],
          AD_ece_width_range=[min(r["ece_width"] for r in ad), max(r["ece_width"] for r in ad)],
          AD_disp_max=max(r["disp"] for r in ad),
          dep_cells=len(dep), dep_max_abs_diff_width=max(r["abs_diff_width"] for r in dep),
          dropped_for_audit=dict(note="AD AUC at L=256 are NOT to be quoted (slow residual modes, dispersion <= 0.02)",
                                 AD_auc=[dict(dataset=r["dataset"], N=r["N"], lam=r["lam"], auc=r["auc"]) for r in ad]))
save("X5_X6_reported", dict(source="HW_analysis.json (HW1), E4_calibration.json"), dict(X5_HW1=x5, X6_E4=x6), t0)
print(json.dumps(x5["mean_200us"], indent=0)); print(json.dumps(x5["mean_100us"], indent=0))
print(x6["N_values_in_E4"], x6["N8_run"], x6["AD_ece_width_range"], x6["AD_disp_max"], x6["dep_max_abs_diff_width"])
