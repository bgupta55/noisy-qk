"""W1 run: interleaved jobs on ibm_marrakesh (PRESPEC_marrakesh_chains.md section 4). Order A-R, B-R, A-A, B-A, A-T, B-T, A-Rrep; one SamplerV2 job each (job mode).
Budget rule: run a job only if remaining - est(job) - sum(est of later jobs with priority number <= this job's) >= 3 s; est uses 0.29 ms/shot, replaced by the measured
rate after the first job (x 1.1 safety). Raw counts + job ids + usage -> results/hardware_w1/raw_<job>.json. End-of-session calibration snapshot -> snapshot_end.json.
No DD, no twirling, no mitigation, optimisation level 0 (done in prep)."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qiskit_ibm_runtime import SamplerV2
from nqk.hw import get_service, snapshot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware_w1")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
isa_all = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))
PRIO = {"A-R": 0, "B-R": 0, "A-Rrep": 1, "B-A": 2, "A-A": 3, "B-T": 4, "A-T": 5}
ORDER = meta["job_order"]
RATE = 0.29e-3
OVER = 2.0          # per-job overhead (s)
MARGIN = 3.0
svc = get_service()
backend = svc.backend(meta["backend"])
log = []


def est(name, rate):
    return sum(meta["jobs"][name]["shots"]) * rate + OVER


for k, name in enumerate(ORDER):
    fn = os.path.join(HW, f"raw_{name}.json")
    if os.path.exists(fn):
        print(name, "already done"); continue
    rem = svc.usage()["usage_remaining_seconds"]
    later = sum(est(n2, RATE) for n2 in ORDER[k + 1:] if PRIO[n2] <= PRIO[name] and not os.path.exists(os.path.join(HW, f"raw_{n2}.json")))
    e = est(name, RATE)
    print(f"{name}: est {e:.1f}s, reserve for higher-priority later jobs {later:.1f}s, remaining {rem}s", flush=True)
    log.append(dict(job=name, est=e, reserve=later, remaining=rem))
    if rem - e - later < MARGIN:
        print("SKIP", name, "(budget rule)", flush=True)
        json.dump(dict(job=name, skipped=True, est=e, reserve=later, remaining=rem), open(os.path.join(HW, f"skipped_{name}.json"), "w"))
        continue
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    pubs = [(c, None, s) for c, s in zip(isa_all[name], meta["jobs"][name]["shots"])]
    u0 = svc.usage()["usage_consumed_seconds"]
    t0 = time.time()
    job = sampler.run(pubs)
    print("job", job.job_id(), "n_pubs", len(pubs), flush=True)
    res = job.result()
    t1 = time.time()
    counts = []
    for r in res:
        reg = list(r.data.keys())[0]
        counts.append(getattr(r.data, reg).get_counts())
    try:
        usage = job.usage()
    except Exception as ex:
        usage = str(ex)
    try:
        metrics = job.metrics()
    except Exception as ex:
        metrics = str(ex)
    u1 = svc.usage()["usage_consumed_seconds"]
    out = dict(job=name, job_id=job.job_id(), backend=meta["backend"], layout=meta["jobs"][name]["layout"], n_circuits=len(pubs), shots=meta["jobs"][name]["shots"], counts=counts,
               usage_seconds=usage, metrics=metrics, account_usage_before=u0, account_usage_after=u1, wall_seconds=t1 - t0, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
               options=dict(optimization_level=0, dd=False, twirling=False, mitigation=None))
    json.dump(out, open(fn, "w"), default=str)
    print("saved", fn, "usage", usage, "account delta", u1 - u0, "wall", round(t1 - t0, 1), flush=True)
    if isinstance(usage, (int, float)) and usage > 0:
        RATE = max(RATE, 1.1 * (usage - 0) / sum(meta["jobs"][name]["shots"]))
        print("rate updated to", RATE, flush=True)

end = {c: snapshot(backend, meta[c]) for c in ("A", "B")}
json.dump(end, open(os.path.join(HW, "snapshot_end.json"), "w"), indent=1, default=str)
json.dump(log, open(os.path.join(HW, "budget_log.json"), "w"), indent=1)
print("final remaining", svc.usage()["usage_remaining_seconds"], "s")
