"""X4 run: submit the pre-transpiled ibm_marrakesh circuits (x4_prep.py) in 3 jobs; a usage guard stops before a job if the remaining account
allowance after its *estimated* cost would fall below 150 s. Raw counts, job ids, usage, timestamps -> results/hardware_x4/raw_x4_<k>.json.
Credentials only from IBMQ_KEY / IBMQ_CRN. No DD, no twirling, no mitigation, optimisation level 0 (done in prep)."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qiskit_ibm_runtime import SamplerV2
from nqk.hw import get_service

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware_x4")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
P = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))
isa, NP = P["isa"], P["n_pairs"]
settings = [tuple(s) for s in meta["settings"]]
shots_of = {tuple(map(int, k.split(","))): v for k, v in meta["shots"].items()}
MS_PER_SHOT = 0.29e-3       # V1 calibration: 58 s / (198*1024) shots
GROUPS = [[0, 1, 2], [3, 4, 5], [6]]
svc = get_service()
backend = svc.backend(meta["backend"])
for gi, g in enumerate(GROUPS):
    fn = os.path.join(HW, f"raw_x4_{gi}.json")
    if os.path.exists(fn):
        print("group", gi, "already done, skipping"); continue
    est = sum(NP * shots_of[settings[k]] for k in g) * MS_PER_SHOT
    u = svc.usage()
    rem = u["usage_remaining_seconds"]
    print(f"group {gi}: settings {[settings[k] for k in g]} est {est:.0f}s; account remaining {rem}s", flush=True)
    if rem - est < 150:
        print("GUARD: stop (remaining after job would be < 150 s)"); break
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    pubs = []
    for k in g:
        for c in isa[k * NP:(k + 1) * NP]:
            pubs.append((c, None, shots_of[settings[k]]))
    t0 = time.time()
    job = sampler.run(pubs)
    print("job", job.job_id(), "n_pubs", len(pubs), flush=True)
    res = job.result()
    t1 = time.time()
    counts = [r.data.meas.get_counts() for r in res]
    try:
        usage = job.usage()
    except Exception as e:
        usage = str(e)
    try:
        metrics = job.metrics()
    except Exception as e:
        metrics = str(e)
    u1 = svc.usage()["usage_consumed_seconds"]
    out = dict(group=gi, settings=[settings[k] for k in g], shots=[shots_of[settings[k]] for k in g], backend=meta["backend"], layout=meta["layout"],
               job_id=job.job_id(), n_circuits=len(pubs), counts=counts, usage_seconds=usage, metrics=metrics,
               account_usage_before=u["usage_consumed_seconds"], account_usage_after=u1, wall_seconds=t1 - t0,
               utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), options=dict(optimization_level=0, dd=False, twirling=False, mitigation=None))
    json.dump(out, open(fn, "w"), default=str)
    print("saved", fn, "usage", usage, "account delta", u1 - u["usage_consumed_seconds"], "wall", round(t1 - t0, 1), flush=True)
print("final usage", svc.usage()["usage_remaining_seconds"], "s remaining")
