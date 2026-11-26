"""Y2 + Y7a run: one job, 28 pre-transpiled circuits on ibm_marrakesh (HW4 repeat at 2048 shots, Hahn echo at 1024 shots). Guard: stop if the remaining account allowance after the
estimated cost would fall below 150 s. Raw counts, job id, usage -> results/hardware_y2/raw_y2.json. No DD, no twirling, no mitigation; optimisation level 0 (done in prep)."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qiskit_ibm_runtime import SamplerV2
from nqk.hw import get_service

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware_y2")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
P = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))
isa, n4 = P["isa"], P["n4"]
fn = os.path.join(HW, "raw_y2.json")
assert not os.path.exists(fn), "already run"
svc = get_service()
backend = svc.backend(meta["backend"])
est = 16.0 * n4 / 18 + 0.29e-3 * (len(isa) - n4) * meta["shots_echo"] + 3
u = svc.usage()
rem = u["usage_remaining_seconds"]
print(f"est {est:.0f}s; account remaining {rem}s", flush=True)
assert rem - est >= 150, "GUARD: remaining after job would be < 150 s"
sampler = SamplerV2(mode=backend)
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False
pubs = [(c, None, meta["shots_hw4"] if k < n4 else meta["shots_echo"]) for k, c in enumerate(isa)]
t0 = time.time()
job = sampler.run(pubs)
print("job", job.job_id(), flush=True)
res = job.result()
t1 = time.time()
counts = [r.data.meas.get_counts() if hasattr(r.data, "meas") else r.data[list(r.data.keys())[0]].get_counts() for r in res]
try:
    usage = job.usage()
except Exception as e:
    usage = str(e)
u1 = svc.usage()["usage_consumed_seconds"]
out = dict(job_id=job.job_id(), backend=meta["backend"], layout=meta["layout"], n_circuits=len(pubs), n4=n4, shots=[p[2] for p in pubs], counts=counts, usage_seconds=usage,
           account_usage_before=u["usage_consumed_seconds"], account_usage_after=u1, wall_seconds=t1 - t0, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           options=dict(optimization_level=0, dd=False, twirling=False, mitigation=None))
json.dump(out, open(fn, "w"), default=str)
print("saved", fn, "usage", usage, "account delta", u1 - u["usage_consumed_seconds"], "wall", round(t1 - t0, 1), "remaining", svc.usage()["usage_remaining_seconds"])
