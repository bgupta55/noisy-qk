"""Submit pre-transpiled ISA circuits (hw_prep.py) to the real QPU, one stage per invocation: python hw_run.py {hw4|hw1|hw2|hw2r}
Credentials from env IBMQ_KEY / IBMQ_CRN only. Saves raw counts, job id, usage, timestamps to results/hardware/raw_<stage>.json.
Protocol: optimisation_level 0 (done in prep), fixed layout, no error mitigation, no dynamical decoupling, no twirling."""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qiskit_ibm_runtime import SamplerV2
from nqk.hw import get_service

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW = os.path.join(ROOT, "results", "hardware")
stage = sys.argv[1]
SHOTS = dict(hw4=2048, hw1=512, hw2=1024, hw2r=1024)[stage]
key = dict(hw4="isa4", hw1="isa1", hw2="isa2", hw2r="isa2r")[stage]
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
circs = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))[key]
if len(sys.argv) > 2:  # optional truncation for dry mini-runs
    circs = circs[: int(sys.argv[2])]
svc = get_service()
backend = svc.backend(meta["backend"])
u0 = svc.usage()["usage_consumed_seconds"]
sampler = SamplerV2(mode=backend)
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False
t0 = time.time()
job = sampler.run(circs, shots=SHOTS)
print(stage, "job", job.job_id(), "n_circuits", len(circs), flush=True)
res = job.result()
t1 = time.time()
counts = [r.data.meas.get_counts() if hasattr(r.data, "meas") else r.data.c.get_counts() for r in res] if stage != "hw1" else [r.data.c.get_counts() for r in res]
try:
    usage = job.usage()
except Exception as e:
    usage = str(e)
try:
    metrics = job.metrics()
except Exception as e:
    metrics = str(e)
u1 = svc.usage()["usage_consumed_seconds"]
out = dict(stage=stage, backend=meta["backend"], layout=meta["layout"], job_id=job.job_id(), shots=SHOTS, n_circuits=len(circs),
           counts=counts, usage_seconds=usage, metrics=metrics, account_usage_before=u0, account_usage_after=u1,
           wall_seconds=t1 - t0, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           options=dict(optimization_level=0, dd=False, twirling=False, mitigation=None))
fn = os.path.join(HW, f"raw_{stage}{'_mini' if len(sys.argv) > 2 else ''}.json")
json.dump(out, open(fn, "w"), default=str)
print("saved", fn, "usage", usage, "account delta", u1 - u0, "wall", round(t1 - t0, 1))
