import os
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fvv_sim as F
class A: pass
args = A(); args.out = "/home/claude/fvv/res"; args.cache = "/home/claude/fvv/res/cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"/home/claude/fvv/cells/{m}.csv.gz" for m in (434, 401, 437)]
job = sys.argv[1]; scen = sys.argv[2]
out = f"/home/claude/fvv/rev/out_{job}_{scen}"; os.makedirs(out, exist_ok=True)
if job == "end60":
    F.P["endurance_min"] = 60.0
W = F.build_world(scen, args, np.random.default_rng(0))
if job == "fig5":
    seeds, Ks, meths = range(30), [10], ["Proposed", "Proposed-noMemory", "Ferry", "ConnCov-static", "KMeans"]
elif job == "scarce":
    seeds, Ks, meths = range(30), [3, 4], ["Proposed", "ConnCov-dynamic"]
elif job == "end60":
    seeds, Ks, meths = range(15), [10], ["Proposed", "ConnCov-dynamic", "ConnCov-static", "Ferry"]
rows = []
for seed in seeds:
    R = F.realise(W, np.random.default_rng(1000 + seed))
    for K in Ks:
        for m in meths:
            t1 = time.time()
            res, deliv, ts = F.simulate(W, R, K, m, np.random.default_rng(seed), log_ts=(job == "fig5"))
            res.update(scenario=scen, seed=seed, K=K, method=m, wall_s=time.time() - t1)
            rows.append(res)
            if job == "fig5":
                np.savez_compressed(f"{out}/s{seed}_{m}.npz", deliv=deliv, ts=np.array(ts), va=R.va, cls=R.vcls)
    pd.DataFrame(rows).to_csv(f"{out}/runs.csv", index=False)
    print(scen, job, seed, f"{time.time():.0f}", flush=True)
