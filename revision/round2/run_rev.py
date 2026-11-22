"""Revision experiments for the STAR manuscript (run on Kaggle, Internet ON).
Usage: python run_rev.py --exp E1 E2 E3 E4 E5 --out /kaggle/working/rev --procs 4"""
import argparse, glob, os, sys, time, json, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvv_sim2 as F

SC = ["sardoba", "fergana", "karakalpak"]
CORE = ["Proposed", "ConnCov-dynamic", "ConnCov-static", "Ferry"]
EXPS = {
    # E1: corrected assignment (tree grown with the available UAVs), full main experiment + connected sweep
    "E1": dict(seeds=30, K=[6, 10, 14], save=True,
               methods=["Proposed", "ConnCov-dynamic", "Proposed-noMemory", "Proposed-noTerrain", "Proposed-static",
                        "ConnCov-static", "KMeans", "Ferry", "Sweep"],
               configs={"fix": {}}),
    # E2: planner's prior is wrong (zone geometry, damage and severity parameters, BS status)
    "E2": dict(seeds=15, K=[10], methods=CORE,
               configs={"shift3": dict(prior_shift_km=3), "shift6": dict(prior_shift_km=6), "rot30": dict(prior_rot_deg=30),
                        "scale0.7": dict(prior_scale=0.7), "scale1.4": dict(prior_scale=1.4),
                        "dmg1": dict(prior_dmg0=1.0), "pc1_0": dict(prior_pc1=0.0), "bsflip0.2": dict(bs_flip=0.2)}),
    # E3: stochastic delivery, retry interval, phone battery; Bayesian belief variant
    "E3": dict(seeds=15, K=[10], methods=["Proposed", "Proposed-Bayes", "Proposed-noMemory", "ConnCov-dynamic", "Ferry"],
               configs={"p0.5": dict(p_deliv=0.5), "retry5": dict(retry_min=5), "retry10_p0.7": dict(retry_min=10, p_deliv=0.7),
                        "batt6h": dict(battery_h=6), "mix": dict(p_deliv=0.7, retry_min=5, battery_h=12)}),
    # E4: few gateways / short backhaul: role of multi-hop relaying
    "E4": dict(seeds=15, K=[10], methods=["Proposed", "ConnCov-dynamic", "Proposed-noMemory", "ConnCov-static", "Ferry", "Sweep"],
               configs={"satonly": dict(bs_fail_all=1), "pbh20": dict(p_bh=20.0), "satonly_pbh20": dict(bs_fail_all=1, p_bh=20.0)}),
    # E5: algorithm and model parameters
    "E5": dict(seeds=15, K=[10], methods=CORE,
               configs={"delta10": dict(epoch_min=10), "delta30": dict(epoch_min=30), "D30": dict(urg_D_min=30), "D120": dict(urg_D_min=120),
                        "beta0": dict(beta=0.0), "beta2": dict(beta=2.0), "N100": dict(n_victims=100), "N1000": dict(n_victims=1000),
                        "grid1000": dict(site_res=1000), "grid2000": dict(site_res=2000), "stagger0.5": dict(stagger=0.5),
                        "ferryhover5": dict(ferry_hover_min=5), "ferryhover20": dict(ferry_hover_min=20), "ferrydeliv1000": dict(ferry_deliv_m=1000)}),
}
BASE = {k: v for k, v in F.P.items()}
_W = {}

class A: pass

def world(scen, cfg, a):
    key = (scen, json.dumps(cfg, sort_keys=True))
    if key not in _W:
        _W.clear()
        F.P.clear(); F.P.update(BASE); F.P.update(cfg); F.P["assign_fix"] = 1.0
        _W[key] = F.build_world(scen, a, np.random.default_rng(0))
    F.P.clear(); F.P.update(BASE); F.P.update(cfg); F.P["assign_fix"] = 1.0
    F.P["_pen"] = F.SCENARIOS[scen]["penetration_db"]
    return _W[key]

def job(args):
    exp, cname, cfg, scen, seed, Ks, methods, save, a = args
    W = world(scen, cfg, a)
    R = F.realise(W, np.random.default_rng(1000 + seed))
    rows, dl = [], {}
    for K in Ks:
        for m in methods:
            t = time.time()
            res, deliv, _ = F.simulate(W, R, K, m, np.random.default_rng(seed))
            res.update(exp=exp, config=cname, scenario=scen, seed=seed, K=K, method=m, wall_s=time.time() - t)
            rows.append(res)
            if save: dl[f"K{K}_{m}"] = deliv
    if save:
        os.makedirs(f"{a.out}/{exp}/deliv", exist_ok=True)
        np.savez_compressed(f"{a.out}/{exp}/deliv/{scen}_s{seed}.npz", vcls=R.vcls, vtau=R.vtau, vTg=R.vTg, va=R.va, **dl)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=list(EXPS)); ap.add_argument("--out", default="rev")
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--cache", default=None)
    ap.add_argument("--cells", nargs="*", default=None); ap.add_argument("--seeds", type=int, default=0, help="override seeds (smoke test)")
    o = ap.parse_args()
    a = A(); a.out = o.out; a.cache = o.cache or os.path.join(o.out, "cache"); a.synthetic = False; a.pop_tif = None; a.cells = o.cells
    os.makedirs(o.out, exist_ok=True)
    print("pre-building worlds (downloads DEM and WorldPop once)", flush=True)
    for s in SC:
        F.P.clear(); F.P.update(BASE); F.build_world(s, a, np.random.default_rng(0))
    for exp in o.exp:
        E = EXPS[exp]; t0 = time.time()
        ns = o.seeds or E["seeds"]
        tasks = [(exp, c, cfg, s, sd, E["K"], E["methods"], E.get("save", False), a)
                 for c, cfg in E["configs"].items() for s in SC for sd in range(ns)]
        rows = []
        with Pool(o.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
                rows += r
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print(f"{exp}: {i+1}/{len(tasks)} tasks, {time.time()-t0:.0f} s", flush=True)
                    os.makedirs(f"{o.out}/{exp}", exist_ok=True)
                    pd.DataFrame(rows).to_csv(f"{o.out}/{exp}/runs.csv", index=False)
        print(f"{exp} done in {time.time()-t0:.0f} s", flush=True)

if __name__ == "__main__":
    main()
