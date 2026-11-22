import os
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fvv_sim as F
class A: pass
args = A(); args.out = "/home/claude/fvv/res"; args.cache = "/home/claude/fvv/res/cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"/home/claude/fvv/cells/{m}.csv.gz" for m in (434, 401, 437)]
rows = []
for h in (150.0, 100.0, 60.0):
    F.P["h_uav"] = h
    for s in F.SCENARIOS:
        W = F.build_world(s, args, np.random.default_rng(0))
        # sites relevant: within 10 km of zone cells? report all
        r = dict(h=h, scenario=s,
                 uu_terr=W.adj.nnz // 2, uu_fs=W.adj_f.nnz // 2,
                 ug_terr=W.sg.nnz, ug_fs=W.sg_f.nnz,
                 acc_terr=W.cov.nnz, acc_fs=W.cov_f.nnz)
        # cells covered by at least one site
        r["cells_cov_terr"] = float((np.asarray(W.cov.sum(axis=0)).ravel() > 0).mean())
        r["cells_cov_fs"] = float((np.asarray(W.cov_f.sum(axis=0)).ravel() > 0).mean())
        r["mean_sites_per_cell_terr"] = float(W.cov.sum(axis=0).mean())
        rows.append(r)
df = pd.DataFrame(rows)
for a in ("uu", "ug", "acc"):
    df[a + "_blocked_pct"] = 100 * (1 - df[a + "_terr"] / df[a + "_fs"])
df.to_csv("link_stats.csv", index=False)
print(df.round(3).to_string())
