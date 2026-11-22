import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fvv_sim as F
class A: pass
args = A(); args.out = "/home/claude/fvv/res"; args.cache = "/home/claude/fvv/res/cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"/home/claude/fvv/cells/{m}.csv.gz" for m in (434, 401, 437)]
for h in (150.0, 60.0):
    F.P["h_uav"] = h
    W = F.build_world("fergana", args, np.random.default_rng(0))
    # sites covering at least one zone cell (useful access sites)
    use = np.asarray(W.cov_f.sum(axis=1)).ravel() > 0
    nt = np.asarray(W.sg.sum(axis=1)).ravel(); nf = np.asarray(W.sg_f.sum(axis=1)).ravel()
    lost = (nf > 0) & (nt == 0)
    print(h, "access sites", use.sum(), "with gw link fs", (use & (nf>0)).sum(), "terr", (use & (nt>0)).sum(),
          "lost all gw", (use & lost).sum(), "median gw links/site terr", np.median(nt[use]), "fs", np.median(nf[use]))
    # UAV-UAV degree among access sites
    dt = np.asarray(W.adj.sum(axis=1)).ravel(); df = np.asarray(W.adj_f.sum(axis=1)).ravel()
    print("  median UAV-UAV degree terr", np.median(dt[use]), "fs", np.median(df[use]), "cells lost cov", (np.asarray(W.cov_f.sum(0)).ravel()>0).sum()-(np.asarray(W.cov.sum(0)).ravel()>0).sum(), "of", W.cov.shape[1])
