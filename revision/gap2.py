import os
"""Optimality check of STAR's single-epoch tree growth against an exact MILP
(connected budgeted maximum coverage, single-commodity flow), on weight vectors
logged from actual STAR runs (served-time memory states at different epochs)."""
import sys, time, numpy as np, pandas as pd, pulp, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fvv_sim as F
class A: pass
args = A(); args.out = "/home/claude/fvv/res"; args.cache = "/home/claude/fvv/res/cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"/home/claude/fvv/cells/{m}.csv.gz" for m in (434, 401, 437)]
scen = sys.argv[1]; margin = 4.0
W = F.build_world(scen, args, np.random.default_rng(0))
orig = F.plan_tree
LOG = []
def logged(W_, w, gw_alive, budget, pen, cov, adj, sg):
    LOG.append((w.copy(), gw_alive.copy(), budget))
    return orig(W_, w, gw_alive, budget, pen, cov, adj, sg)
F.plan_tree = logged
inst = []
for seed, K in ((0, 6), (1, 10)):
    LOG.clear()
    R = F.realise(W, np.random.default_rng(1000 + seed))
    F.simulate(W, R, K, "Proposed", np.random.default_rng(seed))
    for j in (0, 4, 8, 12, 16, 20):
        if j < len(LOG): inst.append((seed, K, j) + LOG[j])
F.plan_tree = orig
x0, x1, y0, y1 = W.cx.min(), W.cx.max(), W.cy.min(), W.cy.max(); m = margin * 1e3
sel = np.where((W.sites[:, 0] > x0 - m) & (W.sites[:, 0] < x1 + m) & (W.sites[:, 1] > y0 - m) & (W.sites[:, 1] < y1 + m))[0]
keep = np.zeros(len(W.sites), bool); keep[sel] = True
adj = W.adj.multiply(keep[:, None]).multiply(keep[None, :]).tocsr()
sg = W.sg.multiply(keep[:, None]).tocsr()
cov = W.cov.multiply(keep[:, None]).tocsr()
S = list(sel); covr = cov[S].tocsc()
a = sp.triu(adj[S][:, S]).tocoo()
rows = []
for seed, K, j, w, gw_alive, budget in inst:
    gw_ok = np.asarray(sg[:, gw_alive].sum(axis=1)).ravel() > 0
    def f(X):
        c = np.zeros(cov.shape[1], bool)
        for s in X: c |= np.asarray(cov[s].todense()).ravel() > 0
        return float((w * c).sum())
    t = time.time(); Xg = orig(W, w, gw_alive, budget, np.ones(len(W.sites)), cov, adj, sg); tg = time.time() - t
    fg = f(Xg)
    cells = np.where((np.asarray(covr.sum(axis=0)).ravel() > 0) & (w > 0))[0]
    prob = pulp.LpProblem("cbmc", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(S))]
    y = {c: pulp.LpVariable(f"y{c}", 0, 1) for c in cells}
    prob += pulp.lpSum(w[c] * y[c] for c in cells)
    for c in cells: prob += y[c] <= pulp.lpSum(x[i] for i in covr[:, c].indices)
    prob += pulp.lpSum(x) <= budget
    fl = {}
    for i, jj in zip(a.row, a.col):
        fl[(i, jj)] = pulp.LpVariable(f"f{i}_{jj}", 0); fl[(jj, i)] = pulp.LpVariable(f"f{jj}_{i}", 0)
    root = [i for i, s in enumerate(S) if gw_ok[s]]
    r = {i: pulp.LpVariable(f"r{i}", 0) for i in root}
    inflow = {i: [] for i in range(len(S))}; outflow = {i: [] for i in range(len(S))}
    for (i, jj), v in fl.items():
        prob += v <= budget * x[jj]; prob += v <= budget * x[i]; outflow[i].append(v); inflow[jj].append(v)
    for i in r: prob += r[i] <= budget * x[i]
    for i in range(len(S)):
        prob += pulp.lpSum(inflow[i]) + (r[i] if i in r else 0) - pulp.lpSum(outflow[i]) == x[i]
    t = time.time(); st = prob.solve(pulp.HiGHS(msg=False, timeLimit=300, mip_rel_gap=1e-4)); tm = time.time() - t
    Xo = [S[i] for i in range(len(S)) if (x[i].value() or 0) > 0.5]
    fo = f(Xo)
    rows.append(dict(scenario=scen, seed=seed, K=budget, epoch=j, n_sites=len(S), n_edges=len(fl) // 2, n_pos_cells=len(cells),
                     f_greedy=fg, f_milp=fo, ratio=fg / max(fo, 1e-12), status=pulp.LpStatus[st], t_greedy_s=tg, t_milp_s=tm))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"gap2_{scen}.csv", index=False)
