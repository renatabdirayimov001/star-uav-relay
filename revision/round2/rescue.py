import numpy as np, pandas as pd, glob, heapq
exec(open('/home/claude/fvv/rev/stats.py').read().split("m = pd.read_csv")[0])
CL = [(10.0, 1.0), (48.0, 1.5), (150.0, 2.0)]; T0 = 60.0; ONSITE = 30.0; TAU0 = 30.0
def S(u, c): lam, k = CL[c]; return np.exp(-(np.maximum(u, 0) / lam) ** k)
def els(deliv, z, M, policy):
    cls, tau, Tg, va = z["vcls"], z["vtau"], z["vTg"], z["va"]
    n = len(cls); arr = np.full(n, np.inf)
    ok = ~np.isnan(deliv)
    if M is None:
        arr[ok] = deliv[ok] + tau[ok]
    else:
        idx = np.where(ok)[0]; idx = idx[np.argsort(deliv[idx])]
        free = [0.0] * M; heapq.heapify(free)
        queue = []; ptr = 0; t = 0.0
        pending = list(idx)
        # event-driven dispatch
        waiting = []
        while ptr < len(pending) or waiting:
            tf = free[0]
            # add all deliveries up to the time a team is free
            tnow = max(tf, deliv[pending[ptr]] if (not waiting and ptr < len(pending)) else tf)
            while ptr < len(pending) and deliv[pending[ptr]] <= tnow:
                i = pending[ptr]; ptr += 1
                key = (cls[i], deliv[i]) if policy == "priority" else (deliv[i],)
                heapq.heappush(waiting, (key, i))
            if not waiting: continue
            _, i = heapq.heappop(waiting); tf = heapq.heappop(free)
            start = max(tf, deliv[i]); a = start + tau[i]
            arr[i] = a
            heapq.heappush(free, a + ONSITE + (tau[i] - TAU0))
    g = np.zeros(n)
    for c in range(3):
        m = cls == c
        gr = S((Tg[m] + tau[m]) / 60, c)
        net = np.where(arr[m] < Tg[m] + tau[m], S(arr[m] / 60, c), gr)
        g[m] = np.maximum(net - gr, 0)
    ideal = np.zeros(n)
    for c in range(3):
        m = cls == c
        ideal[m] = np.maximum(S((np.maximum(va[m], T0) + tau[m]) / 60, c) - S((Tg[m] + tau[m]) / 60, c), 0)
    return g.sum() / ideal.sum()
rows = []
for f in sorted(glob.glob("rev/E1/deliv/*.npz")):
    z = np.load(f); scen, s = f.split("/")[-1][:-4].split("_s"); s = int(s)
    for m in ["Proposed", "ConnCov-dynamic", "ConnCov-static", "Ferry"]:
        d = z[f"K10_{m}"]
        for M in [None, 20, 10, 5]:
            for pol in (["fifo"] if M is None else ["fifo", "priority"]):
                rows.append(dict(scenario=scen, seed=s, method=m, M=M or 0, policy=pol, ELS_norm=els(d, z, M, pol)))
r = pd.DataFrame(rows); r.to_csv("rescue_runs.csv", index=False)
print(r.groupby(["scenario", "M", "policy", "method"]).ELS_norm.mean().unstack().round(3).to_string())
r["config"] = r.M.astype(str) + "_" + r.policy
t = table(r, ["config", "scenario"], "stats_rescue.csv")
print(t.pivot_table(index=["config", "scenario"], columns="baseline", values="gain_pct").round(1).to_string())
print(t.pivot_table(index=["config", "scenario"], columns="baseline", values="p_holm").round(3).to_string())
