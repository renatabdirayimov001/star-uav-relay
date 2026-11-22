import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata
def holm(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j]); adj[j] = min(1, run)
    return adj
def rbc(a, b):
    d = a - b; d = d[d != 0]
    if len(d) == 0: return np.nan
    r = rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()
def boot(a, b, B=10000, rng=np.random.default_rng(1)):
    n = len(a); idx = rng.integers(0, n, (B, n))
    g = 100 * (a[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1)
    return np.percentile(g, [2.5, 97.5])
def table(df, group, label):
    rows = []
    for key, d in df.groupby(group):
        a = d[d.method == "Proposed"].sort_values("seed").ELS_norm.values
        for m in d.method.unique():
            if m == "Proposed": continue
            b = d[d.method == m].sort_values("seed").ELS_norm.values
            diff = a - b
            p = wilcoxon(a, b).pvalue if np.any(diff != 0) else np.nan
            lo, hi = boot(a, b)
            rows.append(dict(zip(group, key if isinstance(key, tuple) else (key,)), baseline=m, n=len(a),
                             gain_pct=100 * (a.mean() - b.mean()) / b.mean(), ci_lo=lo, ci_hi=hi,
                             mean_diff=diff.mean(), wins=(diff > 0).sum(), ties=(diff == 0).sum(), losses=(diff < 0).sum(),
                             p=p, r_rb=rbc(a, b)))
    t = pd.DataFrame(rows); ok = t.p.notna()
    t.loc[ok, "p_holm"] = holm(t.p[ok].values)
    t.to_csv(label, index=False); return t
m = pd.read_csv("/home/claude/fvv/res2/results/runs.csv")
t1 = table(m, ["scenario", "K"], "stats_main.csv")
s = pd.read_csv("/home/claude/fvv/sens/all_runs.csv")
t2 = table(s, ["config", "scenario"], "stats_sens.csv")
pd.set_option("display.width", 250)
print(t1.round(4).to_string()); print(t2.round(4).to_string())
# per-seed CI of normalised ELS mean (t-based) for Table 5
from scipy.stats import t as tdist
g = m[m.K == 10].groupby(["scenario", "method"]).ELS_norm
ci = (g.std() / np.sqrt(g.count()) * tdist.ppf(0.975, 29)).rename("ci95")
pd.concat([g.mean().rename("mean"), g.std().rename("sd"), ci], axis=1).round(4).to_csv("table5_ci.csv"); print(pd.concat([g.mean(), ci], axis=1).round(4))
