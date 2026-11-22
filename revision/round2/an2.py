import pandas as pd, numpy as np
exec(open('/home/claude/fvv/rev/stats.py').read().split("m = pd.read_csv")[0])
pd.set_option("display.width", 250)
e1 = pd.read_csv("rev/E1/runs.csv")
t1 = table(e1, ["scenario","K"], "stats_E1.csv")
for b in t1.baseline.unique():
    x=t1[t1.baseline==b]; print(b, round(x.gain_pct.min(),1), round(x.gain_pct.max(),1), "sig", (x.p_holm<0.05).sum(), "/", x.p_holm.notna().sum(), "wins min", x.wins.min())
print(t1[t1.baseline=="ConnCov-dynamic"][["scenario","K","gain_pct","ci_lo","ci_hi","p_holm","r_rb"]].round(3).to_string())
print(t1[t1.baseline=="Proposed-noTerrain"][["scenario","K","gain_pct","ties","p_holm"]].round(3).to_string())
for e in ["E2","E3","E4","E5"]:
    d = pd.read_csv(f"rev/{e}/runs.csv")
    t = table(d, ["config","scenario"], f"stats_{e}.csv")
    m = d[d.method=="Proposed"].groupby(["config","scenario"]).ELS_norm.mean().unstack()
    print("\n==", e); print(m.round(3).to_string())
    print(t.pivot_table(index=["config","scenario"], columns="baseline", values="gain_pct").round(1).to_string())
    sig = t.pivot_table(index=["config","scenario"], columns="baseline", values="p_holm")
    print((sig<0.05).astype(int).to_string())
