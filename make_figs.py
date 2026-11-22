import sys, os, math, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import fvv_sim as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

OUT = "figures_out"; os.makedirs(OUT, exist_ok=True)
RES = "results/main"; SENS = "results/sensitivity"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False})
COL = {"Proposed": "#2a78d6", "Ferry": "#eb6834", "ConnCov-static": "#1baf7a", "KMeans": "#eda100",
       "Proposed-static": "#e87ba4", "Proposed-noMemory": "#008300", "ConnCov-dynamic": "#2a78d6"}
MK = {"Proposed": "o", "Ferry": "s", "ConnCov-static": "^", "KMeans": "v", "Proposed-static": "D",
      "Proposed-noMemory": "P", "ConnCov-dynamic": "x"}
LAB = {"Proposed": "Proposed (STAR)", "Ferry": "Ferry (SCF)", "ConnCov-static": "Connected coverage (static)",
       "KMeans": "Weighted k-means", "Proposed-static": "STAR, single plan", "Proposed-noMemory": "STAR w/o memory",
       "ConnCov-dynamic": "STAR, count weights"}
SCN = {"sardoba": "(a) Sardoba dam-breach flood", "fergana": "(b) Fergana valley earthquake",
       "karakalpak": "(c) Amu Darya delta flood"}


class A: pass
args = A(); args.out = "run_output"; args.cache = "cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"opencellid/{m}.csv.gz" for m in (434, 401, 437)]


def worlds():
    return {k: F.build_world(k, args, np.random.default_rng(0)) for k in F.SCENARIOS}


def ticks_ll(ax, W):
    xt = np.linspace(0, W.g.W / 1e3, 4)[:-1] + W.g.W / 1e3 / 6
    yt = np.linspace(0, W.g.Hh / 1e3, 5)[1:-1]
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{W.g.lon0 + x * 1e3 / W.g.mx:.2f}" for x in xt])
    ax.set_yticklabels([f"{W.g.lat0 + y * 1e3 / W.g.my:.2f}" for y in yt])
    ax.set_xlim(0, W.g.W / 1e3); ax.set_ylim(0, W.g.Hh / 1e3)


def base_map(ax, W, pop=True):
    ext = [0, W.g.W / 1e3, 0, W.g.Hh / 1e3]
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.hillshade(W.dem, vert_exag=3, dx=F.P["dem_res"], dy=F.P["dem_res"]), cmap="gray",
              origin="lower", extent=ext, alpha=0.9, vmin=0.2, vmax=1.0)
    im = None
    if pop:
        p = np.ma.masked_less(np.log10(W.pop + 1), 0.3)
        im = ax.imshow(p, origin="lower", extent=ext, cmap="Oranges", alpha=0.55, vmin=0, vmax=3)
    z = W.sc["zone"]; th = np.linspace(0, 2 * np.pi, 200)
    cx, cy = W.g.xy(z["lat"], z["lon"]); r = math.radians(z["rot_deg"])
    u, v = z["b_km"] * np.cos(th), z["a_km"] * np.sin(th)
    ax.plot(cx / 1e3 + u * math.cos(r) + v * math.sin(r), cy / 1e3 - u * math.sin(r) + v * math.cos(r),
            "--", color="#1c5cab", lw=1.0)
    ax.grid(False)
    return im


def fig_maps(Ws):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.38))
    for ax, (k, W) in zip(axs, Ws.items()):
        im = base_map(ax, W)
        ax.scatter(W.bs[:, 0] / 1e3, W.bs[:, 1] / 1e3, marker="^", s=7, c="k", lw=0, label="Base station")
        ax.scatter(*W.base / 1e3, marker="*", s=90, c="#1baf7a", edgecolor="k", lw=0.5, label="Rescue base / gateway")
        ax.set_title(SCN[k], fontsize=8); ticks_ll(ax, W)
        ax.set_xlabel("Longitude [°E]")
        if k == "sardoba":
            ax.set_ylabel("Latitude [°N]")
    axs[0].plot([], [], "--", color="#1c5cab", lw=1.0, label="Affected zone")
    axs[0].legend(loc="upper left", fontsize=6, frameon=True, framealpha=0.85)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01); cb.set_label("log$_{10}$(people per cell + 1)", fontsize=7)
    fig.savefig(f"{OUT}/fig2_study_areas.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig2_study_areas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_net(ax, W, R, snap, title):
    base_map(ax, W, pop=False)
    d = snap["delivered"]
    ax.scatter(R.vx[~d] / 1e3, R.vy[~d] / 1e3, s=3, c="#9a9a9a", lw=0, label="SOS pending")
    ax.scatter(R.vx[d] / 1e3, R.vy[d] / 1e3, s=4, c="#d6282b", lw=0, label="SOS delivered")
    con = set(snap["con"].tolist()); hov = snap["hover"]
    gw_alive = W.gw[R.gw_alive]
    for i, a in enumerate(hov):
        for b in hov[i + 1:]:
            if W.adj[a, b] and a in con and b in con:
                ax.plot(W.sites[[a, b], 0] / 1e3, W.sites[[a, b], 1] / 1e3, "-", color="#2a78d6", lw=0.8)
        if a in con:
            gidx = np.where(np.asarray(W.sg[a].todense()).ravel() & R.gw_alive)[0]
            if len(gidx):
                gg = W.gw[gidx[np.argmin(np.hypot(*(W.gw[gidx, :2] - W.sites[a, :2]).T))]]
                ax.plot([W.sites[a, 0] / 1e3, gg[0] / 1e3], [W.sites[a, 1] / 1e3, gg[1] / 1e3], ":", color="#2a78d6", lw=0.7)
    c = np.array([a in con for a in hov], bool)
    if len(hov):
        ax.scatter(W.sites[hov[c], 0] / 1e3, W.sites[hov[c], 1] / 1e3, marker="o", s=22, facecolor="#2a78d6", edgecolor="w", lw=0.6, zorder=5, label="UAV (connected)")
        ax.scatter(W.sites[hov[~c], 0] / 1e3, W.sites[hov[~c], 1] / 1e3, marker="o", s=22, facecolor="w", edgecolor="#2a78d6", lw=0.9, zorder=5, label="UAV (isolated)")
    ax.scatter(*W.base / 1e3, marker="*", s=70, c="#1baf7a", edgecolor="k", lw=0.5, zorder=6)
    ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, W.g.W / 1e3); ax.set_ylim(0, W.g.Hh / 1e3)


def fig_snapshots(Ws):
    T0 = F.P["T0_min"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for row, key in enumerate(["sardoba", "fergana"]):
        W = Ws[key]; F.P["_pen"] = W.sc["penetration_db"]
        R = F.realise(W, np.random.default_rng(1000))
        ta, tb = int(T0 + 67), int(T0 + 247)          # mid-epoch (re-planning occurs every 15 min)
        s1 = {ta: None, tb: None}
        F.simulate(W, R, 10, "Proposed", np.random.default_rng(0), snaps=s1)
        s2 = {tb: None}
        F.simulate(W, R, 10, "ConnCov-static", np.random.default_rng(0), snaps=s2)
        L = "abc" if row == 0 else "def"
        draw_net(axs[row, 0], W, R, s1[ta], f"({L[0]}) STAR, t = {ta//60} h {ta%60} min")
        draw_net(axs[row, 1], W, R, s1[tb], f"({L[1]}) STAR, t = {tb//60} h {tb%60} min")
        draw_net(axs[row, 2], W, R, s2[tb], f"({L[2]}) Static coverage, t = {tb//60} h {tb%60} min")
        axs[row, 0].set_ylabel(["Sardoba flood", "Fergana earthquake"][row], fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/fig4_snapshots.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig4_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_els_vs_k():
    d = pd.read_csv(f"{RES}/runs.csv")
    meths = ["Proposed", "Proposed-noMemory", "Ferry", "Proposed-static", "ConnCov-static", "KMeans"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, k in zip(axs, SCN):
        for m in meths:
            g = d[(d.scenario == k) & (d.method == m)].groupby("K").ELS_norm
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker=MK[m], ms=4, lw=1.2,
                        capsize=2, color=COL[m], label=LAB[m])
        ax.set_title(SCN[k], fontsize=8); ax.set_xticks([6, 10, 14]); ax.set_xlabel("Number of UAVs $K$")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("Normalised ELS")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/fig5_els_vs_K.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig5_els_vs_K.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_time_cdf(Ws, seeds=6):
    meths = ["Proposed", "Proposed-noMemory", "Ferry", "ConnCov-static", "KMeans"]
    T0, H = F.P["T0_min"], F.P["H_min"]
    tg = np.arange(0, T0 + H + 1)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2))
    rows = []
    for j, (k, W) in enumerate(Ws.items()):
        F.P["_pen"] = W.sc["penetration_db"]
        curves = {m: [] for m in meths}; delays = {m: [] for m in meths}
        for s in range(seeds):
            R = F.realise(W, np.random.default_rng(1000 + s))
            ideal = F.victim_gain(R, np.maximum(R.va, T0)).sum()
            for m in meths:
                _, dl, _ = F.simulate(W, R, 10, m, np.random.default_rng(s))
                ok = ~np.isnan(dl)
                g = F.victim_gain(R, np.where(ok, dl, 0)); g[~ok] = 0
                curves[m].append([g[ok & (dl <= t)].sum() / ideal for t in tg])
                crit = R.vcls == 0
                dd = np.where(ok, dl - np.maximum(R.va, T0), np.inf)[crit]
                delays[m].append(dd)
        for m in meths:
            c = np.mean(curves[m], axis=0)
            axs[0, j].plot(tg / 60, c, color=COL[m], lw=1.3, label=LAB[m], marker=MK[m], markevery=60, ms=3.5)
            dd = np.sort(np.concatenate(delays[m]))
            x = np.linspace(0, 360, 361)
            axs[1, j].plot(x / 60, [(dd <= v).mean() for v in x], color=COL[m], lw=1.3, marker=MK[m], markevery=60, ms=3.5)
            rows.append(dict(scenario=k, method=m, crit_within_1h=float((dd <= 60).mean()), crit_within_3h=float((dd <= 180).mean())))
        axs[0, j].axvline(T0 / 60, color="k", lw=0.6, ls=":")
        axs[0, j].set_title(SCN[k], fontsize=8); axs[0, j].set_xlabel("Time since onset [h]"); axs[0, j].set_ylim(0, 1)
        axs[1, j].set_xlabel("Delivery delay of critical SOS [h]"); axs[1, j].set_ylim(0, 1)
    axs[0, 0].set_ylabel("Cumulative normalised ELS"); axs[1, 0].set_ylabel("Empirical CDF")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/fig6_time_cdf.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig6_time_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{OUT}/fig6_numbers.csv", index=False)


def fig_sens():
    d = pd.read_csv(f"{SENS}/all_runs.csv")
    alt = {"h60": 60, "h100": 100, "base": 150}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, k in zip(axs, SCN):
        for m in ["Proposed", "Ferry", "ConnCov-static"]:
            x, y, e = [], [], []
            for c, h in alt.items():
                v = d[(d.config == c) & (d.scenario == k) & (d.method == m)].ELS_norm
                x.append(h); y.append(v.mean()); e.append(v.std())
            ax.errorbar(x, y, yerr=e, marker=MK[m], ms=4, lw=1.2, capsize=2, color=COL[m], label=LAB[m])
        ax.set_title(SCN[k], fontsize=8); ax.set_xticks([60, 100, 150]); ax.set_xlabel("UAV altitude [m AGL]"); ax.set_ylim(0, 1)
    axs[0].set_ylabel("Normalised ELS")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/fig7_altitude.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig7_altitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["maps", "snap", "k", "time", "sens"]
    if "k" in which: fig_els_vs_k()
    if "sens" in which: fig_sens()
    if set(which) & {"maps", "snap", "time"}:
        Ws = worlds()
        if "maps" in which: fig_maps(Ws)
        if "snap" in which: fig_snapshots(Ws)
        if "time" in which: fig_time_cdf(Ws)
    print("done")
