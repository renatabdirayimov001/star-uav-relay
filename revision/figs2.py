import os
import sys, os, math, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fvv_sim as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrow
from matplotlib.lines import Line2D
OUT = "/home/claude/fvv/paper/figs"
REV = "/home/claude/fvv/rev"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
                     "pdf.fonttype": 42})
COL = {"Proposed": "#1f5fbf", "Ferry": "#e0612a", "ConnCov-static": "#139a6b", "KMeans": "#c99000",
       "Proposed-static": "#d0609a", "Proposed-noMemory": "#6b3fa0", "ConnCov-dynamic": "#5a9bd8"}
MK = {"Proposed": "o", "Ferry": "s", "ConnCov-static": "^", "KMeans": "v", "Proposed-static": "D",
      "Proposed-noMemory": "P", "ConnCov-dynamic": "x"}
LAB = {"Proposed": "STAR (proposed)", "Ferry": "Ferry (SCF)", "ConnCov-static": "Connected coverage (static)",
       "KMeans": "Weighted k-means", "Proposed-static": "STAR, single plan", "Proposed-noMemory": "STAR w/o served-time memory",
       "ConnCov-dynamic": "Dynamic connected coverage"}
SCN = {"sardoba": "(a) Sardoba dam-breach flood", "fergana": "(b) Fergana valley earthquake",
       "karakalpak": "(c) Amu Darya delta flood"}
SH = {"sardoba": "Sardoba", "fergana": "Fergana", "karakalpak": "Amu Darya"}
class A: pass
args = A(); args.out = "/home/claude/fvv/res"; args.cache = "/home/claude/fvv/res/cache"; args.synthetic = False
args.pop_tif = None; args.cells = [f"/home/claude/fvv/cells/{m}.csv.gz" for m in (434, 401, 437)]

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

def worlds():
    return {k: F.build_world(k, args, np.random.default_rng(0)) for k in F.SCENARIOS}

def ell(W):
    z = W.sc["zone"]; th = np.linspace(0, 2 * np.pi, 200)
    cx, cy = W.g.xy(z["lat"], z["lon"]); r = math.radians(z["rot_deg"])
    u, v = z["b_km"] * np.cos(th), z["a_km"] * np.sin(th)
    return cx / 1e3 + u * math.cos(r) + v * math.sin(r), cy / 1e3 - u * math.sin(r) + v * math.cos(r)

def ticks_ll(ax, W, left=True):
    xt = np.linspace(0, W.g.W / 1e3, 4)[:-1] + W.g.W / 1e3 / 6
    yt = np.linspace(0, W.g.Hh / 1e3, 5)[1:-1]
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{W.g.lon0 + x * 1e3 / W.g.mx:.2f}°E" for x in xt], fontsize=6.5)
    ax.set_yticklabels([f"{W.g.lat0 + y * 1e3 / W.g.my:.2f}°N" for y in yt] if left else [], fontsize=6.5)
    ax.set_xlim(0, W.g.W / 1e3); ax.set_ylim(0, W.g.Hh / 1e3); ax.grid(False)

def scalebar(ax, W, km=10, color="k"):
    x0, y0 = 0.05 * W.g.W / 1e3, 0.06 * W.g.Hh / 1e3
    ax.add_patch(Rectangle((x0, y0), km / 2, 0.9, fc=color, ec=color, lw=0.5, zorder=9))
    ax.add_patch(Rectangle((x0 + km / 2, y0), km / 2, 0.9, fc="white", ec=color, lw=0.5, zorder=9))
    ax.text(x0 + km / 2, y0 + 1.6, f"{km} km", ha="center", va="bottom", fontsize=6.5, zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))

def north(ax, W):
    x, y = 0.9 * W.g.W / 1e3, 0.82 * W.g.Hh / 1e3
    ax.annotate("", xy=(x, y + 5), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="k"), zorder=9)
    ax.text(x, y + 5.6, "N", ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=9)

def load_geo(fn):
    return json.load(open(fn))["features"]

def draw_poly(ax, geom, **kw):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for p in polys:
        ax.add_patch(Polygon(np.array(p[0]), closed=True, **kw))

def fig_maps(Ws):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1, 1], wspace=0.42, hspace=0.36)
    axl = fig.add_subplot(gs[0, 0:2])
    cs = load_geo(f"{REV}/ne/countries50.geojson")
    for f in cs:
        n = f["properties"]["ADMIN"]
        draw_poly(axl, f["geometry"], fc="#f3e7c8" if n == "Uzbekistan" else "#ececec", ec="#777", lw=0.4, zorder=1)
    for f in load_geo(f"{REV}/ne/ne_50m_lakes.geojson"):
        draw_poly(axl, f["geometry"], fc="#bcd7ee", ec="#7aa6cc", lw=0.3, zorder=2)
    lab = {"UZBEKISTAN": (63.6, 41.2), "KAZAKHSTAN": (67.0, 45.4), "TURKMENISTAN": (58.8, 39.0),
           "KYRGYZSTAN": (75.2, 42.3), "TAJIKISTAN": (71.3, 38.5), "Aral Sea": (59.9, 45.2)}
    for n, (x, y) in lab.items():
        axl.text(x, y, n, fontsize=6, ha="center", style="italic", color="#2b5d8a" if n == "Aral Sea" else "#555", zorder=4)
    for k, L in zip(Ws, "abc"):
        b = F.SCENARIOS[k]["bbox"]
        axl.add_patch(Rectangle((b[2], b[0]), b[3] - b[2], b[1] - b[0], fc="#c0392b", alpha=0.25, ec="#c0392b", lw=1.2, zorder=5))
        ox, oy = {"a": (-1.25, -0.55), "b": (0.25, 0.1), "c": (-1.3, 0.05)}[L]
        axl.text(b[2] + ox, b[0] + oy, f"({L})", color="#c0392b", fontsize=8, fontweight="bold", zorder=5)
    axl.plot(69.24, 41.30, "ks", ms=3, zorder=6); axl.text(69.0, 41.95, "Tashkent", fontsize=6, zorder=6, ha="right")
    axl.set_xlim(55.0, 77.0); axl.set_ylim(37.0, 46.5); axl.set_aspect(1 / math.cos(math.radians(41.8)))
    axl.set_xticks([56, 60, 64, 68, 72, 76]); axl.set_yticks([38, 40, 42, 44, 46])
    axl.set_xticklabels([f"{v}°E" for v in [56, 60, 64, 68, 72, 76]], fontsize=6.5)
    axl.set_yticklabels([f"{v}°N" for v in [38, 40, 42, 44, 46]], fontsize=6.5)
    axl.grid(alpha=0.3); axl.set_facecolor("#e3eef7")
    axl.set_title("Location of the three study areas", fontsize=8)
    # legend panel
    axg = fig.add_subplot(gs[0, 2]); axg.axis("off")
    h = [Line2D([], [], ls="--", color="#0b3d91", lw=1.2, label="Affected zone (ellipse)"),
         Line2D([], [], marker="^", ls="", color="k", ms=4.5, label="Base station"),
         Line2D([], [], marker="*", ls="", mfc="#ffd400", mec="k", ms=9, label="Rescue base /\nsatellite gateway")]
    axg.legend(handles=h, loc="upper left", fontsize=7, frameon=True, framealpha=1, edgecolor="#bbb", borderpad=0.8, labelspacing=1.0)
    bounds = [0, 10, 50, 100, 250, 500, 1000, 1e7]
    pcm = ListedColormap(["#f7f4ee", "#fde7b8", "#fcc983", "#f99d58", "#e6612f", "#bd3020", "#7a1410"])
    pnorm = BoundaryNorm(bounds, pcm.N)
    cax = axg.inset_axes([0.05, 0.30, 0.9, 0.08])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pnorm, cmap=pcm), cax=cax, orientation="horizontal",
                      ticks=bounds[:-1] + [1e7])
    cb.ax.set_xticklabels(["0", "10", "50", "100", "250", "500", "1000", ""], fontsize=5.8)
    cb.set_label("Population density [people km$^{-2}$]", fontsize=6.5); cb.outline.set_linewidth(0.4)
    for j, (k, W) in enumerate(Ws.items()):
        ext = [0, W.g.W / 1e3, 0, W.g.Hh / 1e3]
        ax = fig.add_subplot(gs[1, j])
        ls = LightSource(azdeg=315, altdeg=45)
        lo, hi = np.percentile(W.dem, 1), np.percentile(W.dem, 99.5)
        cm2 = LinearSegmentedColormap.from_list("t2", plt.get_cmap("terrain")(np.linspace(0.25, 0.95, 256)))
        rgb = ls.shade(W.dem, cmap=cm2, norm=Normalize(lo, hi), vert_exag=3 if k != "fergana" else 1.5,
                       dx=F.P["dem_res"], dy=F.P["dem_res"], blend_mode="soft")
        ax.imshow(rgb, origin="lower", extent=ext)
        ex, ey = ell(W); ax.plot(ex, ey, "--", color="#0b3d91", lw=1.2)
        ax.scatter(W.bs[:, 0] / 1e3, W.bs[:, 1] / 1e3, marker="^", s=8, c="k", lw=0, zorder=5)
        ax.scatter(*W.base / 1e3, marker="*", s=100, c="#ffd400", edgecolor="k", lw=0.6, zorder=6)
        src = "OpenCelliD base stations" if W.bs_src.startswith("OpenCelliD") else "synthetic base-station layout"
        ax.set_title(SCN[k] + "\n" + src, fontsize=7.3); ticks_ll(ax, W); scalebar(ax, W); north(ax, W)
        cax = ax.inset_axes([0.05, 0.86, 0.42, 0.045])
        sm = plt.cm.ScalarMappable(cmap=cm2, norm=Normalize(lo, hi))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_ticks([lo, hi])
        cb.set_ticklabels([f"{lo:.0f}", f"{hi:.0f} m"]); cb.ax.tick_params(labelsize=5.8, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)
        cax.text(0.5, -2.6, "elevation", transform=cax.transAxes, ha="center", fontsize=5.8,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
        ax2 = fig.add_subplot(gs[2, j])
        dens = W.pop / (F.P["cell_res"] ** 2 / 1e6)
        ax2.imshow(np.maximum(dens, 0), origin="lower", extent=ext, cmap=pcm, norm=pnorm, interpolation="nearest")
        ax2.plot(ex, ey, "--", color="#0b3d91", lw=1.2)
        ax2.scatter(*W.base / 1e3, marker="*", s=100, c="#ffd400", edgecolor="k", lw=0.6, zorder=6)
        ticks_ll(ax2, W); scalebar(ax2, W)
        ax2.set_title(f"population in zone: {W.cpop.sum():,.0f}", fontsize=7)
    fig.text(0.04, 0.505, "Terrain (hill-shaded DEM)", rotation=90, ha="center", va="center", fontsize=8)
    fig.text(0.04, 0.215, "Population (WorldPop 2020)", rotation=90, ha="center", va="center", fontsize=8)
    save(fig, "fig2_study_areas")

def fig_els_vs_k():
    d = pd.read_csv("/home/claude/fvv/res2/results/runs.csv")
    meths = ["Proposed", "Proposed-noMemory", "Ferry", "Proposed-static", "ConnCov-static", "KMeans"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    off = dict(zip(meths, np.linspace(-0.5, 0.5, len(meths))))
    for ax, k in zip(axs, SCN):
        for m in meths:
            g = d[(d.scenario == k) & (d.method == m)].groupby("K").ELS_norm
            ax.errorbar(g.mean().index + off[m], g.mean().values, yerr=g.std().values, marker=MK[m], ms=4.5, lw=1.3,
                        capsize=2.5, elinewidth=0.9, color=COL[m], label=LAB[m], mfc="white" if m != "Proposed" else COL[m])
        ax.set_title(SCN[k], fontsize=8.5); ax.set_xticks([6, 10, 14]); ax.set_xlabel("Number of UAVs $K$")
        ax.set_ylim(0, 1); ax.set_xlim(4.8, 15.2)
    axs[0].set_ylabel("Normalised ELS")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_els_vs_K")

def fig_sens():
    d = pd.read_csv("/home/claude/fvv/sens/all_runs.csv")
    alt = {"h60": 60, "h100": 100, "base": 150}
    rad = {60: 0.68, 100: 0.88, 150: 1.13}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, k in zip(axs, SCN):
        for m in ["Proposed", "Ferry", "ConnCov-static"]:
            x, y, e = [], [], []
            for c, h in alt.items():
                v = d[(d.config == c) & (d.scenario == k) & (d.method == m)].ELS_norm
                x.append(h); y.append(v.mean()); e.append(v.std())
            ax.errorbar(x, y, yerr=e, marker=MK[m], ms=4.5, lw=1.3, capsize=2.5, color=COL[m], label=LAB[m],
                        mfc="white" if m != "Proposed" else COL[m])
        ax.set_title(SCN[k], fontsize=8.5); ax.set_xticks([60, 100, 150]); ax.set_xlabel("UAV altitude $h$ [m above ground]"); ax.set_ylim(0, 1)
        sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
        sec.set_xticks([60, 100, 150]); sec.set_xticklabels([f"{rad[h]:.2f} km" for h in (60, 100, 150)], fontsize=6.3)
        sec.tick_params(length=2, pad=1)
    axs[0].set_ylabel("Normalised ELS")
    fig.text(0.5, 0.93, "effective access radius on flat ground (10 dB penetration loss)", ha="center", fontsize=6.5, color="#444")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, "fig6_altitude")

def draw_net(ax, W, R, snap, title, rad_km):
    ext = [0, W.g.W / 1e3, 0, W.g.Hh / 1e3]
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.hillshade(W.dem, vert_exag=3, dx=F.P["dem_res"], dy=F.P["dem_res"]), cmap="gray",
              origin="lower", extent=ext, alpha=0.55, vmin=0.0, vmax=1.15)
    ex, ey = ell(W); ax.plot(ex, ey, "--", color="#0b3d91", lw=0.9)
    d = snap["delivered"]
    con = set(snap["con"].tolist()); hov = snap["hover"]
    for a in hov:
        if a in con:
            ax.add_patch(Circle((W.sites[a, 0] / 1e3, W.sites[a, 1] / 1e3), rad_km, fc="#1f5fbf", ec="#0b3d91", alpha=0.30, lw=0.6, zorder=2))
    alive = W.gw[:-1][R.gw_alive[:-1]]
    ax.scatter(alive[:, 0] / 1e3, alive[:, 1] / 1e3, marker="^", s=10, c="k", lw=0, zorder=4.5)
    ax.scatter(R.vx[~d] / 1e3, R.vy[~d] / 1e3, s=6, marker="x", c="#666666", lw=0.5, zorder=4)
    ax.scatter(R.vx[d] / 1e3, R.vy[d] / 1e3, s=6, c="#d6282b", lw=0, zorder=4)
    # draw one uplink per connected UAV: direct to the nearest live gateway if feasible, else to the nearest connected UAV
    conl = [a for a in hov if a in con]
    for a in conl:
        gidx = np.where(np.asarray(W.sg[a].todense()).ravel() & R.gw_alive)[0]
        if len(gidx):
            gg = W.gw[gidx[np.argmin(np.hypot(*(W.gw[gidx, :2] - W.sites[a, :2]).T))]]
            ax.plot([W.sites[a, 0] / 1e3, gg[0] / 1e3], [W.sites[a, 1] / 1e3, gg[1] / 1e3], ":", color="#0e8a5f", lw=1.3, zorder=5)
        else:
            nb = [b for b in conl if b != a and W.adj[a, b]]
            if nb:
                b = nb[int(np.argmin([np.hypot(*(W.sites[a, :2] - W.sites[q, :2])) for q in nb]))]
                ax.plot(W.sites[[a, b], 0] / 1e3, W.sites[[a, b], 1] / 1e3, "-", color="#1f5fbf", lw=1.1, zorder=5)
    c = np.array([a in con for a in hov], bool)
    if len(hov):
        ax.scatter(W.sites[hov[c], 0] / 1e3, W.sites[hov[c], 1] / 1e3, marker="o", s=13, facecolor="#1f5fbf", edgecolor="w", lw=0.5, zorder=6)
        ax.scatter(W.sites[hov[~c], 0] / 1e3, W.sites[hov[~c], 1] / 1e3, marker="o", s=13, facecolor="w", edgecolor="#1f5fbf", lw=0.9, zorder=6)
    ax.scatter(*W.base / 1e3, marker="*", s=90, c="#ffd400", edgecolor="k", lw=0.6, zorder=7)
    n_d = int(d.sum()); ax.text(0.03, 0.97, f"{n_d}/{len(d)} delivered", transform=ax.transAxes, fontsize=6.3, va="top",
                                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    ax.set_title(title, fontsize=7.8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    # zoom to zone + base
    xs = np.r_[ex, W.base[0] / 1e3]; ys = np.r_[ey, W.base[1] / 1e3]
    pad = 4
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    w = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    xb, yb = cx - w / 2 + 0.06 * w, cy - w / 2 + 0.05 * w
    ax.add_patch(Rectangle((xb, yb), 5, 0.012 * w, fc="k", zorder=9)); ax.add_patch(Rectangle((xb + 5, yb), 5, 0.012 * w, fc="w", ec="k", lw=0.5, zorder=9))
    ax.text(xb + 5, yb + 0.025 * w, "10 km", ha="center", fontsize=6, zorder=9, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

def fig_snapshots(Ws):
    T0 = F.P["T0_min"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4))
    for row, key in enumerate(["sardoba", "fergana"]):
        W = Ws[key]; F.P["_pen"] = W.sc["penetration_db"]
        rad = {"sardoba": 1.13, "fergana": 0.93}[key]
        R = F.realise(W, np.random.default_rng(1000))
        ta, tb = int(T0 + 67), int(T0 + 247)
        s1 = {ta: None, tb: None}; F.simulate(W, R, 10, "Proposed", np.random.default_rng(0), snaps=s1)
        s2 = {tb: None}; F.simulate(W, R, 10, "ConnCov-static", np.random.default_rng(0), snaps=s2)
        L = "abc" if row == 0 else "def"
        draw_net(axs[row, 0], W, R, s1[ta], f"({L[0]}) STAR, t = {ta//60} h {ta%60:02d} min", rad)
        draw_net(axs[row, 1], W, R, s1[tb], f"({L[1]}) STAR, t = {tb//60} h {tb%60:02d} min", rad)
        draw_net(axs[row, 2], W, R, s2[tb], f"({L[2]}) Static coverage, t = {tb//60} h {tb%60:02d} min", rad)
        axs[row, 0].set_ylabel(["Sardoba flood", "Fergana earthquake"][row], fontsize=8.5)
    h = [Line2D([], [], marker="x", ls="", color="#555555", ms=4, label="SOS pending"),
         Line2D([], [], marker="o", ls="", color="#d6282b", ms=3.5, label="SOS delivered"),
         Line2D([], [], marker="o", ls="", mfc="#1f5fbf", mec="w", ms=6, label="UAV (connected)"),
         Line2D([], [], marker="o", ls="", mfc="w", mec="#1f5fbf", ms=6, label="UAV (not connected)"),
         plt.Rectangle((0, 0), 1, 1, fc="#1f5fbf", alpha=0.15, label="Access footprint"),
         Line2D([], [], color="#1f5fbf", lw=1, label="UAV–UAV uplink"),
         Line2D([], [], color="#139a6b", lw=1, ls=":", label="UAV–gateway uplink"),
         Line2D([], [], marker="^", ls="", color="k", ms=4, label="Live base station"),
         Line2D([], [], marker="*", ls="", mfc="#ffd400", mec="k", ms=8, label="Rescue base"),
         Line2D([], [], ls="--", color="#0b3d91", label="Affected zone")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, "fig4_snapshots")

def fig_time(Ws):
    meths = ["Proposed", "Proposed-noMemory", "Ferry", "ConnCov-static", "KMeans"]
    T0, H = F.P["T0_min"], F.P["H_min"]
    tg = np.arange(0, T0 + H + 1)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    rows = []
    for j, (k, W) in enumerate(Ws.items()):
        F.P["_pen"] = W.sc["penetration_db"]
        d = f"{REV}/out_fig5_{k}"
        curves = {m: [] for m in meths}; delays = {m: [] for m in meths}; pre = []
        seeds = sorted({int(os.path.basename(p).split("_")[0][1:]) for p in glob.glob(f"{d}/s*_Proposed.npz")})
        for s in seeds:
            R = F.realise(W, np.random.default_rng(1000 + s))
            ideal = F.victim_gain(R, np.maximum(R.va, T0)).sum()
            for m in meths:
                z = np.load(f"{d}/s{s}_{m}.npz"); dl = z["deliv"]
                ok = ~np.isnan(dl)
                g = F.victim_gain(R, np.where(ok, dl, 0)); g[~ok] = 0
                order = np.argsort(np.where(ok, dl, np.inf)); cs = np.cumsum(g[order]); td = np.where(ok, dl, np.inf)[order]
                idx = np.searchsorted(td, tg, side="right")
                curves[m].append(np.where(idx > 0, cs[np.maximum(idx - 1, 0)], 0) / ideal)
                crit = R.vcls == 0
                delays[m].append(np.where(ok, dl - np.maximum(R.va, T0), np.inf)[crit])
        for m in meths:
            C = np.array(curves[m]); c = C.mean(0); lo, hi = np.percentile(C, [2.5, 97.5], axis=0)
            axs[0, j].plot(tg / 60, c, color=COL[m], lw=1.3, label=LAB[m], marker=MK[m], markevery=60, ms=3.8,
                           mfc="white" if m != "Proposed" else COL[m])
            axs[0, j].fill_between(tg / 60, lo, hi, color=COL[m], alpha=0.12, lw=0)
            dd = np.concatenate(delays[m]); x = np.linspace(0, 360, 361)
            axs[1, j].plot(x / 60, [(dd <= v).mean() for v in x], color=COL[m], lw=1.3, marker=MK[m], markevery=60, ms=3.8,
                           mfc="white" if m != "Proposed" else COL[m])
            rows.append(dict(scenario=k, method=m, n_seeds=len(seeds), crit_within_1h=float((dd <= 60).mean()),
                             crit_within_3h=float((dd <= 180).mean()), gain_at_T0=float(C[:, int(T0)].mean()),
                             gain_3h=float(C[:, 180].mean()), gain_2h=float(C[:, 120].mean()), gain_end=float(C[:, -1].mean())))
        axs[0, j].axvline(T0 / 60, color="k", lw=0.6, ls=":")
        axs[0, j].set_title(SCN[k], fontsize=8.5); axs[0, j].set_xlabel("Time since onset [h]"); axs[0, j].set_ylim(0, 1)
        axs[1, j].set_xlabel("Delivery delay of critical SOS [h]"); axs[1, j].set_ylim(0, 1)
        if j: axs[0, j].set_yticklabels([]); axs[1, j].set_yticklabels([])
    axs[0, 0].set_ylabel("Cumulative normalised ELS"); axs[1, 0].set_ylabel("Fraction of critical SOS")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig5_time_cdf")
    pd.DataFrame(rows).to_csv(f"{REV}/fig5_numbers.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

if __name__ == "__main__":
    which = sys.argv[1:]
    if "k" in which: fig_els_vs_k()
    if "sens" in which: fig_sens()
    if set(which) & {"maps", "snap", "time"}:
        Ws = worlds()
        if "maps" in which: fig_maps(Ws)
        if "snap" in which: fig_snapshots(Ws)
        if "time" in which: fig_time(Ws)
