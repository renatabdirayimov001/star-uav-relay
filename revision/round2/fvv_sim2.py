#!/usr/bin/env python3
"""
Survival-time-aware UAV swarm relaying in infrastructure-damaged areas
Terrain-realistic case studies: Uzbekistan / Central Asia

Real data (downloaded automatically on Kaggle with Internet ON):
  * Terrain: AWS Terrain Tiles (Terrarium, SRTM-based), no account needed
  * Population: WorldPop 2020, 1 km
  * Base stations: OpenCelliD CSV (optional, --cells). Without it a synthetic
    population-driven layout is used and flagged as such in the outputs.

Run (Kaggle, Internet ON):
  python fvv_sim.py --out /kaggle/working/results
Quick smoke test without internet:
  python fvv_sim.py --synthetic --seeds 2 --K 6 --out test_results
"""
import argparse, io, json, math, os, sys, time, gzip
import numpy as np

try:
    import scipy.sparse as sp
    from scipy.ndimage import map_coordinates
    from scipy.sparse.csgraph import dijkstra, breadth_first_order
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import wilcoxon
except ImportError:
    sys.exit("scipy is required")

# =============================================================================
# 1. SCENARIOS (edit coordinates here if needed)
# =============================================================================
SCENARIOS = {
    # Real event replay: Sardoba dam breach, 1 May 2020. Reservoir at
    # 40.3446N 68.4658E; flood ran north through Sardoba/Oqoltin/Mirzaobod
    # into Maktaaral (Kazakhstan). Footprint below is an approximate ellipse.
    "sardoba": dict(
        name="Sardoba dam-breach flood (UZ-KZ)",
        bbox=(40.30, 40.88, 68.15, 68.85),          # lat_min, lat_max, lon_min, lon_max
        iso=["UZB", "KAZ"], mcc=[434, 401],
        zone=dict(lat=40.58, lon=68.42, a_km=24.0, b_km=9.0, rot_deg=-10.0),  # a along N-S
        base=dict(lat=40.49, lon=68.78, label="Guliston (regional FVV HQ)"),
        bs_fail_in=0.85, bs_fail_out=0.05,
        penetration_db=10.0,     # flood victims: rooftops / inside houses
        bs_km2_per_site=25.0,    # used only if OpenCelliD is too sparse
        ground_speed_kmh=12.0,   # boats / damaged roads
    ),
    # Hypothetical M7 earthquake in real terrain: Namangan / Kosonsoy foothills
    "fergana": dict(
        name="Fergana valley earthquake (Namangan foothills)",
        bbox=(40.95, 41.42, 71.28, 71.92),
        iso=["UZB", "KGZ"], mcc=[434, 437],
        zone=dict(lat=41.20, lon=71.56, a_km=16.0, b_km=12.0, rot_deg=20.0),
        base=dict(lat=41.00, lon=71.67, label="Namangan city"),
        bs_fail_in=0.80, bs_fail_out=0.05,
        penetration_db=15.0,     # inside / under damaged buildings
        bs_km2_per_site=15.0,    # dense valley (OpenCelliD has ~0 sites here)
        ground_speed_kmh=15.0,
    ),
    # Hypothetical Amu Darya delta flood, sparse network (Karakalpakstan)
    "karakalpak": dict(
        name="Amu Darya delta flood (Muynak, Karakalpakstan)",
        bbox=(43.45, 43.95, 58.75, 59.40),
        iso=["UZB"], mcc=[434],
        zone=dict(lat=43.68, lon=59.05, a_km=18.0, b_km=10.0, rot_deg=30.0),
        base=dict(lat=43.50, lon=59.33, label="Forward command post (satellite terminal)"),
        bs_fail_in=0.90, bs_fail_out=0.10,
        penetration_db=10.0,
        bs_km2_per_site=80.0,    # sparse network (OpenCelliD has ~0 sites here)
        ground_speed_kmh=10.0,
    ),
}

# =============================================================================
# 2. GLOBAL PARAMETERS (every value should be cited in the paper)
# =============================================================================
P = dict(
    dem_res=100.0,          # m, DEM grid
    cell_res=500.0,         # m, demand grid
    site_res=1500.0,        # m, candidate hovering sites
    h_uav=150.0,            # m AGL
    v_uav=15.0,             # m/s cruise
    endurance_min=180.0,    # VTOL fixed-wing class
    recharge_min=30.0,      # battery swap + checks at base
    reserve_min=5.0,
    T0_min=60.0,            # UAV team ready 1 h after onset
    H_min=360.0,            # mission horizon after T0
    epoch_min=15.0,         # re-planning period
    tau_pen_min=30.0,       # travel-time discount in planner
    urg_D_min=float(os.environ.get("URG_D", 60.0)),   # deferral horizon of the survival-decay index
    beta=float(os.environ.get("BETA", 1.0)),          # weight of the level term g(t)
    n_victims=300,
    act_window_min=120.0,   # victims start SOS retries uniformly in [0, 2 h]
    rescue_setup_min=30.0,
    Tg_range_h=(12.0, 72.0),  # discovery time without network (ground search)
    # survival classes: (probability, Weibull scale [h], shape)  -> calibrate with literature
    classes=dict(critical=(0.2, 10.0, 1.0), serious=(0.3, 48.0, 1.5), minor=(0.5, 150.0, 2.0)),
    # hazard intensity I in [0,1] (1 at zone centre, 0 at boundary) shapes victims and severity:
    #   victims per cell ~ pop * (dmg0 + (1-dmg0) I);  p_critical = pc0 + pc1 I, p_serious fixed
    dmg0=0.3, pc0=0.05, pc1=0.40, ps=0.30,
    # access link: victim phone -> UAV small cell (LTE 1.8 GHz)
    f_acc=1.8e9, p_ue=23.0, g_uav_acc=3.0, b_acc=1.4e6, nf_uav=7.0, snr_acc=0.0, margin_acc=10.0,
    # backhaul: UAV <-> UAV / gateway (5.8 GHz)
    f_bh=5.8e9, p_bh=30.0, g_bh=6.0, g_gw=12.0, b_bh=10e6, nf_bh=6.0, snr_bh=5.0, margin_bh=10.0,
    h_bs=30.0, h_base=10.0,
    # direct victim -> surviving BS
    g_bs=15.0, nf_bs=3.0, clutter_bs=15.0,
    # A2G LoS probability (Al-Hourani, suburban) and excess losses
    a_los=4.88, b_los=0.43, eta_los=0.1, eta_nlos=21.0,
    k_earth=4.0 / 3.0,
    # ---- revision options (defaults reproduce the original simulator exactly) ----
    assign_fix=0.0,        # 1: grow the tree with the available UAVs and store paths near-to-far (connected prefix)
    p_deliv=1.0,           # per-attempt success probability of an SOS when a delivery path exists
    retry_min=1.0,         # phones attempt every retry_min minutes (random phase); 1 = every minute
    battery_h=0.0,         # mean phone battery life from onset [h], exponential; 0 = unlimited
    prior_shift_km=0.0,    # planner's belief: zone centre shifted across the major axis [km]
    prior_rot_deg=0.0,     # planner's belief: extra rotation of the zone ellipse [deg]
    prior_scale=1.0,       # planner's belief: scale of the zone axes
    prior_dmg0=-1.0,       # planner's belief of delta_0 (-1: true value)
    prior_pc1=-1.0,        # planner's belief of pi_1 (-1: true value)
    bs_flip=0.0,           # probability that the planner's belief about each BS status is wrong
    bs_fail_all=0.0,       # 1: all base stations fail (satellite gateway only)
    stagger=0.0,           # UAV k ready at T0 + stagger*E*k/K
    ferry_hover_min=10.0,  # ferry collection time at a site [min]
    ferry_deliv_m=3000.0,  # ferry delivery radius around the base [m]
    sweep_stripe_m=2000.0, # stripe width of the connected-sweep baseline [m]
)
C = 3e8
RE = 6371e3


# =============================================================================
# 3. GEO HELPERS
# =============================================================================
class Grid:
    def __init__(self, bbox):
        self.lat0, self.lat1, self.lon0, self.lon1 = bbox
        self.latc = 0.5 * (self.lat0 + self.lat1)
        self.mx = 111320.0 * math.cos(math.radians(self.latc))
        self.my = 110540.0
        self.W = (self.lon1 - self.lon0) * self.mx
        self.Hh = (self.lat1 - self.lat0) * self.my

    def xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.mx, (np.asarray(lat) - self.lat0) * self.my

    def ll(self, x, y):
        return self.lat0 + np.asarray(y) / self.my, self.lon0 + np.asarray(x) / self.mx


def ell_radius(g, x, y, z):
    cx, cy = g.xy(z["lat"], z["lon"])
    th = math.radians(z["rot_deg"])
    dx, dy = x - cx, y - cy
    u = dx * math.cos(th) - dy * math.sin(th)
    v = dx * math.sin(th) + dy * math.cos(th)
    return np.sqrt((u / (z["b_km"] * 1e3)) ** 2 + (v / (z["a_km"] * 1e3)) ** 2)


def in_ellipse(g, x, y, z):
    cx, cy = g.xy(z["lat"], z["lon"])
    th = math.radians(z["rot_deg"])
    dx, dy = x - cx, y - cy
    u = dx * math.cos(th) - dy * math.sin(th)     # across
    v = dx * math.sin(th) + dy * math.cos(th)     # along (N-S before rotation)
    return (u / (z["b_km"] * 1e3)) ** 2 + (v / (z["a_km"] * 1e3)) ** 2 <= 1.0


# =============================================================================
# 4. DATA ACQUISITION
# =============================================================================
def _get(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "fvv-sim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_dem(g, cache, z=11):
    """AWS Terrarium tiles -> DEM on local grid (m)."""
    fn = os.path.join(cache, "dem.npy")
    if os.path.exists(fn):
        return np.load(fn)
    from PIL import Image

    def tile_xy(lat, lon):
        n = 2 ** z
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        xt = (lon + 180.0) / 360.0 * n
        yt = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
        return xt, yt
    x0, y0 = tile_xy(g.lat1, g.lon0)
    x1, y1 = tile_xy(g.lat0, g.lon1)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    mosaic = np.zeros((len(ty) * 256, len(tx) * 256), np.float32)
    for j, yy in enumerate(ty):
        for i, xx in enumerate(tx):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{xx}/{yy}.png"
            im = np.asarray(Image.open(io.BytesIO(_get(url))).convert("RGB"), np.float32)
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
    nx, ny = int(g.W // P["dem_res"]) + 1, int(g.Hh // P["dem_res"]) + 1
    X, Y = np.meshgrid(np.arange(nx) * P["dem_res"], np.arange(ny) * P["dem_res"])
    lat, lon = g.ll(X, Y)
    fx, fy = tile_xy(lat, lon)
    cols = (fx - tx[0]) * 256 - 0.5
    rows = (fy - ty[0]) * 256 - 0.5
    dem = map_coordinates(mosaic, [rows, cols], order=1, mode="nearest").astype(np.float32)
    np.save(fn, dem)
    return dem


WORLDPOP_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/{I}/{i}_ppp_2020_1km_Aggregated.tif",
    "https://worldpop-public-data.soton.ac.uk/GIS/Population/Global_2000_2020_1km/2020/{I}/{i}_ppp_2020_1km_Aggregated.tif",
]


def fetch_pop(g, iso_list, cache, pop_tifs=None):
    """WorldPop people per 1 km pixel -> people per demand cell."""
    fn = os.path.join(cache, "pop.npy")
    if os.path.exists(fn):
        return np.load(fn)
    try:
        import rasterio
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q rasterio")
        import rasterio
    from rasterio.windows import from_bounds
    nx, ny = int(g.W // P["cell_res"]), int(g.Hh // P["cell_res"])
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * P["cell_res"], (np.arange(ny) + 0.5) * P["cell_res"])
    lat, lon = g.ll(X, Y)
    pop = np.zeros((ny, nx), np.float64)
    paths = list(pop_tifs or [])
    if not paths:
        for iso in iso_list:
            ok = False
            for u in WORLDPOP_URLS:
                url = u.format(I=iso, i=iso.lower())
                try:
                    data = _get(url, timeout=300)
                    p = os.path.join(cache, f"{iso}.tif")
                    open(p, "wb").write(data)
                    paths.append(p); ok = True
                    print(f"  WorldPop {iso}: {len(data)/1e6:.1f} MB")
                    break
                except Exception as e:
                    print(f"  WorldPop {iso} failed at {url}: {e}")
            if not ok:
                raise RuntimeError(f"WorldPop download failed for {iso}. Upload the tif and pass --pop-tif.")
    for p in paths:
        with rasterio.open(p) as ds:
            win = from_bounds(g.lon0 - 0.02, g.lat0 - 0.02, g.lon1 + 0.02, g.lat1 + 0.02, ds.transform)
            win = win.round_offsets().round_lengths()
            arr = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
            arr[(arr < 0) | ~np.isfinite(arr)] = 0
            tr = ds.window_transform(win)
            col = (lon - tr.c) / tr.a - 0.5
            row = (lat - tr.f) / tr.e - 0.5
            dens = map_coordinates(arr, [row, col], order=0, mode="constant", cval=0.0)
            pix_area = abs(tr.a * tr.e) * g.mx * g.my  # m2 of one source pixel
            pop = np.maximum(pop, dens * (P["cell_res"] ** 2) / pix_area)  # overlapping countries: take max
    np.save(fn, pop)
    return pop


def load_cells(g, mcc, cell_paths):
    """OpenCelliD CSV(s) -> unique site coordinates inside bbox."""
    import pandas as pd
    cols = ["radio", "mcc", "net", "area", "cell", "unit", "lon", "lat", "range", "samples",
            "changeable", "created", "updated", "averageSignal"]
    frames = []
    for p in cell_paths:
        df = pd.read_csv(p, header=None, names=cols, low_memory=False)
        if str(df.iloc[0, 0]).lower() == "radio":
            df = df.iloc[1:]
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
        frames.append(df[df.mcc.isin(mcc)])
    df = pd.concat(frames)
    df = df[(df.lat > g.lat0) & (df.lat < g.lat1) & (df.lon > g.lon0) & (df.lon < g.lon1)]
    x, y = g.xy(df.lat.values, df.lon.values)
    key = np.round(x / 300).astype(int) * 100000 + np.round(y / 300).astype(int)   # merge cells of one site
    _, idx = np.unique(key, return_index=True)
    return np.c_[x[idx], y[idx]]


def synthetic_bs(g, pop, rng, km2_per_site=30.0):
    """Population-driven synthetic BS layout (used if OpenCelliD is missing or too sparse)."""
    area_km2 = g.W * g.Hh / 1e6
    n = max(8, int(area_km2 / km2_per_site))
    w = np.sqrt(pop.ravel() + 1e-9); w /= w.sum()
    ny, nx = pop.shape
    out = []
    for k in rng.choice(pop.size, size=n * 5, p=w):
        yy, xx = divmod(k, nx)
        p = np.array([(xx + rng.random()) * P["cell_res"], (yy + rng.random()) * P["cell_res"]])
        if all(np.hypot(*(p - q)) > 2500 for q in out):
            out.append(p)
        if len(out) >= n:
            break
    return np.array(out)


def synthetic_dem_pop(g, rng, mountain=False):
    nx, ny = int(g.W // P["dem_res"]) + 1, int(g.Hh // P["dem_res"]) + 1
    X, Y = np.meshgrid(np.arange(nx) * P["dem_res"], np.arange(ny) * P["dem_res"])
    dem = 300 + 0.002 * X
    for _ in range(25 if mountain else 6):
        cx, cy, s = rng.random() * g.W, rng.random() * g.Hh, 2000 + rng.random() * 6000
        dem += (900 if mountain else 40) * rng.random() * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    cnx, cny = int(g.W // P["cell_res"]), int(g.Hh // P["cell_res"])
    Xc, Yc = np.meshgrid((np.arange(cnx) + .5) * P["cell_res"], (np.arange(cny) + .5) * P["cell_res"])
    pop = np.full((cny, cnx), 5.0)
    for _ in range(30):
        cx, cy, s = rng.random() * g.W, rng.random() * g.Hh, 500 + rng.random() * 1500
        pop += 3000 * rng.random() * np.exp(-((Xc - cx) ** 2 + (Yc - cy) ** 2) / (2 * s * s))
    return dem.astype(np.float32), pop


# =============================================================================
# 5. TERRAIN-AWARE PROPAGATION
# =============================================================================
class Terrain:
    def __init__(self, dem):
        self.dem = dem

    def h(self, x, y):
        return map_coordinates(self.dem, [np.asarray(y) / P["dem_res"], np.asarray(x) / P["dem_res"]],
                               order=1, mode="nearest")

    def knife_edge(self, x1, y1, z1, x2, y2, z2, f, ns=64, chunk=20000):
        """Max Fresnel-Kirchhoff parameter v over the terrain profile (with 4/3 earth bulge)."""
        lam = C / f
        out = np.empty(len(x1))
        t = (np.arange(1, ns + 1) / (ns + 1))[None, :]
        for s in range(0, len(x1), chunk):
            e = slice(s, s + chunk)
            d = np.hypot(x2[e] - x1[e], y2[e] - y1[e])[:, None] + 1e-6
            xs = x1[e][:, None] + t * (x2[e] - x1[e])[:, None]
            ys = y1[e][:, None] + t * (y2[e] - y1[e])[:, None]
            zl = z1[e][:, None] + t * (z2[e] - z1[e])[:, None]
            d1, d2 = t * d, (1 - t) * d
            zt = self.h(xs.ravel(), ys.ravel()).reshape(xs.shape) + d1 * d2 / (2 * P["k_earth"] * RE)
            v = (zt - zl) * np.sqrt(2.0 * d / (lam * d1 * d2))
            out[e] = v.max(axis=1)
        return out


def J(v):
    v = np.asarray(v, float)
    L = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
    return np.where(v > -0.78, L, 0.0)


def fspl(d, f):
    return 20 * np.log10(np.maximum(d, 1.0)) + 20 * np.log10(f) - 147.55


def a2g_excess(dh, d3):
    th = np.degrees(np.arcsin(np.clip(dh / np.maximum(d3, 1), 0, 1)))
    plos = 1.0 / (1 + P["a_los"] * np.exp(-P["b_los"] * (th - P["a_los"])))
    return plos * P["eta_los"] + (1 - plos) * P["eta_nlos"]


def noise(b, nf):
    return -174 + 10 * np.log10(b) + nf


def link_snr(terr, a, b, kind, terrain_on=True):
    """a, b: arrays (n,3) absolute positions. kind in {'acc','bh','gw','bs'}"""
    x1, y1, z1 = a.T; x2, y2, z2 = b.T
    d3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if kind == "acc":
        f = P["f_acc"]; pt = P["p_ue"]; g = P["g_uav_acc"]; n = noise(P["b_acc"], P["nf_uav"])
        ex = a2g_excess(np.abs(z2 - z1), d3) + P["_pen"] + P["margin_acc"]
    elif kind in ("bh", "gw"):
        f = P["f_bh"]; pt = P["p_bh"]; g = P["g_bh"] + (P["g_gw"] if kind == "gw" else P["g_bh"])
        n = noise(P["b_bh"], P["nf_bh"]); ex = P["margin_bh"]
    else:  # victim -> surviving BS (ground-ground)
        f = P["f_acc"]; pt = P["p_ue"]; g = P["g_bs"]; n = noise(P["b_acc"], P["nf_bs"])
        ex = P["clutter_bs"] + P["_pen"] + P["margin_acc"]
    pl = fspl(d3, f) + ex
    if terrain_on:
        pl = pl + J(terr.knife_edge(x1, y1, z1, x2, y2, z2, f))
    return pt + g - pl - n


def max_range(kind):
    if kind == "acc":
        budget = P["p_ue"] + P["g_uav_acc"] - noise(P["b_acc"], P["nf_uav"]) - P["snr_acc"] - P["margin_acc"] - P["_pen"] - P["eta_los"]
        f = P["f_acc"]
    elif kind == "bs":
        budget = P["p_ue"] + P["g_bs"] - noise(P["b_acc"], P["nf_bs"]) - P["snr_acc"] - P["margin_acc"] - P["_pen"] - P["clutter_bs"]
        f = P["f_acc"]
    else:
        g = P["g_bh"] + (P["g_gw"] if kind == "gw" else P["g_bh"])
        budget = P["p_bh"] + g - noise(P["b_bh"], P["nf_bh"]) - P["snr_bh"] - P["margin_bh"]
        f = P["f_bh"]
    return 10 ** ((budget - 20 * np.log10(f) + 147.55) / 20)


def pairs_within(A, B, r):
    from scipy.spatial import cKDTree
    t = cKDTree(B[:, :2])
    res = t.query_ball_point(A[:, :2], r)
    ii = np.repeat(np.arange(len(A)), [len(x) for x in res])
    jj = np.concatenate([np.asarray(x, int) for x in res]) if len(ii) else np.zeros(0, int)
    return ii, jj


def feasibility(terr, A, B, kind, thr, symmetric=False):
    """Sparse boolean matrices (true terrain, free-space planner view)."""
    r = min(max_range(kind), 40e3)
    ii, jj = pairs_within(A, B, r)
    if symmetric:
        m = ii < jj; ii, jj = ii[m], jj[m]
    snr_t = link_snr(terr, A[ii], B[jj], kind, True) if len(ii) else np.zeros(0)
    snr_f = link_snr(terr, A[ii], B[jj], kind, False) if len(ii) else np.zeros(0)
    out = []
    for snr in (snr_t, snr_f):
        k = snr >= thr
        M = sp.csr_matrix((np.ones(k.sum(), bool), (ii[k], jj[k])), shape=(len(A), len(B)))
        if symmetric:
            M = (M + M.T).astype(bool).tocsr()
        out.append(M)
    return out


# =============================================================================
# 6. SURVIVAL MODEL
# =============================================================================
def surv(t_h, lam, k):
    return np.exp(-(np.maximum(t_h, 0) / lam) ** k)


def class_probs(I):
    pc = P["pc0"] + P["pc1"] * I
    return np.c_[pc, np.full_like(I, P["ps"]), 1 - pc - P["ps"]]     # critical, serious, minor


def gain_curve(tau_min, tgrid_min, pcls, rng_q=np.linspace(0.01, 0.99, 41)):
    """Expected survival gain per expected victim (cell-specific class mix) if delivered at t."""
    Tg = P["Tg_range_h"][0] + rng_q * (P["Tg_range_h"][1] - P["Tg_range_h"][0])
    out = np.zeros((len(tau_min), len(tgrid_min)))
    for c, (_, lam, k) in enumerate(P["classes"].values()):
        p = pcls[:, c:c + 1]
        s_net = surv((tgrid_min[None, :] + tau_min[:, None]) / 60.0, lam, k)
        s_gnd = surv((Tg[None, None, :] * 60 + tau_min[:, None, None]) / 60.0, lam, k).mean(axis=2)
        out += p * np.maximum(s_net - s_gnd, 0)
    return out


# =============================================================================
# 7. SCENARIO BUILD (static part)
# =============================================================================
class World:
    pass


def build_world(key, args, rng):
    sc = SCENARIOS[key]
    P["_pen"] = sc["penetration_db"]
    g = Grid(sc["bbox"])
    cache = os.path.join(args.cache or os.path.join(args.out, "cache"), key + ("_syn" if args.synthetic else ""))
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    if args.synthetic:
        dem, pop = synthetic_dem_pop(g, np.random.default_rng(7), mountain=(key == "fergana"))
        data_src = "synthetic"
    else:
        dem = fetch_dem(g, cache)
        pop = fetch_pop(g, sc["iso"], cache, args.pop_tif)
        data_src = "AWS Terrarium DEM + WorldPop 2020"
    terr = Terrain(dem)
    W = World(); W.key, W.sc, W.g, W.terr, W.dem, W.pop = key, sc, g, terr, dem, pop

    # demand cells inside affected zone
    ny, nx = pop.shape
    Xc, Yc = np.meshgrid((np.arange(nx) + .5) * P["cell_res"], (np.arange(ny) + .5) * P["cell_res"])
    zmask = in_ellipse(g, Xc, Yc, sc["zone"]) & (pop > 0.5)
    W.cx, W.cy, W.cpop = Xc[zmask], Yc[zmask], pop[zmask]
    W.cz = terr.h(W.cx, W.cy) + 1.5
    W.cI = np.clip(1 - ell_radius(g, W.cx, W.cy, sc["zone"]), 0, 1)       # hazard intensity
    W.cw = W.cpop * (P["dmg0"] + (1 - P["dmg0"]) * W.cI)                  # victim propensity
    W.cm = P["n_victims"] * W.cw / W.cw.sum()                             # expected victims per cell
    W.cp = class_probs(W.cI)

    # base / BS
    bx, by = g.xy(sc["base"]["lat"], sc["base"]["lon"])
    W.base = np.array([float(bx), float(by)])
    W.bs, W.bs_src = None, ""
    if args.cells:
        bs = load_cells(g, sc["mcc"], args.cells)
        if len(bs) >= 20:
            W.bs, W.bs_src = bs, f"OpenCelliD ({len(bs)} sites)"
        else:
            W.bs_src = f"synthetic (OpenCelliD too sparse: {len(bs)} sites)"
    if W.bs is None:
        W.bs = synthetic_bs(g, pop, np.random.default_rng(11), sc["bs_km2_per_site"])
        W.bs_src = W.bs_src or "synthetic (population-driven)"
    W.bs_in = in_ellipse(g, W.bs[:, 0], W.bs[:, 1], sc["zone"])
    gw_xy = np.vstack([W.bs, W.base[None, :]])
    gw_h = np.r_[np.full(len(W.bs), P["h_bs"]), P["h_base"]]
    W.gw = np.c_[gw_xy, terr.h(gw_xy[:, 0], gw_xy[:, 1]) + gw_h]

    # candidate sites
    xs = np.arange(P["site_res"] / 2, g.W, P["site_res"])
    ys = np.arange(P["site_res"] / 2, g.Hh, P["site_res"])
    SX, SY = np.meshgrid(xs, ys)
    W.sites = np.c_[SX.ravel(), SY.ravel(), terr.h(SX.ravel(), SY.ravel()) + P["h_uav"]]
    S = len(W.sites)
    print(f"  [{key}] area {g.W/1e3:.0f}x{g.Hh/1e3:.0f} km, DEM {dem.min():.0f}-{dem.max():.0f} m, "
          f"zone pop {W.cpop.sum():.0f}, cells {len(W.cx)}, sites {S}, BS {len(W.bs)} ({W.bs_src})")
    print(f"  ranges (free space): acc {max_range('acc')/1e3:.1f} km, bh {max_range('bh')/1e3:.1f} km, gw {max_range('gw')/1e3:.1f} km")

    W.adj, W.adj_f = feasibility(terr, W.sites, W.sites, "bh", P["snr_bh"], symmetric=True)
    W.sg, W.sg_f = feasibility(terr, W.sites, W.gw, "gw", P["snr_bh"])
    cells3 = np.c_[W.cx, W.cy, W.cz]
    W.cov, W.cov_f = feasibility(terr, W.sites, cells3, "acc", P["snr_acc"])
    # gain curve per cell (tau = response time from base)
    W.tgrid = np.arange(0, P["T0_min"] + P["H_min"] + P["urg_D_min"] + 2)
    W.ctau = P["rescue_setup_min"] + np.hypot(W.cx - W.base[0], W.cy - W.base[1]) / (W.sc["ground_speed_kmh"] / 3.6) / 60
    W.cgain = gain_curve(W.ctau, W.tgrid, W.cp)
    W.site_base_t = np.hypot(W.sites[:, 0] - W.base[0], W.sites[:, 1] - W.base[1]) / P["v_uav"] / 60
    W.data_src = data_src
    print(f"  links: site-site {W.adj.nnz//2} (free-space view {W.adj_f.nnz//2}), "
          f"site-gw {W.sg.nnz}, site-cell {W.cov.nnz} (fs {W.cov_f.nnz}); {time.time()-t0:.1f} s")
    return W


# =============================================================================
# 8. RANDOM REALISATION (victims, BS failures)
# =============================================================================
def realise(W, rng):
    R = World()
    p = W.cw / W.cw.sum()
    ci = rng.choice(len(W.cx), size=P["n_victims"], p=p)
    R.vci = ci
    R.vx = W.cx[ci] + (rng.random(len(ci)) - .5) * P["cell_res"]
    R.vy = W.cy[ci] + (rng.random(len(ci)) - .5) * P["cell_res"]
    R.vz = W.terr.h(R.vx, R.vy) + 1.5
    cp = W.cp[ci]
    R.vcls = (rng.random(len(ci))[:, None] > np.cumsum(cp, axis=1)).sum(axis=1)
    R.va = rng.random(len(ci)) * P["act_window_min"]
    R.vTg = rng.uniform(*P["Tg_range_h"], size=len(ci)) * 60
    R.vtau = P["rescue_setup_min"] + np.hypot(R.vx - W.base[0], R.vy - W.base[1]) / (W.sc["ground_speed_kmh"] / 3.6) / 60
    fail = np.where(W.bs_in, rng.random(len(W.bs)) < W.sc["bs_fail_in"], rng.random(len(W.bs)) < W.sc["bs_fail_out"])
    if P.get("bs_fail_all", 0):
        fail = np.ones(len(W.bs), bool)
    R.gw_alive = np.r_[~fail, True]
    V = np.c_[R.vx, R.vy, R.vz]
    R.vcov, _ = feasibility(W.terr, W.sites, V, "acc", P["snr_acc"])
    R.vcov = R.vcov.T.tocsr()                       # victims x sites
    alive_bs = W.gw[:-1][~fail]
    R.direct = np.zeros(len(ci), bool)
    if len(alive_bs):
        Mbs, _ = feasibility(W.terr, V, alive_bs, "bs", P["snr_acc"])
        R.direct = np.asarray(Mbs.sum(axis=1)).ravel() > 0
    rx = np.random.default_rng(int(1e6 * R.va[0]) % (2 ** 31) + 17)   # separate stream for revision options
    flip = rx.random(len(W.bs)) < P.get("bs_flip", 0.0)
    R.gw_plan = np.r_[np.where(flip, ~R.gw_alive[:-1], R.gw_alive[:-1]), True]
    R.vphase = rx.integers(0, max(1, int(P.get("retry_min", 1))), size=len(ci))
    R.vbatt = rx.exponential(P["battery_h"] * 60, size=len(ci)) if P.get("battery_h", 0) > 0 else np.full(len(ci), np.inf)
    return R


def make_prior(W):
    """Planner's (possibly wrong) belief about the demand: perturbed zone and parameters."""
    if not (P["prior_shift_km"] or P["prior_rot_deg"] or P["prior_scale"] != 1 or P["prior_dmg0"] >= 0 or P["prior_pc1"] >= 0):
        W.cm_plan, W.cgain_plan, W.cp_plan = W.cm, W.cgain, W.cp
        return
    z = dict(W.sc["zone"]); r = math.radians(z["rot_deg"])
    dx, dy = P["prior_shift_km"] * math.cos(r), -P["prior_shift_km"] * math.sin(r)   # across the major axis
    z["lat"] += dy / (W.g.my / 1e3); z["lon"] += dx / (W.g.mx / 1e3)
    z["rot_deg"] += P["prior_rot_deg"]; z["a_km"] *= P["prior_scale"]; z["b_km"] *= P["prior_scale"]
    I = np.clip(1 - ell_radius(W.g, W.cx, W.cy, z), 0, 1)
    d0 = P["dmg0"] if P["prior_dmg0"] < 0 else P["prior_dmg0"]
    w = W.cpop * (d0 + (1 - d0) * I)
    W.cm_plan = P["n_victims"] * w / w.sum()
    pc1 = P["pc1"]
    if P["prior_pc1"] >= 0: P["pc1"] = P["prior_pc1"]
    W.cp_plan = class_probs(I); P["pc1"] = pc1
    W.cgain_plan = gain_curve(W.ctau, W.tgrid, W.cp_plan)


def victim_gain(R, t_min):
    names = list(P["classes"])
    g = np.zeros(len(R.vx))
    for c, n in enumerate(names):
        _, lam, k = P["classes"][n]
        m = R.vcls == c
        g[m] = np.maximum(surv((t_min[m] + R.vtau[m]) / 60, lam, k) - surv((R.vTg[m] + R.vtau[m]) / 60, lam, k), 0)
    return g


# =============================================================================
# 9. PLANNERS
# =============================================================================
def plan_tree(W, w, gw_alive, budget, pen, cov, adj, sg):
    """Ratio-greedy growth of a gateway-rooted relay tree (connected budgeted coverage)."""
    S = len(W.sites)
    gw_sites = np.asarray(sg[:, gw_alive].sum(axis=1)).ravel() > 0
    root = S
    A = sp.lil_matrix((S + 1, S + 1), dtype=float)
    A[:S, :S] = adj.astype(float)
    idx = np.where(gw_sites)[0]
    A[root, idx] = 1.0; A[idx, root] = 1.0
    A = A.tocsr()
    covered = np.zeros(cov.shape[1], bool)
    chosen, in_tree = [], np.zeros(S + 1, bool); in_tree[root] = True
    while budget > 0:
        nv = cov @ (w * ~covered) * pen                          # marginal value per site
        dist, pred, _ = dijkstra(A, indices=np.where(in_tree)[0], min_only=True, unweighted=True, return_predecessors=True)
        cand = np.where(np.isfinite(dist[:S]) & (dist[:S] >= 1) & (dist[:S] <= budget))[0]
        if len(cand) == 0:
            break
        order = np.argsort(dist[:S][cand])
        cum = np.zeros(S + 1)
        for s in cand[order]:
            p = pred[s]
            cum[s] = nv[s] + (cum[p] if (p >= 0 and not in_tree[p]) else 0.0)
        ratio = cum[cand] / dist[cand]
        b = cand[np.argmax(ratio)]
        if ratio.max() <= 1e-9:
            break
        path, s = [], b
        while s >= 0 and not in_tree[s]:
            path.append(s); s = pred[s]
        for s in (path[::-1] if P.get("assign_fix", 0) else path):
            in_tree[s] = True; chosen.append(s)
            covered |= np.asarray(cov[s].todense()).ravel().astype(bool)
        budget -= len(path)
    return chosen


def weighted_kmeans(W, K, rng, it=50):
    X = np.c_[W.cx, W.cy]; w = getattr(W, "cm_plan", W.cm)
    c = X[rng.choice(len(X), K, replace=False, p=w / w.sum())]
    for _ in range(it):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            m = lab == k
            if m.any():
                c[k] = (X[m] * w[m, None]).sum(0) / w[m].sum()
    d = ((W.sites[None, :, :2] - c[:, None]) ** 2).sum(-1)
    out = []
    for k in range(K):
        for s in np.argsort(d[k]):
            if s not in out:
                out.append(s); break
    return out


# =============================================================================
# 10. SIMULATOR
# =============================================================================
METHODS = ["Proposed", "Proposed-Bayes", "Sweep", "Proposed-noMemory", "Proposed-static", "Proposed-noTerrain", "ConnCov-dynamic", "ConnCov-static", "KMeans", "Ferry"]


def simulate(W, R, K, method, rng, log_ts=False, snaps=None):
    S = len(W.sites); T0, H, E = P["T0_min"], P["H_min"], P["endurance_min"]
    v = P["v_uav"] * 60.0                                   # m/min
    pos = np.tile(W.base, (K, 1)).astype(float)
    mode = np.array(["base"] * K, dtype=object)             # base, transit, hover, ret, charge
    if not hasattr(W, "cm_plan"):
        make_prior(W)
    Wcm, Wg = W.cm_plan, W.cgain_plan
    ready = T0 + P.get("stagger", 0.0) * E * np.arange(K) / K
    Ebel = np.zeros(len(W.cx))                                # Bayes belief: expected undelivered active victims
    p_eff = P.get("p_deliv", 1.0) / max(1.0, P.get("retry_min", 1.0))
    rngd = np.random.default_rng(int(1e6 * R.va[1]) % (2 ** 31) + 29)
    stat = dict(trunc=0, epochs=0, hov=0, con=0, hops=0, multi=0, relay=0)
    if method == "Sweep" and not hasattr(W, "sweep_rank"):
        zz = W.sc["zone"]; cx0, cy0 = W.g.xy(zz["lat"], zz["lon"]); rr = math.radians(zz["rot_deg"])
        dxs, dys = W.cx - cx0, W.cy - cy0
        u = dxs * math.sin(rr) + dys * math.cos(rr)            # along the major axis
        vv = dxs * math.cos(rr) - dys * math.sin(rr)           # across
        stripe = np.floor((vv - vv.min()) / P["sweep_stripe_m"]).astype(int)
        key = np.where(stripe % 2 == 0, u, -u)
        order = np.lexsort((key, stripe)); W.sweep_rank = np.empty(len(order), int); W.sweep_rank[order] = np.arange(len(order))
    tgt = -np.ones(K, int); en = np.full(K, E); chg = np.zeros(K)
    carry = [set() for _ in range(K)]                        # ferry payload
    ferry_state = np.array(["go"] * K, dtype=object)
    ferry_until = np.zeros(K)
    delivered = np.full(len(R.vx), np.nan)
    cell_deliv = np.zeros(len(W.cx))
    cell_last = np.full(len(W.cx), -1e9)        # last minute each demand cell was served (connected coverage)
    Fa = lambda tt: np.clip(tt / P["act_window_min"], 0, 1)
    use_f = method == "Proposed-noTerrain"
    cov_p = W.cov_f if use_f else W.cov
    adj_p = W.adj_f if use_f else W.adj
    sg_p = W.sg_f if use_f else W.sg
    sg_alive = W.sg[:, R.gw_alive]
    gw_ok_site = np.asarray(sg_alive.sum(axis=1)).ravel() > 0
    plan_times, static_plan = [], None
    ts = []
    # direct BS delivery happens as soon as active (independent of UAVs)
    for t in np.arange(0, T0 + H + 1):
        # ---------------- planning ----------------
        if t >= T0 and (t - T0) % P["epoch_min"] == 0 and method != "Ferry":
            avail = [k for k in range(K) if mode[k] in ("base", "transit", "hover")
                     and en[k] > 20 + np.hypot(*(pos[k] - W.base)) / v and t >= ready[k]]
            budget = len(avail) if P.get("assign_fix", 0) else K
            tp = time.time()
            if method in ("Proposed-static", "ConnCov-static", "KMeans") and static_plan is not None:
                plan = static_plan
            else:
                # belief: victims activated after the cell was last served are still waiting
                rem = Wcm * (Fa(t) - Fa(np.maximum(cell_last, 0)) * (cell_last >= 0)) + Wcm * (Fa(t + P["epoch_min"]) - Fa(t))
                if method == "Proposed-noMemory":   # ablation: belief only from delivered SOS counts
                    rem = np.maximum(Wcm * Fa(t) - cell_deliv, 0) + Wcm * (Fa(t + P["epoch_min"]) - Fa(t))
                if method == "Proposed-Bayes":      # posterior mean under a Poisson prior with imperfect delivery
                    rem = Ebel + Wcm * (Fa(t + P["epoch_min"]) - Fa(t))
                if len(avail):
                    dmin = np.min(np.hypot(W.sites[:, None, 0] - pos[avail, 0][None], W.sites[:, None, 1] - pos[avail, 1][None]), axis=1) / v
                else:
                    dmin = W.site_base_t
                pen = np.exp(-dmin / P["tau_pen_min"])
                tt = min(int(t + P["epoch_min"] / 2), len(W.tgrid) - 1)
                if method in ("Proposed", "Proposed-noTerrain", "Proposed-noMemory", "Proposed-Bayes"):
                    t2 = min(int(tt + P["urg_D_min"]), len(W.tgrid) - 1)
                    # survival-decay index (c-mu rule): loss if deferred by D + level term
                    w = rem * ((Wg[:, tt] - Wg[:, t2]) + P["beta"] * Wg[:, tt])
                elif method == "Proposed-static":
                    w = Wcm * Wg[:, int(T0)]; pen = np.ones(S)
                elif method == "ConnCov-dynamic":
                    w = rem
                elif method == "ConnCov-static":
                    w = Wcm.copy(); pen = np.ones(S)
                elif method == "Sweep":
                    keyc = np.where(cell_last >= 0, cell_last, -1e9) * 1e6 + W.sweep_rank
                    nxt = np.argsort(keyc)[:12 * K]
                    w = np.zeros(len(W.cx)); w[nxt] = 1.0
                if method == "KMeans":
                    plan = weighted_kmeans(W, K, rng)
                else:
                    plan = plan_tree(W, w, R.gw_plan, budget, pen, cov_p, adj_p, sg_p)
                if method in ("Proposed-static", "ConnCov-static", "KMeans"):
                    static_plan = plan
            plan_times.append(time.time() - tp)
            # assignment (Hungarian on travel time), keep plan size <= available UAVs
            stat["epochs"] += 1; stat["trunc"] += int(len(plan) > len(avail))
            plan = list(plan)[:len(avail)]
            if len(plan) and len(avail):
                cost = np.hypot(pos[avail][:, None, 0] - W.sites[plan][None, :, 0], pos[avail][:, None, 1] - W.sites[plan][None, :, 1])
                ri, ci = linear_sum_assignment(cost)
                assigned = set()
                for r_, c_ in zip(ri, ci):
                    k = avail[r_]; s = plan[c_]; assigned.add(k)
                    if tgt[k] != s or mode[k] == "base":
                        tgt[k] = s; mode[k] = "transit" if np.hypot(*(pos[k] - W.sites[s, :2])) > 1 else "hover"
                for k in avail:
                    if k not in assigned and mode[k] != "base":
                        mode[k] = "ret"
        # ---------------- ferry policy ----------------
        if method == "Ferry" and t >= T0:
            rem = Wcm * (Fa(t) - Fa(np.maximum(cell_last, 0)) * (cell_last >= 0))
            tt = min(int(t), len(W.tgrid) - 1)
            val = W.cov @ (rem * Wg[:, tt])
            for k in range(K):
                if mode[k] == "base" and en[k] >= E - 1e-9 and ferry_state[k] == "go" and t >= ready[k]:
                    taken = set(tgt[[j for j in range(K) if mode[j] in ("transit", "hover") and j != k]])
                    d = np.hypot(W.sites[:, 0] - pos[k, 0], W.sites[:, 1] - pos[k, 1]) / v
                    feas = (2 * d + 15 < en[k])
                    sc_ = np.where(feas, val * np.exp(-d / P["tau_pen_min"]), -1)
                    for s in taken:
                        if s >= 0: sc_[s] = -1
                    s = int(np.argmax(sc_))
                    if sc_[s] > 0:
                        tgt[k] = s; mode[k] = "transit"; ferry_state[k] = "collect"
        # ---------------- kinematics & energy ----------------
        for k in range(K):
            if mode[k] == "charge":
                chg[k] -= 1
                if chg[k] <= 0:
                    mode[k] = "base"; en[k] = E; ferry_state[k] = "go"
                continue
            if mode[k] == "base":
                continue
            en[k] -= 1
            home_t = np.hypot(*(pos[k] - W.base)) / v
            if mode[k] != "ret" and en[k] <= home_t + P["reserve_min"]:
                mode[k] = "ret"
            if mode[k] == "transit":
                d = W.sites[tgt[k], :2] - pos[k]; dist = np.hypot(*d)
                if dist <= v:
                    pos[k] = W.sites[tgt[k], :2].copy(); mode[k] = "hover"
                    if method == "Ferry":
                        ferry_until[k] = t + P["ferry_hover_min"]
                else:
                    pos[k] += d / dist * v
            elif mode[k] == "ret":
                d = W.base - pos[k]; dist = np.hypot(*d)
                if dist <= v:
                    pos[k] = W.base.copy(); mode[k] = "charge"; chg[k] = P["recharge_min"]; tgt[k] = -1
                else:
                    pos[k] += d / dist * v
            elif mode[k] == "hover" and method == "Ferry" and t >= ferry_until[k]:
                mode[k] = "ret"
        # ---------------- connectivity & delivery ----------------
        active = R.va <= t
        if P.get("retry_min", 1) > 1 or P.get("p_deliv", 1) < 1 or P.get("battery_h", 0) > 0:
            att = ((np.floor(t - R.va).astype(int) % max(1, int(P["retry_min"]))) == R.vphase) & (t < R.vbatt)
            att &= rngd.random(len(R.va)) < P["p_deliv"]
            active = active & att
        Ebel += Wcm * (Fa(t) - Fa(t - 1))
        newly = active & np.isnan(delivered) & R.direct
        hov = [k for k in range(K) if mode[k] == "hover" and tgt[k] >= 0]
        con, hs = np.zeros(0, int), np.zeros(0, int)
        if method == "Ferry":
            for k in hov:
                cell_last[W.cov[tgt[k]].indices] = t
                vic = R.vcov[:, tgt[k]].nonzero()[0]
                carry[k].update(int(i) for i in vic if active[i] and np.isnan(delivered[i]))
            for k in range(K):
                if carry[k]:
                    # delivery when within gateway range (base counts) or at a gateway-linked site
                    near_base = np.hypot(*(pos[k] - W.base)) < P["ferry_deliv_m"]
                    at_gw = mode[k] == "hover" and gw_ok_site[tgt[k]]
                    if near_base or at_gw or mode[k] == "charge":
                        idx = np.array(sorted(carry[k])); idx = idx[np.isnan(delivered[idx])]
                        delivered[idx] = t; np.add.at(cell_deliv, R.vci[idx], 1); carry[k].clear()
        elif hov:
            hs = np.array(sorted(set(tgt[hov])))
            sub = W.adj[hs][:, hs]
            root_ok = gw_ok_site[hs]
            n = len(hs)
            Aext = sp.bmat([[sub, sp.csr_matrix(root_ok[:, None].astype(float))],
                            [sp.csr_matrix(root_ok[None, :].astype(float)), None]]).tocsr()
            reach, predb = breadth_first_order(Aext, n, directed=False, return_predecessors=True)
            con = hs[[r_ for r_ in reach if r_ < n]]
            depth = {n: 0}
            for r_ in reach:
                if r_ != n: depth[r_] = depth[predb[r_]] + 1
            parents = {predb[r_] for r_ in reach if r_ != n and predb[r_] != n}
            stat["hov"] += n; stat["con"] += len(con)
            stat["hops"] += sum(depth[r_] for r_ in reach if r_ < n)
            stat["multi"] += sum(1 for r_ in reach if r_ < n and depth[r_] >= 2)
            stat["relay"] += len(parents)
            if len(con):
                cidx = np.unique(W.cov[con].indices)
                cell_last[cidx] = t
                Ebel[cidx] *= (1 - p_eff)
                cv = np.asarray(R.vcov[:, con].sum(axis=1)).ravel() > 0
                newly |= active & np.isnan(delivered) & cv
        if newly.any():
            delivered[newly] = t
            np.add.at(cell_deliv, R.vci[newly], 1)
        if snaps is not None and t in snaps:
            hs_ = np.array(sorted(set(tgt[hov]))) if hov else np.zeros(0, int)
            con_ = con if (method != "Ferry" and hov and len(hs)) else np.zeros(0, int)
            snaps[t] = dict(hover=hs_, con=np.asarray(con_), delivered=~np.isnan(delivered), pos=pos.copy())
        if log_ts and t >= T0 and (t - T0) % 5 == 0:
            m = ~np.isnan(delivered)
            gsum = victim_gain(R, np.where(m, delivered, 0))[m].sum()
            ts.append((t, m.sum(), gsum, len(hov)))
    # ---------------- metrics ----------------
    m = ~np.isnan(delivered)
    g_real = victim_gain(R, np.where(m, delivered, 0)); g_real[~m] = 0
    g_ideal = victim_gain(R, np.maximum(R.va, T0))
    delay = delivered[m] - np.maximum(R.va[m], T0)
    crit = R.vcls == 0
    res = dict(ELS=g_real.sum(), ELS_ideal=g_ideal.sum(), ELS_norm=g_real.sum() / max(g_ideal.sum(), 1e-9),
               DR=m.mean(), DR_crit=(m & crit).sum() / max(crit.sum(), 1),
               crit_1h=((m & crit) & (np.nan_to_num(delivered - np.maximum(R.va, T0), nan=1e9) <= 60)).sum() / max(crit.sum(), 1),
               delay_med=float(np.median(delay)) if m.any() else np.nan,
               delay_mean=float(np.mean(delay)) if m.any() else np.nan,
               direct=R.direct.mean(), plan_ms=1000 * float(np.mean(plan_times)) if plan_times else 0.0,
               trunc_epochs=stat["trunc"], n_epochs=stat["epochs"],
               iso_frac=1 - stat["con"] / max(stat["hov"], 1), mean_hops=stat["hops"] / max(stat["con"], 1),
               frac_multihop=stat["multi"] / max(stat["con"], 1), frac_relay=stat["relay"] / max(stat["con"], 1))
    return res, delivered, ts


# =============================================================================
# 11. EXPERIMENT DRIVER + PLOTS
# =============================================================================
def run(args):
    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    keys = list(SCENARIOS) if args.scenarios == ["all"] else args.scenarios
    rows, meta = [], {}
    for key in keys:
        print(f"\n=== {SCENARIOS[key]['name']} ===")
        W = build_world(key, args, np.random.default_rng(0))
        meta[key] = dict(data=W.data_src, bs=W.bs_src, n_bs=int(len(W.bs)), zone_pop=float(W.cpop.sum()),
                         sites=int(len(W.sites)), dem_min=float(W.dem.min()), dem_max=float(W.dem.max()))
        plot_map(W, args)
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            R = realise(W, rng)
            for K in args.K:
                for meth in args.methods:
                    t1 = time.time()
                    res, deliv, ts = simulate(W, R, K, meth, np.random.default_rng(seed), log_ts=(seed == 0))
                    res.update(scenario=key, seed=seed, K=K, method=meth, wall_s=time.time() - t1)
                    rows.append(res)
                    if seed == 0:
                        pd.DataFrame(ts, columns=["t", "delivered", "ELS", "hovering"]).to_csv(
                            os.path.join(args.out, f"ts_{key}_K{K}_{meth}.csv"), index=False)
                        np.save(os.path.join(args.out, f"deliv_{key}_K{K}_{meth}.npy"), deliv)
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.out, "runs.csv"), index=False)
            last = df[(df.scenario == key) & (df.seed == seed)]
            print(f"  seed {seed}: " + ", ".join(f"{r.method}@K{r.K}={r.ELS_norm:.3f}" for r in last.itertuples()
                                                   if r.K == args.K[len(args.K)//2]))
    json.dump(dict(meta=meta, params={k: v for k, v in P.items() if not k.startswith("_")},
                   scenarios=SCENARIOS), open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=str)
    summarise(pd.DataFrame(rows), args)


def summarise(df, args):
    import pandas as pd
    met = ["ELS", "ELS_norm", "DR", "DR_crit", "crit_1h", "delay_med", "plan_ms"]
    s = df.groupby(["scenario", "K", "method"])[met].agg(["mean", "std"]).round(4)
    s.to_csv(os.path.join(args.out, "summary.csv"))
    tests = []
    for (sc, K), d in df.groupby(["scenario", "K"]):
        a = d[d.method == "Proposed"].sort_values("seed").ELS.values
        for m in d.method.unique():
            if m == "Proposed":
                continue
            b = d[d.method == m].sort_values("seed").ELS.values
            if len(a) == len(b) and len(a) >= 5 and np.any(a != b):
                p = wilcoxon(a, b).pvalue
            else:
                p = np.nan
            tests.append(dict(scenario=sc, K=K, baseline=m, gain_pct=100 * (a.mean() - b.mean()) / max(b.mean(), 1e-9),
                              wins=int((a > b).sum()), n=len(a), p_wilcoxon=p))
    pd.DataFrame(tests).round(5).to_csv(os.path.join(args.out, "tests.csv"), index=False)
    print("\n" + s.xs("ELS_norm", axis=1, level=0).to_string())
    print("\n" + pd.DataFrame(tests).round(4).to_string())
    plot_results(df, args)


def plot_map(W, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ext = [0, W.g.W / 1e3, 0, W.g.Hh / 1e3]
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.hillshade(W.dem, vert_exag=2, dx=P["dem_res"], dy=P["dem_res"]), cmap="gray", origin="lower", extent=ext, alpha=0.8)
    pe = ax.imshow(np.log10(W.pop + 1), origin="lower", extent=ext, cmap="YlOrRd", alpha=0.45)
    th = np.linspace(0, 2 * np.pi, 200); z = W.sc["zone"]
    cx, cy = W.g.xy(z["lat"], z["lon"]); r = math.radians(z["rot_deg"])
    u, vv = z["b_km"] * np.cos(th), z["a_km"] * np.sin(th)
    ax.plot(cx / 1e3 + u * math.cos(r) + vv * math.sin(r), cy / 1e3 - u * math.sin(r) + vv * math.cos(r), "b--", lw=1.5, label="affected zone")
    ax.scatter(W.bs[:, 0] / 1e3, W.bs[:, 1] / 1e3, marker="^", s=18, c="k", label=f"BS ({W.bs_src.split()[0]})")
    ax.scatter(*W.base / 1e3, marker="*", s=180, c="lime", edgecolor="k", label=W.sc["base"]["label"])
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_title(W.sc["name"], fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    plt.colorbar(pe, ax=ax, fraction=0.04, label="log10(pop per 0.25 km$^2$+1)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"map_{W.key}.png"), dpi=200); plt.close(fig)


def plot_results(df, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sc, d in df.groupby("scenario"):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for m, dm in d.groupby("method"):
            g = dm.groupby("K").ELS_norm
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker="o", capsize=3, label=m)
        ax.set_xlabel("Number of UAVs K"); ax.set_ylabel("Normalised expected lives saved")
        ax.grid(alpha=.3); ax.legend(fontsize=6); ax.set_title(SCENARIOS[sc]["name"], fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"els_vs_K_{sc}.png"), dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--K", type=int, nargs="+", default=[6, 10, 14])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cells", nargs="*", default=None, help="OpenCelliD CSV(.gz) files")
    ap.add_argument("--pop-tif", nargs="*", default=None, help="local WorldPop tif(s)")
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test (NOT for the paper)")
    ap.add_argument("--cache", default=None, help="shared cache dir for DEM/population")
    ap.add_argument("--set", nargs="*", default=[], help="override parameters, e.g. h_uav=60 act_window_min=240")
    ap.add_argument("--surv-scale", type=float, default=1.0, help="multiply all Weibull scales")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=")
        P[k] = tuple(float(x) for x in v.split(",")) if "," in v else float(v)
        print(f"override {k} = {P[k]}")
    if args.surv_scale != 1.0:
        P["classes"] = {n: (p, lam * args.surv_scale, k) for n, (p, lam, k) in P["classes"].items()}
        print("survival classes:", P["classes"])
    t = time.time(); run(args); print(f"\nTotal {time.time()-t:.0f} s")


if __name__ == "__main__":
    main()
