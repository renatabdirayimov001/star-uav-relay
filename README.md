# STAR: survival-time-aware UAV swarm relaying

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24433055.svg)](https://doi.org/10.5281/zenodo.24433055)

Code and results for the paper

> R. Abdirayimov, "Survival-time-aware UAV swarm relaying for life-critical communications in infrastructure-damaged areas: Terrain-realistic case studies from Uzbekistan and Central Asia". Manuscript under preparation.

The simulator evaluates emergency UAV relay networks by the **expected number of lives saved (ELS)**. It couples SOS delivery times with severity-dependent survival curves and compares them with the time ground teams would need without the network. STAR, the proposed planner, grows a gateway-rooted relay tree every 15 min with survival-decay weights and a served-time memory.

## Scenarios

| Scenario | Type | Base stations |
|---|---|---|
| `sardoba` | replay of the 1 May 2020 Sardoba dam breach (Uzbekistan and Kazakhstan) | OpenCelliD (188 sites) |
| `fergana` | hypothetical earthquake, Namangan foothills | population-driven layout (OpenCelliD too sparse) |
| `karakalpak` | hypothetical flood, Amu Darya delta near Muynak | population-driven layout (OpenCelliD too sparse) |

Coordinates, affected zones and all model parameters are defined at the top of `fvv_sim.py` (`SCENARIOS` and `P`).

## Data sources

| Data | Source | Licence | How it is obtained |
|---|---|---|---|
| Terrain | Terrain Tiles on AWS (Terrarium, SRTM-based) | public | downloaded automatically |
| Population | WorldPop 2020, 1 km | CC BY 4.0 | downloaded automatically |
| Base stations | OpenCelliD (MCC 434, 401, 437) | CC BY-SA 4.0 | **download yourself** from https://opencellid.org (free account) and pass with `--cells` |

OpenCelliD files are not redistributed in this repository.

## Running

Requirements: Python 3.10+, see `requirements.txt`. Internet access is needed on the first run to fetch terrain and population (cached afterwards).

```bash
pip install -r requirements.txt
# main experiment: 3 scenarios x K in {6,10,14} x 30 seeds x 8 methods
python fvv_sim.py --out results_run --cells opencellid/434.csv.gz opencellid/401.csv.gz opencellid/437.csv.gz
# one sensitivity setting, e.g. 60 m altitude
python fvv_sim.py --K 10 --seeds 15 --set h_uav=60 --out sens_h60 --cells opencellid/*.csv.gz
# offline smoke test (synthetic terrain, not used in the paper)
python fvv_sim.py --synthetic --seeds 2 --K 6 --out test_results
```

On Kaggle, import `notebooks/fvv_kaggle.ipynb` (main experiment) or `notebooks/fvv_sensitivity.ipynb` (sensitivity), attach a dataset with the OpenCelliD files, turn on Internet and run all cells. The main experiment takes about 90 min on a CPU session.

`make_figs.py` reproduces the figures from the files in `results/` and a cached run (paths at the top of the script).

## Methods implemented

`Proposed` (STAR), `ConnCov-dynamic` (count weights), `Proposed-noMemory`, `Proposed-noTerrain`, `Proposed-static`, `ConnCov-static`, `KMeans`, `Ferry` (store-carry-forward).

## Results included

- `results/main/`: per-run metrics (`runs.csv`), means and standard deviations (`summary.csv`), paired Wilcoxon tests (`tests.csv`), scenario metadata (`meta.json`).
- `results/sensitivity/`: eight one-at-a-time settings (altitude, survival scale, SOS onset window, ground discovery time).
- `figures/`: figures used in the paper.

## Citation

If you use this code, please cite the paper and the archived software: R. Abdirayimov, STAR simulator and results, Zenodo, doi:10.5281/zenodo.24433055 (see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). Result files: CC BY 4.0.
