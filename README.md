# Q-learning_Trival

This repository contains the reproducible code and sample data for a Q-learning based travel route planning experiment.

The project studies how to select tourist attractions under a limited travel-time budget. It combines POI utility scores, dwell time, and inter-attraction commute time to generate feasible routes, then compares the proposed method with several baseline algorithms.

## Repository Contents

```text
.
+-- route_methods.py                # Core route-planning algorithms and metrics
+-- run_experiments.py              # Main experiment runner
+-- data_pipeline.py                # POI and commute-matrix utilities
+-- prepare_data.py                 # Data preparation command-line tools
+-- export_poi_utility_table.py     # Export POI utility table
+-- generate_osrm_road_commute.py   # Generate road-network commute matrix with OSRM
+-- generate_osrm_route_geometry.py # Generate route geometry from OSRM
+-- beijing/                        # Beijing sample POI and commute data
+-- wuhan/                          # Wuhan sample POI and commute data
```

## Main Methods

The core implementation is in `route_methods.py`, including:

- Q-learning route planning
- Greedy utility baseline
- Greedy utility-time ratio baseline
- Ant colony optimization baseline
- Exact dynamic programming route search
- Route evaluation metrics

## Data Format

Each city folder contains:

- `pois.csv`: POI attributes, including id, name, rating, review count, positive rate, dwell time, longitude, latitude, and start-point flag.
- `road_commute.csv`: road-network commute-time matrix.
- `euclidean_commute.csv`: Euclidean-distance commute-time matrix for ablation experiments.

The commute-time matrices are stored in hours.

## Run Experiments

Example for Beijing:

```bash
python run_experiments.py \
  --pois beijing/pois.csv \
  --road-commute beijing/road_commute.csv \
  --euclidean-commute beijing/euclidean_commute.csv \
  --output-dir results/beijing
```

Example for Wuhan:

```bash
python run_experiments.py \
  --pois wuhan/pois.csv \
  --road-commute wuhan/road_commute.csv \
  --euclidean-commute wuhan/euclidean_commute.csv \
  --output-dir results/wuhan
```

The experiment script generates CSV outputs for the main comparison, dynamic programming comparison, budget sensitivity analysis, reward-weight sensitivity analysis, distance-model ablation, robustness analysis, and training history.

## Generate POI Utility Table

```bash
python export_poi_utility_table.py \
  --pois beijing/pois.csv \
  --output results/beijing/table2_poi_utility.csv
```

## Generate Road-Network Matrix

To regenerate a road-network commute matrix with OSRM:

```bash
python generate_osrm_road_commute.py \
  --pois beijing/pois.csv \
  --output beijing/road_commute.csv
```

## Notes

- The current repository includes Beijing and Wuhan sample datasets.
- OSRM-based scripts require access to an OSRM service.
- `prepare_data.py` also supports optional OSM-based matrix generation when `osmnx` and `networkx` are installed.

## License

This repository is intended for academic research and experiment reproduction.
