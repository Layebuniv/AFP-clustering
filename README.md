# Front Propagation Clustering Algorithms

Novel clustering algorithms based on wavefront propagation dynamics, with configurable seed selection and speed calculation strategies.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements two front-propagation clustering algorithms:

1. **Adaptive Front Propagation (AFP)** (`afp.py`) — greedy, speed-modulated propagation from seed points via a priority queue.
2. **Arrival Time Front Propagation (ATFP)** (`atfp.py`) — true Dijkstra-style shortest-arrival-time propagation.

Both support multiple seed selection strategies and speed calculation variants for ablation studies, plus connectivity-aware seeding and noise resolution.

## What's new in this version

- **Gated seeding** (`seed_strategy="gated"`) — a 7th seed strategy that restricts seed placement to connected components above a minimum size, using `scipy.sparse.csgraph.connected_components`.
- **Noise/straggler resolution** (`resolve_unlabeled=True`) — an optional post-propagation pass that reassigns or discards unlabeled points using component-size gating and a relative-density-gap criterion.
- **ATFP propagation fix** — the arrival-time loop is now true Dijkstra (a label-locking bug previously made it behave as greedy BFS).
- Notebooks (`Full_AFP.ipynb`, `Full_ATFP.ipynb`) replaced with standalone, importable modules (`afp.py`, `atfp.py`).

## Features

- 🎯 **7 Seed Selection Strategies**: Random, Density-Maximin, Density Peaks, Speed-Farthest, K-means++, Crowded-Farthest, Gated (component-size-aware)
- ⚡ **8 Speed Calculation Variants**: From simple density to log-scaled formulations
- 🧹 **Optional Noise Resolution**: Component-size gating + relative-density-gap criterion
- 🔬 **Scikit-learn Compatible**: Familiar API with `fit()` and `fit_predict()` methods
- 📊 **Flexible Configuration**: Easy parameter tuning for different datasets

## Installation

```bash
git clone https://github.com/Layebuniv/AFP-clustering.git
cd AFP-clustering
pip install numpy scipy scikit-learn
```

No package build needed — `afp.py` and `atfp.py` can be imported directly once cloned.

## Quick Start

```python
from afp import AdaptiveFrontPropagation
from atfp import ArrivalTimeFrontPropagation
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

model = AdaptiveFrontPropagation(
    n_clusters=3,
    seed_strategy="gated",
    speed_variant="density",
    k_neighbors=15,
    resolve_unlabeled=True,
    random_state=42
)

labels = model.fit_predict(X)
```

## Algorithm Parameters

### Common Parameters

| Parameter            | Type  | Default    | Description                                                        |
| --------------------- | ----- | ---------- | -------------------------------------------------------------------- |
| `n_clusters`           | int   | required   | Number of clusters to form                                          |
| `k_neighbors`          | int   | 15         | Number of neighbors for local statistics                            |
| `seed_strategy`        | str   | 'gated'    | Seed selection strategy                                              |
| `speed_variant`        | str   | 'density'  | Speed calculation variant                                            |
| `min_cluster_size`     | int   | None       | Minimum component size for gated seeding / noise resolution         |
| `min_cluster_frac`     | float | 0.005      | Used to derive `min_cluster_size` when not set explicitly           |
| `relative_gap`         | float | 0.4        | Density-gap threshold used by noise resolution                      |
| `resolve_unlabeled`    | bool  | False      | Whether to run the post-propagation noise/straggler resolution pass |
| `random_state`         | int   | 42         | Random seed for reproducibility                                     |

### Seed Selection Strategies

| Strategy            | Description                                       | Best For                    |
| -------------------- | -------------------------------------------------- | ---------------------------- |
| `random`              | Pure random selection                              | Baseline comparison          |
| `density_maximin`     | Balanced density + distance                        | General purpose               |
| `density_peaks`       | Rodriguez-Laio (2014) method                       | Well-separated clusters      |
| `speed_farthest`      | Speed-based farthest point                         | Recommended default          |
| `kmeans_plusplus`     | K-means++ probabilistic                            | Similar to k-means cases     |
| `crowded_farthest`    | Diversity in crowded regions                       | Dense, overlapping clusters  |
| `gated`               | Component-size-gated seeding                       | Noisy / disconnected data    |

### Speed Calculation Variants

| Variant              | Formula                     | Characteristics          |
| --------------------- | ---------------------------- | -------------------------- |
| `density`              | ρ                             | Simple density-based       |
| `rho_sigma`            | ρ × exp(-σ_norm)              | Penalizes high variance    |
| `sigmoid_sigma`        | ρ × sigmoid(-σ_norm)          | Smooth transition          |
| `no_normalization`     | ρ × exp(-σ)                   | Raw sigma penalty          |
| `quadratic_sigma`      | ρ × exp(-σ²_norm)             | Stronger penalty           |
| `inverse_sigma`        | ρ / (σ_norm + 1)              | Alternative penalty        |
| `sqrt_sigma`           | ρ × exp(-√σ_norm)             | Weaker penalty             |
| `log_density`          | log(1+ρ) × exp(-γσ_norm)      | Distance-modulated         |

## Examples

### Basic Usage

```python
from afp import AdaptiveFrontPropagation
from atfp import ArrivalTimeFrontPropagation

# AFP with custom parameters
afp = AdaptiveFrontPropagation(
    n_clusters=4,
    seed_strategy="density_peaks",
    speed_variant="sigmoid_sigma",
    k_neighbors=20
)
labels_afp = afp.fit_predict(X)

# ATFP with gated seeding and noise resolution
atfp = ArrivalTimeFrontPropagation(
    n_clusters=4,
    seed_strategy="gated",
    speed_variant="rho_sigma",
    resolve_unlabeled=True
)
labels_atfp = atfp.fit_predict(X)
print(atfp.noise_count_, atfp.unreachable_count_)
```

### Comparing Different Configurations

```python
from sklearn.metrics import adjusted_rand_score

configurations = [
    ("speed_farthest", "density"),
    ("speed_farthest", "rho_sigma"),
    ("density_peaks", "sigmoid_sigma"),
    ("gated", "density"),
]

for seed_strategy, speed_variant in configurations:
    afp = AdaptiveFrontPropagation(
        n_clusters=3,
        seed_strategy=seed_strategy,
        speed_variant=speed_variant
    )
    labels = afp.fit_predict(X)
    score = adjusted_rand_score(y_true, labels)
    print(f"{seed_strategy:20s} | {speed_variant:20s} | ARI: {score:.3f}")
```

## Algorithm Details

### Adaptive Front Propagation (AFP)

AFP uses a priority queue to propagate cluster labels from seed points based on local speed values. Points with higher speeds propagate their labels more effectively.

**Key steps:**
1. Compute local density (ρ) and statistics (σ, d_avg) for each point.
2. Calculate propagation speeds using the selected variant.
3. Select seed points using the chosen strategy (optionally gated by connected components).
4. Propagate labels via a priority queue, ordered by speed.
5. Optionally resolve remaining unlabeled points using component-size and relative-density-gap criteria.

### Arrival Time Front Propagation (ATFP)

ATFP uses a Dijkstra-style approach where arrival time determines cluster assignment — the first wavefront to reach a point assigns its cluster label.

**Key steps:**
1. Compute local density and statistics.
2. Calculate propagation speeds.
3. Select seed points (optionally gated by connected components).
4. Propagate using arrival times: `t_new = t_current + distance / speed`.
5. Optionally resolve remaining unlabeled points as in AFP.

## Performance Considerations

- **Time Complexity**: O(n log n) for priority queue operations
- **Space Complexity**: O(n × k) for k-NN graph storage
- **Recommended `k_neighbors`**: 10–20 for most datasets
- **Scalability**: Efficient for datasets up to ~100K points

## Citation

If you use this code in your research, please cite:

```bibtex
@software{front_propagation_clustering,
  author = {Abdesslem Layeb},
  title  = {Front Propagation Clustering Algorithms (AFP / ATFP)},
  year   = {2026},
  url    = {https://github.com/Layebuniv/AFP-clustering}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Density Peaks method inspired by Rodriguez & Laio (2014)
- K-means++ seeding from Arthur & Vassilvitskii (2007)
- Crowding distance from NSGA-II (Deb et al., 2002)

## Author

Abdesslem Layeb — LISIA Laboratory, Faculté NTIC, Université Constantine 2 – Abdelhamid Mehri, Algeria

## Changelog

### Version 2.0.0 (2026-08-27)
- Added gated (component-size-aware) seeding strategy
- Added optional noise/straggler resolution pass
- Fixed ATFP propagation loop to be true Dijkstra
- Replaced notebooks with standalone `afp.py` / `atfp.py` modules

### Version 1.0.0 (2025-02-12)
- Initial release
- AFP and ATFP algorithms, 6 seed strategies, 8 speed variants
