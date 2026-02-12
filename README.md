 Front Propagation Clustering Algorithms

Novel clustering algorithms based on wavefront propagation dynamics with configurable seed selection and speed calculation strategies.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements two innovative clustering algorithms:

1. **Adaptive Front Propagation (AFP)** - Uses local density and propagation speeds to identify and separate clusters
2. **Arrival Time Front Propagation (ATFP)** - Dijkstra-style shortest path propagation for clustering

Both algorithms support multiple seed selection strategies and speed calculation variants for comprehensive ablation studies.

## Features

- 🎯 **6 Seed Selection Strategies**: Random, Density-Maximin, Density Peaks, Speed-Farthest, K-means++, Crowded-Farthest
- ⚡ **8 Speed Calculation Variants**: From simple density to complex log-scaled formulations
- 🔬 **Scikit-learn Compatible**: Familiar API with `fit()` and `fit_predict()` methods
- 📊 **Flexible Configuration**: Easy parameter tuning for different datasets
- 🚀 **Efficient Implementation**: Optimized with priority queues and k-NN graphs

## Installation

### From Source

```bash
git clone https://github.com/yourusername/front-propagation-clustering.git
cd front-propagation-clustering
pip install -r requirements.txt
```

### Using pip (if published to PyPI)

```bash
pip install front-propagation-clustering
```

## Quick Start

```python
from clustering_algorithms_enhanced import AdaptiveFrontPropagation
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Create and fit the model
afp = AdaptiveFrontPropagation(
    n_clusters=3,
    seed_method='speed_farthest',
    speed_variant='rho_sigma',
    k_neighbors=15,
    random_state=42
)

# Get cluster labels
labels = afp.fit_predict(X)
```

## Algorithm Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 2 | Number of clusters to form |
| `k_neighbors` | int | 15 | Number of neighbors for local statistics |
| `seed_method` | str | 'speed_farthest' | Seed selection strategy |
| `speed_variant` | str | 'original' | Speed calculation variant |
| `random_state` | int | 42 | Random seed for reproducibility |

### Seed Selection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `random` | Pure random selection | Baseline comparison |
| `density_maximin` | Balanced density + distance | General purpose |
| `density_peaks` | Rodriguez-Laio 2014 method | Well-separated clusters |
| `speed_farthest` | Speed-based farthest point | **Recommended default** |
| `kmeans_plusplus` | K-means++ probabilistic | Similar to k-means cases |
| `crowded_farthest` | Diversity in crowded regions | Dense, overlapping clusters |

### Speed Calculation Variants

| Variant | Formula | Characteristics |
|---------|---------|-----------------|
| `original` | ρ | Simple density-based |
| `rho_sigma` | ρ × exp(-σ_norm) | Penalizes high variance |
| `sigmoid_sigma` | ρ × sigmoid(-σ_norm) | Smooth transition |
| `no_normalization` | ρ × exp(-σ) | Raw sigma penalty |
| `quadratic_sigma` | ρ × exp(-σ²_norm) | Stronger penalty |
| `inverse_sigma` | ρ / (σ_norm + 1) | Alternative penalty |
| `sqrt_sigma` | ρ × exp(-√σ_norm) | Weaker penalty |
| `log_density` | log(1+ρ) × exp(-γσ_norm) | Distance-modulated |

## Examples

### Basic Usage

```python
from clustering_algorithms_enhanced import (
    AdaptiveFrontPropagation,
    ArrivalTimeFrontPropagation
)
import numpy as np

# AFP with custom parameters
afp = AdaptiveFrontPropagation(
    n_clusters=4,
    seed_method='density_peaks',
    speed_variant='sigmoid_sigma',
    k_neighbors=20
)
labels_afp = afp.fit_predict(X)

# ATFP with different settings
atfp = ArrivalTimeFrontPropagation(
    n_clusters=4,
    seed_method='kmeans_plusplus',
    speed_variant='rho_sigma'
)
labels_atfp = atfp.fit_predict(X)
```

### Comparing Different Configurations

```python
from sklearn.metrics import adjusted_rand_score

configurations = [
    ('speed_farthest', 'original'),
    ('speed_farthest', 'rho_sigma'),
    ('density_peaks', 'sigmoid_sigma'),
    ('kmeans_plusplus', 'original')
]

for seed_method, speed_variant in configurations:
    afp = AdaptiveFrontPropagation(
        n_clusters=3,
        seed_method=seed_method,
        speed_variant=speed_variant
    )
    labels = afp.fit_predict(X)
    score = adjusted_rand_score(y_true, labels)
    print(f"{seed_method:20s} | {speed_variant:20s} | ARI: {score:.3f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

afp = AdaptiveFrontPropagation(n_clusters=3)
labels = afp.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Adaptive Front Propagation Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
```

## Algorithm Details

### Adaptive Front Propagation (AFP)

AFP uses a priority queue to propagate cluster labels from seed points based on local speed values. Points with higher speeds propagate their labels more effectively.

**Key Steps:**
1. Compute local density (ρ) and statistics (σ, d_avg) for each point
2. Calculate propagation speeds using the selected variant
3. Select seed points using the chosen strategy
4. Propagate labels via priority queue based on speeds

### Arrival Time Front Propagation (ATFP)

ATFP uses a Dijkstra-style approach where arrival times determine cluster assignment. The first wavefront to reach a point assigns its cluster label.

**Key Steps:**
1. Compute local density and statistics
2. Calculate propagation speeds
3. Select seed points
4. Propagate using arrival times: t_new = t_current + distance/speed

## Performance Considerations

- **Time Complexity**: O(n log n) for priority queue operations
- **Space Complexity**: O(n × k) for k-NN graph storage
- **Recommended k_neighbors**: 10-20 for most datasets
- **Scalability**: Efficient for datasets up to 100K points

## Citation

If you use this code in your research, please cite:

```bibtex
@software{front_propagation_clustering,
  author = {Your Name},
  title = {Front Propagation Clustering Algorithms},
  year = {2025},
  url = {https://github.com/yourusername/front-propagation-clustering}
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Density Peaks method inspired by Rodriguez & Laio (2014)
- K-means++ seeding from Arthur & Vassilvitskii (2007)
- Crowding distance from NSGA-II (Deb et al., 2002)

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/front-propagation-clustering](https://github.com/yourusername/front-propagation-clustering)

## Changelog

### Version 1.0.0 (2025-02-12)
- Initial release
- Implementation of AFP and ATFP algorithms
- 6 seed selection strategies
- 8 speed calculation variants
- Scikit-learn compatible API

