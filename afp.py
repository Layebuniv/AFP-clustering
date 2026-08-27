"""
Adaptive Front Propagation (AFP) Clustering
===========================================

A single, unified front-propagation clustering framework that
includes:

  * 8 speed modulation strategies
  * 7 seed selection strategies (the original 6 plus component-
    size-gated seeding via ``seed_strategy="gated"``)
  * An optional post-propagation noise/straggler resolution pass
    (``resolve_unlabeled``) using component-size and relative-
    density-gap criteria.

Author: Abdesslem Layeb 
License: MIT
"""

import numpy as np
import heapq
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


class AdaptiveFrontPropagation:

    def __init__(self,
                 n_clusters,
                 k_neighbors=15,
                 speed_variant="density",
                 seed_strategy="gated",
                 min_cluster_size=None,
                 min_cluster_frac=0.005,
                 relative_gap=0.4,
                 resolve_unlabeled=False,
                 random_state=42):

        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.speed_variant = speed_variant
        self.seed_strategy = seed_strategy
        self.min_cluster_size = min_cluster_size
        self.min_cluster_frac = min_cluster_frac
        self.relative_gap = relative_gap
        self.resolve_unlabeled = resolve_unlabeled
        self.random_state = random_state
        self.labels_ = None

    # ==========================================================
    # LOCAL STATISTICS
    # ==========================================================

    def _local_stats(self, X):
        k = min(self.k_neighbors, len(X))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, idx = nbrs.kneighbors(X)

        d_avg = dists.mean(axis=1)
        rho = 1.0 / (d_avg + 1e-9)
        sigma = dists.std(axis=1)

        return rho, sigma, d_avg, idx

    # ==========================================================
    # SPEED VARIANTS
    # ==========================================================

    def _speed(self, rho, sigma, d_avg):
        EPS = 1e-9

        rho_safe = np.maximum(rho, EPS)
        sigma_safe = np.maximum(sigma, EPS)

        if self.speed_variant == "density":
            return rho_safe

        elif self.speed_variant == "rho_sigma":
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return rho_safe * np.exp(-sigma_norm)

        elif self.speed_variant == "sigmoid_sigma":
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return rho_safe * (1.0 / (1.0 + np.exp(0.5 * (sigma_norm - 0.5))))

        elif self.speed_variant == "no_normalization":
            sigma_clipped = np.minimum(sigma_safe, 50.0)
            return rho_safe * np.exp(-sigma_clipped)

        elif self.speed_variant == "quadratic_sigma":
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return rho_safe * np.exp(-sigma_norm ** 2)

        elif self.speed_variant == "inverse_sigma":
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return rho_safe / (sigma_norm + 1.0)

        elif self.speed_variant == "sqrt_sigma":
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return rho_safe * np.exp(-np.sqrt(sigma_norm))

        elif self.speed_variant == "log_density":
            gamma = np.clip(d_avg, 0.1, 10.0)
            sigma_norm = sigma_safe / (sigma_safe.max() + EPS)
            return np.log1p(rho_safe) * np.exp(-gamma * sigma_norm)

        else:
            raise ValueError(f"Unknown speed variant: {self.speed_variant}")

    # ==========================================================
    # SEED DISPATCHER
    # ==========================================================

    def _select_seeds(self, X, rho, speeds, knn_idx=None):
        if self.seed_strategy == "random":
            return self._select_seeds_random(X)

        elif self.seed_strategy == "density_maximin":
            return self._select_seeds_density_maximin(X, rho)

        elif self.seed_strategy == "density_peaks":
            return self._select_seeds_density_peaks(X, rho)

        elif self.seed_strategy == "speed_farthest":
            return self._select_seeds_speed_farthest(X, speeds)

        elif self.seed_strategy == "kmeans_plusplus":
            return self._select_seeds_kmeans_plusplus(X, rho)

        elif self.seed_strategy == "crowded_farthest":
            return self._select_seeds_crowded_farthest(X)

        elif self.seed_strategy == "gated":
            if knn_idx is None:
                raise ValueError("knn_idx is required for the 'gated' seed strategy")
            return self._select_seeds_gated(X, rho, speeds, knn_idx)

        else:
            raise ValueError(f"Unknown seed strategy: {self.seed_strategy}")

    # ==========================================================
    # SEED STRATEGIES
    # ==========================================================

    def _select_seeds_random(self, X):
        rng = np.random.RandomState(self.random_state)
        return rng.choice(len(X), self.n_clusters, replace=False).tolist()

    def _select_seeds_density_maximin(self, X, rho):
        seeds = [np.argmax(rho)]

        for _ in range(self.n_clusters - 1):
            dists = np.min([np.linalg.norm(X - X[s], axis=1)
                            for s in seeds], axis=0)
            score = dists * rho
            seeds.append(np.argmax(score))

        return seeds

    def _select_seeds_density_peaks(self, X, rho):
        delta = np.zeros(len(X))

        for i in range(len(X)):
            higher = np.where(rho > rho[i])[0]
            if len(higher) > 0:
                delta[i] = np.min(np.linalg.norm(X[i] - X[higher], axis=1))
            else:
                delta[i] = np.max(np.linalg.norm(X[i] - X, axis=1))

        gamma = rho * delta
        return np.argsort(-gamma)[:self.n_clusters].tolist()

    def _select_seeds_speed_farthest(self, X, speeds):
        seeds = [np.argmax(speeds)]

        while len(seeds) < self.n_clusters:
            dists = np.min([np.linalg.norm(X - X[s], axis=1)
                            for s in seeds], axis=0)
            score = speeds * (dists ** 1.5)
            seeds.append(np.argmax(score))

        return seeds

    def _select_seeds_kmeans_plusplus(self, X, rho):
        rng = np.random.RandomState(self.random_state)
        seeds = [np.argmax(rho)]

        for _ in range(self.n_clusters - 1):
            dists_sq = np.min([np.linalg.norm(X - X[s], axis=1)**2
                               for s in seeds], axis=0)
            probs = dists_sq / np.sum(dists_sq)
            next_seed = rng.choice(len(X), p=probs)
            seeds.append(next_seed)

        return seeds

    def _crowding_distance(self, X):
        n, m = X.shape
        distances = np.zeros(n)

        for i in range(m):
            order = np.argsort(X[:, i])
            distances[order[0]] = distances[order[-1]] = np.inf

            for j in range(1, n - 1):
                distances[order[j]] += (
                    X[order[j+1], i] - X[order[j-1], i]
                )

        return distances

    def _select_seeds_crowded_farthest(self, X):
        crowd = self._crowding_distance(X)
        crowd = np.maximum(crowd, 1e-9)

        seeds = [np.argmin(crowd)]

        while len(seeds) < self.n_clusters:
            dists = np.min([np.linalg.norm(X - X[s], axis=1)
                            for s in seeds], axis=0)
            score = dists / crowd
            seeds.append(np.argmax(score))

        return seeds

    # ==========================================================
    # CONNECTED COMPONENTS (for gated seeding & noise resolution)
    # ==========================================================

    def _connected_components(self, n, knn_idx):
        k = knn_idx.shape[1]
        rows = np.repeat(np.arange(n), k)
        cols = knn_idx.ravel()
        data = np.ones(len(rows))
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        A = A.maximum(A.T)
        n_comp, comp_labels = connected_components(A, directed=False)
        return n_comp, comp_labels

    # ==========================================================
    # GATED SEEDING
    # ==========================================================

    def _select_by_strategy(self, X, rho, speeds, index_pool, n_seeds=None):
        """Speed-farthest seeding restricted to `index_pool`."""
        n_seeds = n_seeds if n_seeds is not None else self.n_clusters
        pool = np.asarray(index_pool)
        if len(pool) <= n_seeds:
            return pool.tolist()
        X_sub, speeds_sub = X[pool], speeds[pool]

        local = [int(np.argmax(speeds_sub))]
        while len(local) < n_seeds:
            dists = np.min([np.linalg.norm(X_sub - X_sub[s], axis=1) for s in local], axis=0)
            score = speeds_sub * (dists ** 1.5)
            score[local] = -np.inf
            local.append(int(np.argmax(score)))
        return pool[local].tolist()

    def _select_seeds_gated(self, X, rho, speeds, knn_idx):
        n = len(X)
        n_comp, comp_labels = self._connected_components(n, knn_idx)
        comp_sizes = np.bincount(comp_labels, minlength=n_comp)

        min_size = self.min_cluster_size
        if min_size is None:
            min_size = max(2, int(self.min_cluster_frac * n))

        self._n_components_ = n_comp
        self._component_labels_ = comp_labels
        self._comp_sizes_ = comp_sizes
        self._min_cluster_size_ = min_size

        if n_comp == 1:
            return self._select_by_strategy(X, rho, speeds, list(range(n)))

        eligible_comps = np.where(comp_sizes >= min_size)[0]

        seeds = []
        if len(eligible_comps) >= self.n_clusters:
            top = eligible_comps[np.argsort(-comp_sizes[eligible_comps])[: self.n_clusters]]
            for c in top:
                pool = np.where(comp_labels == c)[0].tolist()
                seeds.append(self._select_by_strategy(X, rho, speeds, pool, n_seeds=1)[0])
        else:
            for c in eligible_comps:
                pool = np.where(comp_labels == c)[0].tolist()
                seeds.append(self._select_by_strategy(X, rho, speeds, pool, n_seeds=1)[0])

            remaining = self.n_clusters - len(seeds)
            if remaining > 0 and len(eligible_comps) > 0:
                big_pool = np.where(np.isin(comp_labels, eligible_comps))[0].tolist()
                extra = self._select_by_strategy(
                    X, rho, speeds, [i for i in big_pool if i not in seeds],
                    n_seeds=remaining
                )
                seeds.extend(extra)
            elif remaining > 0:
                pool = [i for i in range(n) if i not in seeds]
                extra = self._select_by_strategy(X, rho, speeds, pool, n_seeds=remaining)
                seeds.extend(extra)

        return seeds

    # ==========================================================
    # NOISE / STRAGGLER RESOLUTION (optional)
    # ==========================================================

    def _resolve_unlabeled(self, X, labels, rho, knn_idx):
        n = len(X)

        # Use cached component info from gated seeding if available
        if hasattr(self, '_component_labels_'):
            comp_labels = self._component_labels_
            comp_sizes = self._comp_sizes_
            min_size = self._min_cluster_size_
        else:
            n_comp, comp_labels = self._connected_components(n, knn_idx)
            comp_sizes = np.bincount(comp_labels, minlength=n_comp)
            min_size = self.min_cluster_size
            if min_size is None:
                min_size = max(2, int(self.min_cluster_frac * n))

        unlabeled_idx = np.where(labels == -1)[0]
        if len(unlabeled_idx) == 0:
            self.noise_count_ = 0
            self.unreachable_count_ = 0
            return labels

        labeled_idx = np.where(labels != -1)[0]
        if len(labeled_idx) == 0:
            self.noise_count_ = len(unlabeled_idx)
            self.unreachable_count_ = 0
            return labels

        cluster_ids = np.unique(labels[labeled_idx])
        cluster_core_rho = {cid: np.median(rho[labels == cid]) for cid in cluster_ids}

        noise_count = 0
        unreachable_count = 0

        for c in np.unique(comp_labels):
            comp_mask = comp_labels == c
            comp_unlabeled = np.where(comp_mask & (labels == -1))[0]
            if len(comp_unlabeled) == 0:
                continue

            if comp_sizes[c] < min_size:
                labels[comp_unlabeled] = -1
                noise_count += len(comp_unlabeled)
                continue

            comp_labeled = np.where(comp_mask & (labels != -1))[0]
            if len(comp_labeled) == 0:
                labels[comp_unlabeled] = -1
                noise_count += len(comp_unlabeled)
                continue

            dists = cdist(X[comp_unlabeled], X[comp_labeled])
            for ui, i in enumerate(comp_unlabeled):
                nearest = comp_labeled[np.argmin(dists[ui])]
                candidate_cluster = labels[nearest]
                is_sparse = rho[i] < self.relative_gap * cluster_core_rho[candidate_cluster]

                if is_sparse:
                    labels[i] = -1
                    noise_count += 1
                else:
                    labels[i] = candidate_cluster
                    unreachable_count += 1

        self.noise_count_ = noise_count
        self.unreachable_count_ = unreachable_count
        return labels

    # ==========================================================
    # FIT (PROPAGATION)
    # ==========================================================

    def fit(self, X):
        X = np.asarray(X)
        n = len(X)

        rho, sigma, d_avg, knn_idx = self._local_stats(X)
        speeds = self._speed(rho, sigma, d_avg)

        seeds = self._select_seeds(X, rho, speeds, knn_idx)

        labels = -np.ones(n, dtype=int)
        arrival = np.zeros(n)
        pq = []

        for cid, s in enumerate(seeds):
            labels[s] = cid
            arrival[s] = speeds[s]
            heapq.heappush(pq, (-speeds[s], s, cid))

        while pq:
            neg_v, i, cid = heapq.heappop(pq)
            for j in knn_idx[i]:
                if labels[j] == -1 or speeds[j] > arrival[j]:
                    labels[j] = cid
                    arrival[j] = speeds[j]
                    heapq.heappush(pq, (-speeds[j], j, cid))

        if self.resolve_unlabeled:
            labels = self._resolve_unlabeled(X, labels, rho, knn_idx)
        else:
            self.noise_count_ = None
            self.unreachable_count_ = None

        self.labels_ = labels
        self.seeds_ = seeds
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)

    # Classic behaviour (non-gated seeding, no resolution)
    base = AdaptiveFrontPropagation(
        n_clusters=4,
        seed_strategy="speed_farthest",
        resolve_unlabeled=False
    ).fit(X)

    # Gated seeding + optional noise resolution
    resolved = AdaptiveFrontPropagation(
        n_clusters=4,
        seed_strategy="gated",
        resolve_unlabeled=True
    ).fit(X)

    # Gated seeding, but resolution disabled
    gated_only = AdaptiveFrontPropagation(
        n_clusters=4,
        seed_strategy="gated",
        resolve_unlabeled=False
    ).fit(X)

    print("base labels (unique):", np.unique(base.labels_))
    print("resolved labels (unique):", np.unique(resolved.labels_))
    print("resolved noise/unreachable:", resolved.noise_count_, resolved.unreachable_count_)
    print("gated_only labels (unique):", np.unique(gated_only.labels_))
    print("gated_only noise/unreachable (should be None, None):",
          gated_only.noise_count_, gated_only.unreachable_count_)
