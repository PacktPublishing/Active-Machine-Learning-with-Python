import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# Generate sample 2D data
np.random.seed(1)
X = np.concatenate((np.random.randn(100, 2) + [2, 2],
                    np.random.randn(50, 2)))

# kNN density
knn = NearestNeighbors(n_neighbors=5).fit(X)
distances, _ = knn.kneighbors(X)
knn_density = 1 / distances.sum(axis=1)

# Kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
kde_dens = np.exp(kde.score_samples(X))

# K-means clustering
km = KMeans(n_clusters=5).fit(X)
km_density = 1 / pairwise_distances(X, km.cluster_centers_).sum(axis=1)

# Maximum Mean Discrepancy
mmd = pairwise_distances(X).mean()
mmd_density = 1 / pairwise_distances(X).sum(axis=1)

# Plot the density estimations
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].scatter(X[:, 0], X[:, 1], c=knn_density)
axs[0, 0].set_title('kNN Density')

axs[0, 1].scatter(X[:, 0], X[:, 1], c=kde_dens)
axs[0, 1].set_title('Kernel Density')

axs[1, 0].scatter(X[:, 0], X[:, 1], c=km_density)
axs[1, 0].set_title('K-Means Density')

axs[1, 1].scatter(X[:, 0], X[:, 1], c=mmd_density)
axs[1, 1].set_title('Maximum mean discrepancy (MMD) Density')

fig.suptitle('Density-Weighted Sampling methods')
plt.show()
