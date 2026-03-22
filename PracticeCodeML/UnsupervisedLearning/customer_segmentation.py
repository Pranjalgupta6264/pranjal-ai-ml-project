import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Submitted by Pranjal Gupta
# K-Means Clustering - Customer Segmentation

X = np.array([
    [2, 10], [3, 12], [4, 8],
    [10, 25], [11, 30], [12, 28],
    [20, 40], [22, 42], [24, 45]
])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster Labels:")
print(labels)

print("Cluster Centers:")
print(centers)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering: Customer Segmentation")
plt.show()