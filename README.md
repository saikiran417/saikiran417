import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare the data
# Sample data - this would be replaced by your actual data
# Assume we have a DataFrame with columns: 'CustomerID', 'AnnualIncome', 'SpendingScore'
data = {
    'CustomerID': range(1, 11),
    'AnnualIncome': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)

# Dropping CustomerID as it's not relevant for clustering
X = df.drop('CustomerID', axis=1)

# Step 2: Preprocess the data
# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Implement K-means clustering
# Finding the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sums of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method result
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow plot, choose the optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Adding the cluster assignments to the original DataFrame
df['Cluster'] = y_kmeans

# Step 4: Evaluate and interpret the clusters
print(df)

# Plotting the clusters
plt.figure(figsize=(10, 5))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()

