"""
Task 1: Basel Climate Dataset Clustering
Student: Average Level Implementation
This script performs clustering analysis on the Basel climate dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
os.makedirs('task1data', exist_ok=True)
os.makedirs('task1plt', exist_ok=True)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 80)
print("TASK 1: BASEL CLIMATE DATASET CLUSTERING")
print("=" * 80)

# Define column names based on the assignment description
column_names = [
    'Temp_Min', 'Temp_Max', 'Temp_Mean',
    'Humidity_Min', 'Humidity_Max', 'Humidity_Mean',
    'Pressure_Min', 'Pressure_Max', 'Pressure_Mean',
    'Precipitation', 'Snowfall', 'Sunshine',
    'WindGust_Min', 'WindGust_Max', 'WindGust_Mean',
    'WindSpeed_Min', 'WindSpeed_Max', 'WindSpeed_Mean'
]

# Load the dataset
df = pd.read_csv('ClimateDataBasel.csv', names=column_names)
print(f"\n1. Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display first few rows
print("\n   First 5 rows:")
print(df.head())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# 2.1 Check for missing values
print("\n2.1 Checking for missing values...")
missing_values = df.isnull().sum()
print(f"    Total missing values: {missing_values.sum()}")
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("    No missing values found!")

# 2.2 Basic statistics
print("\n2.2 Basic Statistics:")
print(df.describe())

# 2.3 Outlier Detection using IQR method
print("\n2.3 Outlier Detection (using IQR method)...")
outlier_counts = {}
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_counts[col] = outliers

print("    Outliers per feature:")
for col, count in outlier_counts.items():
    if count > 0:
        print(f"    {col}: {count} outliers")

# For this simple implementation, we keep outliers but note them
# In a real scenario, we might remove or transform them
print(f"\n    Note: Keeping outliers in data for this analysis")

# 2.4 Standardization
# This is important because features have different scales
print("\n2.4 Standardization...")
print("    Applying StandardScaler to normalize features")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=column_names)
print("    Standardization completed!")
print(f"    Mean values after scaling (should be ~0): {df_scaled.mean().mean():.6f}")
print(f"    Std values after scaling (should be ~1): {df_scaled.std().mean():.6f}")

# 2.5 Dimensionality Reduction for Visualization
print("\n2.5 Dimensionality Reduction with PCA...")
print("    Reducing 18 dimensions to 2 for visualization")
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled.values)
print(f"    Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"    Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# ============================================================================
# 3. CLUSTERING ALGORITHM 1: K-MEANS
# ============================================================================
print("\n" + "=" * 80)
print("3. CLUSTERING ALGORITHM 1: K-MEANS")
print("=" * 80)

print("\nK-Means is a simple and popular clustering algorithm.")
print("It partitions data into K clusters by minimizing within-cluster variance.")

# 3.1 Determine optimal number of clusters using Elbow Method
print("\n3.1 Finding optimal number of clusters (Elbow Method)...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True)
plt.tight_layout()
plt.savefig('task1plt/kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
print("    Saved elbow method plot to 'task1plt/kmeans_elbow_method.png'")

# 3.2 Apply K-Means with optimal K (choosing K=3 as reasonable choice)
optimal_k = 3
print(f"\n3.2 Applying K-Means with K={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

# Evaluate K-Means
kmeans_silhouette = silhouette_score(df_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(df_scaled, kmeans_labels)

print(f"    K-Means Results:")
print(f"    - Silhouette Score: {kmeans_silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"    - Davies-Bouldin Index: {kmeans_db:.4f} (lower is better)")
print(f"    - Cluster sizes: {np.bincount(kmeans_labels)}")

# ============================================================================
# 4. CLUSTERING ALGORITHM 2: DBSCAN
# ============================================================================
print("\n" + "=" * 80)
print("4. CLUSTERING ALGORITHM 2: DBSCAN")
print("=" * 80)

print("\nDBSCAN (Density-Based Spatial Clustering) is different from K-Means.")
print("It can find clusters of arbitrary shape and identify outliers as noise.")

# 4.1 Apply DBSCAN
# Parameters chosen through simple experimentation
print("\n4.1 Applying DBSCAN...")
print("    Parameters: eps=2.5, min_samples=10")
dbscan = DBSCAN(eps=2.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(df_scaled)

# Count clusters and noise points
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\n    DBSCAN Results:")
print(f"    - Number of clusters found: {n_clusters_dbscan}")
print(f"    - Number of noise points: {n_noise}")
print(f"    - Cluster sizes: {np.bincount(dbscan_labels[dbscan_labels >= 0])}")

# Evaluate DBSCAN (only if we have more than 1 cluster and not all noise)
if n_clusters_dbscan > 1 and n_noise < len(dbscan_labels):
    # Filter out noise points for evaluation
    valid_mask = dbscan_labels != -1
    if valid_mask.sum() > 0:
        dbscan_silhouette = silhouette_score(df_scaled[valid_mask], dbscan_labels[valid_mask])
        dbscan_db = davies_bouldin_score(df_scaled[valid_mask], dbscan_labels[valid_mask])
        print(f"    - Silhouette Score: {dbscan_silhouette:.4f} (excluding noise)")
        print(f"    - Davies-Bouldin Index: {dbscan_db:.4f} (excluding noise)")
    else:
        print("    - Cannot compute metrics: all points are noise")
else:
    print("    - Cannot compute metrics: only 1 cluster or all noise")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("5. VISUALIZATION")
print("=" * 80)

# 5.1 Visualize clusters in 2D PCA space
print("\n5.1 Creating cluster visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# K-Means visualization
axes[0].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=50)
axes[0].scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                pca.transform(kmeans.cluster_centers_)[:, 1],
                c='red', marker='X', s=200, label='Centroids', edgecolors='black')
axes[0].set_xlabel('First Principal Component')
axes[0].set_ylabel('Second Principal Component')
axes[0].set_title(f'K-Means Clustering (K={optimal_k})\nSilhouette: {kmeans_silhouette:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DBSCAN visualization
scatter = axes[1].scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6, s=50)
axes[1].set_xlabel('First Principal Component')
axes[1].set_ylabel('Second Principal Component')
title_text = f'DBSCAN Clustering\n{n_clusters_dbscan} clusters, {n_noise} noise points'
axes[1].set_title(title_text)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1plt/clustering_results.png', dpi=300, bbox_inches='tight')
print("    Saved clustering visualization to 'task1plt/clustering_results.png'")

# 5.2 Feature importance analysis for K-Means
print("\n5.2 Analyzing cluster characteristics...")

# Add cluster labels to original dataframe
df_with_clusters = df.copy()
df_with_clusters['KMeans_Cluster'] = kmeans_labels

# Calculate mean values for each cluster
cluster_means = df_with_clusters.groupby('KMeans_Cluster').mean()
print("\n    Mean values per cluster (K-Means):")
print(cluster_means)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Heatmap of cluster means
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0], cbar_kws={'label': 'Value'})
axes[0].set_title('K-Means Cluster Characteristics (Mean Values per Feature)')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Feature')

# Box plots for selected features
selected_features = ['Temp_Mean', 'Humidity_Mean', 'Pressure_Mean', 'Precipitation']
df_melted = df_with_clusters.melt(id_vars=['KMeans_Cluster'], value_vars=selected_features)
sns.boxplot(data=df_melted, x='variable', y='value', hue='KMeans_Cluster', ax=axes[1])
axes[1].set_title('Distribution of Key Features by Cluster')
axes[1].set_xlabel('Feature')
axes[1].set_ylabel('Value')
axes[1].legend(title='Cluster')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('task1plt/cluster_characteristics.png', dpi=300, bbox_inches='tight')
print("    Saved cluster characteristics to 'task1plt/cluster_characteristics.png'")

# 5.3 Save clustering results to CSV
print("\n5.3 Saving clustering results to CSV...")

# Save original data with cluster labels
df_with_clusters['DBSCAN_Cluster'] = dbscan_labels
df_with_clusters.to_csv('task1data/clustering_results.csv', index=False)
print("    Saved clustering results to 'task1data/clustering_results.csv'")

# Save cluster statistics
cluster_stats = df_with_clusters.groupby('KMeans_Cluster').agg(['mean', 'std', 'min', 'max'])
cluster_stats.to_csv('task1data/cluster_statistics.csv')
print("    Saved cluster statistics to 'task1data/cluster_statistics.csv'")

# ============================================================================
# 6. COMPARISON AND CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("6. ALGORITHM COMPARISON AND SUMMARY")
print("=" * 80)

print("\nComparison of Clustering Algorithms:")
print("-" * 50)
print(f"K-Means:")
print(f"  - Clusters: {optimal_k}")
print(f"  - Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  - Davies-Bouldin Index: {kmeans_db:.4f}")
print(f"  - Advantages: Simple, fast, works well with spherical clusters")
print(f"  - Limitations: Requires specifying K, assumes spherical clusters")

print(f"\nDBSCAN:")
print(f"  - Clusters: {n_clusters_dbscan}")
print(f"  - Noise points: {n_noise}")
if n_clusters_dbscan > 1 and n_noise < len(dbscan_labels) and valid_mask.sum() > 0:
    print(f"  - Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"  - Davies-Bouldin Index: {dbscan_db:.4f}")
print(f"  - Advantages: Finds arbitrary shapes, identifies outliers")
print(f"  - Limitations: Sensitive to parameters, struggles with varying densities")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("Generated files:")
print("  Plots:")
print("    - task1plt/kmeans_elbow_method.png")
print("    - task1plt/clustering_results.png")
print("    - task1plt/cluster_characteristics.png")
print("  Data:")
print("    - task1data/clustering_results.csv")
print("    - task1data/cluster_statistics.csv")
print("=" * 80)
