"""
Task 1: Basel Climate Dataset - Data Preprocessing and Clustering
This script performs data preprocessing and K-means clustering on Basel climate data.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output folders
os.makedirs('task1data', exist_ok=True)
os.makedirs('task1plt', exist_ok=True)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 80)
print("TASK 1: BASEL CLIMATE DATASET - DATA PREPROCESSING")
print("=" * 80)

# Define 18 climate features
column_names = [
    'Temp_Min', 'Temp_Max', 'Temp_Mean',
    'Humidity_Min', 'Humidity_Max', 'Humidity_Mean',
    'Pressure_Min', 'Pressure_Max', 'Pressure_Mean',
    'Precipitation', 'Snowfall', 'Sunshine',
    'WindGust_Min', 'WindGust_Max', 'WindGust_Mean',
    'WindSpeed_Min', 'WindSpeed_Max', 'WindSpeed_Mean'
]

# Load CSV file
df = pd.read_csv('ClimateDataBasel.csv', names=column_names)
print(f"\n1. Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n   First 5 rows:")
print(df.head())

# ============================================================================
# 2. DEALING WITH MISSING DATA
# ============================================================================
print("\n" + "=" * 80)
print("2. DEALING WITH MISSING DATA")
print("=" * 80)

# Check how many missing values
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print(f"\n   Total missing values: {total_missing}")

# Fill missing values with median (if any exist)
df_clean = df.fillna(df.median())
print(f"   Strategy: Fill with median")
print(f"   After handling: {df_clean.isnull().sum().sum()} missing values")

# ============================================================================
# 3. OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("3. OUTLIER DETECTION")
print("=" * 80)

print("\n   Using IQR (Interquartile Range) Method")
print("   Outliers: values < Q1-1.5*IQR or > Q3+1.5*IQR")

# Detect outliers for each feature
outlier_counts = {}
total_outliers = 0

for col in df_clean.columns:
    # Calculate Q1, Q3, and IQR
    Q1 = df_clean[col].quantile(0.25)  # 25th percentile
    Q3 = df_clean[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    outlier_counts[col] = outliers.sum()

    if outliers.sum() > 0:
        percentage = (outliers.sum() / len(df_clean)) * 100
        print(f"   - {col}: {outliers.sum()} outliers ({percentage:.2f}%)")
        total_outliers += outliers.sum()

print(f"\n   Total outlier instances: {total_outliers}")
print(f"   Strategy: Keep all outliers (climate extremes are valid)")

# Save outlier report
outlier_report = pd.DataFrame({
    'Feature': list(outlier_counts.keys()),
    'Outlier_Count': list(outlier_counts.values()),
    'Percentage': [(count / len(df_clean)) * 100 for count in outlier_counts.values()]
})
outlier_report.to_csv('task1data/outlier_report.csv', index=False)
print(f"   Saved: task1data/outlier_report.csv")

# ============================================================================
# 4. STANDARDIZATION
# ============================================================================
print("\n" + "=" * 80)
print("4. STANDARDIZATION")
print("=" * 80)

print("\n   Applying Z-score normalization")
print("   Formula: z = (x - mean) / std")
print("   Result: mean=0, std=1 for all features")

# Standardize the data
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_clean)
df_scaled = pd.DataFrame(df_scaled_array, columns=column_names)

print(f"\n   Mean after scaling: {df_scaled.mean().mean():.6f}")
print(f"   Std after scaling: {df_scaled.std().mean():.6f}")

# Create visualization: before vs after standardization
fig, axes = plt.subplots(2, 9, figsize=(20, 6))

for idx, feat in enumerate(column_names):
    row = idx // 9  # row 0 or 1
    col = idx % 9   # column 0-8

    # Top row: original data
    if row == 0:
        axes[row, col].hist(df_clean[feat], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[row, col].set_title(feat, fontsize=8)
        axes[row, col].tick_params(labelsize=6)
        axes[row, col].grid(True, alpha=0.3)

    # Bottom row: standardized data
    if row == 1:
        axes[row, col].hist(df_scaled[feat], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[row, col].axvline(x=0, color='red', linestyle='--', linewidth=1)  # mean line
        axes[row, col].tick_params(labelsize=6)
        axes[row, col].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Standardized', fontsize=10)

plt.suptitle('Standardization Comparison - All 18 Features', fontsize=14)
plt.tight_layout()
plt.savefig('task1plt/standardization_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: task1plt/standardization_comparison.png")

# Save standardized data
df_scaled.to_csv('task1data/data_standardized.csv', index=False)
print(f"   Saved: task1data/data_standardized.csv")

# ============================================================================
# 5. FEATURE EXTRACTION WITH PCA
# ============================================================================
print("\n" + "=" * 80)
print("5. FEATURE EXTRACTION WITH PCA")
print("=" * 80)

print("\n   PCA = Principal Component Analysis")
print("   Goal: Reduce 18 features to fewer components")
print("   Method: Combine correlated features")

# Step 1: Fit PCA on all components to see variance
pca_full = PCA()
pca_full.fit(df_scaled.values)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components needed for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\n   Components for 95% variance: {n_components_95} out of 18")

# Step 2: Use 7 components
n_components = 7
pca = PCA(n_components=n_components)
df_pca = pca.fit_transform(df_scaled.values)

print(f"   Using {n_components} principal components")
print(f"\n   Variance explained by each component:")
for i in range(n_components):
    var = pca.explained_variance_ratio_[i]
    print(f"      PC{i+1}: {var:.4f} ({var*100:.2f}%)")

total_var = sum(pca.explained_variance_ratio_)
print(f"   Total variance: {total_var:.4f} ({total_var*100:.2f}%)")

# Visualize PCA variance
plt.figure(figsize=(10, 6))
x_vals = range(1, 19)  # 1 to 18

# Bar chart: individual variance
plt.bar(x_vals, pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue', label='Individual')

# Line plot: cumulative variance
plt.plot(x_vals, cumulative_variance, 'ro-', linewidth=2, markersize=8, label='Cumulative')

# Reference lines
plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
plt.axvline(x=n_components, color='orange', linestyle='--', linewidth=2, label=f'Selected: {n_components} PCs')

plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('PCA Scree Plot - Variance Explained', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(x_vals)
plt.tight_layout()
plt.savefig('task1plt/pca_scree_plot.png', dpi=300, bbox_inches='tight')
print(f"   Saved: task1plt/pca_scree_plot.png")

# Create DataFrame with PCA results
pca_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca_final = pd.DataFrame(df_pca, columns=pca_columns)

print(f"\n   Reduced: 18 features -> {n_components} components")

# Save PCA data
df_pca_final.to_csv('task1data/pca_data.csv', index=False)
print(f"   Saved: task1data/pca_data.csv")

# ============================================================================
# 6. CLUSTERING WITH K-MEANS
# ============================================================================
print("\n" + "=" * 80)
print("6. CLUSTERING WITH K-MEANS")
print("=" * 80)

print("\n   Finding optimal number of clusters")
print("   Testing K from 2 to 10")

# Test different K values
k_range = range(2, 11)
inertias = []          # Within-cluster sum of squares
silhouette_scores = []  # Quality metric

for k in k_range:
    # Fit K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca_final)

    # Calculate metrics
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_pca_final, labels))

# Find best K by silhouette score
best_k_idx = silhouette_scores.index(max(silhouette_scores))
optimal_k_silhouette = k_range[best_k_idx]

# Use K=4 for seasonal patterns
optimal_k = 4

print(f"\n   Best K by silhouette: {optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
print(f"   Using K={optimal_k} (for 4 seasons)")

# Visualize cluster optimization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Elbow method
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Selected: K={optimal_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(k_range)

# Right plot: Silhouette score
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Selected: K={optimal_k}')
ax2.axvline(x=optimal_k_silhouette, color='orange', linestyle='--', linewidth=2, label=f'Best: K={optimal_k_silhouette}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score Analysis', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(k_range)

plt.tight_layout()
plt.savefig('task1plt/optimal_clusters.png', dpi=300, bbox_inches='tight')
print(f"   Saved: task1plt/optimal_clusters.png")

# Apply final K-means clustering
print(f"\n   Applying K-means with K={optimal_k}")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(df_pca_final)

# Add cluster labels to data
df_pca_final['Cluster'] = clusters

# Calculate silhouette score
final_silhouette = silhouette_score(df_pca_final.drop('Cluster', axis=1), clusters)
print(f"   Silhouette Score: {final_silhouette:.3f}")

# Show cluster distribution
print(f"\n   Cluster distribution:")
for i in range(optimal_k):
    count = (clusters == i).sum()
    percentage = (count / len(clusters)) * 100
    print(f"      Cluster {i}: {count} samples ({percentage:.1f}%)")

# Visualize clusters in 2D space
plt.figure(figsize=(10, 8))

# Scatter plot using first 2 principal components
scatter = plt.scatter(df_pca_final['PC1'], df_pca_final['PC2'],
                     c=clusters, cmap='viridis', s=50, alpha=0.6, edgecolor='black')

# Plot cluster centers
centers = kmeans_final.cluster_centers_[:, :2]  # Use only PC1 and PC2
plt.scatter(centers[:, 0], centers[:, 1],
           c='red', marker='X', s=300, edgecolor='black', linewidth=2,
           label='Cluster Centers')

plt.xlabel('PC1 (First Principal Component)', fontsize=12)
plt.ylabel('PC2 (Second Principal Component)', fontsize=12)
plt.title(f'K-means Clustering Results (K={optimal_k})', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1plt/clustering_results.png', dpi=300, bbox_inches='tight')
print(f"   Saved: task1plt/clustering_results.png")

# Save clustered data
df_pca_final.to_csv('task1data/clustered_data.csv', index=False)
print(f"   Saved: task1data/clustered_data.csv")

# Calculate and save cluster statistics
cluster_stats = df_pca_final.groupby('Cluster').agg(['mean', 'std'])
cluster_stats.to_csv('task1data/cluster_statistics.csv')
print(f"   Saved: task1data/cluster_statistics.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING AND CLUSTERING COMPLETE!")
print("=" * 80)

print("\nSummary:")
print(f"  1. Missing Data: {total_missing} (filled with median)")
print(f"  2. Outliers: {total_outliers} detected and kept")
print(f"  3. Standardization: Z-score normalization applied")
print(f"  4. PCA: Reduced 18 features to {n_components} components ({total_var*100:.2f}% variance)")
print(f"  5. Clustering: K-means with {optimal_k} clusters (silhouette: {final_silhouette:.3f})")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/standardization_comparison.png")
print("    - task1plt/pca_scree_plot.png")
print("    - task1plt/optimal_clusters.png")
print("    - task1plt/clustering_results.png")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/pca_data.csv")
print("    - task1data/clustered_data.csv")
print("    - task1data/cluster_statistics.csv")
print("=" * 80)
