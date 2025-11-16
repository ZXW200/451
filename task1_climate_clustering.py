"""
Task 1: Basel Climate Dataset - Data Preprocessing
Student: Average Level Implementation
This script performs data preprocessing on the Basel climate dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
print("TASK 1: BASEL CLIMATE DATASET - DATA PREPROCESSING")
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
# 2. DEALING WITH MISSING DATA
# ============================================================================
print("\n" + "=" * 80)
print("2. DEALING WITH MISSING DATA")
print("=" * 80)

# Check for missing values
print("\n2.1 Checking for missing values...")
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print(f"    Total missing values: {total_missing}")

if total_missing > 0:
    print("\n    Missing values per feature:")
    for col in missing_values[missing_values > 0].index:
        count = missing_values[col]
        percentage = (count / len(df)) * 100
        print(f"    - {col}: {count} ({percentage:.2f}%)")

    # Handle missing values (if any)
    print("\n    Strategy: Fill missing values with median")
    df_clean = df.fillna(df.median())
    print(f"    After handling: {df_clean.isnull().sum().sum()} missing values")
else:
    print("    No missing values found!")
    df_clean = df.copy()

# ============================================================================
# 3. OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("3. OUTLIER DETECTION")
print("=" * 80)

print("\n3.1 Using IQR (Interquartile Range) Method...")
print("    Outliers defined as: Q1 - 1.5*IQR  or  Q3 + 1.5*IQR")

outlier_counts = {}
outlier_indices = {}

for col in df_clean.columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    outlier_counts[col] = outliers_mask.sum()
    outlier_indices[col] = df_clean[outliers_mask].index.tolist()

print("\n    Outliers detected per feature:")
total_outliers = 0
for col, count in outlier_counts.items():
    if count > 0:
        percentage = (count / len(df_clean)) * 100
        print(f"    - {col}: {count} outliers ({percentage:.2f}%)")
        total_outliers += count

print(f"\n    Total outlier instances: {total_outliers}")
print(f"    Strategy: Keep outliers (climate extremes are valid data points)")

# Save outlier report
outlier_report = pd.DataFrame({
    'Feature': list(outlier_counts.keys()),
    'Outlier_Count': list(outlier_counts.values()),
    'Percentage': [(count / len(df_clean)) * 100 for count in outlier_counts.values()]
})
outlier_report.to_csv('task1data/outlier_report.csv', index=False)
print("    Saved outlier report to 'task1data/outlier_report.csv'")

# ============================================================================
# 4. NORMALIZATION/STANDARDIZATION
# ============================================================================
print("\n" + "=" * 80)
print("4. NORMALIZATION/STANDARDIZATION")
print("=" * 80)

print("\n4.1 Applying Standardization (Z-score normalization)...")
print("    Formula: z = (x - μ) / σ")
print("    This transforms data to have mean=0 and std=1")

# Apply StandardScaler
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_clean)
df_scaled = pd.DataFrame(df_scaled_array, columns=column_names)

print("\n    Standardization completed!")
print(f"    Mean after scaling (should be ~0): {df_scaled.mean().mean():.6f}")
print(f"    Std after scaling (should be ~1): {df_scaled.std().mean():.6f}")

# Visualize before/after standardization - show all features
fig, axes = plt.subplots(2, 9, figsize=(20, 6))

for idx, feat in enumerate(column_names):
    row = idx // 9
    col = idx % 9

    # Before standardization
    if row == 0:
        axes[row, col].hist(df_clean[feat], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[row, col].set_title(feat, fontsize=8)
        axes[row, col].tick_params(labelsize=6)
        axes[row, col].grid(True, alpha=0.3)

    # After standardization
    if row == 1:
        axes[row, col].hist(df_scaled[feat], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[row, col].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[row, col].tick_params(labelsize=6)
        axes[row, col].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Standardized', fontsize=10)

plt.suptitle('Standardization Comparison - All Features', fontsize=14)
plt.tight_layout()
plt.savefig('task1plt/standardization_comparison.png', dpi=300, bbox_inches='tight')
print("    Saved standardization comparison to 'task1plt/standardization_comparison.png'")

# Save scaled data
df_scaled.to_csv('task1data/data_standardized.csv', index=False)
print("    Saved standardized data to 'task1data/data_standardized.csv'")

# ============================================================================
# 5. FEATURE EXTRACTION WITH PCA
# ============================================================================
print("\n" + "=" * 80)
print("5. FEATURE EXTRACTION WITH PCA")
print("=" * 80)

print("\n5.1 Applying PCA...")
print("    PCA (Principal Component Analysis) reduces dimensionality")
print("    Combines correlated features into independent components")

# Apply PCA to extract components
pca = PCA()
pca_data = pca.fit_transform(df_scaled.values)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\n    Components needed for 95% variance: {n_components_95} out of {len(column_names)}")

# Use 7 components (or n_components_95 if less)
n_components = min(7, n_components_95)
print(f"    Using {n_components} principal components")

# Extract the selected number of components
pca_selected = PCA(n_components=n_components)
df_pca = pca_selected.fit_transform(df_scaled.values)

print(f"\n    Variance explained by each component:")
for i, var_ratio in enumerate(pca_selected.explained_variance_ratio_, 1):
    print(f"      PC{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
print(f"    Total variance explained: {sum(pca_selected.explained_variance_ratio_):.4f} ({sum(pca_selected.explained_variance_ratio_)*100:.2f}%)")

# Visualize PCA results with simple scree plot
plt.figure(figsize=(10, 6))
x_vals = range(1, len(pca.explained_variance_ratio_) + 1)

plt.bar(x_vals, pca.explained_variance_ratio_, alpha=0.7, color='steelblue', label='Individual')
plt.plot(x_vals, cumulative_variance, 'ro-', linewidth=2, markersize=8, label='Cumulative')
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
print("    Saved PCA scree plot to 'task1plt/pca_scree_plot.png'")

# Create PCA dataframe
pca_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca_final = pd.DataFrame(df_pca, columns=pca_columns)

print(f"\n    Reduced dimensionality: {len(column_names)} features -> {n_components} components")

# Save PCA results
df_pca_final.to_csv('task1data/pca_data.csv', index=False)
print("    Saved PCA transformed data to 'task1data/pca_data.csv'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE!")
print("=" * 80)

print("\nPreprocessing Summary:")
print(f"  1. Missing Data:")
print(f"     - Total missing values: {total_missing}")
print(f"     - Strategy: {'Filled with median' if total_missing > 0 else 'None needed'}")

print(f"\n  2. Outlier Detection:")
print(f"     - Total outlier instances: {total_outliers}")
print(f"     - Strategy: Kept (climate extremes are valid)")

print(f"\n  3. Standardization:")
print(f"     - Method: StandardScaler (Z-score)")
print(f"     - Result: Mean ≈ 0, Std ≈ 1")

print(f"\n  4. Feature Extraction (PCA):")
print(f"     - Original features: {len(column_names)}")
print(f"     - Principal components: {n_components}")
print(f"     - Method: PCA (Principal Component Analysis)")
print(f"     - Variance explained: {sum(pca_selected.explained_variance_ratio_)*100:.2f}%")
print(f"     - Components needed for 95% variance: {n_components_95}")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/standardization_comparison.png (before/after all 18 features)")
print("    - task1plt/pca_scree_plot.png (variance explained by each component)")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/pca_data.csv (PCA transformed data)")
print("=" * 80)
