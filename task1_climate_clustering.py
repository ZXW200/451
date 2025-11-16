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
# 5. FEATURE SELECTION/EXTRACTION
# ============================================================================
print("\n" + "=" * 80)
print("5. FEATURE SELECTION/EXTRACTION")
print("=" * 80)

# 5.1 Correlation Analysis
print("\n5.1 Correlation Analysis...")
print("    Finding redundant features (high correlation)")

# Calculate correlation matrix
correlation_matrix = df_scaled.corr()

# Visualize correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix\n(High correlation = Redundant features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1plt/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("    Saved correlation matrix to 'task1plt/correlation_matrix.png'")

# 5.2 Feature Selection - Automatic selection based on correlation
print("\n5.2 Automatic Feature Selection...")
print("    Strategy: Remove features with high correlation (> 0.9)")
print("    If two features are highly correlated, keep the one with higher variance")

# Calculate variance for each feature
feature_variance = df_scaled.var().sort_values(ascending=False)

# Greedy algorithm to remove redundant features
selected_features = list(column_names)
removed_features = {}

for i in range(len(correlation_matrix.columns)):
    feat1 = correlation_matrix.columns[i]
    if feat1 not in selected_features:
        continue

    for j in range(i+1, len(correlation_matrix.columns)):
        feat2 = correlation_matrix.columns[j]
        if feat2 not in selected_features:
            continue

        corr = abs(correlation_matrix.iloc[i, j])
        if corr > 0.9:
            # Remove the feature with lower variance
            var1 = feature_variance[feat1]
            var2 = feature_variance[feat2]

            if var1 >= var2:
                to_remove = feat2
                to_keep = feat1
            else:
                to_remove = feat1
                to_keep = feat2

            if to_remove in selected_features:
                selected_features.remove(to_remove)
                removed_features[to_remove] = {
                    'correlated_with': to_keep,
                    'correlation': corr,
                    'variance': feature_variance[to_remove]
                }
                print(f"    Removed {to_remove} (corr={corr:.3f} with {to_keep}, var={feature_variance[to_remove]:.3f})")

print(f"\n    Selected {len(selected_features)} features from original {len(column_names)}:")
for i, feat in enumerate(selected_features, 1):
    var = feature_variance[feat]
    print(f"      {i}. {feat} (variance: {var:.3f})")

print(f"\n    Removed {len(removed_features)} redundant features (correlation > 0.9):")

# Visualize feature selection process
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all features ranked by variance
all_features = feature_variance.index.tolist()
variances = feature_variance.values

y_positions = range(len(all_features))
colors = []
labels = []

for feat in all_features:
    if feat in selected_features:
        colors.append('green')
        labels.append(f'{feat}\n✓ SELECTED\nVar={feature_variance[feat]:.3f}')
    else:
        removed_info = removed_features[feat]
        colors.append('red')
        labels.append(f'{feat}\n✗ Removed\nCorr with {removed_info["correlated_with"][:12]}...\n{removed_info["correlation"]:.2f}')

bars = ax.barh(y_positions, variances, color=colors, alpha=0.7)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Variance', fontsize=12)
ax.set_title(f'Automatic Feature Selection Based on Correlation\nGreen = Selected ({len(selected_features)}), Red = Removed ({len(removed_features)})',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label=f'Selected ({len(selected_features)})'),
    Patch(facecolor='red', alpha=0.7, label=f'Removed - High correlation ({len(removed_features)})')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('task1plt/feature_selection_process.png', dpi=300, bbox_inches='tight')
print("    Saved feature selection process to 'task1plt/feature_selection_process.png'")

# Visualize correlation for selected features only
plt.figure(figsize=(8, 7))
selected_corr = correlation_matrix.loc[selected_features, selected_features]
sns.heatmap(selected_corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Selected Features Correlation Matrix\n(After removing redundant features)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('task1plt/selected_features_correlation.png', dpi=300, bbox_inches='tight')
print("    Saved selected features correlation to 'task1plt/selected_features_correlation.png'")

# Create dataset with selected features
df_selected = df_clean[selected_features]
df_scaled_selected = df_scaled[selected_features]

print(f"\n    Reduced dimensionality: {len(column_names)} -> {len(selected_features)} features")

# Save correlation matrix and feature selection details
correlation_matrix.to_csv('task1data/correlation_matrix.csv')
print("    Saved correlation matrix to 'task1data/correlation_matrix.csv'")

# Save selected features
pd.DataFrame({'Selected_Features': selected_features,
              'Variance': [feature_variance[f] for f in selected_features]}).to_csv(
    'task1data/selected_features.csv', index=False)
print("    Saved selected features to 'task1data/selected_features.csv'")

# Save removed features with reasons
if removed_features:
    removed_df = pd.DataFrame([
        {
            'Removed_Feature': feat,
            'Correlated_With': info['correlated_with'],
            'Correlation': info['correlation'],
            'Variance': info['variance']
        }
        for feat, info in removed_features.items()
    ])
    removed_df.to_csv('task1data/removed_features.csv', index=False)
    print("    Saved removed features to 'task1data/removed_features.csv'")

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

print(f"\n  4. Feature Selection:")
print(f"     - Original features: {len(column_names)}")
print(f"     - Selected features: {len(selected_features)}")
print(f"     - Removed features: {len(removed_features)}")
print(f"     - Method: Automatic correlation-based selection")
print(f"     - Strategy: Remove features with |corr| > 0.9, keep higher variance")
print(f"     - Selected: {', '.join(selected_features)}")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/standardization_comparison.png (before/after all 18 features)")
print("    - task1plt/correlation_matrix.png (18x18 correlation heatmap)")
print("    - task1plt/feature_selection_process.png (why selected/removed)")
print("    - task1plt/selected_features_correlation.png (correlation after selection)")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/correlation_matrix.csv")
print("    - task1data/selected_features.csv (with variance values)")
print("    - task1data/removed_features.csv (with removal reasons)")
print("=" * 80)
