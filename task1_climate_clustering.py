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

# Visualize missing data pattern (if any existed)
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Data Pattern (Before Cleaning)')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.tight_layout()
plt.savefig('task1plt/missing_data_pattern.png', dpi=300, bbox_inches='tight')
print("    Saved missing data pattern to 'task1plt/missing_data_pattern.png'")

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

# Visualize outliers with box plots
fig, axes = plt.subplots(3, 6, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(df_clean.columns):
    axes[idx].boxplot(df_clean[col], vert=True)
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_ylabel('Value')
    if outlier_counts[col] > 0:
        axes[idx].set_facecolor('#fff3cd')  # Highlight if has outliers

plt.suptitle('Outlier Detection - Box Plots for All Features', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('task1plt/outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
print("    Saved outlier detection plots to 'task1plt/outlier_detection_boxplots.png'")

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

# Visualize before/after standardization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before standardization
axes[0].boxplot([df_clean[col] for col in df_clean.columns], labels=column_names)
axes[0].set_title('Before Standardization', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Original Scale')
axes[0].set_xlabel('Features')
axes[0].tick_params(axis='x', rotation=90)
axes[0].grid(True, alpha=0.3)

# After standardization
axes[1].boxplot([df_scaled[col] for col in df_scaled.columns], labels=column_names)
axes[1].set_title('After Standardization', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Standardized Scale (Z-score)')
axes[1].set_xlabel('Features')
axes[1].tick_params(axis='x', rotation=90)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Mean=0')
axes[1].legend()

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

# 5.1 Feature Selection using Domain Knowledge
print("\n5.1 Feature Selection using Domain Knowledge...")
print("    Strategy: Select only 'Mean' values to avoid redundancy with Min/Max")

selected_features = [
    'Temp_Mean',
    'Humidity_Mean',
    'Pressure_Mean',
    'Precipitation',
    'Snowfall',
    'Sunshine',
    'WindSpeed_Mean'
]

print(f"\n    Selected {len(selected_features)} features from original {len(column_names)}:")
for i, feat in enumerate(selected_features, 1):
    print(f"      {i}. {feat}")

# Create dataset with selected features only
df_selected = df_clean[selected_features]
df_scaled_selected = df_scaled[selected_features]

print(f"\n    Reduced dimensionality: {len(column_names)} -> {len(selected_features)} features")
print(f"    This removes redundancy (Min/Max correlated with Mean)")

# Visualize feature selection
fig, ax = plt.subplots(figsize=(10, 6))

feature_status = ['Selected' if feat in selected_features else 'Excluded'
                  for feat in column_names]
colors = ['green' if status == 'Selected' else 'lightgray'
          for status in feature_status]

y_pos = np.arange(len(column_names))
ax.barh(y_pos, [1]*len(column_names), color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(column_names)
ax.set_xlabel('Feature Status')
ax.set_title(f'Feature Selection: {len(selected_features)} of {len(column_names)} Features Selected')
ax.set_xticks([])

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label=f'Selected ({len(selected_features)})'),
                   Patch(facecolor='lightgray', alpha=0.7, label=f'Excluded ({len(column_names) - len(selected_features)})')]
ax.legend(handles=legend_elements, loc='lower right')

ax.text(0.5, -0.05, 'Strategy: Keep Mean values, exclude Min/Max to reduce redundancy',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('task1plt/feature_selection.png', dpi=300, bbox_inches='tight')
print("    Saved feature selection plot to 'task1plt/feature_selection.png'")

# Save selected features
pd.DataFrame({'Selected_Features': selected_features}).to_csv(
    'task1data/selected_features.csv', index=False)
print("    Saved selected features to 'task1data/selected_features.csv'")

# 5.2 Feature Extraction using PCA
print("\n5.2 Feature Extraction using PCA...")
print(f"    Applying PCA to extract principal components from {len(selected_features)} selected features")

pca_full = PCA()
pca_full.fit(df_scaled_selected.values)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\n    Total components: {len(selected_features)}")
print(f"    Components for 95% variance: {n_components_95}")
print(f"    Variance explained by each PC:")
for i, var in enumerate(pca_full.explained_variance_ratio_, 1):
    print(f"      PC{i}: {var:.4f} ({var*100:.2f}%)")

# Visualize PCA results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scree plot
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue',
            label='Individual')
axes[0].plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 'ro-', linewidth=2, label='Cumulative')
axes[0].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot - PCA Explained Variance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))

# Component loadings heatmap
loadings = pca_full.components_.T
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(len(selected_features))],
    index=selected_features
)

sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes[1], cbar_kws={'label': 'Loading'})
axes[1].set_title('PCA Component Loadings')
axes[1].set_xlabel('Principal Components')
axes[1].set_ylabel('Features')

plt.tight_layout()
plt.savefig('task1plt/pca_analysis.png', dpi=300, bbox_inches='tight')
print("    Saved PCA analysis to 'task1plt/pca_analysis.png'")

# Apply PCA transformation
pca_transformed = pca_full.transform(df_scaled_selected.values)
pca_df = pd.DataFrame(
    pca_transformed,
    columns=[f'PC{i+1}' for i in range(len(selected_features))]
)

# Save PCA results
pca_df.to_csv('task1data/pca_transformed_data.csv', index=False)
print("    Saved PCA transformed data to 'task1data/pca_transformed_data.csv'")

loadings_df.to_csv('task1data/pca_loadings.csv')
print("    Saved PCA loadings to 'task1data/pca_loadings.csv'")

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
print(f"     - Selected: {', '.join(selected_features)}")

print(f"\n  5. Feature Extraction (PCA):")
print(f"     - Components for 95% variance: {n_components_95}/{len(selected_features)}")
print(f"     - Total variance by PC1: {pca_full.explained_variance_ratio_[0]*100:.2f}%")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/missing_data_pattern.png")
print("    - task1plt/outlier_detection_boxplots.png")
print("    - task1plt/standardization_comparison.png")
print("    - task1plt/feature_selection.png")
print("    - task1plt/pca_analysis.png")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/selected_features.csv")
print("    - task1data/pca_transformed_data.csv")
print("    - task1data/pca_loadings.csv")
print("=" * 80)
