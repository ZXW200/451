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

# 5.1 Calculate feature importance based on variance
print("\n5.1 Calculating Feature Importance (Variance-based)...")
print("    Higher variance = more information content")

# Calculate variance for each feature (using standardized data)
feature_variance = df_scaled.var()
feature_importance = pd.DataFrame({
    'Feature': column_names,
    'Variance': feature_variance.values
}).sort_values('Variance', ascending=False)

print("\n    Top 10 features by variance:")
print(feature_importance.head(10).to_string(index=False))

# 5.2 Feature Selection - Select top 7 features by variance
print("\n5.2 Feature Selection Strategy...")
print("    Strategy: Select top 7 features by variance (highest information content)")

# Automatically select top 7 features based on variance
top_n = 7
selected_features = feature_importance.head(top_n)['Feature'].tolist()

print(f"\n    Selected top {top_n} features from original {len(column_names)}:")
for i, feat in enumerate(selected_features, 1):
    var = feature_variance[feat]
    print(f"      {i}. {feat} (variance: {var:.4f})")

# Visualize why these features were selected - show actual data distributions
fig, axes = plt.subplots(3, 6, figsize=(18, 10))
axes = axes.flatten()

for idx, feat in enumerate(column_names):
    ax = axes[idx]

    # Plot data distribution for each feature
    ax.hist(df_clean[feat], bins=25, color='steelblue', alpha=0.7, edgecolor='black')

    # Highlight selected features
    if feat in selected_features:
        ax.set_facecolor('#e8f5e9')  # Light green background
        ax.set_title(f'{feat}\n✓ SELECTED\nVar={feature_variance[feat]:.3f}',
                    fontsize=8, fontweight='bold', color='green')
    else:
        ax.set_title(f'{feat}\nVar={feature_variance[feat]:.3f}',
                    fontsize=8, color='gray')

    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature Selection: Data Distribution & Variance\n(Green background = Selected top 7 by variance)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1plt/feature_selection_data.png', dpi=300, bbox_inches='tight')
print("    Saved feature selection visualization to 'task1plt/feature_selection_data.png'")

# Create dataset with selected features
df_selected = df_clean[selected_features]
df_scaled_selected = df_scaled[selected_features]

print(f"\n    Reduced dimensionality: {len(column_names)} -> {len(selected_features)} features")

# Save feature importance and selected features
feature_importance.to_csv('task1data/feature_importance.csv', index=False)
print("    Saved feature importance to 'task1data/feature_importance.csv'")

pd.DataFrame({'Selected_Features': selected_features}).to_csv(
    'task1data/selected_features.csv', index=False)
print("    Saved selected features to 'task1data/selected_features.csv'")

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
print(f"     - Method: Variance-based ranking")
print(f"     - Selected: {', '.join(selected_features)}")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/standardization_comparison.png (before/after standardization)")
print("    - task1plt/feature_selection_data.png (data distributions with variance)")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/feature_importance.csv")
print("    - task1data/selected_features.csv")
print("=" * 80)
