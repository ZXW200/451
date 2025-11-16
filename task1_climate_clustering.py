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

# 5.2 Feature Selection - Remove redundant features based on correlation
print("\n5.2 Feature Selection Strategy...")
print("    Strategy: Remove highly correlated features (correlation > 0.9)")
print("    Keep one representative from each correlated group")

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"\n    Found {len(high_corr_pairs)} highly correlated pairs (|corr| > 0.9):")
for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
    print(f"    - {feat1} <-> {feat2}: {corr:.3f}")

# Strategy: For each weather type, keep only Mean value (most representative)
# Independent features: Precipitation, Snowfall, Sunshine (no Min/Max)
selected_features = [
    'Temp_Mean',        # Temperature representative (drop Min, Max)
    'Humidity_Mean',    # Humidity representative (drop Min, Max)
    'Pressure_Mean',    # Pressure representative (drop Min, Max)
    'Precipitation',    # Independent feature
    'Snowfall',         # Independent feature
    'Sunshine',         # Independent feature
    'WindSpeed_Mean'    # Wind representative (drop Min, Max, Gust)
]

print(f"\n    Selected {len(selected_features)} features from original {len(column_names)}:")
print("    Selection rule: Keep 'Mean' from correlated groups, keep independent features")
for i, feat in enumerate(selected_features, 1):
    print(f"      {i}. {feat}")

# Show which features were dropped and why
dropped_features = [f for f in column_names if f not in selected_features]
print(f"\n    Dropped {len(dropped_features)} redundant features:")
print("    - Temperature: Temp_Min, Temp_Max (correlated with Temp_Mean)")
print("    - Humidity: Humidity_Min, Humidity_Max (correlated with Humidity_Mean)")
print("    - Pressure: Pressure_Min, Pressure_Max (correlated with Pressure_Mean)")
print("    - Wind: WindSpeed_Min, WindSpeed_Max, WindGust_* (correlated with WindSpeed_Mean)")

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

# Save correlation matrix and selected features
correlation_matrix.to_csv('task1data/correlation_matrix.csv')
print("    Saved correlation matrix to 'task1data/correlation_matrix.csv'")

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
print(f"     - Method: Correlation-based (remove redundancy)")
print(f"     - Strategy: Keep Mean from correlated groups")
print(f"     - Selected: {', '.join(selected_features)}")

print("\nGenerated Files:")
print("  Plots:")
print("    - task1plt/standardization_comparison.png (before/after all 18 features)")
print("    - task1plt/correlation_matrix.png (18x18 correlation heatmap)")
print("    - task1plt/selected_features_correlation.png (7x7 after selection)")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/correlation_matrix.csv")
print("    - task1data/selected_features.csv")
print("=" * 80)
