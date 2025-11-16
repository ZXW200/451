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

# Visualize missing data with simple bar chart
plt.figure(figsize=(10, 5))
plt.bar(range(len(missing_values)), missing_values.values, color='steelblue', alpha=0.7)
plt.xticks(range(len(missing_values)), missing_values.index, rotation=90)
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.title('Missing Values per Feature')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('task1plt/missing_data.png', dpi=300, bbox_inches='tight')
print("    Saved missing data chart to 'task1plt/missing_data.png'")

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

# Visualize outliers with simple bar chart
plt.figure(figsize=(10, 5))
colors = ['red' if count > 0 else 'steelblue' for count in outlier_counts.values()]
plt.bar(range(len(outlier_counts)), outlier_counts.values(), color=colors, alpha=0.7)
plt.xticks(range(len(outlier_counts)), outlier_counts.keys(), rotation=90)
plt.xlabel('Features')
plt.ylabel('Number of Outliers')
plt.title('Outliers Detected per Feature (IQR Method)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('task1plt/outlier_detection.png', dpi=300, bbox_inches='tight')
print("    Saved outlier detection chart to 'task1plt/outlier_detection.png'")

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

# Visualize before/after standardization using histograms
# Pick 4 representative features for clearer visualization
sample_features = ['Temp_Mean', 'Humidity_Mean', 'Pressure_Mean', 'Precipitation']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, feat in enumerate(sample_features):
    # Before standardization
    axes[0, idx].hist(df_clean[feat], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, idx].set_title(f'{feat}\n(Original)', fontsize=10)
    axes[0, idx].set_xlabel('Value')
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].grid(True, alpha=0.3)

    # After standardization
    axes[1, idx].hist(df_scaled[feat], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, idx].set_title(f'{feat}\n(Standardized)', fontsize=10)
    axes[1, idx].set_xlabel('Z-score')
    axes[1, idx].set_ylabel('Frequency')
    axes[1, idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean=0')
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].legend()

plt.suptitle('Before and After Standardization (Sample Features)', fontsize=14, y=0.995)
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

# Visualize feature importance with simple bar chart
plt.figure(figsize=(10, 6))
colors = ['green' if i < 10 else 'lightgray' for i in range(len(feature_importance))]
plt.barh(range(len(feature_importance)), feature_importance['Variance'].values, color=colors, alpha=0.7)
plt.yticks(range(len(feature_importance)), feature_importance['Feature'].values)
plt.xlabel('Variance (Importance Score)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance Ranking (Top 10 in Green)', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('task1plt/feature_importance_ranking.png', dpi=300, bbox_inches='tight')
print("    Saved feature importance ranking to 'task1plt/feature_importance_ranking.png'")

# Save feature importance
feature_importance.to_csv('task1data/feature_importance.csv', index=False)
print("    Saved feature importance to 'task1data/feature_importance.csv'")

# 5.2 Feature Selection - Select top features avoiding redundancy
print("\n5.2 Feature Selection Strategy...")
print("    Strategy: Select top variance features, but avoid redundancy (Min/Max/Mean)")
print("    For each weather type, keep only the Mean value")

# Select features: prioritize Mean values from high variance features
selected_features = [
    'Temp_Mean',        # Temperature representative
    'Humidity_Mean',    # Humidity representative
    'Pressure_Mean',    # Pressure representative
    'Precipitation',    # Independent feature
    'Snowfall',         # Independent feature
    'Sunshine',         # Independent feature
    'WindSpeed_Mean'    # Wind representative
]

print(f"\n    Selected {len(selected_features)} features from original {len(column_names)}:")
for i, feat in enumerate(selected_features, 1):
    var = feature_variance[feat]
    rank = feature_importance[feature_importance['Feature'] == feat].index[0] + 1
    print(f"      {i}. {feat} (variance: {var:.4f}, rank: {rank})")

# Create dataset with selected features
df_selected = df_clean[selected_features]
df_scaled_selected = df_scaled[selected_features]

print(f"\n    Reduced dimensionality: {len(column_names)} -> {len(selected_features)} features")

# Save selected features
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
print("    - task1plt/missing_data.png (bar chart)")
print("    - task1plt/outlier_detection.png (bar chart)")
print("    - task1plt/standardization_comparison.png (histograms)")
print("    - task1plt/feature_importance_ranking.png (bar chart)")
print("\n  Data:")
print("    - task1data/outlier_report.csv")
print("    - task1data/data_standardized.csv")
print("    - task1data/feature_importance.csv")
print("    - task1data/selected_features.csv")
print("=" * 80)
