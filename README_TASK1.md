# Task 1: Basel Climate Dataset Clustering

## Overview
This is a simple implementation of clustering analysis on the Basel Climate Dataset for SCC451 Machine Learning coursework.

## Files
- `task1_climate_clustering.py` - Main Python script for Task 1
- `ClimateDataBasel.csv` - Input dataset (1763 rows, 18 features)

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn

## Installation
```bash
pip install pandas numpy matplotlib scikit-learn seaborn
```

## How to Run
```bash
python task1_climate_clustering.py
```

## What the Script Does

### 1. Data Loading
- Loads the climate dataset with 18 features
- Features include temperature, humidity, pressure, precipitation, wind speed, etc.

### 2. Data Preprocessing
- **Missing Values**: Checks for missing data (none found)
- **Outlier Detection**: Uses IQR method to identify outliers
- **Standardization**: Applies StandardScaler to normalize all features
- **Dimensionality Reduction**: Uses PCA to reduce to 2D for visualization

### 3. Clustering Algorithms

#### K-Means
- Simple and popular algorithm
- Tested K values from 2 to 10 using Elbow Method
- Selected K=3 as optimal number of clusters
- Evaluated using Silhouette Score and Davies-Bouldin Index

#### DBSCAN
- Density-based clustering algorithm
- Can find arbitrary-shaped clusters
- Identifies outliers as noise points
- Parameters: eps=2.5, min_samples=10

### 4. Results
The script generates three visualization files:
- `kmeans_elbow_method.png` - Shows optimal K selection
- `clustering_results.png` - 2D visualization of both clustering methods
- `cluster_characteristics.png` - Analysis of cluster features

## Results Summary

### K-Means (K=3)
- Silhouette Score: 0.2725
- Davies-Bouldin Index: 1.3760
- Cluster 0: Cold weather with high wind (390 samples)
- Cluster 1: Warm weather with low wind (824 samples)
- Cluster 2: Cool weather with moderate wind (549 samples)

### DBSCAN
- Found 1 main cluster
- Identified 147 noise points (outliers)
- Shows that most data points are densely grouped

## Interpretation
The K-Means algorithm successfully separated the climate data into three distinct seasons/weather patterns:
1. **Cluster 0**: Winter conditions (cold, windy)
2. **Cluster 1**: Summer conditions (warm, calm)
3. **Cluster 2**: Transitional seasons (cool, moderate)

DBSCAN shows most data follows similar patterns with some outlier days.

## Code Features
- Well-commented code suitable for average student level
- Clear explanations of each step
- Simple visualizations
- Comprehensive evaluation metrics
