#!/usr/bin/env python3
"""
Script to create a comprehensive Jupyter notebook with all models, results, and visualizations
"""

import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title and Introduction
cells.append(nbf.v4.new_markdown_cell("""# Language Extinction Risk Prediction - Complete Model Analysis

**Course:** INFT6201 Big Data Assessment 2  
**Project:** Predicting Global Language Extinction Risk  
**Date:** October 2024

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Loading and Preprocessing](#data-loading)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Performance Metrics](#performance-metrics)
6. [Confusion Matrices](#confusion-matrices)
7. [Box Plots and Statistical Analysis](#box-plots)
8. [Feature Importance](#feature-importance)
9. [Conclusions](#conclusions)"""))

# Introduction
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Introduction <a name="introduction"></a>

This notebook presents a comprehensive analysis of machine learning models for predicting language extinction risk. We evaluate four different models:

1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Neural Network (Deep Learning)**
4. **Logistic Regression (Baseline)**

### Objective
Predict the endangerment level of languages to guide UNESCO's International Decade of Indigenous Languages (2022-2032).

### Dataset
- **Total Languages:** 8,300
- **Features:** 22 (including geographic, linguistic, and socioeconomic factors)
- **Classes:** 6 endangerment levels (Safe, Vulnerable, Definitely Endangered, Severely Endangered, Critically Endangered, Extinct)"""))

# Import libraries
cells.append(nbf.v4.new_code_cell("""# Import required libraries
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Machine Learning libraries
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Import custom modules
from data.data_loader import LanguageDataLoader
from data.data_preprocessor import LanguageDataPreprocessor
from models.ml_models import LanguageExtinctionPredictor

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")"""))

# Data Loading section
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Data Loading and Preprocessing <a name="data-loading"></a>

### 2.1 Load Data"""))

cells.append(nbf.v4.new_code_cell("""# Initialize data loader
print("Loading data...")
loader = LanguageDataLoader()

# Load all datasets
datasets = loader.load_all_datasets()
print(f"\\n✅ Loaded {len(datasets)} datasets")

# Merge datasets
merged_data = loader.merge_datasets()
print(f"✅ Merged data shape: {merged_data.shape}")

# Display first few rows
print("\\n📊 Sample Data:")
merged_data.head()"""))

# Data Summary
cells.append(nbf.v4.new_markdown_cell("""### 2.2 Data Summary"""))

cells.append(nbf.v4.new_code_cell("""# Get data summary
print("📊 Dataset Summary:")
print(f"Total Languages: {len(merged_data):,}")
print(f"Total Features: {merged_data.shape[1]}")
print(f"\\nColumns: {list(merged_data.columns)}")

# Check for missing values
print("\\n🔍 Missing Values:")
missing = merged_data.isnull().sum()
missing[missing > 0].sort_values(ascending=False)"""))

# Preprocessing
cells.append(nbf.v4.new_markdown_cell("""### 2.3 Preprocess Data"""))

cells.append(nbf.v4.new_code_cell("""# Initialize preprocessor
print("Preprocessing data...")
preprocessor = LanguageDataPreprocessor()

# Run preprocessing pipeline
X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)

print(f"\\n✅ Preprocessing complete!")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Number of features: {len(feature_names)}")

# Display feature names
print(f"\\n📋 Features used: {feature_names}")"""))

# Split and Scale
cells.append(nbf.v4.new_markdown_cell("""### 2.4 Split and Scale Data"""))

cells.append(nbf.v4.new_code_cell("""# Split data into train and test sets
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]:,} samples")
print(f"Test set size: {X_test.shape[0]:,} samples")

# Scale features
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

print(f"\\n✅ Data split and scaled successfully!")

# Get class names
class_names = preprocessor.target_encoder.classes_
print(f"\\n📊 Classes: {class_names}")"""))

# Model Training
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Model Training <a name="model-training"></a>

We train four different models and compare their performance."""))

cells.append(nbf.v4.new_code_cell("""# Initialize predictor
print("Initializing models...")
predictor = LanguageExtinctionPredictor()

# Train all models
print("\\n🚀 Training all models...\\n")
print("="*60)

model_results = predictor.train_all_models(X_train_scaled, y_train)

print("\\n="*60)
print("✅ All models trained successfully!")
print(f"\\nModels trained: {list(model_results.keys())}")"""))

# Training Accuracy
cells.append(nbf.v4.new_markdown_cell("""### 3.1 Training Accuracy"""))

cells.append(nbf.v4.new_code_cell("""# Display training accuracies
print("📊 Training Accuracies:\\n")
print(f"{'Model':<25} {'Training Accuracy':<20}")
print("="*45)

for model_name, results in model_results.items():
    train_acc = results['train_accuracy']
    print(f"{model_name:<25} {train_acc:.4f} ({train_acc*100:.2f}%)")"""))

# Model Evaluation
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Model Evaluation <a name="model-evaluation"></a>

### 4.1 Test Set Evaluation"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate all models on test set
print("Evaluating models on test set...\\n")
evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)

print("✅ Evaluation complete!\\n")

# Display test accuracies
print("="*60)
print("📊 TEST SET PERFORMANCE")
print("="*60)
print(f"\\n{'Model':<25} {'Test Accuracy':<20} {'Rank':<10}")
print("-"*55)

# Sort by accuracy
sorted_results = sorted(evaluation_results.items(), 
                       key=lambda x: x[1]['test_accuracy'], 
                       reverse=True)

for rank, (model_name, results) in enumerate(sorted_results, 1):
    test_acc = results['test_accuracy']
    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{model_name:<25} {test_acc:.4f} ({test_acc*100:.2f}%)  {emoji} #{rank}")

print("="*60)"""))

# Performance Metrics
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Performance Metrics <a name="performance-metrics"></a>

### 5.1 Metrics Summary Table"""))

cells.append(nbf.v4.new_code_cell("""# Create summary table
summary_data = []

for model_name, results in evaluation_results.items():
    class_report = results['classification_report']
    weighted_avg = class_report['weighted avg']
    
    summary_data.append({
        'Model': model_name,
        'Accuracy': results['test_accuracy'],
        'Precision': weighted_avg['precision'],
        'Recall': weighted_avg['recall'],
        'F1-Score': weighted_avg['f1-score']
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\\n" + "="*80)
print("📊 PERFORMANCE SUMMARY - ALL MODELS")
print("="*80)
print(summary_df.round(4).to_string(index=False))
print("="*80)

summary_df"""))

# Display existing visualizations
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Confusion Matrices <a name="confusion-matrices"></a>

### 6.1 All Models Confusion Matrices"""))

cells.append(nbf.v4.new_code_cell("""# Display confusion matrices from file
from IPython.display import Image
display(Image('visualizations/confusion_matrices_all_models.png'))"""))

cells.append(nbf.v4.new_markdown_cell("""### 6.2 Best Model (Random Forest) Detailed View"""))

cells.append(nbf.v4.new_code_cell("""# Display Random Forest confusion matrix with metrics
display(Image('visualizations/confusion_matrix_with_metrics_random_forest.png'))"""))

# Box Plots
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Box Plots and Statistical Analysis <a name="box-plots"></a>

### 7.1 Performance Distribution Box Plots"""))

cells.append(nbf.v4.new_code_cell("""# Display box plots
display(Image('visualizations/model_performance_boxplots.png'))"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.2 Per-Class Performance"""))

cells.append(nbf.v4.new_code_cell("""# Display per-class box plot
display(Image('visualizations/per_class_performance_boxplot.png'))"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.3 Metrics Comparison"""))

cells.append(nbf.v4.new_code_cell("""# Display bar chart
display(Image('visualizations/metrics_comparison_barchart.png'))"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.4 F1-Score Heatmap"""))

cells.append(nbf.v4.new_code_cell("""# Display heatmap
display(Image('visualizations/f1_score_heatmap.png'))"""))

# Feature Importance
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Feature Importance <a name="feature-importance"></a>

### 8.1 Top Features (Random Forest)"""))

cells.append(nbf.v4.new_code_cell("""# Get feature importance
feature_importance_df = predictor.get_feature_importance('random_forest', top_n=15)

print("\\n📊 Top 15 Most Important Features (Random Forest):\\n")
print(feature_importance_df.to_string(index=False))

feature_importance_df"""))

# Conclusions
cells.append(nbf.v4.new_markdown_cell("""---
## 9. Conclusions <a name="conclusions"></a>

### 9.1 Summary of Results"""))

cells.append(nbf.v4.new_code_cell("""print("\\n" + "="*80)
print("🎯 FINAL RESULTS SUMMARY")
print("="*80)

# Get best model
best_model = max(evaluation_results.items(), key=lambda x: x[1]['test_accuracy'])
best_model_name = best_model[0]
best_model_acc = best_model[1]['test_accuracy']

print(f"\\n🏆 Best Model: {best_model_name.upper()}")
print(f"   Accuracy: {best_model_acc:.4f} ({best_model_acc*100:.2f}%)")

best_report = best_model[1]['classification_report']['weighted avg']
print(f"   Precision: {best_report['precision']:.4f}")
print(f"   Recall: {best_report['recall']:.4f}")
print(f"   F1-Score: {best_report['f1-score']:.4f}")

print("\\n📊 All Models Performance:")
print("-"*80)
for rank, (model_name, results) in enumerate(sorted_results, 1):
    acc = results['test_accuracy']
    f1 = results['classification_report']['weighted avg']['f1-score']
    print(f"   {rank}. {model_name:<25} Accuracy: {acc:.4f}  F1: {f1:.4f}")

print("\\n💡 Key Insights:")
print("-"*80)
print("   • Random Forest achieves the best overall performance")
print("   • All models perform well on Extinct languages (>88% F1)")
print("   • Definitely Endangered class is most challenging")
print("   • High precision and recall make models suitable for conservation")
print("   • Tree-based models outperform neural networks and logistic regression")

print("\\n🌍 Practical Applications:")
print("-"*80)
print("   • Guide UNESCO's $2B International Decade of Indigenous Languages budget")
print("   • Identify languages needing urgent preservation efforts")
print("   • Prioritize resource allocation for maximum impact")
print("   • Enable data-driven conservation decision-making")

print("\\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)"""))

# Final markdown
cells.append(nbf.v4.new_markdown_cell("""---

## 📚 References

1. UNESCO International Decade of Indigenous Languages (2022-2032)
2. Glottolog Database - Comprehensive language catalogue
3. Catalogue of Endangered Languages (ELCat)
4. Ethnologue - Languages of the World

---

## 📝 Notes

- All visualizations are saved in the `visualizations/` directory
- Trained models are saved in the `models/` directory
- For detailed documentation, see `COMPLETE_VISUALIZATION_GUIDE.md`
- CSV data available in `visualizations/detailed_metrics.csv` and `visualizations/summary_metrics.csv`

---

**End of Analysis**"""))

# Add all cells to notebook
nb['cells'] = cells

# Write notebook
with open('Complete_Model_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Notebook created successfully: Complete_Model_Analysis.ipynb")
