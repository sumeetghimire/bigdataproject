# Confusion Matrix Generation Guide

This guide explains how to generate confusion matrices for all models in the Language Extinction Risk Prediction project.

## 📊 What You'll Get

The scripts will generate comprehensive confusion matrices showing:
- **True Positives, False Positives, True Negatives, False Negatives** for each class
- **Normalized percentages** for better interpretation
- **Model accuracy** displayed on each matrix
- **Individual matrices** for each model (Random Forest, XGBoost, Neural Network, Logistic Regression)
- **Combined visualization** with all models in one figure
- **Error analysis** showing misclassification patterns

## 🚀 Quick Start (Recommended)

If you already have trained models and processed data, use the quick script:

```bash
python generate_confusion_matrices.py
```

This will:
1. Load your trained models from `models/` directory
2. Load processed data from `data/processed_language_data.csv`
3. Generate confusion matrices for all models
4. Save outputs to `visualizations/` directory

### Output Files:
- `confusion_matrices_all_models.png` - Combined view of all models
- `confusion_matrix_random_forest.png` - Individual Random Forest matrix
- `confusion_matrix_xgboost.png` - Individual XGBoost matrix
- `confusion_matrix_neural_network.png` - Individual Neural Network matrix
- `confusion_matrix_logistic_regression.png` - Individual Logistic Regression matrix

## 🔬 Comprehensive Analysis (Full Pipeline)

For a complete analysis with additional visualizations:

```bash
python create_confusion_matrices.py
```

This will:
1. Load and preprocess data from scratch
2. Train all models
3. Generate confusion matrices with multiple views:
   - Count-based matrices
   - Normalized (percentage) matrices
   - Matrices with classification metrics (precision, recall, F1-score)
   - Error analysis showing misclassification patterns

### Output Files:
- `confusion_matrix_{model}_counts.png` - Raw count matrices
- `confusion_matrix_{model}_normalized.png` - Percentage matrices
- `confusion_matrix_with_metrics_{model}.png` - Matrix + metrics table
- `confusion_matrices_combined.png` - All models in one view
- `error_analysis.png` - Misclassification pattern analysis

## 📋 Prerequisites

### Required Files:
1. **Trained Models** (in `models/` directory):
   - `random_forest_model.joblib`
   - `xgboost_model.joblib`
   - `neural_network_model.h5`
   - `logistic_regression_model.joblib`

2. **Data** (in `data/` directory):
   - `processed_language_data.csv`

### If Models Don't Exist:

Train the models first:
```bash
python main.py --step train
```

Or run the full pipeline:
```bash
python main.py --step all
```

## 📊 Understanding the Confusion Matrix

### Matrix Layout:
```
                Predicted
              Safe  Vuln  Def.End  Sev.End  Crit.End  Extinct
Actual  Safe   [TP]  [FP]   [FP]     [FP]     [FP]     [FP]
        Vuln   [FN]  [TP]   [FP]     [FP]     [FP]     [FP]
        ...
```

### Reading the Matrix:
- **Diagonal values (TP)**: Correct predictions
- **Off-diagonal values**: Misclassifications
- **Row totals**: Actual class distribution
- **Column totals**: Predicted class distribution

### Normalized Matrix:
- Shows **percentages** instead of counts
- Each row sums to 100%
- Easier to compare across classes with different sample sizes

### Color Coding:
- **Darker blue**: Higher values (more predictions)
- **Lighter blue**: Lower values (fewer predictions)
- **Diagonal should be darkest**: Indicates good performance

## 🎯 Interpreting Results

### Good Model Performance:
- ✅ **Dark diagonal**: Most predictions are correct
- ✅ **Light off-diagonal**: Few misclassifications
- ✅ **High accuracy** (>85%)

### Areas for Improvement:
- ⚠️ **Bright off-diagonal cells**: Common misclassifications
- ⚠️ **Light diagonal cells**: Classes being confused
- ⚠️ **Imbalanced rows**: Some classes harder to predict

## 🔍 Model Comparison

Compare models by looking at:
1. **Overall accuracy** (shown in title)
2. **Diagonal strength** (darker = better)
3. **Misclassification patterns** (which classes get confused)
4. **Class-specific performance** (some models better for certain classes)

## 📈 Using Results in Your Report

### For Academic Reports:
1. Include the **combined confusion matrix** figure
2. Highlight **best performing model**
3. Discuss **common misclassification patterns**
4. Explain **why certain classes are confused** (e.g., "Vulnerable" vs "Definitely Endangered")

### For Presentations:
1. Show **individual matrices** for your best model
2. Use **normalized matrices** for easier interpretation
3. Include **accuracy metrics** in your slides
4. Discuss **practical implications** of misclassifications

## 🛠️ Troubleshooting

### Error: "Models directory not found"
**Solution**: Train models first
```bash
python main.py --step train
```

### Error: "Processed data not found"
**Solution**: Run data preprocessing
```bash
python main.py --step data
python main.py --step preprocess
```

### Error: "Module not found"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Poor Quality Images
**Solution**: The scripts save at 300 DPI. If you need higher resolution, edit the `dpi` parameter:
```python
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Higher quality
```

## 📝 Customization

### Change Color Scheme:
Edit the `cmap` parameter in the scripts:
```python
sns.heatmap(..., cmap='YlOrRd')  # Yellow-Orange-Red
sns.heatmap(..., cmap='RdYlGn')  # Red-Yellow-Green
sns.heatmap(..., cmap='viridis') # Viridis colormap
```

### Adjust Figure Size:
```python
fig, axes = plt.subplots(2, 2, figsize=(24, 20))  # Larger figure
```

### Change Font Sizes:
```python
axes[idx].set_title(..., fontsize=16)  # Larger title
```

## 📚 Additional Resources

- **Confusion Matrix Explained**: https://en.wikipedia.org/wiki/Confusion_matrix
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- **Seaborn Heatmaps**: https://seaborn.pydata.org/generated/seaborn.heatmap.html

## 💡 Tips

1. **Always use normalized matrices** when comparing models with imbalanced classes
2. **Look at per-class metrics** (precision, recall, F1) not just overall accuracy
3. **Identify systematic errors** - if a model always confuses two specific classes, investigate why
4. **Consider the cost of errors** - is it worse to misclassify "Safe" as "Extinct" or vice versa?

## 🎓 For Your Assignment

Include in your report:
- [ ] Combined confusion matrix figure
- [ ] Discussion of model performance
- [ ] Analysis of misclassification patterns
- [ ] Comparison of all four models
- [ ] Explanation of why certain classes are harder to predict
- [ ] Suggestions for improvement

---

**Need Help?** Check the main README.md or contact your instructor.

