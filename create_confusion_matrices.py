"""
Create Comprehensive Confusion Matrices for All Models

This script generates detailed confusion matrices for all trained models:
- Random Forest
- XGBoost
- Neural Network
- Logistic Regression

Output:
- Individual confusion matrix plots for each model
- Combined confusion matrix visualization
- Normalized confusion matrices (percentages)
- Classification metrics alongside matrices
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_loader import LanguageDataLoader
from data.data_preprocessor import LanguageDataPreprocessor
from models.ml_models import LanguageExtinctionPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(cm, class_names, model_name, normalize=False, save_path=None):
    """
    Plot a single confusion matrix with enhanced visualization
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        model_name: Name of the model
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'{model_name}\nNormalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name}\nConfusion Matrix'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                ax=ax, square=True, linewidths=0.5, linecolor='gray')
    
    # Set labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_all_confusion_matrices(evaluation_results, class_names, output_dir):
    """
    Create a combined plot with all confusion matrices
    
    Args:
        evaluation_results: Dictionary of model evaluation results
        class_names: List of class names
        output_dir: Directory to save outputs
    """
    n_models = len(evaluation_results)
    
    # Create figure with subplots (2 rows x 2 cols for 4 models)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], square=True, linewidths=0.5, linecolor='gray',
                   cbar_kws={'label': 'Percentage'})
        
        # Add accuracy to title
        accuracy = results['test_accuracy']
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {accuracy:.2%}',
                           fontsize=14, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        # Rotate labels
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save combined figure
    save_path = output_dir / "confusion_matrices_combined.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined confusion matrices to {save_path}")
    
    return fig


def create_confusion_matrix_with_metrics(cm, class_names, model_name, classification_rep, output_dir):
    """
    Create confusion matrix with classification metrics side by side
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Model name
        classification_rep: Classification report dictionary
        output_dir: Output directory
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Create grid for subplots
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    # Left: Confusion Matrix
    ax1 = fig.add_subplot(gs[0])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax1, square=True, linewidths=0.5, linecolor='gray',
               cbar_kws={'label': 'Percentage'})
    
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name}\nNormalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Right: Classification Metrics Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    # Prepare metrics data
    metrics_data = []
    for class_name in class_names:
        if class_name in classification_rep:
            metrics = classification_rep[class_name]
            metrics_data.append([
                class_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{int(metrics['support'])}"
            ])
    
    # Add overall metrics
    if 'weighted avg' in classification_rep:
        metrics = classification_rep['weighted avg']
        metrics_data.append([
            'Weighted Avg',
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            f"{int(metrics['support'])}"
        ])
    
    # Create table
    table = ax2.table(cellText=metrics_data,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(metrics_data):  # Last row (weighted avg)
                cell.set_facecolor('#E7E6E6')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax2.set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    save_path = output_dir / f"confusion_matrix_with_metrics_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix with metrics to {save_path}")
    
    return fig


def create_error_analysis(evaluation_results, class_names, output_dir):
    """
    Create error analysis visualization showing misclassification patterns
    
    Args:
        evaluation_results: Dictionary of model evaluation results
        class_names: List of class names
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Error Analysis - Misclassification Patterns', fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        # Calculate misclassification matrix (set diagonal to 0)
        error_matrix = cm.copy().astype(float)
        np.fill_diagonal(error_matrix, 0)
        
        # Normalize by row (true labels)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        error_matrix = error_matrix / row_sums
        
        # Create heatmap
        sns.heatmap(error_matrix, annot=True, fmt='.2%', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], square=True, linewidths=0.5, linecolor='gray',
                   cbar_kws={'label': 'Error Rate'})
        
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nMisclassification Rates',
                           fontsize=14, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        # Rotate labels
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / "error_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved error analysis to {save_path}")
    
    return fig


def main():
    """Main function to create all confusion matrices"""
    logger.info("Starting confusion matrix generation...")
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize components
    data_loader = LanguageDataLoader('config.yaml')
    preprocessor = LanguageDataPreprocessor('config.yaml')
    predictor = LanguageExtinctionPredictor('config.yaml')
    
    # Load data
    logger.info("Loading and preprocessing data...")
    datasets = data_loader.load_all_datasets()
    raw_data = data_loader.merge_datasets()
    
    # Preprocess data
    X, y, feature_names = preprocessor.preprocess_pipeline(raw_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Get class names from the target encoder
    class_names = preprocessor.target_encoder.classes_
    logger.info(f"Class names: {class_names}")
    
    # Train models
    logger.info("Training models...")
    model_results = predictor.train_all_models(X_train_scaled, y_train)
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)
    
    # Create individual confusion matrices for each model
    logger.info("Creating individual confusion matrices...")
    for model_name, results in evaluation_results.items():
        cm = results['confusion_matrix']
        classification_rep = results['classification_report']
        
        # Non-normalized confusion matrix
        fig1 = plot_confusion_matrix(
            cm, class_names, 
            model_name.replace('_', ' ').title(),
            normalize=False,
            save_path=output_dir / f"confusion_matrix_{model_name}_counts.png"
        )
        plt.close(fig1)
        
        # Normalized confusion matrix
        fig2 = plot_confusion_matrix(
            cm, class_names,
            model_name.replace('_', ' ').title(),
            normalize=True,
            save_path=output_dir / f"confusion_matrix_{model_name}_normalized.png"
        )
        plt.close(fig2)
        
        # Confusion matrix with metrics
        fig3 = create_confusion_matrix_with_metrics(
            cm, class_names,
            model_name.replace('_', ' ').title(),
            classification_rep,
            output_dir
        )
        plt.close(fig3)
    
    # Create combined confusion matrix plot
    logger.info("Creating combined confusion matrix visualization...")
    fig4 = plot_all_confusion_matrices(evaluation_results, class_names, output_dir)
    plt.close(fig4)
    
    # Create error analysis
    logger.info("Creating error analysis...")
    fig5 = create_error_analysis(evaluation_results, class_names, output_dir)
    plt.close(fig5)
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("CONFUSION MATRIX GENERATION COMPLETE")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print("-" * 40)
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Weighted F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
        print(f"Weighted Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
        print(f"Weighted Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
    
    print("\n" + "="*60)
    print("FILES GENERATED:")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nIndividual confusion matrices:")
    for model_name in evaluation_results.keys():
        print(f"  - confusion_matrix_{model_name}_counts.png")
        print(f"  - confusion_matrix_{model_name}_normalized.png")
        print(f"  - confusion_matrix_with_metrics_{model_name}.png")
    print("\nCombined visualizations:")
    print("  - confusion_matrices_combined.png")
    print("  - error_analysis.png")
    print("="*60)
    
    logger.info("All confusion matrices created successfully!")


if __name__ == "__main__":
    main()

