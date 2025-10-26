"""
Quick Script to Generate Confusion Matrices

This is a simplified script to quickly generate confusion matrices for all models.
Run this if you just want the confusion matrices without the full pipeline.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_preprocessor import LanguageDataPreprocessor

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_processed_data():
    """Load the processed language data"""
    data_path = Path("data/processed_language_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def generate_confusion_matrices():
    """Generate confusion matrices for all trained models"""
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading processed data...")
    raw_data = load_processed_data()
    
    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = LanguageDataPreprocessor('config.yaml')
    X, y, feature_names = preprocessor.preprocess_pipeline(raw_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Get class names
    class_names = preprocessor.target_encoder.classes_
    logger.info(f"Classes: {class_names}")
    
    # Load trained models
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found. Please train models first.")
    
    # Model files
    model_files = {
        'Random Forest': 'random_forest_model.joblib',
        'XGBoost': 'xgboost_model.joblib',
        'Logistic Regression': 'logistic_regression_model.joblib',
        'Neural Network': 'neural_network_model.h5'
    }
    
    # Create figure for all confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Confusion Matrices - All Models', fontsize=20, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    model_idx = 0
    
    # Process each model
    for model_name, model_file in model_files.items():
        model_path = models_dir / model_file
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        logger.info(f"Loading {model_name}...")
        
        try:
            # Load model and make predictions
            if model_name == 'Neural Network':
                from tensorflow import keras
                model = keras.models.load_model(model_path)
                y_pred_proba = model.predict(X_test_scaled.values, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                # Convert back to original labels
                y_pred = preprocessor.target_encoder.inverse_transform(y_pred)
                y_test_labels = preprocessor.target_encoder.inverse_transform(y_test)
            else:
                model = joblib.load(model_path)
                if model_name == 'XGBoost':
                    y_pred = model.predict(X_test_scaled.values)
                else:
                    y_pred = model.predict(X_test_scaled)
                y_test_labels = preprocessor.target_encoder.inverse_transform(y_test)
                y_pred = preprocessor.target_encoder.inverse_transform(y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test_labels, y_pred, labels=class_names)
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[model_idx], square=True, linewidths=0.5, linecolor='gray',
                       cbar_kws={'label': 'Percentage'})
            
            axes[model_idx].set_title(f'{model_name}\nAccuracy: {accuracy:.2%}',
                                     fontsize=14, fontweight='bold', pad=10)
            axes[model_idx].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            axes[model_idx].set_ylabel('True Label', fontsize=12, fontweight='bold')
            axes[model_idx].set_xticklabels(axes[model_idx].get_xticklabels(), 
                                           rotation=45, ha='right', fontsize=10)
            axes[model_idx].set_yticklabels(axes[model_idx].get_yticklabels(), 
                                           rotation=0, fontsize=10)
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
            
            # Save individual confusion matrix
            fig_individual = plt.figure(figsize=(10, 8))
            ax_individual = fig_individual.add_subplot(111)
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax_individual, square=True, linewidths=0.5, linecolor='gray',
                       cbar_kws={'label': 'Percentage'})
            
            ax_individual.set_title(f'{model_name}\nNormalized Confusion Matrix\nAccuracy: {accuracy:.2%}',
                                   fontsize=14, fontweight='bold', pad=15)
            ax_individual.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax_individual.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax_individual.set_xticklabels(ax_individual.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            individual_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            logger.info(f"Saved individual confusion matrix: {individual_path}")
            
            model_idx += 1
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {str(e)}")
            continue
    
    # Save combined figure
    plt.tight_layout()
    combined_path = output_dir / "confusion_matrices_all_models.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Confusion matrices saved successfully!")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Combined plot: {combined_path.name}")
    logger.info(f"{'='*60}")
    
    print("\n" + "="*60)
    print("CONFUSION MATRICES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput Location: {output_dir.absolute()}")
    print("\nFiles Created:")
    print(f"  ✓ {combined_path.name}")
    for model_name in model_files.keys():
        individual_file = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        if (output_dir / individual_file).exists():
            print(f"  ✓ {individual_file}")
    print("="*60)


if __name__ == "__main__":
    try:
        generate_confusion_matrices()
    except Exception as e:
        logger.error(f"Failed to generate confusion matrices: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Trained the models (run main.py --step train)")
        print("  2. Processed data file exists (data/processed_language_data.csv)")
        sys.exit(1)

