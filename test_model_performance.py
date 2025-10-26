#!/usr/bin/env python3
"""
Comprehensive Model Performance Testing and Evaluation Script

This script demonstrates how to test accuracy and evaluate performance metrics
for all trained models in the Language Extinction Risk Prediction system.

Performance Metrics Explained:
- Accuracy: Overall correctness of predictions
- Precision: Of predicted positive cases, how many were actually positive
- Recall: Of actual positive cases, how many were correctly predicted
- F1-Score: Harmonic mean of precision and recall
- Support: Number of actual occurrences of each class
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_loader import LanguageDataLoader
from data.data_preprocessor import LanguageDataPreprocessor
from models.ml_models import LanguageExtinctionPredictor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceTester:
    """
    Comprehensive model performance testing and evaluation class
    """
    
    def __init__(self):
        self.predictor = None
        self.X_test = None
        self.y_test = None
        self.evaluation_results = {}
        self.class_names = None
        
    def load_data_and_models(self):
        """Load data and trained models"""
        logger.info("Loading data and preprocessing...")
        
        # Load data
        loader = LanguageDataLoader()
        datasets = loader.load_all_datasets()
        merged_data = loader.merge_datasets()
        
        # Preprocess data
        preprocessor = LanguageDataPreprocessor()
        X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)
        
        # Split data (same split as training)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.class_names = sorted(y_test.unique())
        
        # Initialize predictor and load models
        self.predictor = LanguageExtinctionPredictor()
        
        # Check if models exist, if not train them
        models_dir = Path("models")
        if not models_dir.exists() or len(list(models_dir.glob("*.joblib"))) == 0:
            logger.info("No trained models found. Training models...")
            self.predictor.train_all_models(X_train_scaled, y_train)
            self.predictor.save_models()
        else:
            logger.info("Loading existing trained models...")
            self.predictor.load_models()
        
        logger.info(f"Data loaded: {X_test_scaled.shape[0]} test samples, {len(self.class_names)} classes")
        
    def calculate_detailed_metrics(self, model_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a specific model
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            Dict[str, Any]: Detailed performance metrics
        """
        logger.info(f"Calculating detailed metrics for {model_name}...")
        
        # Get model predictions
        evaluation_result = self.predictor.evaluate_model(model_name, self.X_test, self.y_test)
        y_pred = evaluation_result['predictions']
        
        # Calculate basic metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Calculate precision, recall, f1 for each class and overall
        precision_macro = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(self.y_test, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(self.y_test, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(self.y_test, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Get classification report
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_name in self.class_names:
            if class_name in class_report:
                per_class_metrics[class_name] = {
                    'precision': class_report[class_name]['precision'],
                    'recall': class_report[class_name]['recall'],
                    'f1-score': class_report[class_name]['f1-score'],
                    'support': class_report[class_name]['support']
                }
        
        detailed_metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': {
                'macro': precision_macro,
                'micro': precision_micro,
                'weighted': precision_weighted
            },
            'recall': {
                'macro': recall_macro,
                'micro': recall_micro,
                'weighted': recall_weighted
            },
            'f1_score': {
                'macro': f1_macro,
                'micro': f1_micro,
                'weighted': f1_weighted
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred
        }
        
        return detailed_metrics
    
    def test_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Test all trained models and calculate comprehensive metrics
        
        Returns:
            Dict[str, Dict[str, Any]]: Detailed metrics for all models
        """
        logger.info("Testing all models...")
        
        self.evaluation_results = {}
        
        # Get available models
        available_models = list(self.predictor.results.keys())
        logger.info(f"Available models: {available_models}")
        
        for model_name in available_models:
            try:
                self.evaluation_results[model_name] = self.calculate_detailed_metrics(model_name)
                logger.info(f"✅ {model_name} evaluation completed")
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
                continue
        
        return self.evaluation_results
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Overall comparison table
        print("\n📊 OVERALL PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12}")
        print("-" * 60)
        
        for model_name, metrics in self.evaluation_results.items():
            accuracy = metrics['accuracy']
            f1_macro = metrics['f1_score']['macro']
            f1_weighted = metrics['f1_score']['weighted']
            print(f"{model_name:<20} {accuracy:<10.4f} {f1_macro:<10.4f} {f1_weighted:<12.4f}")
        
        # Detailed metrics for each model
        for model_name, metrics in self.evaluation_results.items():
            print(f"\n🔍 DETAILED ANALYSIS: {model_name.upper()}")
            print("-" * 50)
            
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision (Macro): {metrics['precision']['macro']:.4f}")
            print(f"Recall (Macro): {metrics['recall']['macro']:.4f}")
            print(f"F1-Score (Macro): {metrics['f1_score']['macro']:.4f}")
            
            print("\nPer-Class Performance:")
            print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
            print("-" * 65)
            
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                precision = class_metrics['precision']
                recall = class_metrics['recall']
                f1 = class_metrics['f1-score']
                support = class_metrics['support']
                print(f"{class_name:<25} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8.0f}")
    
    def explain_metrics(self):
        """Explain what each performance metric means in the context of language extinction prediction"""
        print("\n" + "="*80)
        print("📚 PERFORMANCE METRICS EXPLAINED")
        print("="*80)
        
        explanations = {
            "Accuracy": {
                "definition": "Percentage of correct predictions out of total predictions",
                "formula": "(True Positives + True Negatives) / Total Predictions",
                "context": "How often our model correctly predicts the endangerment level",
                "importance": "High accuracy means the model is reliable for conservation planning"
            },
            "Precision": {
                "definition": "Of all positive predictions, how many were actually correct",
                "formula": "True Positives / (True Positives + False Positives)",
                "context": "When we predict a language is 'Critically Endangered', how often are we right?",
                "importance": "High precision prevents wasting resources on false alarms"
            },
            "Recall (Sensitivity)": {
                "definition": "Of all actual positive cases, how many did we correctly identify",
                "formula": "True Positives / (True Positives + False Negatives)",
                "context": "Of all truly 'Critically Endangered' languages, how many did we catch?",
                "importance": "High recall ensures we don't miss languages that need urgent help"
            },
            "F1-Score": {
                "definition": "Harmonic mean of precision and recall",
                "formula": "2 × (Precision × Recall) / (Precision + Recall)",
                "context": "Balanced measure of model performance",
                "importance": "Good F1 means balanced precision and recall - ideal for conservation"
            },
            "Support": {
                "definition": "Number of actual occurrences of each class in the test set",
                "formula": "Count of samples in each endangerment category",
                "context": "How many languages of each endangerment level we're testing on",
                "importance": "Shows if our test set is representative of real-world distribution"
            }
        }
        
        for metric, info in explanations.items():
            print(f"\n🔸 {metric.upper()}")
            print(f"   Definition: {info['definition']}")
            print(f"   Formula: {info['formula']}")
            print(f"   In our context: {info['context']}")
            print(f"   Why it matters: {info['importance']}")
    
    def create_confusion_matrix_plots(self):
        """Create confusion matrix visualizations for all models"""
        logger.info("Creating confusion matrix visualizations...")
        
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            if idx >= 4:  # Only plot first 4 models
                break
                
            conf_matrix = metrics['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name.title()} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Confusion matrix plots saved to visualizations/confusion_matrices.png")
    
    def analyze_model_strengths_weaknesses(self):
        """Analyze strengths and weaknesses of each model"""
        print("\n" + "="*80)
        print("🎯 MODEL STRENGTHS & WEAKNESSES ANALYSIS")
        print("="*80)
        
        for model_name, metrics in self.evaluation_results.items():
            print(f"\n🔍 {model_name.upper()} ANALYSIS")
            print("-" * 40)
            
            per_class = metrics['per_class_metrics']
            
            # Find best and worst performing classes
            f1_scores = {class_name: class_metrics['f1-score'] 
                        for class_name, class_metrics in per_class.items()}
            
            best_class = max(f1_scores, key=f1_scores.get)
            worst_class = min(f1_scores, key=f1_scores.get)
            
            print(f"✅ Strongest at predicting: {best_class} (F1: {f1_scores[best_class]:.4f})")
            print(f"⚠️  Weakest at predicting: {worst_class} (F1: {f1_scores[worst_class]:.4f})")
            
            # Overall assessment
            accuracy = metrics['accuracy']
            if accuracy >= 0.95:
                assessment = "Excellent - Ready for production use"
            elif accuracy >= 0.90:
                assessment = "Very Good - Suitable for most applications"
            elif accuracy >= 0.85:
                assessment = "Good - May need some improvements"
            elif accuracy >= 0.80:
                assessment = "Fair - Requires significant improvements"
            else:
                assessment = "Poor - Needs major revisions"
            
            print(f"📊 Overall Assessment: {assessment}")
            
            # Recommendations
            print("💡 Recommendations:")
            if f1_scores[worst_class] < 0.8:
                print(f"   - Collect more training data for '{worst_class}' category")
                print(f"   - Consider feature engineering specific to '{worst_class}' languages")
            
            if metrics['precision']['macro'] < metrics['recall']['macro']:
                print("   - Model tends to over-predict - consider adjusting decision thresholds")
            elif metrics['recall']['macro'] < metrics['precision']['macro']:
                print("   - Model is conservative - might miss some endangered languages")

def main():
    """Main function to run comprehensive model testing"""
    print("🚀 Starting Comprehensive Model Performance Testing...")
    
    # Initialize tester
    tester = ModelPerformanceTester()
    
    # Load data and models
    tester.load_data_and_models()
    
    # Test all models
    evaluation_results = tester.test_all_models()
    
    # Print comprehensive analysis
    tester.print_performance_summary()
    tester.explain_metrics()
    tester.analyze_model_strengths_weaknesses()
    
    # Create visualizations
    try:
        # Create visualizations directory if it doesn't exist
        Path("visualizations").mkdir(exist_ok=True)
        tester.create_confusion_matrix_plots()
    except Exception as e:
        logger.warning(f"Could not create visualizations: {str(e)}")
    
    print("\n🎉 Model performance testing completed!")
    print("📁 Results saved to visualizations/ directory")
    
    return tester, evaluation_results

if __name__ == "__main__":
    tester, results = main()
