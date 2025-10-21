#!/usr/bin/env python3
"""
Demo Script for Language Extinction Risk Prediction

This script demonstrates the complete pipeline from data loading to visualization.
Run this script to see the project in action.

Usage:
    python demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run the complete language extinction prediction demo"""
    
    print("=" * 60)
    print("LANGUAGE EXTINCTION RISK PREDICTION - DEMO")
    print("=" * 60)
    print()
    
    try:
        # Import our modules
        from data.data_loader import LanguageDataLoader
        from data.data_preprocessor import LanguageDataPreprocessor
        from models.ml_models import LanguageExtinctionPredictor
        from visualization.visualizer import LanguageVisualizer
        
        print("✓ Modules imported successfully")
        
        # Step 1: Load data
        print("\n1. Loading datasets...")
        loader = LanguageDataLoader()
        datasets = loader.load_all_datasets()
        merged_data = loader.merge_datasets()
        print(f"   ✓ Loaded {len(datasets)} datasets")
        print(f"   ✓ Merged dataset shape: {merged_data.shape}")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        preprocessor = LanguageDataPreprocessor()
        X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        print(f"   ✓ Preprocessed data: {X_train_scaled.shape[0]} training, {X_test_scaled.shape[0]} test samples")
        print(f"   ✓ Features: {len(feature_names)}")
        
        # Step 3: Train models
        print("\n3. Training machine learning models...")
        predictor = LanguageExtinctionPredictor()
        model_results = predictor.train_all_models(X_train_scaled, y_train)
        evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)
        print(f"   ✓ Trained {len(model_results)} models")
        
        # Display results
        print("\n4. Model Performance Results:")
        print("   " + "-" * 40)
        for model_name, results in evaluation_results.items():
            accuracy = results['test_accuracy']
            f1_score = results['classification_report']['weighted avg']['f1-score']
            print(f"   {model_name.replace('_', ' ').title():20} | Accuracy: {accuracy:.3f} | F1: {f1_score:.3f}")
        
        # Step 4: Create visualizations
        print("\n5. Creating visualizations...")
        visualizer = LanguageVisualizer()
        
        # Get feature importance
        feature_importance = predictor.get_feature_importance('random_forest', top_n=15)
        feature_importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
        
        # Create key visualizations
        try:
            visualizer.create_global_endangerment_map(merged_data)
            print("   ✓ Global endangerment map created")
        except Exception as e:
            print(f"   ⚠ Global map creation failed: {str(e)[:50]}...")
        
        try:
            visualizer.create_feature_importance_chart(feature_importance_dict)
            print("   ✓ Feature importance chart created")
        except Exception as e:
            print(f"   ⚠ Feature importance chart failed: {str(e)[:50]}...")
        
        try:
            visualizer.create_model_performance_comparison(evaluation_results)
            print("   ✓ Model performance comparison created")
        except Exception as e:
            print(f"   ⚠ Performance comparison failed: {str(e)[:50]}...")
        
        # Step 5: Generate insights
        print("\n6. Key Insights:")
        print("   " + "-" * 40)
        
        # Best model
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"   • Best Model: {best_model[0].replace('_', ' ').title()}")
        print(f"   • Best Accuracy: {best_model[1]['test_accuracy']:.1%}")
        
        # Top features
        top_3_features = feature_importance.head(3)
        print(f"   • Top 3 Most Important Features:")
        for _, row in top_3_features.iterrows():
            print(f"     - {row['feature']}: {row['importance']:.3f}")
        
        # Performance vs target
        target_accuracy = 0.85
        meets_target = best_model[1]['test_accuracy'] >= target_accuracy
        print(f"   • Meets 85% accuracy target: {'Yes' if meets_target else 'No'}")
        
        # Step 6: Summary
        print("\n7. Project Summary:")
        print("   " + "-" * 40)
        print(f"   • Total languages analyzed: {len(merged_data)}")
        print(f"   • Endangerment levels: {len(y.unique())}")
        print(f"   • Features engineered: {len(feature_names)}")
        print(f"   • Models trained: {len(model_results)}")
        print(f"   • Visualizations created: visualizations/")
        print(f"   • Models saved: models/")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("• Open visualizations/ folder to view interactive charts")
        print("• Check models/ folder for trained models")
        print("• Run 'python main.py --step all --report' for full analysis")
        print("• Open Language_Extinction_Analysis.ipynb for detailed exploration")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPlease install required dependencies:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check the error and try again.")
        print("For help, see README.md or run with --help")

if __name__ == "__main__":
    main()
