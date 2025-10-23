#!/usr/bin/env python3
"""
Train Deep Learning Models for Language Extinction Prediction

This script trains advanced deep learning models including:
- CNN for geographic pattern recognition
- LSTM for temporal sequence analysis  
- Transformer for complex feature interactions
- Multi-modal model for heterogeneous data fusion
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from models.deep_learning_models import DeepLearningLanguagePredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to train deep learning models"""
    
    try:
        logger.info("🚀 Starting Deep Learning Model Training...")
        
        # Check if data exists
        data_path = Path('data/glottolog_language_data.csv')
        if not data_path.exists():
            logger.error("❌ Glottolog language data not found!")
            logger.info("Please run create_glottolog_dataset.py first to generate the dataset.")
            return
        
        # Load data
        logger.info("📊 Loading Glottolog language data...")
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded {len(df)} language records")
        
        # Filter data for training (remove records with missing critical data)
        df_clean = df.dropna(subset=['endangerment_level', 'speaker_count', 'lat', 'lng'])
        logger.info(f"📋 Using {len(df_clean)} clean records for training")
        
        # Sample data for faster training (remove this for full training)
        if len(df_clean) > 2000:
            df_clean = df_clean.sample(n=2000, random_state=42)
            logger.info(f"🎯 Sampling {len(df_clean)} records for faster training")
        
        # Initialize deep learning predictor
        logger.info("🧠 Initializing Deep Learning Predictor...")
        predictor = DeepLearningLanguagePredictor()
        
        # Train models
        logger.info("🏋️ Training Deep Learning Models...")
        results = predictor.train_deep_learning_models(df_clean)
        
        # Print results
        print("\n" + "="*60)
        print("🎉 DEEP LEARNING MODEL TRAINING COMPLETED!")
        print("="*60)
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            params = result['model'].count_params()
            print(f"\n📈 {model_name.upper()} Model:")
            print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   Parameters: {params:,}")
            print(f"   Architecture: {model_name} neural network")
        
        # Save models
        logger.info("💾 Saving trained models...")
        predictor.save_models()
        
        # Get model summary
        summary = predictor.get_model_summary()
        
        print(f"\n🏆 BEST PERFORMING MODEL:")
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"   {best_model[0].upper()}: {best_model[1]['accuracy']:.3f} accuracy")
        
        print(f"\n📊 MODEL COMPARISON:")
        print(f"   Traditional ML (Random Forest): ~89%")
        print(f"   Deep Learning (Best): {best_model[1]['accuracy']:.1%}")
        
        print(f"\n✨ Deep learning models trained and saved successfully!")
        print(f"📁 Models saved in: models/")
        print(f"🔧 Ready for integration with web application!")
        
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
