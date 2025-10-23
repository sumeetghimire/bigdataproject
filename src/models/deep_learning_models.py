#!/usr/bin/env python3
"""
Deep Learning Models Module for Language Extinction Risk Prediction

This module implements advanced deep learning architectures:
- Convolutional Neural Network (CNN) for geographic pattern recognition
- Long Short-Term Memory (LSTM) for temporal sequence analysis
- Transformer for complex feature interactions
- Multi-Modal Deep Learning for heterogeneous data fusion
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import yaml
from pathlib import Path

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepLearningLanguagePredictor:
    """
    Advanced deep learning models for language extinction prediction
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the deep learning predictor"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Deep learning specific configuration
        self.dl_config = self.config.get('deep_learning', {
            'cnn': {
                'filters': [32, 64, 128],
                'kernel_sizes': [3, 3, 3],
                'pool_sizes': [2, 2, 2],
                'dropout_rate': 0.3,
                'dense_units': [256, 128, 64]
            },
            'lstm': {
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.2,
                'dense_units': [128, 64]
            },
            'transformer': {
                'num_heads': 8,
                'num_layers': 4,
                'd_model': 128,
                'dff': 512,
                'dropout_rate': 0.1,
                'max_sequence_length': 100
            },
            'multimodal': {
                'geographic_branch_units': [64, 32],
                'linguistic_branch_units': [128, 64],
                'socioeconomic_branch_units': [64, 32],
                'fusion_units': [256, 128, 64],
                'dropout_rate': 0.3
            }
        })
    
    def prepare_geographic_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare geographic data for CNN input
        Creates a 2D grid representation of language locations
        """
        logger.info("Preparing geographic data for CNN...")
        
        # Create geographic grid
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lng_min, lng_max = df['lng'].min(), df['lng'].max()
        
        # Create grid resolution
        grid_size = 100
        lat_bins = np.linspace(lat_min, lat_max, grid_size)
        lng_bins = np.linspace(lng_min, lng_max, grid_size)
        
        # Create 3D array: [samples, height, width, channels]
        # Channels: [language_density, endangerment_heat, speaker_density]
        geographic_data = np.zeros((len(df), grid_size, grid_size, 3))
        
        for idx, row in df.iterrows():
            # Find grid position
            lat_idx = np.digitize(row['lat'], lat_bins) - 1
            lng_idx = np.digitize(row['lng'], lng_bins) - 1
            
            # Ensure indices are within bounds
            lat_idx = max(0, min(lat_idx, grid_size - 1))
            lng_idx = max(0, min(lng_idx, grid_size - 1))
            
            # Set values
            geographic_data[idx, lat_idx, lng_idx, 0] = 1.0  # Language presence
            geographic_data[idx, lat_idx, lng_idx, 1] = row.get('lei_score', 0) / 100.0  # Endangerment
            geographic_data[idx, lat_idx, lng_idx, 2] = min(row.get('speaker_count', 0) / 1000000.0, 1.0)  # Speaker density
        
        return geographic_data
    
    def prepare_sequence_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare temporal sequence data for LSTM
        Creates sequences based on language family relationships and geographic proximity
        """
        logger.info("Preparing sequence data for LSTM...")
        
        # Create sequences based on language family and geographic proximity
        sequence_length = 10
        features = ['speaker_count', 'lei_score', 'intergenerational_transmission', 'lat', 'lng']
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features].fillna(0))
        self.scalers['sequence'] = scaler
        
        # Create sequences by grouping similar languages
        sequences = []
        for family in df['family_id'].unique()[:100]:  # Limit for performance
            family_languages = df[df['family_id'] == family].sort_values('speaker_count', ascending=False)
            
            if len(family_languages) >= sequence_length:
                family_indices = family_languages.index[:sequence_length]
                sequence = scaled_features[family_indices]
                sequences.append(sequence)
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in sequences) if sequences else sequence_length
        padded_sequences = np.zeros((len(sequences), max_length, len(features)))
        
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        
        return padded_sequences
    
    def create_cnn_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """
        Create Convolutional Neural Network for geographic pattern recognition
        """
        logger.info("Creating CNN model for geographic pattern recognition...")
        
        config = self.dl_config['cnn']
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Convolutional layers
            layers.Conv2D(config['filters'][0], config['kernel_sizes'][0], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(config['pool_sizes'][0]),
            layers.Dropout(config['dropout_rate']),
            
            layers.Conv2D(config['filters'][1], config['kernel_sizes'][1], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(config['pool_sizes'][1]),
            layers.Dropout(config['dropout_rate']),
            
            layers.Conv2D(config['filters'][2], config['kernel_sizes'][2], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config['dropout_rate']),
            
            # Dense layers
            layers.Dense(config['dense_units'][0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            layers.Dense(config['dense_units'][1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            layers.Dense(config['dense_units'][2], activation='relu'),
            layers.Dropout(config['dropout_rate']),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['cnn'] = model
        return model
    
    def create_lstm_model(self, input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
        """
        Create LSTM model for temporal sequence analysis
        """
        logger.info("Creating LSTM model for sequence analysis...")
        
        config = self.dl_config['lstm']
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # LSTM layers
            layers.LSTM(config['lstm_units'][0], 
                       return_sequences=True,
                       dropout=config['dropout_rate'],
                       recurrent_dropout=config['recurrent_dropout']),
            layers.BatchNormalization(),
            
            layers.LSTM(config['lstm_units'][1], 
                       return_sequences=True,
                       dropout=config['dropout_rate'],
                       recurrent_dropout=config['recurrent_dropout']),
            layers.BatchNormalization(),
            
            layers.LSTM(config['lstm_units'][2], 
                       dropout=config['dropout_rate'],
                       recurrent_dropout=config['recurrent_dropout']),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(config['dense_units'][0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            layers.Dense(config['dense_units'][1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['lstm'] = model
        return model
    
    def create_transformer_model(self, input_dim: int, num_classes: int) -> keras.Model:
        """
        Create Transformer model for complex feature interactions
        """
        logger.info("Creating Transformer model for feature interactions...")
        
        config = self.dl_config['transformer']
        
        # Multi-head attention layer
        class MultiHeadAttention(layers.Layer):
            def __init__(self, d_model, num_heads):
                super(MultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model
                
                self.depth = d_model // num_heads
                
                self.wq = layers.Dense(d_model)
                self.wk = layers.Dense(d_model)
                self.wv = layers.Dense(d_model)
                
                self.dense = layers.Dense(d_model)
                
            def call(self, x):
                batch_size = tf.shape(x)[0]
                
                q = self.wq(x)
                k = self.wk(x)
                v = self.wv(x)
                
                q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth))
                k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth))
                v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
                
                q = tf.transpose(q, perm=[0, 2, 1, 3])
                k = tf.transpose(k, perm=[0, 2, 1, 3])
                v = tf.transpose(v, perm=[0, 2, 1, 3])
                
                attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32)))
                attention = tf.nn.dropout(attention, rate=config['dropout_rate'])
                
                out = tf.matmul(attention, v)
                out = tf.transpose(out, perm=[0, 2, 1, 3])
                out = tf.reshape(out, (batch_size, -1, self.d_model))
                
                return self.dense(out)
        
        # Transformer block
        class TransformerBlock(layers.Layer):
            def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
                super(TransformerBlock, self).__init__()
                self.mha = MultiHeadAttention(d_model, num_heads)
                self.ffn = keras.Sequential([
                    layers.Dense(dff, activation='relu'),
                    layers.Dense(d_model)
                ])
                
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                
                self.dropout1 = layers.Dropout(dropout_rate)
                self.dropout2 = layers.Dropout(dropout_rate)
                
            def call(self, x):
                attn_output = self.mha(x)
                attn_output = self.dropout1(attn_output)
                out1 = self.layernorm1(x + attn_output)
                
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output)
                return self.layernorm2(out1 + ffn_output)
        
        # Build model
        inputs = layers.Input(shape=(input_dim,))
        
        # Embedding layer
        x = layers.Dense(config['d_model'])(inputs)
        x = layers.Reshape((1, config['d_model']))(x)
        
        # Transformer blocks
        for _ in range(config['num_layers']):
            x = TransformerBlock(config['d_model'], config['num_heads'], 
                               config['dff'], config['dropout_rate'])(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(config['dff'], activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
        
        x = layers.Dense(config['dff'] // 2, activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['transformer'] = model
        return model
    
    def create_multimodal_model(self, geographic_shape: Tuple[int, int, int], 
                               linguistic_dim: int, socioeconomic_dim: int, 
                               num_classes: int) -> keras.Model:
        """
        Create multi-modal deep learning model for heterogeneous data fusion
        """
        logger.info("Creating multi-modal deep learning model...")
        
        config = self.dl_config['multimodal']
        
        # Geographic branch (CNN)
        geo_input = layers.Input(shape=geographic_shape, name='geographic')
        geo_x = layers.Conv2D(32, 3, activation='relu')(geo_input)
        geo_x = layers.MaxPooling2D(2)(geo_x)
        geo_x = layers.Conv2D(64, 3, activation='relu')(geo_x)
        geo_x = layers.GlobalAveragePooling2D()(geo_x)
        geo_x = layers.Dense(config['geographic_branch_units'][0], activation='relu')(geo_x)
        geo_x = layers.Dropout(config['dropout_rate'])(geo_x)
        geo_x = layers.Dense(config['geographic_branch_units'][1], activation='relu')(geo_x)
        geo_x = layers.Dropout(config['dropout_rate'])(geo_x)
        
        # Linguistic branch (Dense)
        ling_input = layers.Input(shape=(linguistic_dim,), name='linguistic')
        ling_x = layers.Dense(config['linguistic_branch_units'][0], activation='relu')(ling_input)
        ling_x = layers.BatchNormalization()(ling_x)
        ling_x = layers.Dropout(config['dropout_rate'])(ling_x)
        ling_x = layers.Dense(config['linguistic_branch_units'][1], activation='relu')(ling_x)
        ling_x = layers.Dropout(config['dropout_rate'])(ling_x)
        
        # Socioeconomic branch (Dense)
        socio_input = layers.Input(shape=(socioeconomic_dim,), name='socioeconomic')
        socio_x = layers.Dense(config['socioeconomic_branch_units'][0], activation='relu')(socio_input)
        socio_x = layers.BatchNormalization()(socio_x)
        socio_x = layers.Dropout(config['dropout_rate'])(socio_x)
        socio_x = layers.Dense(config['socioeconomic_branch_units'][1], activation='relu')(socio_x)
        socio_x = layers.Dropout(config['dropout_rate'])(socio_x)
        
        # Fusion layer
        fusion = layers.Concatenate()([geo_x, ling_x, socio_x])
        fusion = layers.Dense(config['fusion_units'][0], activation='relu')(fusion)
        fusion = layers.BatchNormalization()(fusion)
        fusion = layers.Dropout(config['dropout_rate'])(fusion)
        
        fusion = layers.Dense(config['fusion_units'][1], activation='relu')(fusion)
        fusion = layers.BatchNormalization()(fusion)
        fusion = layers.Dropout(config['dropout_rate'])(fusion)
        
        fusion = layers.Dense(config['fusion_units'][2], activation='relu')(fusion)
        fusion = layers.Dropout(config['dropout_rate'])(fusion)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(fusion)
        
        model = models.Model(
            inputs=[geo_input, ling_input, socio_input],
            outputs=outputs
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['multimodal'] = model
        return model
    
    def train_deep_learning_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all deep learning models
        """
        logger.info("Training deep learning models...")
        
        # Prepare data
        X_geo = self.prepare_geographic_data(df)
        X_seq = self.prepare_sequence_data(df)
        
        # Prepare traditional features for transformer
        feature_columns = ['speaker_count', 'lei_score', 'intergenerational_transmission', 'lat', 'lng']
        X_traditional = df[feature_columns].fillna(0).values
        
        # Prepare target
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['endangerment_level'])
        y_categorical = to_categorical(y_encoded)
        self.encoders['target'] = le
        
        # Split data
        X_geo_train, X_geo_test, y_train, y_test = train_test_split(
            X_geo, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_seq_train, X_seq_test, _, _ = train_test_split(
            X_seq, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_traditional_train, X_traditional_test, _, _ = train_test_split(
            X_traditional, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        num_classes = len(le.classes_)
        
        # Training callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
        ]
        
        results = {}
        
        # Train CNN
        if len(X_geo_train) > 0:
            logger.info("Training CNN model...")
            cnn_model = self.create_cnn_model(X_geo_train.shape[1:], num_classes)
            
            cnn_history = cnn_model.fit(
                X_geo_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=callbacks_list,
                verbose=1
            )
            
            cnn_pred = cnn_model.predict(X_geo_test)
            cnn_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnn_pred, axis=1))
            
            results['cnn'] = {
                'model': cnn_model,
                'accuracy': cnn_accuracy,
                'history': cnn_history.history,
                'predictions': cnn_pred
            }
        
        # Train LSTM
        if len(X_seq_train) > 0:
            logger.info("Training LSTM model...")
            lstm_model = self.create_lstm_model(X_seq_train.shape[1:], num_classes)
            
            lstm_history = lstm_model.fit(
                X_seq_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=callbacks_list,
                verbose=1
            )
            
            lstm_pred = lstm_model.predict(X_seq_test)
            lstm_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(lstm_pred, axis=1))
            
            results['lstm'] = {
                'model': lstm_model,
                'accuracy': lstm_accuracy,
                'history': lstm_history.history,
                'predictions': lstm_pred
            }
        
        # Train Transformer
        logger.info("Training Transformer model...")
        transformer_model = self.create_transformer_model(X_traditional_train.shape[1], num_classes)
        
        transformer_history = transformer_model.fit(
            X_traditional_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        transformer_pred = transformer_model.predict(X_traditional_test)
        transformer_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(transformer_pred, axis=1))
        
        results['transformer'] = {
            'model': transformer_model,
            'accuracy': transformer_accuracy,
            'history': transformer_history.history,
            'predictions': transformer_pred
        }
        
        # Train Multi-modal model
        logger.info("Training Multi-modal model...")
        multimodal_model = self.create_multimodal_model(
            X_geo_train.shape[1:], 
            X_traditional_train.shape[1], 
            X_traditional_train.shape[1], 
            num_classes
        )
        
        multimodal_history = multimodal_model.fit(
            [X_geo_train, X_traditional_train, X_traditional_train], y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        multimodal_pred = multimodal_model.predict([X_geo_test, X_traditional_test, X_traditional_test])
        multimodal_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(multimodal_pred, axis=1))
        
        results['multimodal'] = {
            'model': multimodal_model,
            'accuracy': multimodal_accuracy,
            'history': multimodal_history.history,
            'predictions': multimodal_pred
        }
        
        self.results = results
        return results
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained deep learning models
        """
        summary = {}
        
        for model_name, result in self.results.items():
            summary[model_name] = {
                'accuracy': result['accuracy'],
                'model_type': model_name.upper(),
                'parameters': result['model'].count_params(),
                'architecture': result['model'].get_config()
            }
        
        return summary
    
    def save_models(self, save_dir: str = "models"):
        """
        Save all trained models
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        for model_name, result in self.results.items():
            model_path = f"{save_dir}/{model_name}_model.h5"
            result['model'].save(model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scalers and encoders
        joblib.dump(self.scalers, f"{save_dir}/scalers.pkl")
        joblib.dump(self.encoders, f"{save_dir}/encoders.pkl")
        logger.info("Saved scalers and encoders")

def main():
    """
    Main function to demonstrate deep learning models
    """
    # Load data
    df = pd.read_csv('data/glottolog_language_data.csv')
    
    # Initialize predictor
    predictor = DeepLearningLanguagePredictor()
    
    # Train models
    results = predictor.train_deep_learning_models(df)
    
    # Print results
    print("\n=== DEEP LEARNING MODEL RESULTS ===")
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['accuracy']:.3f} accuracy")
    
    # Save models
    predictor.save_models()
    
    print("\nDeep learning models trained and saved successfully!")

if __name__ == "__main__":
    main()
