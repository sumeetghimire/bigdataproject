"""
Machine Learning Models Module for Language Extinction Risk Prediction

This module implements multiple ML models as specified in the project requirements:
- Random Forest Classifier (Primary)
- XGBoost Classifier
- Neural Network (TensorFlow/Keras)
- Logistic Regression (Baseline)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import yaml
from pathlib import Path

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageExtinctionPredictor:
    """
    Main class for training and evaluating language extinction prediction models
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the predictor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.model_configs = self.config['models']
        self.results = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def create_random_forest(self) -> RandomForestClassifier:
        """
        Create and configure Random Forest Classifier
        
        Returns:
            RandomForestClassifier: Configured Random Forest model
        """
        logger.info("Creating Random Forest Classifier...")
        
        config = self.model_configs['random_forest']
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            random_state=config['random_state'],
            class_weight=config['class_weight'],
            n_jobs=-1
        )
        
        self.models['random_forest'] = model
        return model
    
    def create_xgboost(self) -> xgb.XGBClassifier:
        """
        Create and configure XGBoost Classifier
        
        Returns:
            xgb.XGBClassifier: Configured XGBoost model
        """
        logger.info("Creating XGBoost Classifier...")
        
        config = self.model_configs['xgboost']
        model = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            random_state=config['random_state'],
            eval_metric=config['eval_metric'],
            n_jobs=-1
        )
        
        self.models['xgboost'] = model
        return model
    
    def create_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """
        Create and configure Neural Network model
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            
        Returns:
            keras.Model: Configured Neural Network model
        """
        logger.info("Creating Neural Network...")
        
        config = self.model_configs['neural_network']
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(config['hidden_layers'][0], 
                              activation=config['activation'], 
                              input_shape=(input_dim,)))
        model.add(layers.Dropout(config['dropout_rate']))
        
        # Hidden layers
        for units in config['hidden_layers'][1:]:
            model.add(layers.Dense(units, activation=config['activation']))
            model.add(layers.Dropout(config['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['neural_network'] = model
        return model
    
    def create_logistic_regression(self) -> LogisticRegression:
        """
        Create and configure Logistic Regression Classifier
        
        Returns:
            LogisticRegression: Configured Logistic Regression model
        """
        logger.info("Creating Logistic Regression Classifier...")
        
        config = self.model_configs['logistic_regression']
        model = LogisticRegression(
            random_state=config['random_state'],
            max_iter=config['max_iter'],
            class_weight=config['class_weight'],
            multi_class='ovr',
            n_jobs=-1
        )
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: Optional[pd.DataFrame] = None, 
                           y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training Random Forest model...")
        
        model = self.create_random_forest()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        results = {
            'model': model,
            'train_accuracy': train_accuracy,
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
            'train_predictions': y_train_pred
        }
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            results['val_accuracy'] = val_accuracy
            results['val_predictions'] = y_val_pred
        
        self.results['random_forest'] = results
        logger.info(f"Random Forest training completed. Train accuracy: {train_accuracy:.4f}")
        
        return results
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training XGBoost model...")
        
        model = self.create_xgboost()
        
        # Prepare data for XGBoost
        X_train_xgb = X_train.values
        y_train_xgb = y_train.values
        
        # Train model
        model.fit(X_train_xgb, y_train_xgb)
        
        # Make predictions
        y_train_pred = model.predict(X_train_xgb)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        results = {
            'model': model,
            'train_accuracy': train_accuracy,
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
            'train_predictions': y_train_pred
        }
        
        if X_val is not None and y_val is not None:
            X_val_xgb = X_val.values
            y_val_xgb = y_val.values
            y_val_pred = model.predict(X_val_xgb)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            results['val_accuracy'] = val_accuracy
            results['val_predictions'] = y_val_pred
        
        self.results['xgboost'] = results
        logger.info(f"XGBoost training completed. Train accuracy: {train_accuracy:.4f}")
        
        return results
    
    def train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train Neural Network model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training Neural Network model...")
        
        # Encode target labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # Create model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train_encoded))
        model = self.create_neural_network(input_dim, num_classes)
        
        # Prepare data
        X_train_nn = X_train.values
        y_train_nn = y_train_encoded
        
        # Train model
        config = self.model_configs['neural_network']
        history = model.fit(
            X_train_nn, y_train_nn,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split'],
            verbose=1
        )
        
        # Make predictions
        y_train_pred_proba = model.predict(X_train_nn)
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)
        y_train_pred_labels = le.inverse_transform(y_train_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred_labels)
        
        results = {
            'model': model,
            'label_encoder': le,
            'train_accuracy': train_accuracy,
            'history': history.history,
            'train_predictions': y_train_pred_labels
        }
        
        if X_val is not None and y_val is not None:
            y_val_encoded = le.transform(y_val)
            X_val_nn = X_val.values
            y_val_pred_proba = model.predict(X_val_nn)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
            y_val_pred_labels = le.inverse_transform(y_val_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred_labels)
            results['val_accuracy'] = val_accuracy
            results['val_predictions'] = y_val_pred_labels
        
        self.results['neural_network'] = results
        logger.info(f"Neural Network training completed. Train accuracy: {train_accuracy:.4f}")
        
        return results
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train Logistic Regression model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training Logistic Regression model...")
        
        model = self.create_logistic_regression()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        results = {
            'model': model,
            'train_accuracy': train_accuracy,
            'coefficients': dict(zip(X_train.columns, model.coef_[0])),
            'train_predictions': y_train_pred
        }
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            results['val_accuracy'] = val_accuracy
            results['val_predictions'] = y_val_pred
        
        self.results['logistic_regression'] = results
        logger.info(f"Logistic Regression training completed. Train accuracy: {train_accuracy:.4f}")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for all models
        """
        logger.info("Training all models...")
        
        # Train each model
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_neural_network(X_train, y_train, X_val, y_val)
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        logger.info("All models trained successfully")
        return self.results
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a specific model on test data
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating {model_name} model...")
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.results.keys())}")
        
        model_info = self.results[model_name]
        model = model_info['model']
        
        # Make predictions
        if model_name == 'neural_network':
            # Special handling for neural network
            le = model_info['label_encoder']
            X_test_nn = X_test.values
            y_test_pred_proba = model.predict(X_test_nn)
            y_test_pred = np.argmax(y_test_pred_proba, axis=1)
            y_test_pred_labels = le.inverse_transform(y_test_pred)
        elif model_name == 'xgboost':
            # Special handling for XGBoost
            X_test_xgb = X_test.values
            y_test_pred_labels = model.predict(X_test_xgb)
        else:
            # Standard sklearn models
            y_test_pred_labels = model.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred_labels)
        
        # Classification report
        class_report = classification_report(y_test, y_test_pred_labels, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred_labels)
        
        evaluation_results = {
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_test_pred_labels
        }
        
        logger.info(f"{model_name} evaluation completed. Test accuracy: {test_accuracy:.4f}")
        return evaluation_results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results for all models
        """
        logger.info("Evaluating all models...")
        
        evaluation_results = {}
        for model_name in self.results.keys():
            evaluation_results[model_name] = self.evaluate_model(model_name, X_test, y_test)
        
        logger.info("All models evaluated successfully")
        return evaluation_results
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.results[model_name]
        
        if 'feature_importance' in model_info:
            importance_dict = model_info['feature_importance']
            importance_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
    
    def save_models(self, models_dir: str = "models"):
        """
        Save all trained models
        
        Args:
            models_dir (str): Directory to save models
        """
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        for model_name, model_info in self.results.items():
            model = model_info['model']
            model_path = models_path / f"{model_name}_model.joblib"
            
            if model_name == 'neural_network':
                # Save Keras model
                model.save(models_path / f"{model_name}_model.h5")
            else:
                # Save sklearn models
                joblib.dump(model, model_path)
            
            logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, models_dir: str = "models"):
        """
        Load previously saved models
        
        Args:
            models_dir (str): Directory containing saved models
        """
        models_path = Path(models_dir)
        
        for model_file in models_path.glob("*_model.*"):
            model_name = model_file.stem.replace("_model", "")
            
            if model_file.suffix == '.h5':
                # Load Keras model
                model = keras.models.load_model(model_file)
            else:
                # Load sklearn models
                model = joblib.load(model_file)
            
            self.models[model_name] = model
            logger.info(f"Loaded {model_name} model from {model_file}")


def main():
    """Main function to demonstrate model training"""
    from data.data_loader import LanguageDataLoader
    from data.data_preprocessor import LanguageDataPreprocessor
    
    # Load and preprocess data
    loader = LanguageDataLoader()
    datasets = loader.load_all_datasets()
    merged_data = loader.merge_datasets()
    
    preprocessor = LanguageDataPreprocessor()
    X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train models
    predictor = LanguageExtinctionPredictor()
    results = predictor.train_all_models(X_train_scaled, y_train)
    
    # Evaluate models
    evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)
    
    # Print results
    print("\nModel Performance Summary:")
    print("=" * 50)
    for model_name, eval_results in evaluation_results.items():
        print(f"{model_name}: {eval_results['test_accuracy']:.4f}")
    
    # Save models
    predictor.save_models()
    
    return predictor, evaluation_results


if __name__ == "__main__":
    predictor, results = main()
