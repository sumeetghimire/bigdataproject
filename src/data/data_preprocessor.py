"""
Data Preprocessing Module for Language Extinction Risk Prediction

This module handles data cleaning, feature engineering, and preprocessing
for the language endangerment prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDataPreprocessor:
    """
    Main class for preprocessing language endangerment data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the preprocessor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.imputers = {}
        
        # Store processed data
        self.processed_data = None
        self.feature_names = []
        self.target_column = 'endangerment_level'
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Cleaning dataset...")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(cleaned_df)} duplicate rows")
        
        # Handle missing values in critical columns
        critical_columns = ['name', 'country']
        for col in critical_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=[col])
        
        # Clean text columns
        text_columns = ['name', 'country', 'region', 'macroarea']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
        
        # Clean numeric columns (handle different coordinate column names)
        numeric_columns = ['latitude', 'longitude', 'lat', 'lng', 'speaker_count', 'speakers', 'lei_score']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        logger.info(f"Cleaned dataset shape: {cleaned_df.shape}")
        return cleaned_df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for better model performance
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with derived features
        """
        logger.info("Creating derived features...")
        
        df_enhanced = df.copy()
        
        # 1. Speaker Density (speakers per square km)
        speaker_col = None
        for col in ['speaker_count', 'speakers']:
            if col in df_enhanced.columns:
                speaker_col = col
                break
                
        if speaker_col:
            # Assume language area based on country size (simplified)
            country_areas = {
                'USA': 9834000, 'India': 3287000, 'Australia': 7692000,
                'Brazil': 8516000, 'Russia': 17100000, 'China': 9597000,
                'Nigeria': 923800, 'Mexico': 1964000, 'Canada': 9985000
            }
            
            df_enhanced['country_area'] = df_enhanced['country'].map(country_areas)
            df_enhanced['speaker_density'] = df_enhanced[speaker_col] / df_enhanced['country_area']
            df_enhanced['speaker_density'] = df_enhanced['speaker_density'].fillna(0)
        
        # 2. Transmission Rate (if we have children speakers data)
        if 'intergenerational_transmission' in df_enhanced.columns:
            # Convert to transmission rate (0-1 scale)
            df_enhanced['transmission_rate'] = df_enhanced['intergenerational_transmission'] / 4.0
        
        # 3. Geographic Isolation Index
        # Handle different coordinate column names
        lat_col = None
        lng_col = None
        for lat_name in ['latitude', 'lat']:
            if lat_name in df_enhanced.columns:
                lat_col = lat_name
                break
        for lng_name in ['longitude', 'lng']:
            if lng_name in df_enhanced.columns:
                lng_col = lng_name
                break
                
        if lat_col and lng_col:
            # Calculate distance to nearest major city (simplified)
            major_cities = {
                'New York': (40.7128, -74.0060),
                'London': (51.5074, -0.1278),
                'Tokyo': (35.6762, 139.6503),
                'Sydney': (-33.8688, 151.2093),
                'Moscow': (55.7558, 37.6176)
            }
            
            def calculate_min_distance_to_city(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return np.nan
                
                min_dist = float('inf')
                for city, (city_lat, city_lon) in major_cities.items():
                    dist = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
                    min_dist = min(min_dist, dist)
                return min_dist
            
            df_enhanced['distance_to_city'] = df_enhanced.apply(
                lambda row: calculate_min_distance_to_city(row[lat_col], row[lng_col]), 
                axis=1
            )
            df_enhanced['geographic_isolation_index'] = df_enhanced['distance_to_city']
        
        # 4. Economic Pressure Index
        if all(col in df_enhanced.columns for col in ['gdp_per_capita', 'urbanization_rate']):
            # Higher GDP and urbanization = more economic pressure on traditional languages
            df_enhanced['economic_pressure_index'] = (
                df_enhanced['gdp_per_capita'] * df_enhanced['urbanization_rate'] / 10000
            )
        
        # 5. Policy Support Score
        policy_columns = ['official_language_status', 'documentation_level']
        if any(col in df_enhanced.columns for col in policy_columns):
            df_enhanced['policy_support_score'] = 0
            
            if 'official_language_status' in df_enhanced.columns:
                df_enhanced['policy_support_score'] += df_enhanced['official_language_status'].fillna(0)
            
            if 'documentation_level' in df_enhanced.columns:
                df_enhanced['policy_support_score'] += df_enhanced['documentation_level'].fillna(0)
        
        # 6. Language Family Risk (average endangerment of related languages)
        if 'family_id' in df_enhanced.columns and 'aes' in df_enhanced.columns:
            family_risk = df_enhanced.groupby('family_id')['aes'].mean().to_dict()
            df_enhanced['language_family_risk'] = df_enhanced['family_id'].map(family_risk)
        
        # 7. Interaction Features
        if speaker_col and 'transmission_rate' in df_enhanced.columns:
            df_enhanced['speakers_x_transmission'] = (
                df_enhanced[speaker_col] * df_enhanced['transmission_rate']
            )
        
        if all(col in df_enhanced.columns for col in ['years_of_schooling', 'official_language_status']):
            df_enhanced['education_x_official_status'] = (
                df_enhanced['years_of_schooling'] * df_enhanced['official_language_status'].fillna(0)
            )
        
        if all(col in df_enhanced.columns for col in ['road_density', 'geographic_isolation_index']):
            df_enhanced['road_density_x_remoteness'] = (
                df_enhanced['road_density'] * df_enhanced['geographic_isolation_index']
            )
        
        logger.info(f"Created {len([col for col in df_enhanced.columns if col not in df.columns])} derived features")
        return df_enhanced
    
    def create_endangerment_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create standardized endangerment target variable
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with endangerment target
        """
        logger.info("Creating endangerment target variable...")
        
        df_target = df.copy()
        
        # Create endangerment level based on available data
        if 'endangerment' in df_target.columns:
            # Use existing endangerment column (from real_language_data.csv)
            df_target['endangerment_level'] = df_target['endangerment'].str.strip()
            
        elif 'endangerment_level' in df_target.columns:
            # Already exists, just clean it
            df_target['endangerment_level'] = df_target['endangerment_level'].str.strip()
            
        elif 'lei_score' in df_target.columns:
            # Use LEI score to create endangerment levels
            def lei_to_endangerment(score):
                if pd.isna(score):
                    return 'Unknown'
                elif score >= 80:
                    return 'Safe'
                elif score >= 60:
                    return 'Vulnerable'
                elif score >= 40:
                    return 'Definitely Endangered'
                elif score >= 20:
                    return 'Severely Endangered'
                elif score >= 5:
                    return 'Critically Endangered'
                else:
                    return 'Extinct'
            
            df_target['endangerment_level'] = df_target['lei_score'].apply(lei_to_endangerment)
        
        elif 'aes' in df_target.columns:
            # Use AES scale
            aes_mapping = {0: 'Safe', 1: 'Vulnerable', 2: 'Definitely Endangered', 
                          3: 'Severely Endangered', 4: 'Critically Endangered', 5: 'Extinct'}
            df_target['endangerment_level'] = df_target['aes'].map(aes_mapping)
        
        else:
            # Create based on speaker count (fallback)
            speaker_col = None
            for col in ['speakers', 'speaker_count']:
                if col in df_target.columns:
                    speaker_col = col
                    break
            
            if speaker_col:
                def speaker_to_endangerment(count):
                    if pd.isna(count):
                        return 'Unknown'
                    elif count >= 10000:
                        return 'Safe'
                    elif count >= 1000:
                        return 'Vulnerable'
                    elif count >= 100:
                        return 'Definitely Endangered'
                    elif count >= 10:
                        return 'Severely Endangered'
                    else:
                        return 'Critically Endangered'
                
                df_target['endangerment_level'] = df_target[speaker_col].apply(speaker_to_endangerment)
            else:
                logger.error("No suitable column found to create endangerment target")
                raise ValueError("Cannot create endangerment target: no suitable columns found")
        
        # Remove unknown endangerment levels for training
        initial_count = len(df_target)
        df_target = df_target[df_target['endangerment_level'] != 'Unknown']
        df_target = df_target.dropna(subset=['endangerment_level'])
        
        logger.info(f"Removed {initial_count - len(df_target)} rows with unknown/missing endangerment levels")
        logger.info(f"Endangerment level distribution:")
        if len(df_target) > 0:
            logger.info(df_target['endangerment_level'].value_counts().to_string())
        else:
            logger.error("No valid endangerment data remaining after filtering!")
        
        return df_target
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from categorical columns
        if self.target_column in categorical_columns:
            categorical_columns.remove(self.target_column)
        
        # Encode each categorical column using label encoding for simplicity
        for col in categorical_columns:
            if col in df_encoded.columns:
                try:
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str).fillna('Unknown'))
                    self.label_encoders[col] = le
                    logger.debug(f"Label encoded column {col}")
                except Exception as e:
                    logger.warning(f"Failed to encode column {col}: {str(e)}")
                    # Skip this column if encoding fails
                    continue
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with imputed missing values
        """
        logger.info("Handling missing values...")
        
        df_imputed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df_imputed.select_dtypes(include=['object']).columns.tolist()
        
        # Impute numeric columns
        if numeric_columns:
            try:
                # Filter out any columns that might have been corrupted
                valid_numeric_columns = [col for col in numeric_columns if col in df_imputed.columns]
                if valid_numeric_columns:
                    numeric_imputer = KNNImputer(n_neighbors=5)
                    imputed_values = numeric_imputer.fit_transform(df_imputed[valid_numeric_columns])
                    df_imputed[valid_numeric_columns] = imputed_values
                    self.imputers['numeric'] = numeric_imputer
            except Exception as e:
                logger.warning(f"KNN imputation failed: {str(e)}, using simple imputation")
                # Fallback to simple imputation
                try:
                    valid_numeric_columns = [col for col in numeric_columns if col in df_imputed.columns]
                    if valid_numeric_columns:
                        numeric_imputer = SimpleImputer(strategy='median')
                        imputed_values = numeric_imputer.fit_transform(df_imputed[valid_numeric_columns])
                        df_imputed[valid_numeric_columns] = imputed_values
                        self.imputers['numeric'] = numeric_imputer
                except Exception as e2:
                    logger.error(f"All imputation methods failed: {str(e2)}")
                    # Fill NaN values with 0 as last resort
                    for col in valid_numeric_columns:
                        df_imputed[col] = df_imputed[col].fillna(0)
        
        # Impute categorical columns
        for col in categorical_columns:
            if col in df_imputed.columns and col != self.target_column:
                mode_value = df_imputed[col].mode()
                if len(mode_value) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_value[0])
                else:
                    df_imputed[col] = df_imputed[col].fillna('Unknown')
        
        logger.info(f"Handled missing values in {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
        return df_imputed
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for machine learning
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str]]: Features, target, feature names
        """
        logger.info("Preparing features and target...")
        
        # Select only numeric features for ML
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it's numeric
        if self.target_column in feature_columns:
            feature_columns.remove(self.target_column)
        
        # Remove ID columns
        id_columns = ['glottocode', 'iso6393', 'family_id']
        feature_columns = [col for col in feature_columns if col not in id_columns]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Encode target variable to numeric labels for ML models
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            self.target_encoder = target_encoder
            logger.info(f"Encoded target variable. Classes: {target_encoder.classes_}")
            y = pd.Series(y_encoded, index=y.index, name=self.target_column)
        
        # Store feature names
        self.feature_names = feature_columns
        
        logger.info(f"Prepared {len(feature_columns)} features for machine learning")
        logger.info(f"Feature names: {feature_columns}")
        
        return X, y, feature_columns
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str]]: Processed features, target, feature names
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Create derived features
        df_enhanced = self.create_derived_features(df_clean)
        
        # Step 3: Create endangerment target
        df_target = self.create_endangerment_target(df_enhanced)
        
        # Step 4: Encode categorical features
        df_encoded = self.encode_categorical_features(df_target)
        
        # Step 5: Handle missing values
        df_imputed = self.handle_missing_values(df_encoded)
        
        # Step 6: Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df_imputed)
        
        # Final check: ensure no NaN values remain in features
        if X.isnull().any().any():
            logger.warning("NaN values detected in features, filling with 0")
            X = X.fillna(0)
        
        # Store processed data
        self.processed_data = df_imputed
        
        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"NaN values in features: {X.isnull().sum().sum()}")
        
        return X, y, feature_names
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set size
            random_state (int): Random seed
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test features
        """
        logger.info("Scaling features...")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled


def main():
    """Main function to demonstrate preprocessing"""
    from data_loader import LanguageDataLoader
    
    # Load data
    loader = LanguageDataLoader()
    datasets = loader.load_all_datasets()
    merged_data = loader.merge_datasets()
    
    # Preprocess data
    preprocessor = LanguageDataPreprocessor()
    X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"Final dataset shape: {X_train_scaled.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    
    return preprocessor, X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
