"""
Data Loading Module for Language Extinction Risk Prediction

This module handles loading and initial processing of multiple datasets:
- Glottolog Database
- Catalogue of Endangered Languages (ELCat)
- UNESCO Atlas of Endangered Languages
- Our World in Data - Living Languages
- Kaggle - Extinct Languages
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import sqlite3
import os
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from urllib.parse import urljoin
import gzip
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDataLoader:
    """
    Main class for loading and preprocessing language endangerment datasets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data loader with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.datasets = {}
        self.merged_data = None
        
    def load_glottolog_data(self) -> pd.DataFrame:
        """
        Load Glottolog database - comprehensive catalogue of languages
        
        Returns:
            pd.DataFrame: Glottolog language data
        """
        logger.info("Loading Glottolog data...")
        
        # For demonstration, we'll create a sample dataset
        # In practice, you would download from the actual URLs
        sample_data = {
            'glottocode': ['abcd1234', 'efgh5678', 'ijkl9012', 'mnop3456', 'qrst7890'],
            'name': ['Sample Language 1', 'Sample Language 2', 'Sample Language 3', 
                    'Sample Language 4', 'Sample Language 5'],
            'latitude': [40.7128, 34.0522, 51.5074, -33.8688, 35.6762],
            'longitude': [-74.0060, -118.2437, -0.1278, 151.2093, 139.6503],
            'family_id': ['fam001', 'fam002', 'fam001', 'fam003', 'fam002'],
            'level': ['language', 'language', 'dialect', 'language', 'language'],
            'aes': [1, 2, 3, 4, 5],  # Agglomerated Endangerment Scale
            'macroarea': ['North America', 'North America', 'Europe', 'Australia', 'Asia']
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['glottolog'] = df
        logger.info(f"Loaded Glottolog data: {len(df)} records")
        return df
    
    def load_elcat_data(self) -> pd.DataFrame:
        """
        Load Catalogue of Endangered Languages (ELCat) data
        
        Returns:
            pd.DataFrame: ELCat language endangerment data
        """
        logger.info("Loading ELCat data...")
        
        # Sample ELCat data structure
        sample_data = {
            'iso6393': ['abc', 'def', 'ghi', 'jkl', 'mno'],
            'name': ['Endangered Language 1', 'Endangered Language 2', 
                    'Endangered Language 3', 'Endangered Language 4', 'Endangered Language 5'],
            'lei_score': [25.5, 45.2, 67.8, 82.1, 15.3],  # Language Endangerment Index
            'speaker_count': [150, 1200, 50, 5000, 25],
            'speaker_trend': ['decreasing', 'stable', 'decreasing', 'increasing', 'decreasing'],
            'intergenerational_transmission': [1, 2, 0, 3, 0],  # 0-4 scale
            'domains_of_use': [2, 4, 1, 6, 1],  # 0-7 domains
            'documentation_level': [2, 4, 1, 5, 1],  # 0-5 scale
            'country': ['USA', 'Mexico', 'Canada', 'Brazil', 'Australia'],
            'region': ['North America', 'Central America', 'North America', 'South America', 'Oceania']
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['elcat'] = df
        logger.info(f"Loaded ELCat data: {len(df)} records")
        return df
    
    def load_unesco_data(self) -> pd.DataFrame:
        """
        Load UNESCO Atlas of Endangered Languages data
        
        Returns:
            pd.DataFrame: UNESCO language data
        """
        logger.info("Loading UNESCO data...")
        
        # Sample UNESCO data
        sample_data = {
            'name': ['UNESCO Language 1', 'UNESCO Language 2', 'UNESCO Language 3', 
                    'UNESCO Language 4', 'UNESCO Language 5'],
            'country': ['USA', 'India', 'Australia', 'Brazil', 'Russia'],
            'speaker_count': [200, 1500, 80, 3000, 120],
            'endangerment_level': ['Vulnerable', 'Definitely Endangered', 'Severely Endangered', 
                                 'Critically Endangered', 'Extinct'],
            'latitude': [40.7128, 20.5937, -25.2744, -14.2350, 61.5240],
            'longitude': [-74.0060, 78.9629, 133.7751, -51.9253, 105.3188]
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['unesco'] = df
        logger.info(f"Loaded UNESCO data: {len(df)} records")
        return df
    
    def load_our_world_in_data(self) -> pd.DataFrame:
        """
        Load Our World in Data - Living Languages dataset
        
        Returns:
            pd.DataFrame: OWiD language data
        """
        logger.info("Loading Our World in Data...")
        
        # Sample OWiD data
        sample_data = {
            'country': ['USA', 'India', 'Australia', 'Brazil', 'Russia', 'China', 'Nigeria', 'Mexico'],
            'total_languages': [420, 453, 123, 289, 105, 301, 524, 186],
            'endangered_languages': [150, 200, 80, 120, 60, 180, 300, 90],
            'safe_languages': [270, 253, 43, 169, 45, 121, 224, 96],
            'year': [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020]
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['our_world_in_data'] = df
        logger.info(f"Loaded Our World in Data: {len(df)} records")
        return df
    
    def load_kaggle_data(self) -> pd.DataFrame:
        """
        Load Kaggle Extinct Languages dataset
        
        Returns:
            pd.DataFrame: Kaggle language data
        """
        logger.info("Loading Kaggle data...")
        
        # Sample Kaggle data
        sample_data = {
            'name': ['Kaggle Language 1', 'Kaggle Language 2', 'Kaggle Language 3', 
                    'Kaggle Language 4', 'Kaggle Language 5'],
            'country': ['USA', 'Canada', 'Australia', 'Brazil', 'Mexico'],
            'extinction_likelihood': [0.8, 0.6, 0.9, 0.4, 0.7],
            'last_speaker_year': [1995, 2000, 1985, 2010, 1998],
            'language_family': ['Algonquian', 'Athabaskan', 'Pama-Nyungan', 'Tupian', 'Uto-Aztecan']
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['kaggle'] = df
        logger.info(f"Loaded Kaggle data: {len(df)} records")
        return df
    
    def load_socioeconomic_data(self) -> pd.DataFrame:
        """
        Load supplementary socioeconomic data from World Bank
        
        Returns:
            pd.DataFrame: Socioeconomic indicators by country
        """
        logger.info("Loading socioeconomic data...")
        
        # Sample socioeconomic data
        sample_data = {
            'country': ['USA', 'India', 'Australia', 'Brazil', 'Russia', 'China', 'Nigeria', 'Mexico'],
            'gdp_per_capita': [65000, 2000, 55000, 8000, 12000, 10000, 2000, 10000],
            'years_of_schooling': [13.2, 6.5, 12.8, 7.8, 11.5, 7.6, 5.2, 8.9],
            'urbanization_rate': [82.5, 35.0, 86.0, 87.0, 74.0, 60.0, 52.0, 80.0],
            'population_density': [36, 464, 3, 25, 9, 149, 226, 66],
            'road_density': [0.67, 0.45, 0.12, 0.15, 0.08, 0.52, 0.18, 0.25]
        }
        
        df = pd.DataFrame(sample_data)
        self.datasets['socioeconomic'] = df
        logger.info(f"Loaded socioeconomic data: {len(df)} records")
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of all loaded datasets
        """
        logger.info("Loading all datasets...")
        
        # Load all datasets
        self.load_glottolog_data()
        self.load_elcat_data()
        self.load_unesco_data()
        self.load_our_world_in_data()
        self.load_kaggle_data()
        self.load_socioeconomic_data()
        
        logger.info(f"Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into a unified dataset for analysis
        
        Returns:
            pd.DataFrame: Merged dataset with all features
        """
        logger.info("Merging datasets...")
        
        if not self.datasets:
            self.load_all_datasets()
        
        # Start with Glottolog as base dataset
        merged = self.datasets['glottolog'].copy()
        
        # Merge ELCat data on language name (simplified matching)
        if 'elcat' in self.datasets:
            # In practice, you would use more sophisticated matching
            merged = merged.merge(
                self.datasets['elcat'], 
                left_on='name', 
                right_on='name', 
                how='left',
                suffixes=('_glottolog', '_elcat')
            )
        
        # Merge UNESCO data
        if 'unesco' in self.datasets:
            merged = merged.merge(
                self.datasets['unesco'], 
                left_on='name', 
                right_on='name', 
                how='left',
                suffixes=('', '_unesco')
            )
        
        # Merge country-level data
        if 'our_world_in_data' in self.datasets:
            # Add country-level statistics
            country_stats = self.datasets['our_world_in_data'].groupby('country').agg({
                'total_languages': 'first',
                'endangered_languages': 'first',
                'safe_languages': 'first'
            }).reset_index()
            
            merged = merged.merge(
                country_stats,
                left_on='country',
                right_on='country',
                how='left'
            )
        
        # Merge socioeconomic data
        if 'socioeconomic' in self.datasets:
            merged = merged.merge(
                self.datasets['socioeconomic'],
                on='country',
                how='left'
            )
        
        self.merged_data = merged
        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for all loaded datasets
        
        Returns:
            Dict: Summary statistics
        """
        summary = {}
        
        for name, df in self.datasets.items():
            summary[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.to_dict()
            }
        
        if self.merged_data is not None:
            summary['merged'] = {
                'shape': self.merged_data.shape,
                'columns': list(self.merged_data.columns),
                'missing_values': self.merged_data.isnull().sum().to_dict(),
                'dtypes': self.merged_data.dtypes.to_dict()
            }
        
        return summary
    
    def save_processed_data(self, filename: str = "processed_language_data.csv"):
        """
        Save the merged dataset to CSV
        
        Args:
            filename (str): Output filename
        """
        if self.merged_data is not None:
            filepath = self.data_dir / filename
            self.merged_data.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")
        else:
            logger.warning("No merged data to save. Run merge_datasets() first.")


def main():
    """Main function to demonstrate data loading"""
    loader = LanguageDataLoader()
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    # Merge datasets
    merged_data = loader.merge_datasets()
    
    # Get summary
    summary = loader.get_data_summary()
    
    # Print summary
    for name, stats in summary.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Columns: {len(stats['columns'])}")
        print(f"  Missing values: {sum(stats['missing_values'].values())}")
    
    # Save processed data
    loader.save_processed_data()
    
    return loader, merged_data


if __name__ == "__main__":
    loader, data = main()
