"""
Flask Web Application for Language Extinction Risk Prediction
This app provides an interactive web interface for exploring language extinction data
using Canvas.js for visualizations.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
import yaml
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_loader import LanguageDataLoader
from data.data_preprocessor import LanguageDataPreprocessor
from models.ml_models import LanguageExtinctionPredictor
from visualization.visualizer import LanguageVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store data
raw_data = None
processed_data = None
model_results = None
evaluation_results = None
feature_importance = None

def load_data():
    """Load and process data for the web application"""
    global raw_data, processed_data, model_results, evaluation_results, feature_importance
    
    try:
        # Load configuration
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Initialize components
        data_loader = LanguageDataLoader('config.yaml')
        preprocessor = LanguageDataPreprocessor('config.yaml')
        predictor = LanguageExtinctionPredictor('config.yaml')
        
        # Load data
        logger.info("Loading data...")
        datasets = data_loader.load_all_datasets()
        raw_data = data_loader.merge_datasets()
        
        # Map AES values to endangerment levels if needed
        if 'aes' in raw_data.columns and 'endangerment_level' not in raw_data.columns:
            aes_to_endangerment = {
                0: 'Safe',
                1: 'Vulnerable', 
                2: 'Definitely Endangered',
                3: 'Severely Endangered',
                4: 'Critically Endangered',
                5: 'Extinct'
            }
            raw_data['endangerment_level'] = raw_data['aes'].map(aes_to_endangerment)
        
        # Fix coordinates by mapping countries to proper coordinates
        country_coords = {
            'USA': (39.8283, -98.5795),
            'Canada': (56.1304, -106.3468),
            'Mexico': (23.6345, -102.5528),
            'Brazil': (-14.2350, -51.9253),
            'Peru': (-9.1900, -75.0152),
            'Chile': (-35.6751, -71.5430),
            'Argentina': (-38.4161, -63.6167),
            'Ecuador': (-1.8312, -78.1834),
            'India': (20.5937, 78.9629),
            'China': (35.8617, 104.1954),
            'Tibet': (29.6465, 91.1172),
            'Mongolia': (46.8625, 103.8467),
            'Kazakhstan': (48.0196, 66.9237),
            'Kyrgyzstan': (41.2044, 74.7661),
            'Tajikistan': (38.8610, 71.2761),
            'New Zealand': (-40.9006, 174.8860),
            'Australia': (-25.2744, 133.7751),
            'UK': (55.3781, -3.4360),
            'Ireland': (53.4129, -8.2439),
            'France': (46.2276, 2.2137),
            'Spain': (40.4637, -3.7492),
            'Norway': (60.4720, 8.4689),
            'Sweden': (60.1282, 18.6435),
            'Finland': (61.9241, 25.7482),
            'Russia': (61.5240, 105.3188),
            'Nigeria': (9.0820, 8.6753),
            'Ethiopia': (9.1450, 40.4897),
            'Kenya': (-0.0236, 37.9062),
            'Tanzania': (-6.3690, 34.8888),
            'South Africa': (-30.5595, 22.9375),
            'Ghana': (7.9465, -1.0232),
            'Senegal': (14.4974, -14.4524),
            'Mali': (17.5707, -3.9962),
            'Burkina Faso': (12.2383, -1.5616),
            'Niger': (17.6078, 8.0817),
            'Japan': (36.2048, 138.2529),
            'South Korea': (35.9078, 127.7669),
            'Vietnam': (14.0583, 108.2772),
            'Thailand': (15.8700, 100.9925),
            'Saudi Arabia': (23.8859, 45.0792),
            'Israel': (31.0461, 34.8516),
            'Iran': (32.4279, 53.6880),
            'Turkey': (38.9637, 35.2433),
            'Poland': (51.9194, 19.1451),
            'Czech Republic': (49.8175, 15.4730)
        }
        
        # Update coordinates based on country
        if 'country' in raw_data.columns and 'latitude' in raw_data.columns and 'longitude' in raw_data.columns:
            for idx, row in raw_data.iterrows():
                country = row['country']
                if country in country_coords:
                    # Add some random variation around the country center
                    lat, lng = country_coords[country]
                    # Add random variation within country bounds (±2 degrees)
                    raw_data.at[idx, 'latitude'] = lat + np.random.uniform(-2, 2)
                    raw_data.at[idx, 'longitude'] = lng + np.random.uniform(-2, 2)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, y, feature_names = preprocessor.preprocess_pipeline(raw_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }
        
        # Train models
        logger.info("Training models...")
        model_results = predictor.train_all_models(X_train_scaled, y_train)
        evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)
        
        # Get feature importance
        feature_importance_df = predictor.get_feature_importance('random_forest', top_n=20)
        feature_importance = dict(zip(feature_importance_df['feature'], feature_importance_df['importance']))
        
        logger.info("Data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise e

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/data/summary')
def data_summary():
    """Get data summary statistics"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    summary = {
        'total_languages': len(raw_data),
        'endangerment_distribution': raw_data['endangerment_level'].value_counts().to_dict(),
        'family_distribution': raw_data['family_id'].value_counts().head(10).to_dict(),
        'country_distribution': raw_data['country'].value_counts().head(10).to_dict(),
        'avg_speakers': float(raw_data['speaker_count'].mean()),
        'median_speakers': float(raw_data['speaker_count'].median()),
        'avg_lei_score': float(raw_data['lei_score'].mean()),
        'transmission_distribution': raw_data['intergenerational_transmission'].value_counts().to_dict()
    }
    
    return jsonify(summary)

@app.route('/api/data/endangerment-distribution')
def endangerment_distribution():
    """Get endangerment level distribution data for charts"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    distribution = raw_data['endangerment_level'].value_counts()
    
    # Prepare data for Canvas.js
    data_points = []
    colors = {
        'Safe': '#2E8B57',
        'Vulnerable': '#FFD700',
        'Definitely Endangered': '#FF8C00',
        'Severely Endangered': '#FF4500',
        'Critically Endangered': '#DC143C',
        'Extinct': '#8B0000'
    }
    
    for level, count in distribution.items():
        data_points.append({
            'label': level,
            'y': int(count),
            'color': colors.get(level, '#808080')
        })
    
    return jsonify(data_points)

@app.route('/api/data/feature-importance')
def feature_importance_data():
    """Get feature importance data for charts"""
    if feature_importance is None:
        return jsonify({'error': 'Feature importance not available'}), 500
    
    # Sort features by importance and take top 15
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    data_points = []
    for feature, importance in sorted_features:
        data_points.append({
            'label': feature.replace('_', ' ').title(),
            'y': float(importance)
        })
    
    return jsonify(data_points)

@app.route('/api/data/model-performance')
def model_performance():
    """Get model performance data for charts"""
    if evaluation_results is None:
        return jsonify({'error': 'Model results not available'}), 500
    
    models = []
    accuracies = []
    f1_scores = []
    
    for model_name, results in evaluation_results.items():
        models.append(model_name.replace('_', ' ').title())
        accuracies.append(float(results['test_accuracy']))
        f1_scores.append(float(results['classification_report']['weighted avg']['f1-score']))
    
    return jsonify({
        'models': models,
        'accuracies': accuracies,
        'f1_scores': f1_scores
    })

@app.route('/api/data/speaker-vs-endangerment')
def speaker_vs_endangerment():
    """Get speaker count vs endangerment data for scatter plot"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Sample data for performance (take every 10th row)
    sample_data = raw_data.iloc[::10].copy()
    
    data_points = []
    colors = {
        'Safe': '#2E8B57',
        'Vulnerable': '#FFD700',
        'Definitely Endangered': '#FF8C00',
        'Severely Endangered': '#FF4500',
        'Critically Endangered': '#DC143C',
        'Extinct': '#8B0000'
    }
    
    for _, row in sample_data.iterrows():
        if pd.notna(row['speaker_count']) and pd.notna(row['endangerment_level']):
            data_points.append({
                'x': float(row['speaker_count']),
                'y': row['endangerment_level'],
                'color': colors.get(row['endangerment_level'], '#808080'),
                'name': row['name'],
                'country': row['country'],
                'transmission': int(row['intergenerational_transmission']) if pd.notna(row['intergenerational_transmission']) else 0
            })
    
    return jsonify(data_points)

@app.route('/api/data/geographic-distribution')
def geographic_distribution():
    """Get geographic distribution data for map"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Filter data with valid coordinates
    geo_data = raw_data.dropna(subset=['latitude', 'longitude']).copy()
    
    # Sample data for performance
    if len(geo_data) > 1000:
        geo_data = geo_data.sample(n=1000, random_state=42)
    
    data_points = []
    colors = {
        'Safe': '#2E8B57',
        'Vulnerable': '#FFD700',
        'Definitely Endangered': '#FF8C00',
        'Severely Endangered': '#FF4500',
        'Critically Endangered': '#DC143C',
        'Extinct': '#8B0000'
    }
    
    for _, row in geo_data.iterrows():
        data_points.append({
            'lat': float(row['latitude']),
            'lng': float(row['longitude']),
            'name': row['name'],
            'country': row['country'],
            'speakers': int(row['speaker_count']) if pd.notna(row['speaker_count']) else 0,
            'endangerment': row['endangerment_level'],
            'color': colors.get(row['endangerment_level'], '#808080'),
            'lei_score': float(row['lei_score']) if pd.notna(row['lei_score']) else 0
        })
    
    return jsonify(data_points)

@app.route('/api/data/family-distribution')
def family_distribution():
    """Get language family distribution data"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    family_data = raw_data.groupby('family_id').agg({
        'name': 'count',
        'speaker_count': 'sum',
        'endangerment_level': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    family_data.columns = ['family', 'language_count', 'total_speakers', 'endangerment_distribution']
    
    # Sort by language count and take top 15
    family_data = family_data.sort_values('language_count', ascending=False).head(15)
    
    data_points = []
    for _, row in family_data.iterrows():
        data_points.append({
            'label': f"Family {row['family']}",
            'y': int(row['language_count']),
            'total_speakers': int(row['total_speakers']) if pd.notna(row['total_speakers']) else 0
        })
    
    return jsonify(data_points)

@app.route('/api/data/transmission-distribution')
def transmission_distribution():
    """Get intergenerational transmission distribution data"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    transmission_data = raw_data['intergenerational_transmission'].value_counts().sort_index()
    
    data_points = []
    for level, count in transmission_data.items():
        if pd.notna(level):
            data_points.append({
                'label': f"Level {int(level)}",
                'y': int(count)
            })
    
    return jsonify(data_points)

@app.route('/api/data/languages')
def languages_data():
    """Get paginated languages data for the table"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    search = request.args.get('search', '')
    
    # Filter data based on search
    filtered_data = raw_data
    if search:
        filtered_data = raw_data[
            raw_data['name'].str.contains(search, case=False, na=False) |
            raw_data['country'].str.contains(search, case=False, na=False) |
            raw_data['family_id'].str.contains(search, case=False, na=False)
        ]
    
    # Paginate
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_data = filtered_data.iloc[start_idx:end_idx]
    
    # Convert to JSON-serializable format
    languages = []
    for _, row in page_data.iterrows():
        languages.append({
            'name': row['name'],
            'country': row['country'],
            'family_id': row['family_id'],
            'speaker_count': int(row['speaker_count']) if pd.notna(row['speaker_count']) else 0,
            'endangerment_level': row['endangerment_level'],
            'lei_score': float(row['lei_score']) if pd.notna(row['lei_score']) else 0,
            'intergenerational_transmission': int(row['intergenerational_transmission']) if pd.notna(row['intergenerational_transmission']) else 0,
            'latitude': float(row['latitude']) if pd.notna(row['latitude']) else None,
            'longitude': float(row['longitude']) if pd.notna(row['longitude']) else None
        })
    
    return jsonify({
        'languages': languages,
        'total': len(filtered_data),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(filtered_data) + per_page - 1) // per_page
    })

if __name__ == '__main__':
    # Load data when starting the app
    try:
        load_data()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
