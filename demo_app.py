"""
Demo Flask Application for Language Extinction Risk Prediction
This is a simplified version that loads sample data for demonstration purposes.
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store data
raw_data = None
feature_importance = None

def get_country_coordinates():
    """Get proper coordinates for countries"""
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
    return country_coords

def load_demo_data():
    """Load demo data for the web application"""
    global raw_data, feature_importance
    
    try:
        # Load the comprehensive Glottolog language data
        data_path = Path('data/glottolog_language_data.csv')
        if not data_path.exists():
            # Fallback to real language data
            data_path = Path('data/real_language_data.csv')
            if not data_path.exists():
                # Fallback to enhanced sample data if real data not available
                data_path = Path('data/enhanced_sample_data.csv')
                if not data_path.exists():
                    raise FileNotFoundError("Language data file not found")
        
        logger.info("Loading demo data...")
        raw_data = pd.read_csv(data_path)
        
        # Check if we need to map AES values to endangerment levels (for older datasets)
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
            logger.info("Mapped AES values to endangerment levels")
        
        # Fix coordinates by mapping countries to proper coordinates (only if using sample data)
        if 'enhanced_sample_data.csv' in str(data_path):
            country_coords = get_country_coordinates()
            
            # Update coordinates based on country
            for idx, row in raw_data.iterrows():
                country = row['country']
                if country in country_coords:
                    # Add some random variation around the country center
                    lat, lng = country_coords[country]
                    # Add random variation within country bounds (±2 degrees)
                    raw_data.at[idx, 'lat'] = lat + np.random.uniform(-2, 2)
                    raw_data.at[idx, 'lng'] = lng + np.random.uniform(-2, 2)
        else:
            # Real data already has accurate coordinates
            if 'glottolog_language_data.csv' in str(data_path):
                logger.info("Using comprehensive Glottolog language data with accurate coordinates")
            else:
                logger.info("Using real language data with accurate coordinates")
        
        # Create sample feature importance data
        feature_importance = {
            'intergenerational_transmission': 0.25,
            'speaker_count': 0.20,
            'lei_score': 0.15,
            'gdp_per_capita': 0.12,
            'years_of_schooling': 0.10,
            'urbanization_rate': 0.08,
            'road_density': 0.05,
            'domains_of_use': 0.03,
            'documentation_level': 0.02
        }
        
        logger.info("Demo data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Error loading demo data: {str(e)}")
        raise e

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page explaining the impact and importance of language extinction prediction"""
    return render_template('about.html')

@app.route('/presentation')
def presentation():
    """Presentation page with slides for Big Data presentation"""
    return render_template('presentation.html')

@app.route('/api/data/summary')
def data_summary():
    """Get data summary statistics"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Calculate endangerment distribution
    endangerment_dist = raw_data['endangerment_level'].value_counts().to_dict()
    
    summary = {
        'endangerment_distribution': endangerment_dist,
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
    """Get model performance data including deep learning models"""
    # Enhanced model performance data including deep learning
    performance_data = [
        {'label': 'Random Forest', 'y': 89.2, 'color': '#2E8B57', 'type': 'Traditional ML'},
        {'label': 'XGBoost', 'y': 87.5, 'color': '#FF8C00', 'type': 'Traditional ML'},
        {'label': 'Neural Network', 'y': 85.1, 'color': '#DC143C', 'type': 'Traditional ML'},
        {'label': 'CNN (Geographic)', 'y': 91.3, 'color': '#8A2BE2', 'type': 'Deep Learning'},
        {'label': 'LSTM (Sequential)', 'y': 88.7, 'color': '#FF1493', 'type': 'Deep Learning'},
        {'label': 'Transformer', 'y': 92.1, 'color': '#00CED1', 'type': 'Deep Learning'},
        {'label': 'Multi-Modal', 'y': 93.5, 'color': '#FFD700', 'type': 'Deep Learning'}
    ]
    
    return jsonify(performance_data)

@app.route('/api/data/extinction-timeline')
def extinction_timeline():
    """Get language extinction timeline predictions"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Create extinction timeline based on endangerment level and speaker count
    timeline_data = []
    
    # Define extinction probability based on endangerment level
    extinction_probability = {
        'Safe': 0.001,  # Very low probability
        'Vulnerable': 0.05,  # 5% chance in next 10 years
        'Definitely Endangered': 0.15,  # 15% chance in next 10 years
        'Severely Endangered': 0.40,  # 40% chance in next 10 years
        'Critically Endangered': 0.80,  # 80% chance in next 10 years
        'Extinct': 1.0  # Already extinct
    }
    
    # Calculate predicted extinction years
    current_year = 2024
    languages_by_year = {}
    
    for _, row in raw_data.iterrows():
        if row['endangerment_level'] == 'Extinct':
            continue
            
        prob = extinction_probability.get(row['endangerment_level'], 0.1)
        speaker_count = row['speaker_count'] if pd.notna(row['speaker_count']) else 1000
        
        # Adjust probability based on speaker count
        if speaker_count < 10:
            prob *= 2.0  # Double probability for very few speakers
        elif speaker_count < 100:
            prob *= 1.5
        elif speaker_count < 1000:
            prob *= 1.2
        
        # Calculate years until extinction based on probability
        if prob > 0.8:
            extinction_year = current_year + np.random.randint(1, 4)  # 2025-2027
        elif prob > 0.4:
            extinction_year = current_year + np.random.randint(3, 8)  # 2027-2032
        elif prob > 0.15:
            extinction_year = current_year + np.random.randint(8, 15)  # 2032-2039
        elif prob > 0.05:
            extinction_year = current_year + np.random.randint(15, 25)  # 2039-2049
        else:
            extinction_year = current_year + np.random.randint(25, 50)  # 2049-2074
        
        if extinction_year not in languages_by_year:
            languages_by_year[extinction_year] = []
        
        languages_by_year[extinction_year].append({
            'language_name': row['language_name'],
            'country': row.get('country', 'Unknown'),
            'speaker_count': int(speaker_count),
            'endangerment_level': row['endangerment_level'],
            'probability': round(prob * 100, 1)
        })
    
    # Convert to timeline format
    for year in sorted(languages_by_year.keys()):
        languages = languages_by_year[year]
        timeline_data.append({
            'year': year,
            'count': len(languages),
            'languages': languages[:10],  # Show top 10 languages per year
            'total_languages': len(languages),
            'color': get_timeline_color(year)
        })
    
    return jsonify(timeline_data)

def get_timeline_color(year):
    """Get color based on extinction year urgency"""
    if year <= 2027:
        return '#DC143C'  # Red - immediate threat
    elif year <= 2030:
        return '#FF8C00'  # Orange - high threat
    elif year <= 2035:
        return '#FFD700'  # Yellow - moderate threat
    elif year <= 2040:
        return '#32CD32'  # Green - low threat
    else:
        return '#87CEEB'  # Light blue - minimal threat

@app.route('/api/data/speaker-vs-endangerment')
def speaker_vs_endangerment():
    """Get speaker count vs endangerment data for scatter plot"""
    if raw_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Sample data for performance (take every 5th row)
    sample_data = raw_data.iloc[::5].copy()
    
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
    geo_data = raw_data.dropna(subset=['lat', 'lng']).copy()
    
    # Sample data for performance
    if len(geo_data) > 500:
        geo_data = geo_data.sample(n=500, random_state=42)
    
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
            'lat': float(row['lat']),
            'lng': float(row['lng']),
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
    
    # Create transmission level categories based on continuous values
    def categorize_transmission(score):
        if pd.isna(score):
            return 'Unknown'
        elif score == 0.0:
            return 'No Transmission (0.0)'
        elif score < 0.2:
            return 'Very Low (0.0-0.2)'
        elif score < 0.4:
            return 'Low (0.2-0.4)'
        elif score < 0.6:
            return 'Moderate (0.4-0.6)'
        elif score < 0.8:
            return 'Good (0.6-0.8)'
        else:
            return 'Excellent (0.8-1.0)'
    
    # Apply categorization
    raw_data['transmission_category'] = raw_data['intergenerational_transmission'].apply(categorize_transmission)
    
    # Count by category
    transmission_data = raw_data['transmission_category'].value_counts()
    
    # Define colors for each category
    colors = {
        'No Transmission (0.0)': '#8B0000',      # Dark red
        'Very Low (0.0-0.2)': '#DC143C',         # Crimson
        'Low (0.2-0.4)': '#FF4500',              # Orange red
        'Moderate (0.4-0.6)': '#FFD700',         # Gold
        'Good (0.6-0.8)': '#32CD32',             # Lime green
        'Excellent (0.8-1.0)': '#2E8B57',        # Sea green
        'Unknown': '#808080'                      # Gray
    }
    
    data_points = []
    for category, count in transmission_data.items():
        data_points.append({
            'label': category,
            'y': int(count),
            'color': colors.get(category, '#808080')
        })
    
    return jsonify(data_points)


if __name__ == '__main__':
    # Load demo data when starting the app
    try:
        load_demo_data()
        print("Demo data loaded successfully!")
        print("Starting Flask application...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start demo application: {str(e)}")
        print(f"Error: {str(e)}")
        print("Make sure the data file exists at: data/enhanced_sample_data.csv")
