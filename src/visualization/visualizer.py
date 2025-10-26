"""
Data Visualization Module for Language Extinction Risk Prediction

This module provides comprehensive visualization capabilities for:
- Global endangerment heat maps
- Language family trees with endangerment
- Feature importance charts
- Endangerment transition matrices
- Interactive dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
from pathlib import Path
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageVisualizer:
    """
    Main class for creating visualizations for language extinction risk analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the visualizer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.viz_dir = Path(self.config['paths']['visualizations_dir'])
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color scheme from config
        self.colors = self.config['visualization']['color_scheme']
        
    def create_global_endangerment_map(self, df: pd.DataFrame, 
                                     lat_col: str = 'latitude', 
                                     lon_col: str = 'longitude',
                                     endangerment_col: str = 'endangerment_level',
                                     title: str = "Global Language Endangerment Map") -> go.Figure:
        """
        Create interactive global map showing language endangerment
        
        Args:
            df (pd.DataFrame): Dataset with geographic and endangerment data
            lat_col (str): Latitude column name
            lon_col (str): Longitude column name
            endangerment_col (str): Endangerment level column name
            title (str): Map title
            
        Returns:
            go.Figure: Interactive Plotly map
        """
        logger.info("Creating global endangerment map...")
        
        # Prepare data
        map_data = df.dropna(subset=[lat_col, lon_col, endangerment_col]).copy()
        
        # Create color mapping
        color_map = {
            'Safe': self.colors['safe'],
            'Vulnerable': self.colors['vulnerable'],
            'Definitely Endangered': self.colors['definitely_endangered'],
            'Severely Endangered': self.colors['severely_endangered'],
            'Critically Endangered': self.colors['critically_endangered'],
            'Extinct': self.colors['extinct']
        }
        
        map_data['color'] = map_data[endangerment_col].map(color_map)
        
        # Create scatter plot on map
        fig = go.Figure()
        
        for level in map_data[endangerment_col].unique():
            level_data = map_data[map_data[endangerment_col] == level]
            
            fig.add_trace(go.Scattergeo(
                lon=level_data[lon_col],
                lat=level_data[lat_col],
                text=level_data['name'] + '<br>' + 
                     'Speakers: ' + level_data['speaker_count'].astype(str) + '<br>' +
                     'Country: ' + level_data['country'].astype(str),
                mode='markers',
                marker=dict(
                    size=8,
                    color=color_map[level],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=level,
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            geo=dict(
                scope='world',
                showland=True,
                landcolor='lightgray',
                showocean=True,
                oceancolor='lightblue',
                projection_type='equirectangular'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "global_endangerment_map.html")
        logger.info("Global endangerment map saved")
        
        return fig
    
    def create_language_family_tree(self, df: pd.DataFrame, 
                                  family_col: str = 'family_id',
                                  endangerment_col: str = 'endangerment_level',
                                  speaker_col: str = 'speaker_count',
                                  title: str = "Language Family Endangerment Tree") -> go.Figure:
        """
        Create tree diagram showing language family endangerment
        
        Args:
            df (pd.DataFrame): Dataset with family and endangerment data
            family_col (str): Language family column name
            endangerment_col (str): Endangerment level column name
            speaker_col (str): Speaker count column name
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive tree diagram
        """
        logger.info("Creating language family tree...")
        
        # Prepare data
        family_data = df.groupby([family_col, endangerment_col]).agg({
            speaker_col: ['count', 'sum', 'mean']
        }).round(0)
        
        family_data.columns = ['language_count', 'total_speakers', 'avg_speakers']
        family_data = family_data.reset_index()
        
        # Create color mapping
        color_map = {
            'Safe': self.colors['safe'],
            'Vulnerable': self.colors['vulnerable'],
            'Definitely Endangered': self.colors['definitely_endangered'],
            'Severely Endangered': self.colors['severely_endangered'],
            'Critically Endangered': self.colors['critically_endangered'],
            'Extinct': self.colors['extinct']
        }
        
        family_data['color'] = family_data[endangerment_col].map(color_map)
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=family_data[family_col] + '<br>' + 
                   family_data[endangerment_col] + '<br>' +
                   'Languages: ' + family_data['language_count'].astype(str),
            parents=[''] * len(family_data),
            values=family_data['total_speakers'],
            marker=dict(
                colors=family_data['color'],
                line=dict(width=2, color='white')
            ),
            textinfo="label+value",
            hovertemplate='<b>%{label}</b><br>' +
                         'Total Speakers: %{value}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            font=dict(size=12)
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "language_family_tree.html")
        logger.info("Language family tree saved")
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict[str, float],
                                      title: str = "Feature Importance for Language Endangerment Prediction",
                                      top_n: int = 15) -> go.Figure:
        """
        Create horizontal bar chart of feature importance
        
        Args:
            feature_importance (Dict[str, float]): Feature importance dictionary
            title (str): Chart title
            top_n (int): Number of top features to show
            
        Returns:
            go.Figure: Interactive bar chart
        """
        logger.info("Creating feature importance chart...")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "feature_importance.html")
        logger.info("Feature importance chart saved")
        
        return fig
    
    def create_endangerment_transition_matrix(self, df: pd.DataFrame,
                                            current_col: str = 'endangerment_level',
                                            predicted_col: str = 'predicted_endangerment',
                                            title: str = "Language Endangerment Transitions") -> go.Figure:
        """
        Create Sankey diagram showing transitions between endangerment levels
        
        Args:
            df (pd.DataFrame): Dataset with current and predicted endangerment
            current_col (str): Current endangerment column
            predicted_col (str): Predicted endangerment column
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive Sankey diagram
        """
        logger.info("Creating endangerment transition matrix...")
        
        # Create transition counts
        transition_counts = df.groupby([current_col, predicted_col]).size().reset_index(name='count')
        
        # Create node labels
        all_levels = sorted(set(transition_counts[current_col].unique()) | 
                           set(transition_counts[predicted_col].unique()))
        
        node_labels = []
        for level in all_levels:
            node_labels.extend([f"{level} (Current)", f"{level} (Predicted)"])
        
        # Create source and target indices
        source = []
        target = []
        value = []
        
        for _, row in transition_counts.iterrows():
            current_idx = all_levels.index(row[current_col])
            predicted_idx = all_levels.index(row[predicted_col]) + len(all_levels)
            
            source.append(current_idx)
            target.append(predicted_idx)
            value.append(row['count'])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="lightblue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            title=title,
            font_size=12,
            height=600
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "endangerment_transitions.html")
        logger.info("Endangerment transition matrix saved")
        
        return fig
    
    def create_speaker_vs_endangerment_scatter(self, df: pd.DataFrame,
                                             speaker_col: str = 'speaker_count',
                                             endangerment_col: str = 'endangerment_level',
                                             transmission_col: str = 'intergenerational_transmission',
                                             title: str = "Speaker Population vs Endangerment") -> go.Figure:
        """
        Create scatter plot showing relationship between speaker numbers and endangerment
        
        Args:
            df (pd.DataFrame): Dataset with speaker and endangerment data
            speaker_col (str): Speaker count column
            endangerment_col (str): Endangerment level column
            transmission_col (str): Intergenerational transmission column
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive scatter plot
        """
        logger.info("Creating speaker vs endangerment scatter plot...")
        
        # Prepare data
        scatter_data = df.dropna(subset=[speaker_col, endangerment_col]).copy()
        
        # Create color mapping
        color_map = {
            'Safe': self.colors['safe'],
            'Vulnerable': self.colors['vulnerable'],
            'Definitely Endangered': self.colors['definitely_endangered'],
            'Severely Endangered': self.colors['severely_endangered'],
            'Critically Endangered': self.colors['critically_endangered'],
            'Extinct': self.colors['extinct']
        }
        
        scatter_data['color'] = scatter_data[endangerment_col].map(color_map)
        
        # Create scatter plot
        fig = go.Figure()
        
        for level in scatter_data[endangerment_col].unique():
            level_data = scatter_data[scatter_data[endangerment_col] == level]
            
            fig.add_trace(go.Scatter(
                x=level_data[speaker_col],
                y=level_data[endangerment_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color_map[level],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=level,
                text=level_data['name'] + '<br>' + 
                     'Transmission: ' + level_data[transmission_col].astype(str),
                hovertemplate='<b>%{text}</b><br>' +
                             'Speakers: %{x}<br>' +
                             'Endangerment: %{y}<extra></extra>'
            ))
        
        # Add trend line
        if len(scatter_data) > 1:
            # Convert endangerment to numeric for trend line
            endangerment_numeric = pd.Categorical(scatter_data[endangerment_col]).codes
            z = np.polyfit(np.log10(scatter_data[speaker_col] + 1), endangerment_numeric, 1)
            p = np.poly1d(z)
            
            x_trend = np.logspace(0, 6, 100)
            y_trend = p(np.log10(x_trend))
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Trend Line',
                showlegend=True
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Number of Speakers (log scale)",
            yaxis_title="Endangerment Level",
            xaxis_type="log",
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "speaker_vs_endangerment.html")
        logger.info("Speaker vs endangerment scatter plot saved")
        
        return fig
    
    def create_confusion_matrix_heatmap(self, confusion_matrix: np.ndarray,
                                      class_labels: List[str],
                                      title: str = "Confusion Matrix") -> go.Figure:
        """
        Create heatmap of confusion matrix
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            class_labels (List[str]): Class labels
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive heatmap
        """
        logger.info("Creating confusion matrix heatmap...")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_labels,
            y=class_labels,
            colorscale='Blues',
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "confusion_matrix.html")
        logger.info("Confusion matrix heatmap saved")
        
        return fig
    
    def create_model_performance_comparison(self, model_results: Dict[str, Dict[str, Any]],
                                          title: str = "Model Performance Comparison") -> go.Figure:
        """
        Create bar chart comparing model performance
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Model evaluation results
            title (str): Chart title
            
        Returns:
            go.Figure: Interactive bar chart
        """
        logger.info("Creating model performance comparison...")
        
        # Extract metrics
        models = list(model_results.keys())
        accuracies = [model_results[model]['test_accuracy'] for model in models]
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=models,
            y=accuracies,
            marker=dict(
                color=accuracies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Accuracy")
            ),
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title="Test Accuracy",
            height=500
        )
        
        # Save figure
        fig.write_html(self.viz_dir / "model_performance_comparison.html")
        logger.info("Model performance comparison saved")
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame, 
                                   model_results: Dict[str, Dict[str, Any]],
                                   feature_importance: Dict[str, float]) -> None:
        """
        Create comprehensive interactive dashboard
        
        Args:
            df (pd.DataFrame): Main dataset
            model_results (Dict[str, Dict[str, Any]]): Model evaluation results
            feature_importance (Dict[str, float]): Feature importance
        """
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Global Endangerment Map",
                "Feature Importance",
                "Speaker vs Endangerment",
                "Model Performance",
                "Endangerment Distribution",
                "Confusion Matrix"
            ),
            specs=[
                [{"type": "scattergeo"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "heatmap"}]
            ]
        )
        
        # Add global map (simplified for subplot)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            fig.add_trace(
                go.Scattergeo(
                    lon=df['longitude'],
                    lat=df['latitude'],
                    mode='markers',
                    marker=dict(size=5, color='blue'),
                    name='Languages'
                ),
                row=1, col=1
            )
        
        # Add feature importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*top_features)
        
        fig.add_trace(
            go.Bar(x=list(importances), y=list(features), orientation='h', name='Importance'),
            row=1, col=2
        )
        
        # Add speaker vs endangerment scatter
        if 'speaker_count' in df.columns and 'endangerment_level' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['speaker_count'],
                    y=df['endangerment_level'],
                    mode='markers',
                    name='Languages'
                ),
                row=2, col=1
            )
        
        # Add model performance
        models = list(model_results.keys())
        accuracies = [model_results[model]['test_accuracy'] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy'),
            row=2, col=2
        )
        
        # Add endangerment distribution
        if 'endangerment_level' in df.columns:
            endangerment_counts = df['endangerment_level'].value_counts()
            fig.add_trace(
                go.Bar(x=endangerment_counts.index, y=endangerment_counts.values, name='Count'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Language Extinction Risk Prediction Dashboard",
            height=1200,
            showlegend=False
        )
        
        # Save dashboard
        fig.write_html(self.viz_dir / "interactive_dashboard.html")
        logger.info("Interactive dashboard saved")
    
    def create_confusion_matrices_for_all_models(self, model_results: Dict[str, Dict[str, Any]],
                                                 class_names: List[str]) -> None:
        """
        Create comprehensive confusion matrices for all models
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Model evaluation results
            class_names (List[str]): List of class names
        """
        logger.info("Creating confusion matrices for all models...")
        
        # Create combined figure with all confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(model_results.items()):
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], square=True, linewidths=0.5, linecolor='gray',
                       cbar_kws={'label': 'Percentage'})
            
            # Add accuracy to title
            accuracy = results['test_accuracy']
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {accuracy:.2%}',
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
            
            # Rotate labels
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.viz_dir / "confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {save_path}")
    
    def create_static_plots(self, df: pd.DataFrame, 
                          model_results: Dict[str, Dict[str, Any]],
                          feature_importance: Dict[str, float]) -> None:
        """
        Create static matplotlib plots for reports
        
        Args:
            df (pd.DataFrame): Main dataset
            model_results (Dict[str, Dict[str, Any]]): Model evaluation results
            feature_importance (Dict[str, float]): Feature importance
        """
        logger.info("Creating static plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Language Extinction Risk Analysis', fontsize=16, fontweight='bold')
        
        # 1. Endangerment distribution
        if 'endangerment_level' in df.columns:
            endangerment_counts = df['endangerment_level'].value_counts()
            axes[0, 0].bar(endangerment_counts.index, endangerment_counts.values, 
                          color=[self.colors.get(level, 'gray') for level in endangerment_counts.index])
            axes[0, 0].set_title('Endangerment Level Distribution')
            axes[0, 0].set_xlabel('Endangerment Level')
            axes[0, 0].set_ylabel('Number of Languages')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Speaker count distribution
        if 'speaker_count' in df.columns:
            axes[0, 1].hist(np.log10(df['speaker_count'].dropna() + 1), bins=30, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Speaker Count Distribution (log scale)')
            axes[0, 1].set_xlabel('Log10(Speaker Count + 1)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Feature importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*top_features)
        axes[0, 2].barh(features, importances, color='lightcoral')
        axes[0, 2].set_title('Top 10 Feature Importance')
        axes[0, 2].set_xlabel('Importance Score')
        
        # 4. Model performance comparison
        models = list(model_results.keys())
        accuracies = [model_results[model]['test_accuracy'] for model in models]
        axes[1, 0].bar(models, accuracies, color='lightgreen')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Geographic distribution
        if 'latitude' in df.columns and 'longitude' in df.columns:
            scatter = axes[1, 1].scatter(df['longitude'], df['latitude'], 
                                       c=df['endangerment_level'].astype('category').cat.codes, 
                                       cmap='viridis', alpha=0.6, s=20)
            axes[1, 1].set_title('Geographic Distribution of Languages')
            axes[1, 1].set_xlabel('Longitude')
            axes[1, 1].set_ylabel('Latitude')
        
        # 6. Confusion matrix (if available)
        if model_results and 'random_forest' in model_results:
            cm = model_results['random_forest']['confusion_matrix']
            # Normalize for better visualization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1, 2],
                       square=True, linewidths=0.5, cbar_kws={'label': 'Percentage'})
            axes[1, 2].set_title('Random Forest Confusion Matrix')
            axes[1, 2].set_xlabel('Predicted')
            axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save static plots
        plt.savefig(self.viz_dir / "static_analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Static plots saved")


def main():
    """Main function to demonstrate visualization capabilities"""
    from data.data_loader import LanguageDataLoader
    from data.data_preprocessor import LanguageDataPreprocessor
    from models.ml_models import LanguageExtinctionPredictor
    
    # Load and preprocess data
    loader = LanguageDataLoader()
    datasets = loader.load_all_datasets()
    merged_data = loader.merge_datasets()
    
    preprocessor = LanguageDataPreprocessor()
    X, y, feature_names = preprocessor.preprocess_pipeline(merged_data)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train models
    predictor = LanguageExtinctionPredictor()
    model_results = predictor.train_all_models(X_train_scaled, y_train)
    evaluation_results = predictor.evaluate_all_models(X_test_scaled, y_test)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance('random_forest', top_n=15)
    feature_importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
    
    # Create visualizations
    visualizer = LanguageVisualizer()
    
    # Create individual visualizations
    visualizer.create_global_endangerment_map(merged_data)
    visualizer.create_language_family_tree(merged_data)
    visualizer.create_feature_importance_chart(feature_importance_dict)
    visualizer.create_speaker_vs_endangerment_scatter(merged_data)
    visualizer.create_model_performance_comparison(evaluation_results)
    
    # Create comprehensive dashboard
    visualizer.create_interactive_dashboard(merged_data, evaluation_results, feature_importance_dict)
    
    # Create static plots
    visualizer.create_static_plots(merged_data, evaluation_results, feature_importance_dict)
    
    print("All visualizations created successfully!")
    print(f"Visualizations saved in: {visualizer.viz_dir}")
    
    return visualizer


if __name__ == "__main__":
    visualizer = main()
