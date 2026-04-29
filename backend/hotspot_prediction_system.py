# Satellite Data Processing & Hotspot Prediction System
# Complete pipeline for NDVI, LST, and urban land-use analysis

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request, render_template_string
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SatelliteDataProcessor:
    """Process satellite data for hotspot prediction"""
    
    def __init__(self, raw_folder='raw', processed_folder='processed'):
        #self.raw_folder = raw_folder
        self.raw_folder = os.path.join(PROJECT_ROOT, 'data', 'raw','delhi_20250619_144238')
        #self.processed_folder = processed_folder
        self.processed_folder = os.path.join(PROJECT_ROOT, 'data', 'processed','delhi_20250619_144238')
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
    def normalize_temperature(self, lst_data, method='minmax'):
        """
        Normalize temperature values for visualization
        
        Args:
            lst_data: Land Surface Temperature array
            method: 'minmax' or 'zscore'
        """
        if method == 'minmax':
            # Min-Max normalization (0-1)
            lst_min = np.nanmin(lst_data)
            lst_max = np.nanmax(lst_data)
            normalized = (lst_data - lst_min) / (lst_max - lst_min)
        elif method == 'zscore':
            # Z-score normalization
            lst_mean = np.nanmean(lst_data)
            lst_std = np.nanstd(lst_data)
            normalized = (lst_data - lst_mean) / lst_std
        
        return normalized
    
    def preprocess_to_geotiff(self, data_array, output_path, bounds, crs='EPSG:4326'):
        """
        Convert processed data to GeoTIFF format
        
        Args:
            data_array: Processed data array
            output_path: Output GeoTIFF file path
            bounds: Geographic bounds (left, bottom, right, top)
            crs: Coordinate Reference System
        """
        height, width = data_array.shape
        
        # Create transform
        transform = from_bounds(*bounds, width, height)
        
        # Write to GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data_array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data_array, 1)
            
        print(f"GeoTIFF saved to: {output_path}")
    
    def preprocess_to_csv(self, data_dict, output_path):
        """
        Convert processed data to CSV format
        
        Args:
            data_dict: Dictionary containing processed data
            output_path: Output CSV file path
        """
        df = pd.DataFrame(data_dict)
        df.to_csv(output_path, index=False)
        print(f"CSV saved to: {output_path}")
        
    def generate_sample_data(self, size=(100, 100), bounds=(-74.25, 40.5, -73.75, 41.0)):
        """Generate sample satellite data for demonstration"""
        
        # Generate sample NDVI data (0-1 range)
        ndvi = np.random.beta(2, 2, size)  # Beta distribution for realistic NDVI
        
        # Generate sample LST data (Celsius)
        base_temp = 25
        lst = base_temp + np.random.normal(0, 5, size) + (1 - ndvi) * 10
        
        # Generate urban land-use data (0-1, higher = more urban)
        urban = np.random.beta(1.5, 3, size)
        
        # Create coordinate grids
        lats = np.linspace(bounds[1], bounds[3], size[0])
        lons = np.linspace(bounds[0], bounds[2], size[1])
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return {
            'ndvi': ndvi,
            'lst': lst,
            'urban': urban,
            'lat': lat_grid,
            'lon': lon_grid,
            'bounds': bounds
        }

class HotspotPredictor:
    """AI Model for hotspot prediction using satellite data"""
    
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, ndvi, lst, urban):
        """Prepare features for model training"""
        # Flatten arrays for model input
        ndvi_flat = ndvi.flatten()
        lst_flat = lst.flatten()
        urban_flat = urban.flatten()
        
        # Create feature matrix
        features = np.column_stack([ndvi_flat, lst_flat, urban_flat])
        
        # Remove NaN values
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]
        
        return features, valid_mask
    
    def create_hotspot_labels(self, lst, ndvi, urban):
        """Create hotspot risk scores based on domain knowledge"""
        # Normalize inputs
        lst_norm = (lst - np.nanmin(lst)) / (np.nanmax(lst) - np.nanmin(lst))
        ndvi_norm = ndvi
        urban_norm = urban
        
        # Calculate risk score
        # High temperature + Low vegetation + High urban = High risk
        risk_score = (
            0.4 * lst_norm +           # Temperature weight
            0.3 * (1 - ndvi_norm) +    # Inverse vegetation weight
            0.3 * urban_norm           # Urban weight
        )
        
        return risk_score
    
    def train_model(self, ndvi, lst, urban):
        """Train the hotspot prediction model"""
        print("Training hotspot prediction model...")
        
        # Prepare features
        features, valid_mask = self.prepare_features(ndvi, lst, urban)
        
        # Create target variable (hotspot risk scores)
        risk_scores = self.create_hotspot_labels(lst, ndvi, urban)
        risk_scores_flat = risk_scores.flatten()[valid_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, risk_scores_flat, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Feature importance
        feature_names = ['NDVI', 'LST', 'Urban']
        importance = self.model.feature_importances_
        
        print(f"\nFeature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp:.4f}")
        
        self.is_trained = True
        return mse, r2
    
    def predict_hotspots(self, ndvi, lst, urban):
        """Predict hotspot risk scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Prepare features
        features, valid_mask = self.prepare_features(ndvi, lst, urban)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        
        # Reshape predictions back to original shape
        risk_map = np.full(ndvi.shape, np.nan)
        risk_map.flat[valid_mask] = predictions
        
        return risk_map

class MitigationRecommendationSystem:
    """System for generating mitigation recommendations"""
    
    def __init__(self):
        self.recommendations = {
            'tree_cover': {
                'action': 'Increase tree cover',
                'description': 'Plant trees and create green spaces',
                'priority': 'High'
            },
            'cool_pavements': {
                'action': 'Install cool pavements',
                'description': 'Use reflective materials for roads and pavements',
                'priority': 'Medium'
            },
            'green_roofs': {
                'action': 'Implement green roofs',
                'description': 'Install vegetation on building rooftops',
                'priority': 'Medium'
            },
            'water_features': {
                'action': 'Add water features',
                'description': 'Create fountains, ponds, or water bodies',
                'priority': 'Low'
            }
        }
    
    def generate_recommendations(self, risk_score, ndvi, lst, urban, lat, lon):
        """Generate location-specific recommendations"""
        recommendations = []
        
        # Rule-based logic for recommendations
        if risk_score > 0.7:  # High risk
            if ndvi < 0.3:  # Low vegetation
                recommendations.append({
                    **self.recommendations['tree_cover'],
                    'lat': lat,
                    'lon': lon,
                    'risk_score': risk_score,
                    'reason': 'Low vegetation in high-risk area'
                })
            
            if urban > 0.6:  # High urban density
                recommendations.append({
                    **self.recommendations['cool_pavements'],
                    'lat': lat,
                    'lon': lon,
                    'risk_score': risk_score,
                    'reason': 'High urban density contributing to heat'
                })
                
                recommendations.append({
                    **self.recommendations['green_roofs'],
                    'lat': lat,
                    'lon': lon,
                    'risk_score': risk_score,
                    'reason': 'Dense urban area needs cooling'
                })
        
        elif risk_score > 0.5:  # Medium risk
            if ndvi < 0.5:
                recommendations.append({
                    **self.recommendations['tree_cover'],
                    'lat': lat,
                    'lon': lon,
                    'risk_score': risk_score,
                    'reason': 'Medium risk area with limited vegetation'
                })
        
        return recommendations
    
    def batch_recommendations(self, risk_map, ndvi, lst, urban, lat_grid, lon_grid, threshold=0.5):
        """Generate recommendations for all high-risk areas"""
        all_recommendations = []
        
        # Find high-risk pixels
        high_risk_mask = risk_map > threshold
        
        for i in range(risk_map.shape[0]):
            for j in range(risk_map.shape[1]):
                if high_risk_mask[i, j] and not np.isnan(risk_map[i, j]):
                    recommendations = self.generate_recommendations(
                        risk_map[i, j], 
                        ndvi[i, j], 
                        lst[i, j], 
                        urban[i, j],
                        lat_grid[i, j], 
                        lon_grid[i, j]
                    )
                    all_recommendations.extend(recommendations)
        
        return all_recommendations

# Flask API for serving data
app = Flask(__name__)

# Global variables to store processed data
processed_data = {}
hotspot_model = None
mitigation_system = None

@app.route('/api/process_data', methods=['POST'])
def process_data():
    """Process satellite data and train model"""
    global processed_data, hotspot_model, mitigation_system
    
    try:
        # Initialize processors
        processor = SatelliteDataProcessor()
        hotspot_model = HotspotPredictor()
        mitigation_system = MitigationRecommendationSystem()
        
        # Generate sample data (replace with actual data loading)
        data = processor.generate_sample_data()
        
        # Normalize temperature
        normalized_lst = processor.normalize_temperature(data['lst'])
        
        # Save processed data
        processor.preprocess_to_geotiff(
            data['ndvi'], 
            'output/ndvi.tif', 
            data['bounds']
        )
        processor.preprocess_to_geotiff(
            normalized_lst, 
            'output/lst_normalized.tif', 
            data['bounds']
        )
        processor.preprocess_to_geotiff(
            data['urban'], 
            'output/urban.tif', 
            data['bounds']
        )
        
        # Train hotspot prediction model
        mse, r2 = hotspot_model.train_model(data['ndvi'], data['lst'], data['urban'])
        
        # Predict hotspots
        risk_map = hotspot_model.predict_hotspots(data['ndvi'], data['lst'], data['urban'])
        
        # Generate recommendations
        recommendations = mitigation_system.batch_recommendations(
            risk_map, data['ndvi'], data['lst'], data['urban'],
            data['lat'], data['lon']
        )
        
        # Store processed data
        processed_data = {
            'ndvi': data['ndvi'].tolist(),
            'lst': data['lst'].tolist(),
            'urban': data['urban'].tolist(),
            'risk_map': risk_map.tolist(),
            'lat': data['lat'].tolist(),
            'lon': data['lon'].tolist(),
            'bounds': data['bounds'],
            'recommendations': recommendations,
            'model_performance': {'mse': mse, 'r2': r2}
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Data processed successfully',
            'model_performance': {'mse': mse, 'r2': r2},
            'recommendations_count': len(recommendations)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    """Get hotspot prediction data"""
    if not processed_data:
        return jsonify({'status': 'error', 'message': 'No data processed yet'}), 400
    
    return jsonify({
        'status': 'success',
        'data': {
            'risk_map': processed_data['risk_map'],
            'bounds': processed_data['bounds'],
            'coordinates': {
                'lat': processed_data['lat'],
                'lon': processed_data['lon']
            }
        }
    })

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get mitigation recommendations"""
    if not processed_data:
        return jsonify({'status': 'error', 'message': 'No data processed yet'}), 400
    
    return jsonify({
        'status': 'success',
        'recommendations': processed_data['recommendations']
    })

@app.route('/api/satellite_data', methods=['GET'])
def get_satellite_data():
    """Get satellite data (NDVI, LST, Urban)"""
    layer = request.args.get('layer', 'ndvi')
    
    if not processed_data:
        return jsonify({'status': 'error', 'message': 'No data processed yet'}), 400
    
    if layer not in ['ndvi', 'lst', 'urban']:
        return jsonify({'status': 'error', 'message': 'Invalid layer'}), 400
    
    return jsonify({
        'status': 'success',
        'layer': layer,
        'data': processed_data[layer],
        'bounds': processed_data['bounds'],
        'coordinates': {
            'lat': processed_data['lat'],
            'lon': processed_data['lon']
        }
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get data statistics"""
    if not processed_data:
        return jsonify({'status': 'error', 'message': 'No data processed yet'}), 400
    
    # Calculate statistics
    stats = {}
    for layer in ['ndvi', 'lst', 'urban', 'risk_map']:
        data_array = np.array(processed_data[layer])
        stats[layer] = {
            'mean': float(np.nanmean(data_array)),
            'std': float(np.nanstd(data_array)),
            'min': float(np.nanmin(data_array)),
            'max': float(np.nanmax(data_array))
        }
    
    return jsonify({
        'status': 'success',
        'statistics': stats,
        'model_performance': processed_data.get('model_performance', {})
    })


@app.route('/')
def dashboard():
    """Serve the dashboard"""
    return render_template_string(DASHBOARD_HTML)

if __name__ == '__main__':
    print("🌡️ Urban Heat Island Hotspot Prediction System")
    print("=" * 50)
    print("Starting Flask application...")
    print("Dashboard will be available at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("- POST /api/process_data - Process satellite data and train model")
    print("- GET /api/hotspots - Get hotspot predictions")
    print("- GET /api/recommendations - Get mitigation recommendations")
    print("- GET /api/satellite_data?layer=<ndvi|lst|urban> - Get satellite data")
    print("- GET /api/stats - Get data statistics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)