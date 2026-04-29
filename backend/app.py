import os
import sys
import json
import subprocess
import yaml
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from flask import Flask, render_template, send_from_directory, jsonify, request, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules (make sure these files exist and are correct)
try:
    from raster_clip_utils import clip_all_rasters_in_folder
    from geoclassifier import GeoClassifierConfig, LandUseClassifier, GeoClassifierUtils
    from reproject_files import process_folder
    from shapely.geometry import Point, shape, mapping
    import pyproj
    from functools import partial
    from shapely.ops import transform
    from band_process import process_satellite_bands
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("The application will run with limited functionality")

# Absolute path to this file (backend/app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root (one level up from backend/)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Common paths used in the app
template_dir = os.path.join(PROJECT_ROOT, 'frontend')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
SERVER_CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'server_configs')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
TRAINING_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'training_data')
MODAL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create necessary directories
for directory in [CONFIGS_DIR, SERVER_CONFIGS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, 
                  ASSETS_DIR, TRAINING_DATA_FOLDER, MODAL_DIR]:
    os.makedirs(directory, exist_ok=True)

app = Flask(__name__, template_folder=template_dir)

CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/configs/*": {"origins": "*"}
})

app.config['UPLOAD_FOLDER'] = SERVER_CONFIGS_DIR
app.config['ALLOWED_EXTENSIONS'] = {'yml', 'yaml'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Serve main index page
@app.route('/')
def index():
    return render_template('index.html')

# Serve all frontend pages
@app.route('/<page>')
def serve_page(page):
    if page.endswith('.html'):
        return render_template(page)
    return render_template(f'{page}.html')
########################################################################################################################
#####
@app.route('/location-config')
def location_config():
    return render_template('location_config_selector.html')

@app.route('/api/save-config', methods=['POST'])
def save_config():
    try:
        data = request.json
        if not data or 'yaml' not in data:
            return jsonify({'ok': False, 'message': 'Invalid request data'}), 400
            
        yaml_content = data['yaml']
        
        # Parse YAML to get the first city prefix
        try:
            config_data = yaml.safe_load(yaml_content)
            first_city = next(iter(config_data.get('city_boundaries', {})), None)
            prefix = f"{first_city}_" if first_city else "uhi_"
        except Exception as e:
            prefix = "uhi_"
        
        # Generate filename with city prefix and datestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}{timestamp}.yml"
        
        # Ensure filename is safe
        filename = secure_filename(filename)
        if not filename.lower().endswith(('.yml', '.yaml')):
            filename += '.yml'
        
        # Save to configs directory
        os.makedirs('configs', exist_ok=True)
        filepath = os.path.join('configs', filename)
        
        with open(filepath, 'w') as f:
            f.write(yaml_content)
            
        return jsonify({
            'ok': True,
            'message': f'Config saved successfully as {filename}',
            'path': filepath,
            'download_url': f'/configs/{filename}',
            'filename': filename
        })
    except Exception as e:
        return jsonify({
            'ok': False,
            'message': str(e)
        }), 500

# Add these endpoints to your Flask app

@app.route('/api/load-config/<filename>')
def load_config(filename):
    try:
        filepath = os.path.join('configs', secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        return jsonify({
            'name': filename,
            'content': content,
            'yaml': yaml.safe_load(content)  # Return parsed YAML as well
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for config uploads
@app.route('/api/upload-config', methods=['POST'])
def upload_config():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            "status": "success",
            "message": f"File {filename} uploaded successfully"
        })
    
    return jsonify({"error": "Invalid file type"}), 400

#
@app.route('/api/list-configs', methods=['GET'])
def list_configs():
    try:
        config_dir = 'configs'  # Your config directory
        configs = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
        return jsonify({'success': True, 'configs': configs})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    try :
        if not request.json:
            return jsonify({'success': False, 'message': 'No JSON data provided'})
        
        config_filename = request.json.get('config_filename')
        if not config_filename:
            return jsonify({'success': False, 'message': 'config_filename not provided'})
        
        # Check if analysis script exists
        script_path = 'uhi_analysis.py'
        if not os.path.exists(script_path):
            return jsonify({'success': False, 'message': f'Analysis script not found: {script_path}'})
        
        # Check if config file exists
        config_path = os.path.join('configs', config_filename)
        if not os.path.exists(config_path):
            return jsonify({'success': False, 'message': f'Config file not found: {config_filename}'})
        
        result = subprocess.run(['python', script_path, config_filename], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Analysis completed'})
        else:
            return jsonify({'success': False, 'message': f'Analysis failed: {result.stderr}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# Serve static files (CSS, JS, images)
@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('assets', path)

# Serve uploaded config files
@app.route('/configs/<filename>')
def serve_config(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API endpoint for dashboard data
@app.route('/api/dashboard-data')
def dashboard_data():
    # In a real app, you would query your database here
    sample_data = {
        "cities": [
            {"name": "New York", "avg_temp": 24.5, "uhi_intensity": 2.3},
            {"name": "Los Angeles", "avg_temp": 26.1, "uhi_intensity": 1.8}
        ],
        "time_series": {
            "dates": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "temperatures": [22.1, 23.5, 25.2]
        }
    }
    return jsonify(sample_data)

# API endpoint for satellite analysis
@app.route('/api/process-satellite', methods=['POST'])
def process_satellite():
    try:
        # In a real app, you would process the uploaded files here
        return jsonify({
            "status": "success",
            "results": {
                "lst_map": "data/lst_result.tif",
                "ndvi_map": "data/ndvi_result.tif",
                "stats": {
                    "max_temp": 35.2,
                    "min_temp": 18.7,
                    "avg_ndvi": 0.42
                }
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500
################################
@app.route('/save_training_points', methods=['POST'])
def save_training_points():
    """Endpoint to save training points to a JSON file"""
    try:
        # Get data from request
        data = request.get_json()
        name = data.get('name')
        points = data.get('points')
        folder = data.get('folder')
        
        # Validate required fields
        if not all([name, points, folder]):
            return jsonify({
                "success": False,
                "error": "Missing required fields (name, points, or folder)"
            }), 400

        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(TRAINING_DATA_FOLDER, filename)

        # Save the data
        with open(filepath, 'w') as f:
            json.dump({
                "name": name,
                "folder": name,
                "time": name,
                "points": points,
                "created": timestamp,
                "version": "1.0"
            }, f, indent=2)

        return jsonify({
            "success": True,
            "message": "Training points saved successfully",
            "path": filepath,
            "filename": filename
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

##########################################################
@app.route('/training-data', methods=['GET'])
def get_training_data():
    
  print("Files in folder:")
  for filename in os.listdir(TRAINING_DATA_FOLDER):
    print(filename)


    """Endpoint to list available training datasets"""
    path = os.path.join(TRAINING_DATA_FOLDER)
    if not os.path.exists(path):
        return jsonify({"error": "Folder not found"}), 404
    files = [f for f in os.listdir(path)]
    return jsonify(files)

##########################################
@app.route('/folders')
def list_folders():
    folders = [f for f in os.listdir(DATA_RAW_DIR) if os.path.isdir(os.path.join(DATA_RAW_DIR, f))]
    return jsonify(folders)

@app.route('/files/<folder>')
def list_files(folder):
    path = os.path.join(DATA_RAW_DIR, folder)
    if not os.path.exists(path):
        return jsonify({"error": "Folder not found"}), 404
    files = [f for f in os.listdir(path) if f.lower().endswith('.tif')]
    return jsonify(files)

@app.route('/raster/<folder>/<filename>')
def serve_raster(folder, filename):
    """Serve raster files with proper error handling and MIME type"""
    # Secure the folder and filename to prevent path traversal
    folder = secure_filename(folder)
    filename = secure_filename(filename)
    
    file_path = os.path.join(DATA_RAW_DIR, folder, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {folder}/{filename}"}), 404
    
    # Check if it's actually a file (not a directory)
    if not os.path.isfile(file_path):
        return jsonify({"error": f"Path is not a file: {folder}/{filename}"}), 404
    
    try:
        # Use send_file for better performance and proper headers
        return send_file(
            file_path,
            mimetype='image/tiff',
            as_attachment=False,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500

# Alternative implementation using streaming (if send_file doesn't work)
@app.route('/raster-stream/<folder>/<filename>')
def serve_raster_stream(folder, filename):
    """Alternative streaming implementation"""
    folder = secure_filename(folder)
    filename = secure_filename(filename)
    
    file_path = os.path.join(DATA_RAW_DIR, folder, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {folder}/{filename}"}), 404
    
    def generate():
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            print(f"Error reading file: {e}")
            return
                
    return Response(
        generate(), 
        mimetype='image/tiff',
        headers={
            'Content-Disposition': f'inline; filename="{filename}"',
            'Content-Type': 'image/tiff'
        }
    )

# Add a debug endpoint to check file existence
@app.route('/debug/raster/<folder>/<filename>')
def debug_raster(folder, filename):
    """Debug endpoint to check file status"""
    folder = secure_filename(folder)
    filename = secure_filename(filename)
    
    file_path = os.path.join(DATA_RAW_DIR, folder, filename)
    
    debug_info = {
        "folder": folder,
        "filename": filename,
        "file_path": file_path,
        "data_raw_dir": DATA_RAW_DIR,
        "exists": os.path.exists(file_path),
        "is_file": os.path.isfile(file_path) if os.path.exists(file_path) else False,
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "readable": os.access(file_path, os.R_OK) if os.path.exists(file_path) else False
    }
    
    return jsonify(debug_info)

import math

def copy_all_files(src_folder, dst_folder, overwrite=False, backup_existing=False):
    """
    Copy all files from source folder to destination folder
    
    Args:
        src_folder (str): Source directory path
        dst_folder (str): Destination directory path
        overwrite (bool): Whether to overwrite existing files
        backup_existing (bool): Whether to backup existing files before overwriting
        
    Returns:
        dict: Results with counts and error information
    """
    results = {
        'copied': 0,
        'skipped': 0,
        'errors': 0,
        'error_messages': [],
        'backed_up': 0
    }
    
    try:
        # Create destination folder if it doesn't exist
        os.makedirs(dst_folder, exist_ok=True)
        
        # Get list of files in source folder
        files = [f for f in os.listdir(src_folder) 
                if os.path.isfile(os.path.join(src_folder, f))]
        
        for filename in files:
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            
            try:
                # Handle existing files
                if os.path.exists(dst_path):
                    if not overwrite:
                        results['skipped'] += 1
                        continue
                        
                    if backup_existing:
                        # Create backup of existing file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = os.path.join(dst_folder, f"backup_{timestamp}_{filename}")
                        shutil.copy2(dst_path, backup_path)
                        results['backed_up'] += 1
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                results['copied'] += 1
                
            except PermissionError as e:
                results['errors'] += 1
                results['error_messages'].append(f"Permission error copying {filename}: {str(e)}")
            except Exception as e:
                results['errors'] += 1
                results['error_messages'].append(f"Error copying {filename}: {str(e)}")
                
    except Exception as e:
        results['errors'] += 1
        results['error_messages'].append(f"Folder error: {str(e)}")
    
    return results
import os

def delete_all_files(folder_path):
    """Delete all files in the specified folder"""
    try:
        # Verify folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder does not exist - {folder_path}")
            return False
            
        # Verify it's a directory
        if not os.path.isdir(folder_path):
            print(f"Error: Path is not a folder - {folder_path}")
            return False
            
        # Get all files in folder
        files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f))]
        
        # Delete each file
        for file in files:
            try:
                os.remove(os.path.join(folder_path, file))
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
                
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route('/clip', methods=['POST'])
def clip():
    data = request.get_json()
    folder = data.get('folder')
    geometry = data.get('geometry')

    input_folder = os.path.join(DATA_RAW_DIR, folder)
    output_folder = os.path.join(PROJECT_ROOT, "data", "clipped", folder)
    os.makedirs(output_folder, exist_ok=True)
    

    if not os.path.exists(input_folder):
        return jsonify({"error": f"Input folder does not exist: {input_folder}"}), 404

    try:
        copy_all_files(input_folder,output_folder)
        delete_all_files(input_folder)
        result = clip_all_rasters_in_folder(output_folder, input_folder, geometry)
        return jsonify({"clipped": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize geoclassifier components
gconfig = GeoClassifierConfig()
classifier = LandUseClassifier(gconfig)
utils = GeoClassifierUtils(gconfig)

@app.route('/classify', methods=['POST'])
def classify():
    """
    Fixed classify route that properly integrates with geoclassifier classes
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        print(data)
        # Get parameters
        folder = data.get('folder')
        training_filename = data.get('training')
        
        if not folder or not training_filename:
            return jsonify({
                'status': 'error',
                'message': 'Both folder and training data must be specified'
            }), 400
        
        # Check if training file exists
        training_path = Path(TRAINING_DATA_FOLDER) / training_filename
        if not training_path.exists():
            return jsonify({
                'status': 'error',
                'message': f'Training file not found: {training_filename}'
            }), 404
        
        # Check if band folder exists - using DATA_RAW_DIR instead of hardcoded path
        band_folder = Path(DATA_RAW_DIR) / folder
        if not band_folder.exists():
            return jsonify({
                'status': 'error',
                'message': f'Band folder not found: {folder}'
            }), 404
        
        # Load training data
        with open(training_path, 'r') as f:
            training_data = json.load(f)
        
        # ACTUAL CLASSIFICATION LOGIC USING YOUR CLASSES:
        
        # 1. Load and stack band images using the classifier
        print(f"Loading bands from {band_folder}")
        image_data, profile = classifier.load_and_stack_bands(band_folder)
        
        # 2. Extract training samples from the loaded image and training points
        print("Extracting training data from image")
        X_train, y_train = classifier.extract_training_data(image_data, training_data, profile)
        
        # 3. Train the Random Forest classifier
        print(f"Training classifier with {len(X_train)} samples")
        trained_model = classifier.train(X_train, y_train)
        
        # 4. Classify the entire image
        print("Classifying entire image")
        classified_image = classifier.classify_image(image_data, profile)
        
        vegdir=os.makedirs(os.path.join(DATA_PROCESSED_DIR, folder), exist_ok=True)
        # 5. Save the classification result using utils
        output_filename = f"landcover_{folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        output_path = utils.save_classification_result(classified_image, profile, folder, output_filename)
        
        # 6. Optionally save the trained model
        model_filename = f"model_{folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = classifier.save_model(Path(MODAL_DIR) / model_filename)
        
        # 7. Generate classification statistics
        import numpy as np
        unique_classes, class_counts = np.unique(classified_image, return_counts=True)
        class_distribution = {
            f'class_{int(cls)}': int(count) 
            for cls, count in zip(unique_classes, class_counts)
        }
        
        # Calculate accuracy metrics if possible
        total_pixels = classified_image.size
        input_path=os.path.join(PROJECT_ROOT, 'data', 'raw',folder)
        out_path=os.path.join(PROJECT_ROOT, 'data', 'processed',folder)
        results = process_satellite_bands(input_path,out_path)

        return jsonify({
            'status': 'success',
            'output': output_filename,
            'model_saved': model_filename,
            'message': f'Classification completed successfully. Results saved to {output_filename}',
            'statistics': {
                'training_points_used': len(training_data.get('features', training_data.get('points', []))),
                'training_samples_extracted': len(X_train),
                'band_folder': folder,
                'image_dimensions': {
                    'height': image_data.shape[0],
                    'width': image_data.shape[1],
                    'bands': image_data.shape[2]
                },
                'total_pixels_classified': total_pixels,
                'class_distribution': class_distribution,
                'output_path': str(output_path),
                'model_path': str(model_path)
            }
        })
    
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'message': f"Required file not found: {str(e)}"
        }), 404
    
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f"Data validation error: {str(e)}"
        }), 400
    
    except Exception as e:
        print(f"Classification error: {str(e)}")
        import traceback
        traceback.print_exc()  # For debugging
        return jsonify({
            'status': 'error',
            'message': f"Classification failed: {str(e)}"
        }), 500

# Additional helper routes that might be useful
@app.route('/status')
def status():
    """Get API status and configuration"""
    return jsonify({
        'status': 'running',
        'config': {
            'band_names': gconfig.BAND_NAMES,
            'classes': config.CLASSES,
            'directories': {
                'training_data': str(TRAINING_DATA_FOLDER),
                'output': str(OUTPUT_FOLDER)
            }
        }
    })

# Add a separate route for file uploads
@app.route('/upload-usgs-bands', methods=['GET', 'POST'])
def usgs_band_uploadfiles():
    if request.method == 'POST':
        place = request.form.get("place", "").strip()
        b2 = request.files.get("b2")
        b3 = request.files.get("b3")
        b4 = request.files.get("b4")
        b5 = request.files.get("b5")
        b6 = request.files.get("b6")
        b7 = request.files.get("b7")
        b10 = request.files.get("b10")


        if not place or not all([b2, b3, b4, b5, b6, b7, b10]):
            return "Please enter a place name and upload all three files.", 400

        # Timestamped folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{place}_{timestamp}"
        save_path = os.path.join(DATA_RAW_DIR, folder_name)
        os.makedirs(save_path, exist_ok=True)

        # Save files
        b2.save(os.path.join(save_path, "B2.tif"))
        b3.save(os.path.join(save_path, "B3.tif"))
        b4.save(os.path.join(save_path, "B4.tif"))
        b5.save(os.path.join(save_path, "B5.tif"))
        b6.save(os.path.join(save_path, "B6.tif"))
        b7.save(os.path.join(save_path, "B7.tif"))  
        b10.save(os.path.join(save_path, "B10.tif"))

        """Process uploaded files and return status message"""
    try:
        # Ensure path exists
        if not Path(save_path).exists():
            return f"Error: Path not found - {save_path}"
        
        # Process files
        result = process_folder(
            folder_path=save_path,
            raster_target_crs="EPSG:4326",
            vector_target_crs="EPSG:4326",
            backup=True
        )
        
        if result['success']:
            return (
                f"Files processed successfully in <code>{save_path}</code><br>"
                f"Processed: {result['details']['processed']} files<br>"
                f"Rasters: {result['details']['rasters']}<br>"
                f"Vectors: {result['details']['vectors']}<br>"
                f"Errors: {len(result['details']['errors'])}"
            )
        else:
            return (
                f"Error processing files in <code>{save_path}</code><br>"
                f"Error: {result['message']}<br>"
                f"Partial results: {result['details']}"
            )
            
    except Exception as e:
        return f"Unexpected error processing files: {str(e)}"
        
        print(result)
        # Output example:
        # {
        #   'success': True,
        #   'message': 'Processed 3 rasters and 2 vectors',
        #   'details': {'rasters': 3, 'vectors': 2, 'errors': []}
        # }

    return render_template("usgs_band_upload.html")

#########################################################################################################################
class SatelliteDataProcessor:
    """Process satellite data for hotspot prediction"""
    
    def __init__(self, raw_folder='raw', processed_folder='processed'):
        self.output_folder = os.path.join(processed_folder, 'output')  
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
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
            if lst_max == lst_min:
                return np.zeros_like(lst_data)
            normalized = (lst_data - lst_min) / (lst_max - lst_min)
        elif method == 'zscore':
            # Z-score normalization
            lst_mean = np.nanmean(lst_data)
            lst_std = np.nanstd(lst_data)
            if lst_std == 0:
                return np.zeros_like(lst_data)
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
        try:
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
        except Exception as e:
            print(f"Error saving GeoTIFF: {e}")
    
    def preprocess_to_csv(self, data_dict, output_path):
        """
        Convert processed data to CSV format
        
        Args:
            data_dict: Dictionary containing processed data
            output_path: Output CSV file path
        """
        try:
            df = pd.DataFrame(data_dict)
            df.to_csv(output_path, index=False)
            print(f"CSV saved to: {output_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
        
    def generate_sample_data(self, size=(100, 100), bounds=(-74.25, 40.5, -73.75, 41.0)):
        """Generate sample satellite data for demonstration"""
        
        # Generate sample NDVI data (0-1 range)
        np.random.seed(42)  # For reproducible results
        ndvi = np.random.beta(2, 2, size)
        
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

    def load_band_from_folder(self, folder_path, band_name):
        """
        Loads a .tif file matching band_name from the given folder.
        Returns the array and raster profile.
        """
        folder = Path(folder_path)
        tif_files = list(folder.glob("*.tif")) + list(folder.glob("*.TIF"))

        for tif in tif_files:
            if band_name.lower() in tif.name.lower():
                with rasterio.open(tif) as src:
                    arr = src.read(1)
                    profile = src.profile
                return arr, profile

        raise FileNotFoundError(f"No file with '{band_name}' found in {folder_path}")

    def diagnose_data_bounds(self):
        """Diagnose and return the geographic bounds of your data"""
        try:
            ndvi, profile = self.load_band_from_folder(self.raw_folder, "ndvi")
            print(f"Profile keys: {profile.keys()}")
            print(f"CRS: {profile.get('crs')}")
            print(f"Transform: {profile.get('transform')}")

            if 'transform' in profile:
                transform = profile['transform']
                height, width = ndvi.shape
                #bounds = array_bounds(height, width, transform)
                bounds = rasterio.transform.array_bounds(height, width, transform)
                print(f"Calculated bounds: {bounds}")
                print(f"Longitude range: {bounds[0]:.4f} to {bounds[2]:.4f}")
                print(f"Latitude range: {bounds[1]:.4f} to {bounds[3]:.4f}")
                return bounds
            else:
                print("No transform found in profile")
                return None

        except Exception as e:
            print(f"Error: {e}")
            return None

    def generate_actual_data(self):
        """Load actual NDVI, LST, and Urban bands with coordinates."""
        # Diagnose and get bounds
        bounds = self.diagnose_data_bounds()
        if bounds is None:
            raise RuntimeError("Failed to diagnose data bounds.")

        # Load bands
        ndvi, _ = self.load_band_from_folder(self.raw_folder, "ndvi")
        lst, _ = self.load_band_from_folder(self.raw_folder, "lst")
        urban, _ = self.load_band_from_folder(self.raw_folder, "urban")

        # Create coordinate grids
        size = ndvi.shape
        lats = np.linspace(bounds[1], bounds[3], size[0])
        lons = np.linspace(bounds[0], bounds[2], size[1])
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        print("Loaded NDVI data shape:", ndvi.shape)

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
        
        try:
            # Prepare features
            features, valid_mask = self.prepare_features(ndvi, lst, urban)
            
            if len(features) == 0:
                raise ValueError("No valid features found for training")
            
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
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
    
    def predict_hotspots(self, ndvi, lst, urban):
        """Predict hotspot risk scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        try:
            # Prepare features
            features, valid_mask = self.prepare_features(ndvi, lst, urban)
            
            if len(features) == 0:
                raise ValueError("No valid features found for prediction")
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            predictions = self.model.predict(features_scaled)
            
            # Reshape predictions back to original shape
            risk_map = np.full(ndvi.shape, np.nan)
            risk_map.flat[valid_mask] = predictions
            
            return risk_map
            
        except Exception as e:
            print(f"Error predicting hotspots: {e}")
            raise

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
                    'lat': float(lat),
                    'lon': float(lon),
                    'risk_score': float(risk_score),
                    'reason': 'Low vegetation in high-risk area'
                })
            
            if urban > 0.6:  # High urban density
                recommendations.append({
                    **self.recommendations['cool_pavements'],
                    'lat': float(lat),
                    'lon': float(lon),
                    'risk_score': float(risk_score),
                    'reason': 'High urban density contributing to heat'
                })
                
                recommendations.append({
                    **self.recommendations['green_roofs'],
                    'lat': float(lat),
                    'lon': float(lon),
                    'risk_score': float(risk_score),
                    'reason': 'Dense urban area needs cooling'
                })
        
        elif risk_score > 0.5:  # Medium risk
            if ndvi < 0.5:
                recommendations.append({
                    **self.recommendations['tree_cover'],
                    'lat': float(lat),
                    'lon': float(lon),
                    'risk_score': float(risk_score),
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

# Global variables to store processed data
processed_data = {}
hotspot_model = None
mitigation_system = None

@app.route('/api/process_data', methods=['POST'])
def process_data():
    """Process satellite data and train model"""
    global processed_data, hotspot_model, mitigation_system
    data = request.get_json()
    # Check if data was received
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'No data provided'
        }), 400
    print(data)
    # Get the 'folder' parameter
    pfolder = data.get('folder')
    if not pfolder:
        return jsonify({
            'status': 'error',
            'message': 'Missing "folder" parameter'
        }), 400    

    try:
        # Initialize processors
        
        raw_folder = os.path.join(PROJECT_ROOT, 'data', 'raw', pfolder)
        processed_folder = os.path.join(PROJECT_ROOT, 'data', 'processed',pfolder)
        processor = SatelliteDataProcessor(processed_folder,processed_folder)
        hotspot_model = HotspotPredictor()
        mitigation_system = MitigationRecommendationSystem()
        dbound=processor.diagnose_data_bounds()
        print(f"Data bounds: {dbound}" if dbound else "No bounds found")
        # Generate sample data (replace with actual data loading)
        #data = processor.generate_sample_data()
        
        data = processor.generate_actual_data()
        
        # Normalize temperature
        normalized_lst = processor.normalize_temperature(data['lst'])
        
        # Save processed data
        processor.preprocess_to_geotiff(
            data['ndvi'], 
            os.path.join(processor.output_folder, 'ndvi.tif'), 
            data['bounds']
        )
        processor.preprocess_to_geotiff(
            normalized_lst, 
            os.path.join(processor.output_folder, 'lst_normalized.tif'), 
            data['bounds']
        )
        processor.preprocess_to_geotiff(
            data['urban'], 
            os.path.join(processor.output_folder, 'urban.tif'), 
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
        
        # Convert numpy arrays to lists for JSON serialization
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
        print(f"Error in process_data: {e}")
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
#################################################################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)