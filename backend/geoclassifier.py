# geoclassifier.py - ENHANCED VERSION WITH DEBUGGING
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.crs import CRS
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
import gc  # Garbage Collector
from pathlib import Path
           
# Configuration Class
class GeoClassifierConfig:
    """Configuration settings for the geoclassifier"""
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        # Folder configurations
        self.PROJECT_FOLDER = Path(__file__).parent.parent
        
        # Data directories
        self.DATA_DIR = self.PROJECT_FOLDER / 'data'
        self.UPLOAD_FOLDER = self.DATA_DIR / 'uploads'
        self.TRAINING_DATA_FOLDER = self.BASE_DIR / 'training_data'
        self.OUTPUT_FOLDER = self.DATA_DIR / 'output'
        self.DATA_RAW_DIR = self.DATA_DIR / 'raw'
        self.MODAL_DIR = self.PROJECT_FOLDER / 'modals'
        self.dband_folder = None
        self.DATA_PROCESSED_DIR = self.PROJECT_FOLDER / "data/processed"
        # FIXED: More flexible band configuration
        # Define possible band names for different satellite sensors
        self.LANDSAT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
        self.SENTINEL_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        # Default bands to look for (will be auto-detected)
        self.BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        
        self.CLASSES = {
            'Urban': {'value': 0, 'color': 'red'},
            'Vegetation': {'value': 1, 'color': 'green'},
            'Water': {'value': 2, 'color': 'blue'}
        }
        
        # Create directories if they don't exist
        for folder in [self.UPLOAD_FOLDER, self.TRAINING_DATA_FOLDER, self.OUTPUT_FOLDER, 
                      self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

# Classifier Class
class LandUseClassifier:
    """Handles land use classification using Random Forest"""
    def __init__(self, config):
        self.config = config
        #self.model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.detected_bands = []
        self.image_bounds = None  # Store actual image bounds
        self.image_crs = None     # Store image CRS
    
    def detect_available_bands(self, band_folder):
        """FIXED: Auto-detect available band files in the folder"""
        band_folder = Path(band_folder)
        available_bands = []
        
        print(f"Scanning folder: {band_folder}")
        
        # Check for .tif files in the folder
        tif_files = [f for f in band_folder.glob("*.tif") if "b10" not in f.name.lower()]
        tif_files += [f for f in band_folder.glob("*.TIF") if "b10" not in f.name.lower()]

        print(f"Found TIF files: {[f.name for f in tif_files]}")
        
        # Extract band names from filenames
        for tif_file in tif_files:
            filename = tif_file.stem  # Get filename without extension
            
            # Check various band naming patterns
            for possible_band in self.config.LANDSAT_BANDS + self.config.SENTINEL_BANDS:
                if possible_band in filename.upper():
                    available_bands.append(possible_band)
                    print(f"Detected band: {possible_band} from file: {tif_file.name}")
                    break
            else:
                # If no standard band name found, use the filename as band name
                available_bands.append(filename)
                print(f"Using filename as band name: {filename}")
        
        if not available_bands:
            raise FileNotFoundError(f"No band files found in {band_folder}")
        
        self.detected_bands = available_bands
        return available_bands
    
    def get_image_info(self, band_folder):
        """Get detailed information about the first band file to understand the image bounds"""
        band_folder = Path(band_folder)
        tif_files = list(band_folder.glob("*.tif")) + list(band_folder.glob("*.TIF"))
        
        if not tif_files:
            raise FileNotFoundError(f"No TIF files found in {band_folder}")
        
        # Use the first TIF file to get image info
        first_band = tif_files[0]
        with rasterio.open(first_band) as src:
            bounds = src.bounds
            crs = src.crs
            transform_matrix = src.transform
            shape = (src.height, src.width)
            
            # ADDED: Check for NoData values
            nodata = src.nodata
            
            print(f"Image Information:")
            print(f"  File: {first_band.name}")
            print(f"  CRS: {crs}")
            print(f"  Bounds: {bounds}")
            print(f"  Shape: {shape}")
            print(f"  Transform: {transform_matrix}")
            print(f"  NoData value: {nodata}")
            
            # Store for later use
            self.image_bounds = bounds
            self.image_crs = crs
            
            return {
                'bounds': bounds,
                'crs': crs,
                'transform': transform_matrix,
                'shape': shape,
                'nodata': nodata
            }
    
    def load_and_stack_bands(self, band_folder):
        """ENHANCED: Load and stack all available band files with better data validation"""
        band_folder = Path(band_folder)
        self.dband_folder = band_folder
        if not band_folder.exists():
            raise FileNotFoundError(f"Band folder does not exist: {band_folder}")
        
        # Get image info first
        image_info = self.get_image_info(band_folder)
        
        # Auto-detect available bands
        available_bands = self.detect_available_bands(band_folder)
        
        bands = []
        profile = None
        successfully_loaded = []
        
        # Try to load each detected band
        for band_name in available_bands:
            # Try different possible filenames
            possible_files = [
                band_folder / f'{band_name}.tif',
                band_folder / f'{band_name}.TIF',
                band_folder / f'{band_name.upper()}.tif',
                band_folder / f'{band_name.lower()}.tif'
            ]
            
            # Also try filenames that contain the band name
            for tif_file in band_folder.glob("*.tif"):
                if band_name.upper() in tif_file.name.upper():
                    possible_files.append(tif_file)
            
            band_loaded = False
            for band_path in possible_files:
                if band_path.exists():
                    try:
                        print(f"Loading band file: {band_path}")
                        with rasterio.open(band_path) as src:
                            band_data = src.read(1)
                            
                            # ENHANCED: Better data validation
                            print(f"Band {band_name} statistics:")
                            print(f"  Shape: {band_data.shape}")
                            print(f"  Data type: {band_data.dtype}")
                            print(f"  Min: {np.min(band_data)}")
                            print(f"  Max: {np.max(band_data)}")
                            print(f"  Mean: {np.mean(band_data)}")
                            print(f"  NoData value: {src.nodata}")
                            
                            # Handle NoData values
                            if src.nodata is not None:
                                band_data = np.where(band_data == src.nodata, np.nan, band_data)
                                print(f"  Converted NoData ({src.nodata}) to NaN")
                            
                            # Check for valid data
                            valid_pixels = np.isfinite(band_data)
                            valid_count = np.sum(valid_pixels)
                            total_pixels = band_data.size
                            
                            print(f"  Valid pixels: {valid_count}/{total_pixels} ({100*valid_count/total_pixels:.1f}%)")
                            
                            if valid_count == 0:
                                print(f"Warning: No valid pixels in band {band_name}")
                                continue
                                
                            bands.append(band_data)
                            successfully_loaded.append(band_name)
                            
                            if profile is None:
                                profile = src.profile.copy()
                            
                            print(f"Successfully loaded band: {band_name} from {band_path}")
                            band_loaded = True
                            break
                    except Exception as e:
                        print(f"Error loading {band_path}: {e}")
                        continue
            
            if not band_loaded:
                print(f"Warning: Could not load band {band_name}")
        
        if not bands:
            raise FileNotFoundError(f"No valid band files could be loaded from {band_folder}")
        
        print(f"Successfully loaded {len(bands)} bands: {successfully_loaded}")
        
        # Stack bands into a single array
        stacked_bands = np.stack(bands, axis=-1)
        print(f"Stacked bands shape: {stacked_bands.shape}")
        
        # ADDED: Overall data statistics
        print(f"Stacked data statistics:")
        print(f"  Overall shape: {stacked_bands.shape}")
        print(f"  Data type: {stacked_bands.dtype}")
        valid_mask = np.isfinite(stacked_bands).all(axis=-1)
        print(f"  Pixels with all valid bands: {np.sum(valid_mask)}/{valid_mask.size}")
        
        return stacked_bands, profile
    
    def check_point_bounds(self, lon, lat):
        """Check if a point is within the image bounds"""
        if self.image_bounds is None:
            return True  # Can't check without bounds info
        
        minx, miny, maxx, maxy = self.image_bounds
        return minx <= lon <= maxx and miny <= lat <= maxy
    
    def encode_labels(self, labels):
        """Convert string labels to integer codes"""
        # Create label mapping
        if not hasattr(self, 'label_encoder'):
            self.label_encoder = {}
            self.label_decoder = {}
        
        # Handle different label formats
        encoded_labels = []
        unique_labels = set()
        
        for label in labels:
            # Convert to string for consistency
            label_str = str(label).strip().lower()
            unique_labels.add(label_str)
        
        print(f"Unique labels found: {unique_labels}")
        
        # Create mapping based on config or auto-assign
        for i, label_str in enumerate(sorted(unique_labels)):
            if label_str not in self.label_encoder:
                # Try to match with config classes
                matched = False
                for class_name, class_info in self.config.CLASSES.items():
                    if label_str in class_name.lower() or class_name.lower() in label_str:
                        self.label_encoder[label_str] = class_info['value']
                        self.label_decoder[class_info['value']] = class_name
                        matched = True
                        break
                
                if not matched:
                    # Auto-assign integer code
                    self.label_encoder[label_str] = i
                    self.label_decoder[i] = label_str
        
        print(f"Label encoding mapping: {self.label_encoder}")
        
        # Encode all labels
        for label in labels:
            label_str = str(label).strip().lower()
            encoded_labels.append(self.label_encoder[label_str])
        
        return np.array(encoded_labels, dtype=np.int64)

    def extract_training_data(self, image_data, training_points, profile):
        """ENHANCED: Extract pixel values with proper label encoding"""
        transform_matrix = profile['transform']
        samples = []
        labels = []
        
        # Handle both GeoJSON format and simple points format
        points = training_points.get('features', training_points.get('points', []))
        
        print(f"Processing {len(points)} training points")
        print(f"Image bounds: {self.image_bounds}")
        print(f"Transform matrix: {transform_matrix}")
        
        valid_points = 0
        out_of_bounds_points = 0
        invalid_pixel_points = 0
        
        for i, point in enumerate(points):
            try:
                # Handle GeoJSON format
                if 'geometry' in point and 'coordinates' in point['geometry']:
                    lon, lat = point['geometry']['coordinates']
                    class_value = point['properties'].get('class_value', point['properties'].get('class', point['properties'].get('label')))
                # Handle simple format
                elif 'coordinates' in point:
                    lon, lat = point['coordinates']
                    class_value = point.get('class_value', point.get('class', point.get('label')))
                else:
                    print(f"Warning: Invalid point format at index {i}: {point}")
                    continue
                
                if class_value is None:
                    print(f"Warning: No class value found for point {i}: {point}")
                    continue
                
                print(f"Point {i}: lon={lon}, lat={lat}, class={class_value}")
                
                # Check if point is within image bounds
                if not self.check_point_bounds(lon, lat):
                    print(f"Warning: Point {i} is outside image geographic bounds")
                    out_of_bounds_points += 1
                    continue
                
                # Convert geographic coordinates to pixel coordinates
                try:
                    row, col = rasterio.transform.rowcol(transform_matrix, lon, lat)
                    print(f"Point {i}: Converted to pixel row={row}, col={col}")
                except Exception as e:
                    print(f"Error converting coordinates for point {i}: {e}")
                    continue
                
                # Check pixel bounds
                if 0 <= row < image_data.shape[0] and 0 <= col < image_data.shape[1]:
                    pixel_values = image_data[row, col, :]
                    
                    # ENHANCED: Better pixel value validation
                    print(f"Point {i}: Raw pixel values: {pixel_values}")
                    
                    # Check for valid pixel values (not NaN or NoData)
                    if np.isfinite(pixel_values).all() and not np.all(pixel_values == 0):
                        samples.append(pixel_values)
                        labels.append(class_value)
                        valid_points += 1
                        print(f"Point {i}: VALID - Added to training set")
                    else:
                        print(f"Warning: Invalid pixel values at point {i}: {pixel_values}")
                        invalid_pixel_points += 1
                else:
                    print(f"Warning: Point {i} pixel coordinates outside image: row={row}, col={col}, image_shape={image_data.shape[:2]}")
                    out_of_bounds_points += 1
                    
            except (KeyError, IndexError, ValueError) as e:
                print(f"Warning: Skipping invalid training point {i}: {point} - {e}")
                continue
        
        print(f"Training point summary:")
        print(f"  Valid points: {valid_points}")
        print(f"  Out of bounds: {out_of_bounds_points}")
        print(f"  Invalid pixels: {invalid_pixel_points}")
        print(f"  Total processed: {len(points)}")
        
        if not samples:
            raise ValueError("No valid training points found. Check coordinate system and image bounds.")
        
        # Convert samples to numpy array
        X = np.array(samples)
        
        # FIXED: Encode labels properly
        y = self.encode_labels(labels)
        
        print(f"Training data statistics:")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Feature ranges: min={np.min(X, axis=0)}, max={np.max(X, axis=0)}")
        print(f"  Raw labels: {labels[:5]}...")  # Show first 5 raw labels
        print(f"  Encoded labels: {y[:5]}...")    # Show first 5 encoded labels
        print(f"  Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def train(self, X, y):
        """Train the classifier with enhanced validation"""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        print(f"Training model with {len(X)} samples and {X.shape[1]} features")
        
        # ADDED: Check for data quality issues
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("WARNING: Training data contains NaN or Inf values!")
            # Remove problematic samples
            valid_mask = np.isfinite(X).all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"After cleaning: {len(X)} samples remain")
        
        self.model.fit(X, y)
        
        # Print training statistics
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Training class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # ADDED: Model validation
        train_score = self.model.score(X, y)
        print(f"Training accuracy: {train_score:.3f}")
        
        return self.model

    def classify_image(self, image_data, profile):
        """ENHANCED: Classify with comprehensive debugging and data validation"""
        if image_data.size == 0:
            raise ValueError("Image data cannot be empty")
        
        print(f"Classifying image of shape: {image_data.shape}")
        
        height, width, num_bands = image_data.shape
        pixels = image_data.reshape(-1, num_bands).astype(np.float32)
        
        # ENHANCED: Better data validation and debugging
        print(f"Pixel data statistics:")
        print(f"  Total pixels: {pixels.shape[0]}")
        print(f"  Bands: {num_bands}")
        print(f"  Data type: {pixels.dtype}")
        print(f"  Min values per band: {np.min(pixels, axis=0)}")
        print(f"  Max values per band: {np.max(pixels, axis=0)}")
        print(f"  Mean values per band: {np.mean(pixels, axis=0)}")
        print(f"  Std values per band: {np.std(pixels, axis=0)}")
        
        # Compare with training data statistics
        if hasattr(self, 'training_stats'):
            print(f"\nComparing with training data:")
            train_mins = np.min(self.training_stats['feature_means'] - 2*self.training_stats['feature_stds'])
            train_maxs = np.max(self.training_stats['feature_means'] + 2*self.training_stats['feature_stds'])
            image_mins = np.min(pixels, axis=0)
            image_maxs = np.max(pixels, axis=0)
            
            print(f"  Training range (approx): {train_mins:.2f} to {train_maxs:.2f}")
            print(f"  Image range: {image_mins} to {image_maxs}")
            
            # Check if image data is in similar range to training data
            for i in range(num_bands):
                if image_maxs[i] < train_mins or image_mins[i] > train_maxs:
                    print(f"  WARNING: Band {i} values may be outside training range!")
        
        # Create comprehensive validity mask
        finite_mask = np.isfinite(pixels).all(axis=1)
        nonzero_mask = ~np.all(pixels == 0, axis=1)
        valid_mask = finite_mask & nonzero_mask
        
        print(f"  Finite pixels: {np.sum(finite_mask)}")
        print(f"  Non-zero pixels: {np.sum(nonzero_mask)}")
        print(f"  Valid pixels: {np.sum(valid_mask)}")
        
        # Initialize classification result
        classified = np.zeros(pixels.shape[0], dtype=np.uint8)
        
        if valid_mask.any():
            valid_pixels = pixels[valid_mask]
            print(f"Processing {len(valid_pixels)} valid pixels")
            
            # CRITICAL: Test predictions on a small sample first
            test_sample_size = min(100, len(valid_pixels))
            test_pixels = valid_pixels[:test_sample_size]
            
            print(f"\nTesting prediction on sample of {test_sample_size} pixels...")
            print(f"Sample pixel statistics:")
            print(f"  Min: {np.min(test_pixels, axis=0)}")
            print(f"  Max: {np.max(test_pixels, axis=0)}")
            print(f"  Mean: {np.mean(test_pixels, axis=0)}")
            
            try:
                test_predictions = self.model.predict(test_pixels)
                test_probabilities = self.model.predict_proba(test_pixels)
                
                print(f"Sample predictions: {test_predictions[:10]}")
                print(f"Sample prediction distribution: {np.bincount(test_predictions)}")
                print(f"Sample prediction probabilities (first 3):")
                for i in range(min(3, len(test_probabilities))):
                    print(f"  Pixel {i}: {test_probabilities[i]}")
                
                # Check if all predictions are the same
                unique_preds = np.unique(test_predictions)
                if len(unique_preds) == 1:
                    print(f"CRITICAL WARNING: All sample predictions are the same class: {unique_preds[0]}")
                    print("This suggests a problem with the model or data preprocessing!")
                    
                    # Additional debugging
                    print(f"Model classes: {self.model.classes_}")
                    print(f"Model n_classes: {self.model.n_classes_}")
                    
                    # Check decision function if available
                    if hasattr(self.model, 'decision_function'):
                        decision_scores = self.model.decision_function(test_pixels[:3])
                        print(f"Decision scores for first 3 pixels: {decision_scores}")
                
            except Exception as e:
                print(f"ERROR during sample prediction: {e}")
                return np.zeros((height, width), dtype=np.uint8)
            
            # ADDED: Batch processing for large datasets with progress
            batch_size = 10000
            total_batches = (len(valid_pixels) - 1) // batch_size + 1
            
            all_predictions = []
            
            for i in range(0, len(valid_pixels), batch_size):
                batch = valid_pixels[i:i + batch_size]
                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pixels)")
                
                try:
                    # Clear memory if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process batch
                    batch_predictions = self.model.predict(batch)
                    all_predictions.extend(batch_predictions)
                    
                    # ADDED: Check batch prediction statistics
                    unique_preds, pred_counts = np.unique(batch_predictions, return_counts=True)
                    print(f"  Batch {batch_num} predictions: {dict(zip(unique_preds, pred_counts))}")
                    
                    # Manual garbage collection
                    del batch, batch_predictions
                    gc.collect()
                    
                except Exception as e:
                    print(f"Failed on batch {batch_num}: {str(e)}")
                    # Fill with a default class instead of leaving as 0
                    all_predictions.extend([0] * len(batch))
                    continue
            
            # Assign predictions to valid pixels
            classified[valid_mask] = np.array(all_predictions, dtype=np.uint8)
            
        else:
            print("WARNING: No valid pixels found for classification!")
        
        result = classified.reshape(height, width)
        
        # ADDED: Final result statistics with label decoding
        unique_results, result_counts = np.unique(result, return_counts=True)
        print(f"Final classification statistics:")
        for class_val, count in zip(unique_results, result_counts):
            percentage = 100 * count / result.size
            class_name = self.label_decoder.get(class_val, f"Class_{class_val}")
            print(f"  {class_name} (value {class_val}): {count} pixels ({percentage:.1f}%)")
        
        print(f"Classification completed. Result shape: {result.shape}")
        
        return result
    
    def save_model(self, output_path):
        """Save the trained classifier with validation"""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been trained yet")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both the model and metadata
        model_data = {
            'model': self.model,
            'detected_bands': self.detected_bands,
            'image_bounds': self.image_bounds,
            'image_crs': str(self.image_crs) if self.image_crs else None,
            'config': {
                'classes': self.config.CLASSES,
                'n_estimators': self.model.n_estimators
            }
        }
        
        joblib.dump(model_data, output_path)
        print(f"Model saved to: {output_path}")
        return output_path

# Utility Class
class GeoClassifierUtils:
    """Utility functions for geoclassification"""
    
    def __init__(self, config):
        self.config = config
        #self.DATA_PROCESSED_DIR = "data/processed"  # Set your directory here
    
    def save_classification_result(self, classified_image, profile, ofolder, output_filename='classification.tif'):
        """ENHANCED: Save classification result with better validation and error handling"""
        
        # Validate input data
        if classified_image.size == 0:
            raise ValueError("Classification result cannot be empty")
        
        # Construct full file path (not just directory)
        output_dir = self.config.DATA_PROCESSED_DIR / ofolder
        output_path = output_dir / output_filename  # This should be the full file path
        
        # Create directory with proper permissions
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Ensure directory has write permissions
            os.chmod(output_dir, 0o755)
        except PermissionError as e:
            print(f"Permission error creating directory {output_dir}: {e}")
            raise
        
        print(f"Output directory: {output_dir}")
        print(f"Full output path: {output_path}")
        
        # ADDED: Verify classification data before saving
        print(f"Saving classification result:")
        print(f"  Image shape: {classified_image.shape}")
        print(f"  Data type: {classified_image.dtype}")
        print(f"  Value range: {np.min(classified_image)} to {np.max(classified_image)}")
        
        # Update profile for single-band classification output
        profile_copy = profile.copy()  # Don't modify original profile
        profile_copy.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': None,
            'compress': 'lzw'
        })
        
        try:
            # Ensure classified_image is 2D for single band output
            if len(classified_image.shape) == 3:
                classified_image = classified_image.squeeze()
            
            # Save the classification result
            with rasterio.open(output_path, 'w', **profile_copy) as dst:
                dst.write(classified_image.astype('uint8'), 1)
            
            print(f"Classification result successfully saved to: {output_path}")
            
            # ADDED: Verify the saved file
            with rasterio.open(output_path) as src:
                saved_data = src.read(1)
                print(f"Verification - saved file statistics:")
                print(f"  Shape: {saved_data.shape}")
                print(f"  Value range: {np.min(saved_data)} to {np.max(saved_data)}")
                unique_vals, counts = np.unique(saved_data, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    print(f"  Value {val}: {count} pixels")
                    
        except PermissionError as e:
            print(f"Permission error writing file {output_path}: {e}")
            print("Suggestions:")
            print("1. Check if the directory exists and has write permissions")
            print("2. Ensure the file is not currently open in another application")
            print("3. Run the script with appropriate permissions")
            raise
        except Exception as e:
            print(f"Error saving classification result: {e}")
            raise
        
        return output_path
    
    def check_directory_permissions(self, path):
        """Helper method to check directory permissions"""
        path = Path(path)
        
        if not path.exists():
            print(f"Path does not exist: {path}")
            return False
            
        if not path.is_dir():
            print(f"Path is not a directory: {path}")
            return False
            
        if not os.access(path, os.W_OK):
            print(f"No write permission for directory: {path}")
            return False
            
        print(f"Directory permissions OK: {path}")
        return True
    
    def create_safe_output_path(self, base_dir, folder_name, filename):
        """Create a safe output path with proper validation"""
        
        # Sanitize folder name (remove invalid characters)
        safe_folder = "".join(c for c in folder_name if c.isalnum() or c in ('-', '_'))
        
        output_dir = Path(base_dir) / safe_folder
        output_path = output_dir / filename
        
        # Create directory structure
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(output_dir, 0o755)
        except Exception as e:
            print(f"Error creating directory structure: {e}")
            raise
            
        return output_path
# Initialize Flask app and components (rest of the Flask code remains the same)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

config = GeoClassifierConfig()
classifier = LandUseClassifier(config)
utils = GeoClassifierUtils(config)

# ... (rest of the Flask endpoints remain the same)

if __name__ == '__main__':
    print("Starting Enhanced GeoClassifier API with debugging...")
    print(f"Configuration:")
    print(f"  Training data folder: {config.TRAINING_DATA_FOLDER}")
    print(f"  Raw data folder: {config.DATA_RAW_DIR}")
    print(f"  Processed data folder: {config.DATA_PROCESSED_DIR}")
    print(f"  Output folder: {config.OUTPUT_FOLDER}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)