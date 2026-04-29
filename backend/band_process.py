import os
import numpy as np
import rasterio
from pathlib import Path
import glob

def process_satellite_bands(input_path, output_path):
    """
    Single function to process satellite bands and create NDVI and LST files
    
    Args:
        input_path (str): Path to folder containing band files
        output_path (str): Path where processed files will be saved
    
    Returns:
        dict: Results with success status and file paths
    """
    
    def _identify_bands(folder_path):
        """Internal function to identify band files"""
        
        print(f"🔍 Scanning for band files in: {folder_path}")
        
        # Look for common band file patterns
        band_patterns = [
            "*.tif", "*.TIF", "*.tiff", "*.TIFF",
            "*B*.tif", "*B*.TIF", "*band*.tif", "*band*.TIF"
        ]
        
        all_files = []
        for pattern in band_patterns:
            files = list(Path(folder_path).glob(pattern))
            all_files.extend(files)
        
        # Remove duplicates
        all_files = list(set(all_files))
        
        print(f"📁 Found {len(all_files)} potential band files")
        
        band_files = {}
        
        # Identify bands by filename patterns
        for file_path in all_files:
            filename = file_path.name.upper()
            
            # Common band identification patterns
            if any(pattern in filename for pattern in ['B2', 'BAND2', '_02_', 'RED']):
                band_files['red'] = file_path
                print(f"🔴 RED band: {file_path.name}")
            elif any(pattern in filename for pattern in ['B3', 'BAND3', '_03_', 'GREEN']):
                band_files['green'] = file_path
                print(f"🟢 GREEN band: {file_path.name}")
            elif any(pattern in filename for pattern in ['B4', 'BAND4', '_04_', 'NIR', 'NEAR']):
                band_files['nir'] = file_path
                print(f"🌿 NIR band: {file_path.name}")
            elif any(pattern in filename for pattern in ['B5', 'BAND5', '_05_', 'SWIR1']):
                band_files['swir1'] = file_path
                print(f"📡 SWIR1 band: {file_path.name}")
            elif any(pattern in filename for pattern in ['B10', 'BAND10', '_10_', 'TIR', 'THERMAL']):
                band_files['thermal'] = file_path
                print(f"🌡️ THERMAL band: {file_path.name}")
            elif any(pattern in filename for pattern in ['B11', 'BAND11', '_11_', 'TIR2']):
                band_files['thermal2'] = file_path
                print(f"🌡️ THERMAL2 band: {file_path.name}")
        
        return band_files
    
    def _load_band_data(band_files):
        """Internal function to load band data"""
        
        print(f"📊 Loading band data...")
        
        band_data = {}
        profile = None
        
        for band_name, file_path in band_files.items():
            try:
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    band_data[band_name] = data
                    if profile is None:
                        profile = src.profile
                    print(f"✅ Loaded {band_name}: {data.shape}")
            except Exception as e:
                print(f"❌ Error loading {band_name}: {e}")
                continue
        
        return band_data, profile
    
    def _calculate_ndvi(red_band, nir_band):
        """Internal function to calculate NDVI"""
        
        print(f"🧮 Calculating NDVI...")
        
        # Convert to float to avoid overflow
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Handle division by zero
        denominator = nir + red
        valid_mask = denominator != 0
        
        # Initialize NDVI array with nodata value
        ndvi = np.full_like(red, -9999, dtype=np.float32)
        
        # Calculate NDVI for valid pixels
        ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
        
        # Clip to valid range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        # Set invalid pixels back to nodata
        ndvi[~valid_mask] = -9999
        
        print(f"✅ NDVI calculated: {np.sum(valid_mask)} valid pixels")
        print(f"   Range: {np.min(ndvi[valid_mask]):.3f} to {np.max(ndvi[valid_mask]):.3f}")
        
        return ndvi
    
    def _calculate_lst(thermal_band, ndvi=None):
        """Internal function to calculate Land Surface Temperature"""
        
        print(f"🌡️ Calculating LST...")
        
        thermal = thermal_band.astype(np.float32)
        
        # Simple method: Convert DN to temperature (adjust for your sensor)
        # This is a generic conversion - modify based on your satellite sensor
        lst = thermal * 0.1 - 273.15  # Convert from Kelvin to Celsius
        
        # Optional: Apply emissivity correction using NDVI
        if ndvi is not None:
            print("   Applying emissivity correction...")
            emissivity = np.full_like(ndvi, 0.95, dtype=np.float32)
            vegetation_mask = ndvi > 0.2
            soil_mask = (ndvi <= 0.2) & (ndvi != -9999)
            
            emissivity[vegetation_mask] = 0.99
            emissivity[soil_mask] = 0.92
            
            # Apply correction where valid
            valid_mask = ndvi != -9999
            lst[valid_mask] = lst[valid_mask] / emissivity[valid_mask]
        
        print(f"✅ LST calculated")
        print(f"   Temperature range: {np.min(lst):.1f}°C to {np.max(lst):.1f}°C")
        
        return lst
    
    def _save_raster(data, profile, output_file, nodata_value=-9999):
        """Internal function to save raster data"""
        
        print(f"💾 Saving: {output_file}")
        
        # Create output directory if it doesn't exist
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update profile for output
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': nodata_value,
            'compress': 'lzw'
        })
        
        try:
            with rasterio.open(str(output_file), 'w', **output_profile) as dst:
                if len(data.shape) == 3:
                    dst.write(data[0])
                else:
                    dst.write(data, 1)
                
            
            # Verify file was created
            if output_file.exists() and output_file.stat().st_size > 0:
                file_size = output_file.stat().st_size / 1024 / 1024  # MB
                print(f"✅ Saved successfully: {output_file}")
                print(f"   File size: {file_size:.2f} MB")
                return str(output_file)
            else:
                raise RuntimeError(f"File was not created: {output_file}")
                
        except Exception as e:
            print(f"❌ Error saving {output_file}: {e}")
            raise
    
    # Main processing logic starts here
    print("🛰️ SATELLITE BAND PROCESSOR")
    print("=" * 50)
    
    try:
        # Validate input path
        input_path = Path(input_path)
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input path does not exist: {input_path}',
                'results': {}
            }
        
        if not input_path.is_dir():
            return {
                'success': False,
                'error': f'Input path is not a directory: {input_path}',
                'results': {}
            }
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Input folder: {input_path}")
        print(f"📁 Output folder: {output_path}")
        
        # Step 1: Identify band files
        band_files = _identify_bands(input_path)
        
        if not band_files:
            return {
                'success': False,
                'error': 'No band files found or identified',
                'results': {}
            }
        
        # Step 2: Load band data
        band_data, profile = _load_band_data(band_files)
        
        if not band_data or profile is None:
            return {
                'success': False,
                'error': 'Failed to load band data',
                'results': {}
            }
        
        results = {}
        
        # Step 3: Calculate NDVI
        if 'red' in band_data and 'nir' in band_data:
            print(f"\n🌱 PROCESSING NDVI")
            print("-" * 30)
            
            ndvi = _calculate_ndvi(band_data['red'], band_data['nir'])
            ndvi_file = output_path / 'ndvi.tif'
            saved_ndvi = _save_raster(ndvi, profile, ndvi_file, nodata_value=-9999)
            results['ndvi'] = saved_ndvi
            
        else:
            print(f"\n⚠️ Cannot calculate NDVI - missing red or NIR band")
            ndvi = None
        
        # Step 4: Calculate LST
        if 'thermal' in band_data:
            print(f"\n🌡️ PROCESSING LST")
            print("-" * 30)
            
            lst = _calculate_lst(band_data['thermal'], ndvi)
            lst_file = output_path / 'lst.tif'
            saved_lst = _save_raster(lst, profile, lst_file, nodata_value=-9999)
            results['lst'] = saved_lst
            
        else:
            print(f"\n⚠️ Cannot calculate LST - missing thermal band")
        
        # Summary
        print(f"\n🎉 PROCESSING COMPLETE!")
        print("=" * 50)
        
        if results:
            print("📋 Generated files:")
            for file_type, file_path in results.items():
                print(f"   {file_type.upper()}: {file_path}")
            
            return {
                'success': True,
                'message': f'Successfully processed {len(results)} files',
                'results': results
            }
        else:
            return {
                'success': False,
                'error': 'No output files were generated',
                'results': {}
            }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': {}
        }

# Convenience wrapper functions for different use cases
def process_folder_to_folder(input_folder, output_folder):
    """Process bands from input folder to output folder"""
    return process_satellite_bands(input_folder, output_folder)

def process_with_custom_names(input_folder, output_folder, ndvi_name='ndvi.tif', lst_name='lst.tif'):
    """Process bands with custom output file names"""
    
    result = process_satellite_bands(input_folder, output_folder)
    
    if result['success'] and result['results']:
        # Rename files if needed
        output_path = Path(output_folder)
        
        if 'ndvi' in result['results'] and ndvi_name != 'ndvi.tif':
            old_path = Path(result['results']['ndvi'])
            new_path = output_path / ndvi_name
            old_path.rename(new_path)
            result['results']['ndvi'] = str(new_path)
        
        if 'lst' in result['results'] and lst_name != 'lst.tif':
            old_path = Path(result['results']['lst'])
            new_path = output_path / lst_name
            old_path.rename(new_path)
            result['results']['lst'] = str(new_path)
    
    return result

# Example usage and testing
def test_processor():
    """Test function - remove in production"""
    
    # Example 1: Basic usage
    input_folder = "data/raw/delhi_20250619"
    output_folder = "data/processed/delhi_20250619"
    
    result = process_satellite_bands(input_folder, output_folder)
    
    if result['success']:
        print("✅ Test successful!")
        for file_type, path in result['results'].items():
            print(f"   {file_type}: {path}")
    else:
        print(f"❌ Test failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    # Run test
    test_processor()