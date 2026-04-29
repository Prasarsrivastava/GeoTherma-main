import requests
import numpy as np
import rasterio
from datetime import datetime
import tempfile
import os
import shutil
import zipfile
import yaml
from typing import Dict, Any

#def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
#    """Load configuration from YAML file"""
#    with open(config_path, 'r') as f:
#        config = yaml.safe_load(f)
#    return config

def load_config(city_name: str = None) -> Path:
    """Find the most recent config file for a city"""
    config_dir = Path("config")
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir.absolute()}")
    
    pattern = f"{city_name}_*.yml" if city_name else "*.yml"
    config_files = sorted(config_dir.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if not config_files:
        available = [f.name for f in config_dir.glob("*.yml")]
        raise FileNotFoundError(
            f"No config files found for {city_name or 'any city'}. Available: {available}"
        )
    
    return config_files[0]

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration structure"""
    required_sections = ['city_boundaries', 'collection_params', 'date_range']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    
    if 'fgh' not in config['city_boundaries']:
        raise ValueError("City boundary configuration missing 'fgh' section")
    
    boundary = config['city_boundaries']['fgh']
    if boundary.get('type') != 'circle':
        raise ValueError("Only circle boundaries are currently supported")
    
    if 'coordinates' not in boundary or 'center' not in boundary['coordinates']:
        raise ValueError("Circle boundary requires center coordinates")

def download_band(url: str) -> str:
    """Download a Landsat band and return temporary file path"""
    r = requests.get(url)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    with open(temp.name, "wb") as f:
        f.write(r.content)
    return temp.name

def compute_ndvi(b4_path: str, b5_path: str) -> tuple[np.ndarray, dict]:
    """Calculate NDVI from Landsat bands 4 and 5"""
    with rasterio.open(b4_path) as red, rasterio.open(b5_path) as nir:
        red_data = red.read(1).astype('float32')
        nir_data = nir.read(1).astype('float32')
        ndvi = (nir_data - red_data) / (nir_data + red_data + 1e-10)
        profile = red.profile
        profile.update(dtype='float32', count=1)
        return ndvi, profile

def compute_lst(b10_path: str) -> tuple[np.ndarray, dict]:
    """Calculate Land Surface Temperature from band 10"""
    with rasterio.open(b10_path) as band10:
        thermal = band10.read(1).astype('float32')
        lst = 0.00341802 * thermal + 149.0  # Conversion to Kelvin
        profile = band10.profile
        profile.update(dtype='float32', count=1)
        return lst, profile

def save_geotiff(array: np.ndarray, profile: dict, filename: str, folder: str) -> str:
    """Save array as GeoTIFF file"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)
    return path

def zip_folder(folder_path: str) -> str:
    """Create ZIP archive of folder"""
    zip_path = folder_path + ".zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    return zip_path

def analyze_uhi(config: Dict[str, Any]) -> str:
    """Main UHI analysis function using config dictionary"""
    validate_config(config)
    
    # Extract parameters from config
    boundary = config['city_boundaries']['fgh']
    center_lon, center_lat = boundary['coordinates']['center']
    radius_deg = boundary['coordinates']['radius'] / 111320  # Convert meters to degrees
    
    collection = config['collection_params']
    max_cloud = collection['max_cloud_cover']
    output_crs = collection['crs']
    
    date_range = config['date_range']
    start_date = datetime.strptime(date_range['start_date'], '%Y-%m-%d').date()
    end_date = datetime.strptime(date_range['end_date'], '%Y-%m-%d').date()
    
    location_name = config.get('metadata', {}).get('generated_by', 'uhi_analysis')

    # Search for Landsat scenes
    payload = {
        "datasetName": "LANDSAT_8_C1",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                "latitude": center_lat - radius_deg,
                "longitude": center_lon - radius_deg
            },
            "upperRight": {
                "latitude": center_lat + radius_deg,
                "longitude": center_lon + radius_deg
            }
        },
        "temporalFilter": {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d")
        },
        "maxResults": 5,
        "sortOrder": "DESC",
        "maxCloudCover": max_cloud
    }

    print("Searching for Landsat scenes...")
    search = requests.post(
        "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search",
        json=payload
    )

    results = search.json().get("data", {}).get("results", [])
    if not results:
        raise ValueError("No suitable Landsat scenes found for the given parameters")

    scene = results[0]
    base_id = scene["displayId"]
    print(f"Using scene: {base_id} (Cloud cover: {scene.get('cloudCover', 'unknown')}%)")

    # Download required bands
    path = base_id.split("_")[2][:3]
    row = base_id.split("_")[2][3:]
    aws_url = f"https://landsat-pds.s3.amazonaws.com/c1/L8/{path}/{row}/{base_id}"

    print("Downloading Landsat bands...")
    b4 = download_band(f"{aws_url}/{base_id}_B4.TIF")  # Red
    b5 = download_band(f"{aws_url}/{base_id}_B5.TIF")  # NIR
    b10 = download_band(f"{aws_url}/{base_id}_B10.TIF")  # Thermal

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join("output", f"{location_name}_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Process NDVI
    print("Calculating NDVI...")
    ndvi, ndvi_profile = compute_ndvi(b4, b5)
    ndvi_profile.update(crs=output_crs)
    ndvi_path = save_geotiff(ndvi, ndvi_profile, "ndvi.tif", save_folder)

    # Process LST
    print("Calculating Land Surface Temperature...")
    lst, lst_profile = compute_lst(b10)
    lst_profile.update(crs=output_crs)
    lst_path = save_geotiff(lst, lst_profile, "lst.tif", save_folder)

    # Classify UHI
    print("Classifying Urban Heat Island areas...")
    low = np.percentile(lst, 60)
    high = np.percentile(lst, 90)
    uhi_class = np.zeros_like(lst, dtype=np.uint8)
    uhi_class[(lst > low) & (lst <= high)] = 1  # Moderate heat
    uhi_class[lst > high] = 2  # High heat
    uhi_profile = lst_profile.copy()
    uhi_profile.update(dtype='uint8')
    uhi_path = save_geotiff(uhi_class, uhi_profile, "uhi_class.tif", save_folder)

    # Create metadata file
    metadata = {
        "processing_date": datetime.now().isoformat(),
        "landsat_scene": base_id,
        "parameters": {
            "ndvi_thresholds": {"low": float(np.nanmin(ndvi)), "high": float(np.nanmax(ndvi))},
            "lst_thresholds": {"low": float(low), "high": float(high)},
            "crs": output_crs
        },
        "original_config": config
    }
    
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Create ZIP archive
    print("Creating output archive...")
    zip_path = zip_folder(save_folder)

    # Clean up temporary files
    for f in [b4, b5, b10]:
        os.remove(f)

    print(f"Analysis complete. Results saved to: {zip_path}")
    return zip_path

if __name__ == "__main__":
    try:
        import json  # For metadata output
        config = load_config()
        result_path = analyze_uhi(config)
        print(f"Success! Output created at: {result_path}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")