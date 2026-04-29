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
import sys
from pathlib import Path
import json  # Moved to top level imports

# Optional: load most recent config file for a city
def load_config(city_name: str = None) -> Path:
    """Find the most recent config file for a city or token file"""
    config_dir = Path("configs")
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir.absolute()}")
    
    # Special case for token file
    if city_name == "token":
        token_file = config_dir / "token.yml"
        if token_file.exists():
            return token_file
        available = [f.name for f in config_dir.glob("*.yml")]
        raise FileNotFoundError(f"Token file not found. Available configs: {available}")
    
    # Normal case for city configs
    pattern = f"{city_name}_*.yml" if city_name else "*.yml"
    config_files = sorted(config_dir.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if not config_files:
        available = [f.name for f in config_dir.glob("*.yml")]
        raise FileNotFoundError(f"No config files found for {city_name or 'any city'}. Available: {available}")
    
    return config_files[0]

def validate_config(config: Dict[str, Any]) -> str:
    """Validate the configuration structure and return the city key"""
    required_sections = ['city_boundaries', 'collection_params', 'date_range']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")

    boundaries = config['city_boundaries']
    if not isinstance(boundaries, dict) or not boundaries:
        raise ValueError("city_boundaries must contain at least one city")

    # Use the first (and assumed only) city key
    city_key, boundary = next(iter(boundaries.items()))

    if boundary.get('type') != 'circle':
        raise ValueError("Only circle boundaries are currently supported")

    coords = boundary.get('coordinates', {})
    if 'center' not in coords or 'radius' not in coords:
        raise ValueError("Circle boundary requires 'center' and 'radius'")

    return city_key

def download_band(url: str) -> str:
    """Download a Landsat band and save to temp file"""
    print(f"Downloading: {url}")
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Failed to download from URL: {url} (Status code: {r.status_code})")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    with open(temp.name, "wb") as f:
        f.write(r.content)
    print(f"Saved temporary file: {temp.name}")
    return temp.name

def compute_ndvi(b4_path: str, b5_path: str) -> tuple[np.ndarray, dict]:
    """Calculate NDVI from Landsat bands 4 and 5"""
    print(f"Computing NDVI using B4: {b4_path} and B5: {b5_path}")
    with rasterio.open(b4_path) as red, rasterio.open(b5_path) as nir:
        red_data = red.read(1).astype('float32')
        nir_data = nir.read(1).astype('float32')
        ndvi = (nir_data - red_data) / (nir_data + red_data + 1e-10)
        profile = red.profile
        profile.update(dtype='float32', count=1)
        return ndvi, profile

def compute_lst(b10_path: str) -> tuple[np.ndarray, dict]:
    """Calculate LST from band 10"""
    print(f"Computing LST using B10: {b10_path}")
    with rasterio.open(b10_path) as band10:
        thermal = band10.read(1).astype('float32')
        lst = 0.00341802 * thermal + 149.0  # Convert to Kelvin
        profile = band10.profile
        profile.update(dtype='float32', count=1)
        return lst, profile

def save_geotiff(array: np.ndarray, profile: dict, filename: str, folder: str) -> str:
    """Save array to GeoTIFF"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    print(f"Saving GeoTIFF: {path}")
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)
    return path

def zip_folder(folder_path: str) -> str:
    """ZIP the output folder"""
    zip_path = folder_path + ".zip"
    print(f"Zipping folder: {folder_path}")
    shutil.make_archive(folder_path, 'zip', folder_path)
    return zip_path

def analyze_uhi(config: Dict[str, Any]) -> str:
    """Main processing logic"""
    city_key = validate_config(config)
    print("Config validated")

    # Extract parameters using dynamic city_key
    boundary = config['city_boundaries'][city_key]
    center_lon, center_lat = boundary['coordinates']['center']
    radius_deg = boundary['coordinates']['radius'] / 111320  # meters to degrees
    location_name = config.get('metadata', {}).get('generated_by', city_key)

    collection = config['collection_params']
    max_cloud = collection['max_cloud_cover']
    output_crs = collection['crs']

    date_range = config['date_range']
    start_date = datetime.strptime(date_range['start_date'], '%Y-%m-%d').date()
    end_date = datetime.strptime(date_range['end_date'], '%Y-%m-%d').date()

   
    # --- Search Scenes ---
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
#
    def load_token(token_path="configs/token.yml") -> str:
        with open(token_path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "token" not in data:
            raise ValueError("Invalid format in token.yml. Expected: token: <value>")
        return data["token"]
    
    # Get token from config (assuming it's in the main config)
    token_path = Path("configs/token.yml")
    with open(token_path, "r") as f:
        token_data = yaml.safe_load(f)
    print("Search response status:",token_data)
    
    headers = {"X-Auth-Token": token_data}

    print(headers)
    search = requests.post(
        "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search",
        json=payload,
        headers = {
    'X-Auth-Token': 'CPbCbOHeq6VMlmQnWrnd6gca5K3pIl@JV99hhhj0lTSZm!pjcc3Sv0CllGM49FSv'
}
    )
    print("Search response status:", search.status_code)
    print("Search response JSON:", search.text)

    # --- Debug print API response ---
    print("API Status Code:", search.status_code)
    try:
        response_json = search.json()
        print("Response JSON:", json.dumps(response_json, indent=2)[:1000])  # limit output size
    except Exception as e:
        print("Failed to parse JSON:", e)
        raise

    results = response_json.get("data", {}).get("results", [])
    print("Parsed results:", results)
    if not results:
        raise ValueError("❌ No suitable Landsat scenes found for the given parameters")

    scene = results[0]
    base_id = scene["displayId"]
    print(f"Using scene: {base_id} (Cloud cover: {scene.get('cloudCover', 'unknown')}%)")

    # --- Download bands from AWS ---
    path = base_id.split("_")[2][:3]
    row = base_id.split("_")[2][3:]
    aws_url = f"https://landsat-pds.s3.amazonaws.com/c1/L8/{path}/{row}/{base_id}"

    print("Downloading Landsat bands...")
    b4 = download_band(f"{aws_url}/{base_id}_B4.TIF")
    b5 = download_band(f"{aws_url}/{base_id}_B5.TIF")
    b10 = download_band(f"{aws_url}/{base_id}_B10.TIF")

    # --- Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join("output", f"{location_name}_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # --- NDVI ---
    print("Calculating NDVI...")
    ndvi, ndvi_profile = compute_ndvi(b4, b5)
    ndvi_profile.update(crs=output_crs)
    ndvi_path = save_geotiff(ndvi, ndvi_profile, "ndvi.tif", save_folder)

    # --- LST ---
    print("Calculating Land Surface Temperature...")
    lst, lst_profile = compute_lst(b10)
    lst_profile.update(crs=output_crs)
    lst_path = save_geotiff(lst, lst_profile, "lst.tif", save_folder)

    # --- Classify UHI ---
    print("Classifying Urban Heat Island areas...")
    low = np.percentile(lst, 60)
    high = np.percentile(lst, 90)
    uhi_class = np.zeros_like(lst, dtype=np.uint8)
    uhi_class[(lst > low) & (lst <= high)] = 1
    uhi_class[lst > high] = 2
    uhi_profile = lst_profile.copy()
    uhi_profile.update(dtype='uint8')
    uhi_path = save_geotiff(uhi_class, uhi_profile, "uhi_class.tif", save_folder)

    # --- Metadata ---
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

    # --- Archive ---
    print("Creating ZIP archive...")
    zip_path = zip_folder(save_folder)

    # --- Cleanup ---
    for f in [b4, b5, b10]:
        os.remove(f)

    print(f"Analysis complete. Results saved to: {zip_path}")
    return zip_path

def main() -> Dict[str, Any]:
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
        city_name = config_filename.split('_')[0] if '_' in config_filename else config_filename.replace('.yml', '')
        config_path = load_config(city_name)
    else:
        config_path = load_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If token is in a separate file, load it here
    if 'token_config' in config:  # If your config references a token config
        token_config_path = load_config("token")
        with open(token_config_path, 'r') as f:
            token_config = yaml.safe_load(f)
        config['token'] = token_config['token']
    
    return config

if __name__ == "__main__":
    try:
        config = main()
        result_path = analyze_uhi(config)
        print(f"Success! Output created at: {result_path}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")