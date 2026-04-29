import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RASTER_EXTENSIONS = ('.tif', '.tiff', '.img', '.vrt')
VECTOR_EXTENSIONS = ('.geojson', '.shp', '.json', '.gpkg')

def reproject_raster(input_path: str, output_path: str, target_crs: str = 'EPSG:4326') -> bool:
    """Reproject a raster file to target CRS. Returns success status."""
    try:
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear)
        return True
    except Exception as e:
        logger.error(f"Failed to reproject raster {input_path}: {str(e)}")
        return False

def reproject_vector(input_path: str, output_path: str, target_crs: str = 'EPSG:4326') -> bool:
    """Reproject a vector file to target CRS. Returns success status."""
    try:
        gdf = gpd.read_file(input_path)
        if gdf.crs is None:
            logger.warning(f"No CRS defined in {input_path}, assuming EPSG:4326")
            gdf.crs = 'EPSG:4326'
        gdf = gdf.to_crs(target_crs)
        gdf.to_file(output_path)
        return True
    except Exception as e:
        logger.error(f"Failed to reproject vector {input_path}: {str(e)}")
        return False

def process_folder(
    folder_path: str,
    raster_target_crs: str = 'EPSG:4326',
    vector_target_crs: str = 'EPSG:4326',
    backup: bool = True
) -> Dict[str, Any]:
    """
    Process all supported files in a folder.
    Returns dictionary with processing results.
    """
    if not os.path.exists(folder_path):
        return {
            'success': False,
            'message': f"Folder does not exist: {folder_path}",
            'details': {'processed': 0, 'errors': []}
        }

    backup_folder = os.path.join(folder_path, "original_backup")
    results = {
        'processed': 0,
        'rasters': 0,
        'vectors': 0,
        'errors': []
    }

    try:
        if backup:
            os.makedirs(backup_folder, exist_ok=True)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Skip directories and backup folder
            if not os.path.isfile(file_path) or filename == "original_backup":
                continue

            try:
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in RASTER_EXTENSIONS:
                    if backup:
                        backup_path = os.path.join(backup_folder, filename)
                        os.rename(file_path, backup_path)
                        input_path = backup_path
                    else:
                        input_path = file_path

                    if reproject_raster(input_path, file_path, raster_target_crs):
                        results['rasters'] += 1
                        results['processed'] += 1

                elif file_ext in VECTOR_EXTENSIONS:
                    if backup:
                        backup_path = os.path.join(backup_folder, filename)
                        os.rename(file_path, backup_path)
                        input_path = backup_path
                    else:
                        input_path = file_path

                    if reproject_vector(input_path, file_path, vector_target_crs):
                        results['vectors'] += 1
                        results['processed'] += 1

            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)

        return {
            'success': True,
            'message': f"Processed {results['processed']} files ({results['rasters']} rasters, {results['vectors']} vectors)",
            'details': results
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Fatal processing error: {str(e)}",
            'details': results
        }