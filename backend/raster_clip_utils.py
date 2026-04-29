import rasterio
from rasterio.warp import transform_geom
from rasterio.mask import mask
from shapely.geometry import shape
import os
import json

def clip_raster(input_path, output_path, geometry):
    with rasterio.open(input_path) as src:
        # Transform geometry to raster CRS
        geom = transform_geom('EPSG:4326', src.crs.to_string(), geometry)

        out_image, out_transform = mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def clip_all_rasters_in_folder(input_folder, output_folder, geometry):
    clipped_files = []
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".tif"):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            clip_raster(in_path, out_path, geometry)
            clipped_files.append(fname)
    return clipped_files
