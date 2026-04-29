import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt

input_path = 'data/raw/LST_Bangalore_2023.tif'
output_path = 'data/processed/LST_Bangalore_2023_normalized.tif'

# Calculate LST from ST_B10 band
lst = image.select('ST_B10').multiply(0.00341802).add(149.0)
# Open original LST GeoTIFF
with rasterio.open(input_path) as src:
    band = src.read(1).astype('float32')
    profile = src.profile.copy()

# Handle nodata values
band[band == profile.get('nodata', -9999)] = np.nan

# Normalize: (x - min) / (max - min)
min_val = np.nanmin(band)
max_val = np.nanmax(band)
normalized = (band - min_val) / (max_val - min_val)

# Update profile and save new GeoTIFF
profile.update(dtype='float32', nodata=np.nan)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(normalized, 1)

print(f"✅ Normalized GeoTIFF saved to: {output_path}")

# Optional: Plot
plt.imshow(normalized, cmap='hot')
plt.colorbar(label='Normalized Temperature')
plt.title("Normalized LST")
plt.show()
