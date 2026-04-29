// GEE script for LST, NDVI, and Urban Classification

var bangalore = ee.Geometry.Point([77.5946, 12.9716]);
var delhi = ee.Geometry.Point([77.2090, 28.6139]);

var region = bangalore; // change to 'delhi' for Delhi

var image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(region)
    .filterDate('2023-01-01', '2023-12-31')
    .sort('CLOUD_COVER')
    .first();

// LST (Surface Temperature)
var lst = image.select('ST_B10').multiply(0.00341802).add(149.0);
Map.centerObject(region, 10);
Map.addLayer(lst, {min: 290, max: 320, palette: ['blue', 'red']}, 'LST');

// NDVI
var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
Map.addLayer(ndvi, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');

// Urban Land Cover using MODIS
var landcover = ee.ImageCollection("MODIS/006/MCD12Q1")
                  .filterDate('2023-01-01', '2023-12-31')
                  .first()
                  .select('LC_Type1');
Map.addLayer(landcover, {}, 'Urban Cover');
