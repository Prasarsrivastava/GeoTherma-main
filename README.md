
# Urban Heat Island Hackathon Project

**Cities**: Bangalore & Delhi

This project detects urban heat islands using satellite data, predicts hotspots using ML, and recommends mitigation strategies.

---

## Features
- Satellite data ingestion (LST, NDVI, Urban Land Use) via Google Earth Engine
- Flask backend with APIs to serve temperature data and predictions
- Leaflet.js frontend map for visualization
- ML model to predict hotspots
- Chart.js dashboard for historical trend analytics

---

## Deployment

### Backend (Flask)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
Just open `frontend/index.html` in a browser (or serve with any HTTP server).

### Cloud Deployment
- **Backend**: [Render](https://render.com/) or [Heroku](https://heroku.com/)
- **Frontend**: [Vercel](https://vercel.com/) or GitHub Pages

---

## Google Earth Engine Data Fetch

Use `scripts/fetch_lst_data.js` in the GEE Code Editor:
1. Go to https://code.earthengine.google.com/
2. Paste the script
3. Export the processed LST image for Bangalore or Delhi
4. Repeat for NDVI and land cover (sample below)

```javascript
// NDVI Example
var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
Map.addLayer(ndvi, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');
```

```javascript
// Urban classification example using MODIS
var landcover = ee.ImageCollection("MODIS/006/MCD12Q1")
                  .filterDate('2023-01-01', '2023-12-31')
                  .first()
                  .select('LC_Type1');
Map.addLayer(landcover, {}, 'Urban Cover');
```

---

## Authors & Credits
Built during a 9-day hackathon challenge using Open Data and open-source tools.
