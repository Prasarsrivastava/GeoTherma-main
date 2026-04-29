# 🌆 Urban Heat Island (UHI) Analysis Web App

This project is a satellite-data-powered web application designed to detect Urban Heat Islands (UHIs), predict heat hotspots using AI models, and recommend mitigation strategies. It leverages NDVI, LST, and land-use classification for UHI severity analysis.

---

## 🚀 Features

- 🔍 Select a location and fetch satellite imagery
- 🛰️ Preprocess NDVI, LST, and Urban data to GeoTIFF
- 🔥 Detect heat hotspots with a trained AI model
- 🌡️ Normalize temperature (LST)
- 🧠 Classify UHI severity using decision trees or random forest
- 💡 Recommend mitigation strategies
- 💾 Download processed layers and results
- 🧰 REST API for integration

---

## 🛠️ Tech Stack

- **Backend**: Flask, Scikit-learn, Rasterio, NumPy
- **Frontend**: Streamlit (optional), HTML (for Flask templates)
- **AI Models**: Decision Tree, Random Forest, TensorFlow (optional)
- **Geospatial**: Rasterio, GeoPandas, Shapely
- **Visualization**: Matplotlib, Seaborn

---

## 📂 Project Structure

uhi-project/
├── app.py # Main Flask app
├── requirements.txt
├── README.md
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ └── hotspot_model.pkl
├── static/
├── templates/
├── utils/
│ └── satellite_processor.py


---

## ⚙️ Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/uhi-project.git
    cd uhi-project
    ```

2. **Set up a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**:
    ```bash
    python app.py
    ```

---

## 🧪 API Usage

### `/api/process_data` `POST`

**Description**: Process satellite data and train model.

**Request JSON:**
```json
{
  "folder": "2024_city_data"
}

{
  "status": "success",
  "message": "Data processing started for folder: 2024_city_data"
}

Response:

json
Copy
Edit
{
  "status": "success",
  "message": "Data processing started for folder: 2024_city_data"
}
📥 Sample Data
Put your Landsat or Sentinel satellite imagery in:

bash
Copy
Edit
data/raw/<your-folder-name>/
Processed output and GeoTIFFs will appear in:

bash
Copy
Edit
data/processed/<your-folder-name>/output/
🧠 Model Training
Model training uses LST, NDVI, and urban land classification for UHI hotspot prediction. Trained models are stored in /models.

🧹 Cleanup & Tips
To clear temporary files: rm -rf data/processed/*

Always activate the virtual environment when working.

🙌 Credits
USGS EarthExplorer for imagery: https://earthexplorer.usgs.gov/

Libraries: Flask, Rasterio, Scikit-learn, GeoPandas, etc.

📃 License
MIT License (or your preferred license)