# 🌍 AI-Powered Earthquake Prediction System

An intelligent seismic analysis and prediction platform that identifies earthquake risk zones, estimates magnitudes, and provides early warnings using real-time and historical seismic data.


---

## 🚀 Live Application

🔗 [Deployed App](https://earthquakeprediction-f8rddmsonf3h2fkv6abutv.streamlit.app/)

---

## 📌 Key Features

- **Real-time earthquake monitoring** via USGS API
- **Magnitude prediction** using Random Forest & Gradient Boosting models
- **Weather integration** (WeatherAPI) for environmental context
- **Region-based risk assessment** using DBSCAN spatial clustering
- **Early warning system** with impact countdown and safety recommendations
- **Interactive dashboards & maps** (Streamlit + Folium)
- **Probability estimation** for significant seismic events (M≥4.5)
- **Historical and real-time data fusion**

---

## 🧠 Machine Learning Pipeline

- **Feature Engineering:**  
  - Latitude, Longitude, Depth
  - Year, Month, Day, Hour, Weekday
  - Spatial cluster ID (DBSCAN)
  - Rolling means for magnitude and depth (per cluster)
  - Magnitude-to-depth ratio
  - Prior quakes in region (cumulative count)
- **Models:**  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - *Best model auto-selected based on R² score*
- **Outputs:**  
  - Predicted magnitude
  - Probability of significant event (sigmoid-transformed)

---

## 📊 Technologies Used

| Backend      | Frontend       | Data Sources               |
|--------------|----------------|----------------------------|
| Python       | Streamlit      | USGS Earthquake API        |
| Scikit-learn | Folium Maps    | WeatherAPI (climate data)  |
| Pandas       | Matplotlib     | USGS CSV (historical data) |
| DBSCAN       | Seaborn        |                            |

---

---

## ⚙️ Installation

1. **Clone the repository:**

2. **Install dependencies:**

3. **Configure API keys:**
- Set your WeatherAPI key in `streamlit.py` (`WEATHER_API_KEY`).

4. **Prepare data and models:**
- Place your raw data in `data/data.csv`.
- Run the training script to generate models and processed data:
  ```
  python earthquake_model_training.py
  ```

5. **Launch the application:**

---

## 🖥️ Usage Overview

- **Prediction:**  
Enter a location, latitude/longitude, and depth to get a predicted earthquake magnitude and risk probability.
- **Early Warnings:**  
Real-time alerts for significant seismic events, including estimated impact time and safety guidance.
- **Risk Dashboard:**  
Visualizes high-risk regions, average magnitudes, and event frequencies.
- **Regional Assessment:**  
Select a region to view detailed seismic statistics and event history.
- **Probability Estimator:**  
Calculate the probability of significant earthquakes for any location and time window.

---

## 📈 Model Training Details

- Reads and preprocesses historical earthquake data (`preprocessed_data.csv`)
- Performs feature engineering (temporal, spatial, rolling statistics)
- Clusters events spatially using DBSCAN
- Trains and evaluates Random Forest and Gradient Boosting models
- Selects and saves the best model based on R² score
- Outputs feature importance and cluster visualizations

---


## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---




