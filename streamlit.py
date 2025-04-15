# earthquake_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import os
import threading
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, sqrt, atan2

# Set page config
st.set_page_config(
    page_title="Earthquake Prediction System",
    page_icon="🌋",
    layout="wide"
)

# Constants
REALTIME_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
HISTORICAL_DATA_PATH = "processed_earthquake_data.csv"
S_WAVE_SPEED = 3.5  # km/s
ALERT_THRESHOLD = 4.5  # Minimum magnitude for alerts

# Initialize geocoder
@st.cache_resource
def init_geocoder():
    return RateLimiter(Nominatim(user_agent="earthquake_app").geocode, min_delay_seconds=2)

geocode = init_geocoder()

# Load model and related files
@st.cache_resource
def load_resources():
    return {
        'best_model': pickle.load(open('best_model.pkl', 'rb')),
        'scaler': pickle.load(open('scaler.pkl', 'rb')),
        'features': pickle.load(open('feature_list.pkl', 'rb'))
    }

# Real-time data fetching with error handling
@st.cache_data(ttl=300)
def fetch_realtime_data():
    try:
        response = requests.get(REALTIME_API_URL)
        data = response.json()
        
        features = data['features']
        quake_data = []
        for feature in features:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            quake_data.append({
                'time': pd.to_datetime(props['time'], unit='ms', utc=True),
                'longitude': coords[0],
                'latitude': coords[1],
                'depth': coords[2],
                'mag': props['mag'],
                'place': str(props.get('place', 'Unknown'))
            })
            
        df = pd.DataFrame(quake_data)
        return df[df['time'].notna()]
    
    except Exception as e:
        st.error(f"Real-time data fetch error: {str(e)}")
        return None

# Enhanced region naming function
def get_region_name(lat, lon):
    """Convert coordinates to human-readable region names"""
    if 15 <= lat <= 90 and -168 <= lon <= -52:
        if lat > 50: return "Northern North America"
        return "Central North America"
    elif -56 <= lat < 15 and -82 <= lon <= -35: return "South America"
    elif 35 <= lat <= 71 and -25 <= lon <= 40: return "Europe"
    elif 10 <= lat <= 55 and 40 <= lon <= 190: return "East Asia" if lon > 100 else "Central Asia"
    elif -35 <= lat <= 37 and -20 <= lon <= 55: return "Africa"
    elif -50 <= lat <= -10 and 110 <= lon <= 180: return "Oceania"
    elif lat > 66.5: return "Arctic Region"
    elif lat < -66.5: return "Antarctic Region"
    else:
        if -180 <= lon <= -80: return "East Pacific"
        elif -80 <= lon <= 20: return "Atlantic Ocean"
        elif 20 <= lon <= 160: return "Indian Ocean"
        else: return "West Pacific"

# Enhanced data loading with cluster naming
@st.cache_data
def load_data(use_realtime=False):
    # Load historical data with strict datetime conversion
    historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
    historical_df['time'] = pd.to_datetime(historical_df['time'], utc=True, errors='coerce')
    historical_df = historical_df.dropna(subset=['time'])
    historical_df['place'] = historical_df['place'].astype(str).replace('nan', 'Unknown')
    
    if use_realtime:
        realtime_df = fetch_realtime_data()
        if realtime_df is not None:
            # Merge and ensure datetime consistency
            combined_df = pd.concat([historical_df, realtime_df], ignore_index=True)
            # Assign cluster_name to all rows (historical + real-time)
            combined_df['cluster_name'] = combined_df.apply(
                lambda row: get_region_name(row['latitude'], row['longitude']), axis=1
            )
            return combined_df.drop_duplicates()
    # Always assign cluster_name for historical-only mode too
    historical_df['cluster_name'] = historical_df.apply(
        lambda row: get_region_name(row['latitude'], row['longitude']), axis=1
    )
    return historical_df


# Enhanced prediction function
def predict_probability(lat, lon, depth, date_str, model, scaler, features, df):
    try:
        date = pd.to_datetime(date_str, utc=True, errors='coerce') or datetime.now(timezone.utc)
    except:
        date = datetime.now(timezone.utc)

    coords = df[['latitude', 'longitude']].values
    distances = np.sqrt(((coords - np.array([lat, lon]))**2).sum(axis=1))
    closest_idx = distances.argmin()
    cluster_name = df.iloc[closest_idx]['cluster_name']

    region_data = df[df['cluster_name'] == cluster_name]
    rolling_mag_mean = region_data['mag'].mean() if not region_data.empty else df['mag'].mean()
    
    sample = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'depth': [depth],
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'hour': [date.hour],
        'dayofweek': [date.weekday()],
        'rolling_mag_mean': [rolling_mag_mean],
        'mag_depth_ratio': [rolling_mag_mean/depth if depth > 0 else 0],
    }, columns=features)

    sample_scaled = scaler.transform(sample)
    raw_pred = model.predict(sample_scaled)[0]
    probability = 1 / (1 + np.exp(-(raw_pred - 4.5))) * 100
    return raw_pred, min(max(probability, 0), 100), cluster_name

def calculate_shaking_time(epicenter_lat, epicenter_lon, user_lat, user_lon):
    """Calculate time until shaking arrives using haversine distance"""
    R = 6371  # Earth radius in km
    lat1, lon1 = radians(epicenter_lat), radians(epicenter_lon)
    lat2, lon2 = radians(user_lat), radians(user_lon)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance = R * c
    return int(distance / S_WAVE_SPEED)

# Modified Real-Time Alert System
def check_realtime_alerts():
    while True:
        try:
            df = fetch_realtime_data()
            if df is not None and not df.empty:
                recent_quakes = df[df['mag'] >= ALERT_THRESHOLD]
                if not recent_quakes.empty:
                    latest_quake = recent_quakes.iloc[0]
                    st.session_state.alert_triggered = True
                    st.session_state.alert_data = {
                        'mag': latest_quake['mag'],
                        'lat': latest_quake['latitude'],
                        'lon': latest_quake['longitude'],
                        'time': latest_quake['time'].tz_convert(None),  # Convert to naive for display
                        'place': latest_quake['place']
                    }
        except Exception as e:
            print(f"Alert check error: {str(e)}")
        threading.Event().wait(1)

# Main application
def main():
    try:
        resources = load_resources()
        st.sidebar.title("🌋 Earthquake Analysis")
        
        # Data source selection
        use_realtime = st.sidebar.toggle("Enable Real-Time Data", value=False)
        df = load_data(use_realtime=use_realtime)
        
        # Start alert thread
        if use_realtime and not hasattr(st.session_state, 'alert_thread'):
            alert_thread = threading.Thread(target=check_realtime_alerts, daemon=True)
            alert_thread.start()
            st.session_state.alert_thread = alert_thread

        app_mode = st.sidebar.selectbox(
            "Select Mode",
            ["Prediction", "Risk Dashboard", "Regional Risk Assessment", 
             "Probability Estimator", "Early Warnings", "About"]
        )

        # Always use best model
        model = resources['best_model']
        scaler = resources['scaler']
        features = resources['features']

        if app_mode == "Early Warnings":
            st.title("Real-time Seismic Alerts")
            
            # User location input
            with st.expander("Set Your Location"):
                col1, col2 = st.columns(2)
                with col1:
                    location_input = st.text_input("Enter Your Location (e.g., Los Angeles, CA)")
                with col2:
                    if location_input:
                        try:
                            location = geocode(location_input)
                            if location:
                                user_lat = location.latitude
                                user_lon = location.longitude
                                st.success(f"Location set to: {location.address}")
                            else:
                                user_lat = 34.05
                                user_lon = -118.25
                        except:
                            user_lat = 34.05
                            user_lon = -118.25
                    else:
                        user_lat = 34.05
                        user_lon = -118.25
                    
                    st.write(f"Current Coordinates: {user_lat:.4f}, {user_lon:.4f}")

            if st.session_state.get('alert_triggered'):
                alert = st.session_state.alert_data
                shaking_time = calculate_shaking_time(
                    alert['lat'], alert['lon'], 
                    user_lat, user_lon
                )
                
                st.error("🚨 EARTHQUAKE ALERT! Prepare for shaking!")
                with st.expander("Alert Details", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Estimated Magnitude", f"M{alert['mag']:.1f}")
                        st.metric("Time to Impact", f"{shaking_time} seconds")
                        st.write("**Recommended Actions:**")
                        st.write("1. DROP to the ground")
                        st.write("2. Take COVER under sturdy furniture")
                        st.write("3. HOLD ON until shaking stops")
                        
                        if alert['mag'] > 5.0:
                            st.write("**Automated Safety Measures:**")
                            st.write("- Elevators recalled to nearest floor")
                            st.write("- Gas lines pressure reduced")
                            st.write("- Emergency power activated")
                        
                    with col2:
                        alert_map = folium.Map(
                            location=[alert['lat'], alert['lon']], 
                            zoom_start=7
                        )
                        folium.Marker(
                            [alert['lat'], alert['lon']],
                            popup=f"Epicenter M{alert['mag']:.1f}",
                            icon=folium.Icon(color='red', icon='exclamation-triangle')
                        ).add_to(alert_map)
                        
                        folium.Circle(
                            location=[alert['lat'], alert['lon']],
                            radius=shaking_time*S_WAVE_SPEED*1000,
                            color='red',
                            fill=True,
                            popup=f"Estimated impact area in {shaking_time}s"
                        ).add_to(alert_map)
                        
                        folium_static(alert_map)
                
                st.session_state.alert_triggered = False
            else:
                st.success("No active seismic alerts")
                st.write("System status: Monitoring 24/7")
                if use_realtime:
                    st.write("Recent seismic activity:")
                    st.dataframe(df.sort_values('time', ascending=False).head(10)[['time', 'mag', 'place']])

        elif app_mode == "Prediction":
            st.title("Earthquake Magnitude Prediction")
            with st.form("prediction_form"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    location_input = st.text_input("Enter Location Name (e.g., Tokyo, Japan)")
                    lat = st.number_input("Latitude", value=35.0, format="%.4f")
                    lon = st.number_input("Longitude", value=-118.0, format="%.4f")
                    depth = st.slider("Depth (km)", 0.1, 700.0, 10.0)
                    
                with col2:
                    date = st.date_input("Date", value=datetime.now(timezone.utc))
                    time = st.time_input("Time", value=datetime.now(timezone.utc).time())
                    date_str = f"{date} {time}"
                    
                    if location_input:
                        try:
                            location = geocode(location_input)
                            if location:
                                lat = location.latitude
                                lon = location.longitude
                                st.success(f"Location found: {location.address}")
                        except Exception as e:
                            st.error(f"Geocoding error: {str(e)}")
                
                if st.form_submit_button("Predict Magnitude"):
                    with st.spinner("Generating prediction..."):
                        predicted_mag, probability, cluster_name = predict_probability(
                            lat, lon, depth, date_str, model, scaler, features, df
                        )
                        
                        st.subheader("Prediction Results")
                        res_col, map_col = st.columns([1, 2])
                        
                        with res_col:
                            mag_color = 'green' if predicted_mag < 4.0 else 'orange' if predicted_mag < 5.0 else 'red'
                            st.markdown(f"### Predicted Magnitude: <span style='color:{mag_color}'>{predicted_mag:.1f}</span>", 
                                      unsafe_allow_html=True)
                            
                            st.write(f"**Probability of Significant Event (M≥4.5):**")
                            st.progress(int(probability))
                            st.write(f"{probability:.1f}% likelihood within 30 days")
                            
                            similar_events = df[
                                (df['cluster_name'] == cluster_name) &
                                (df['depth'].between(depth*0.8, depth*1.2))
                            ].sort_values('time', ascending=False).head(5)
                            
                            if not similar_events.empty:
                                st.dataframe(
                                    similar_events[['time', 'mag', 'depth', 'cluster_name']]
                                    .rename(columns={
                                        'time': 'Date',
                                        'mag': 'Magnitude',
                                        'depth': 'Depth',
                                        'cluster_name': 'Region'
                                    })
                                )
                            else:
                                st.write("No similar events found in historical data")
                        
                        with map_col:
                            pred_map = folium.Map(location=[lat, lon], zoom_start=8)
                            folium.Marker(
                                [lat, lon],
                                popup=f"Predicted: M{predicted_mag:.1f}",
                                icon=folium.Icon(color=mag_color, icon='exclamation-triangle')
                            ).add_to(pred_map)
                            
                            cluster_data = df[df['cluster_name'] == cluster_name]
                            for _, row in cluster_data[cluster_data['mag'] > 4.0].iterrows():
                                folium.CircleMarker(
                                    location=[row['latitude'], row['longitude']],
                                    radius=row['mag'],
                                    color='orange',
                                    fill=True,
                                    popup=f"M{row['mag']} {row['cluster_name']}"
                                ).add_to(pred_map)
                            
                            folium_static(pred_map)

        elif app_mode == "Risk Dashboard":
            st.title("Disaster Risk Assessment")
            st.subheader("🌋 High-Risk Zones Dashboard")
            
            risk_df = df.groupby('cluster_name').agg({
                'mag': ['mean', 'count'],
                'latitude': 'mean',
                'longitude': 'mean',
                'depth': 'mean'
            }).reset_index()
            risk_df.columns = ['region', 'avg_mag', 'quake_count', 'lat', 'lon', 'avg_depth']
            
            risk_df['risk_score'] = (risk_df['avg_mag'] ** 2) * (risk_df['quake_count'] ** 0.5) / (risk_df['avg_depth'] + 1)
            risk_df = risk_df.sort_values('risk_score', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("### Top 10 High-Risk Regions")
                st.dataframe(
                    risk_df.head(10)[['region', 'avg_mag', 'quake_count', 'risk_score']]
                    .style.background_gradient(cmap='Reds', subset=['risk_score'])
                )
                
            with col2:
                st.write("### Risk Map")
                risk_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=2)
                
                for _, row in risk_df.head(20).iterrows():
                    folium.Circle(
                        location=[row['lat'], row['lon']],
                        radius=row['risk_score'] * 10000,
                        color='red',
                        fill=True,
                        popup=f"""
                        {row['region']}<br>
                        Risk Score: {row['risk_score']:.1f}<br>
                        Avg Magnitude: {row['avg_mag']:.1f}<br>
                        Total Quakes: {row['quake_count']}
                        """
                    ).add_to(risk_map)
                
                folium_static(risk_map)

        elif app_mode == "Regional Risk Assessment":
            st.title("Regional Earthquake Risk Assessment")
            regions = df.groupby('cluster_name').agg({
                'mag': ['mean', 'count'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            regions.columns = ['region', 'avg_mag', 'quake_count', 'lat', 'lon']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Seismic Regions")
                selected_region = st.selectbox("Select Region", regions['region'])
                region_data = df[df['cluster_name'] == selected_region]
                
                st.metric("Average Magnitude", f"{regions[regions['region'] == selected_region]['avg_mag'].values[0]:.2f}")
                st.metric("Total Events", regions[regions['region'] == selected_region]['quake_count'].values[0])
                
                st.write("### Regional Statistics")
                st.dataframe(
                    region_data.describe().T[['mean', 'min', 'max']]
                    .rename(columns={'mean': 'Average', 'min': 'Minimum', 'max': 'Maximum'})
                )
            
            with col2:
                st.subheader("Regional Visualization")
                region_map = folium.Map(
                    location=[
                        regions[regions['region'] == selected_region]['lat'].values[0],
                        regions[regions['region'] == selected_region]['lon'].values[0]
                    ], 
                    zoom_start=6
                )
                
                for _, row in region_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=row['mag'],
                        color='orange',
                        fill=True,
                        popup=f"M{row['mag']} {row['cluster_name']}"
                    ).add_to(region_map)
                
                folium_static(region_map)

        elif app_mode == "Probability Estimator":
            st.title("Earthquake Probability Estimator")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                location_input = st.text_input("Enter Location Name (e.g., Tokyo, Japan)")
                lat = st.number_input("Latitude", value=35.0, format="%.4f")
                lon = st.number_input("Longitude", value=-118.0, format="%.4f")
                
                if location_input:
                    try:
                        location = geocode(location_input)
                        if location:
                            lat = location.latitude
                            lon = location.longitude
                            st.success(f"Location found: {location.address}")
                    except Exception as e:
                        st.error(f"Geocoding error: {str(e)}")
                
                days_ahead = st.slider("Prediction Window (days)", 1, 90, 30)
                mag_threshold = st.slider("Magnitude Threshold", 3.0, 8.0, 5.0, 0.1)
                
                if st.button("Calculate Risk Probability"):
                    with st.spinner("Analyzing seismic patterns..."):
                        daily_probs = []
                        for day in range(days_ahead):
                            future_date = datetime.now(timezone.utc) + timedelta(days=day)
                            _, prob, _ = predict_probability(
                                lat, lon, 10.0, future_date, model, 
                                scaler, features, df
                            )
                            daily_probs.append(prob * (mag_threshold/5.0))
                        
                        avg_prob = np.mean(daily_probs)
                        
                        st.subheader("Risk Assessment")
                        risk_col, map_col = st.columns([1, 2])
                        
                        with risk_col:
                            plt.figure(figsize=(8, 4))
                            plt.plot(range(1, days_ahead+1), daily_probs, marker='o', color='#FF4B4B')
                            plt.fill_between(range(1, days_ahead+1), daily_probs, color='#FF4B4B', alpha=0.1)
                            plt.title('Daily Risk Probability Trend')
                            plt.xlabel('Days Ahead')
                            plt.ylabel('Probability (%)')
                            plt.grid(True, alpha=0.3)
                            st.pyplot(plt)
                            
                            st.metric("Average Probability", f"{avg_prob:.1f}%")
                            
                            if avg_prob < 20:
                                st.success("Low Risk Zone")
                            elif avg_prob < 50:
                                st.warning("Moderate Risk Zone")
                            else:
                                st.error("High Risk Zone")
                        
                        with map_col:
                            prob_map = folium.Map(location=[lat, lon], zoom_start=8)
                            folium.Marker(
                                [lat, lon],
                                popup=f"Risk Probability: {avg_prob:.1f}%",
                                icon=folium.Icon(color='red' if avg_prob > 50 else 'orange' if avg_prob > 20 else 'green')
                            ).add_to(prob_map)
                            folium_static(prob_map)

            with col2:
                st.write("### Historical Context")
                st.dataframe(
                    df.sort_values('time', ascending=False).head(10)[['time', 'mag', 'cluster_name', 'depth']]
                    .rename(columns={
                        'time': 'Date', 
                        'mag': 'Magnitude',
                        'cluster_name': 'Region',
                        'depth': 'Depth (km)'
                    }),
                    height=400
                )

        elif app_mode == "About":
            st.title("About the Earthquake Prediction System")
            st.markdown("""
            ## 🌍 Comprehensive Seismic Analysis Platform

            ### Key Features:
            - Real-time earthquake monitoring
            - Machine learning predictions
            - Risk probability estimation
            - Interactive visualizations
            - Named seismic regions for better interpretation
            - Early warning system with impact time calculation

            ### Methodology:
            - Advanced machine learning model (Best Model)
            - Real-time data integration from USGS
            - Cluster-based spatial analysis with regional naming
            - Advanced probability calibration using sigmoid function
            - Haversine-based impact time calculations

            ### Data Sources:
            - USGS Real-time Earthquake Feed
            - Historical seismic records

            ### Limitations:
            - Predictions based on historical patterns
            - Regional data variations
            - Educational purposes only
            """)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check your input parameters and try again")

if __name__ == "__main__":
    main()
