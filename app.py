import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        # Try local file first
        return joblib.load('gwamz_stream_predictor.pkl')
    except:
        st.error("Model file not found. Please ensure gwamz_stream_predictor.pkl is in the same directory.")
        return None

model = load_model()

# App UI
st.set_page_config(page_title="Gwamz Song Predictor", layout="wide")
st.title("🎵 Gwamz Song Performance Predictor")

# Input form
with st.form("prediction_form"):
    st.subheader("Release Details")
    col1, col2 = st.columns(2)
    
    with col1:
        release_year = st.slider("Release Year", 2021, 2025, 2024)
        release_month = st.slider("Release Month", 1, 12, 6)
        track_popularity = st.slider("Track Popularity (0-100)", 0, 100, 50)
        
    with col2:
        album_type = st.selectbox("Album Type", ["single", "album", "compilation"])
        version_type = st.selectbox("Version Type", ["original", "sped_up", "remix"])
        is_explicit = st.checkbox("Explicit Content")
    
    submit_button = st.form_submit_button("Predict Streams")

# Prediction logic
if submit_button and model:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'artist_followers': [7937],
            'artist_popularity': [41],
            'release_year': [release_year],
            'total_tracks_in_album': [1],
            'available_markets_count': [185],
            'track_number': [1],
            'disc_number': [1],
            'track_popularity': [track_popularity],
            'release_month': [release_month],
            'days_since_first_release': [(pd.to_datetime(f'{release_year}-{release_month}-01') - pd.to_datetime('2021-04-29')).days],
            'is_single': [1 if album_type == 'single' else 0],
            'is_explicit': [1 if is_explicit else 0],
            'album_type': [album_type]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"Predicted Streams: {int(prediction):,}")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.barh(['Prediction'], [prediction], color='skyblue')
        ax.set_xlabel('Streams')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
elif submit_button and not model:
    st.error("Model not loaded - cannot make predictions")
