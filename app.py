import streamlit as st
import pandas as pd
import joblib
import os
from io import BytesIO
import requests

# --- Model Loading with Error Handling ---
@st.cache_resource
def load_model():
    MODEL_PATH = "gwamz_stream_predictor_v2.pkl"
    
    # Check if file exists locally
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Local model loading failed: {str(e)}")
    
    # Fallback: Load from URL (for demo purposes)
    try:
        model_url = "https://github.com/yourusername/yourrepo/raw/main/gwamz_stream_predictor_v2.pkl"
        response = requests.get(model_url)
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"Remote model loading failed: {str(e)}")
        return None

model = load_model()

# --- App UI ---
st.set_page_config(page_title="Gwamz Predictor", layout="wide")
st.title("🎵 Gwamz Song Performance Predictor")

if model:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            release_year = st.slider("Release Year", 2021, 2025, 2024)
            track_popularity = st.slider("Track Popularity (0-100)", 0, 100, 50)
            
        with col2:
            album_type = st.selectbox("Album Type", ["single", "album"])
            is_explicit = st.checkbox("Explicit Content")
        
        if st.form_submit_button("Predict Streams"):
            input_data = pd.DataFrame({
                'artist_followers': [7937],
                'artist_popularity': [41],
                'release_year': [release_year],
                'total_tracks_in_album': [1 if album_type == "single" else 10],
                'available_markets_count': [185],
                'track_popularity': [track_popularity],
                'release_month': [6],  # Default month
                'is_single': [1 if album_type == "single" else 0],
                'is_explicit': [1 if is_explicit else 0],
                'album_type': [album_type]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Streams: {int(prediction):,}")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
else:
    st.warning("Model not loaded - check error messages above")

# --- File Verification Section ---
st.sidebar.header("Debug Info")
if st.sidebar.checkbox("Show file status"):
    st.sidebar.write("Current directory contents:")
    st.sidebar.write(os.listdir("."))
    
    if st.sidebar.button("Verify model file"):
        if os.path.exists("gwamz_stream_predictor_v2.pkl"):
            st.sidebar.success("Model file found!")
            try:
                test_load = joblib.load("gwamz_stream_predictor_v2.pkl")
                st.sidebar.success("Model loads successfully")
            except Exception as e:
                st.sidebar.error(f"Load test failed: {str(e)}")
        else:
            st.sidebar.error("Model file NOT found")
