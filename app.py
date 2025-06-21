import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# Custom unpickler function
def load_model_safely():
    try:
        model = joblib.load('gwamz_stream_predictor_v2.pkl')
        # Verify it's a pipeline
        if isinstance(model, Pipeline):
            return model
        else:
            st.error("Loaded object is not a scikit-learn Pipeline")
            return None
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load model with cache
model = st.cache_resource(load_model_safely)()

# App UI
st.title("🎵 Gwamz Song Performance Predictor")

if model:
    with st.form("prediction_form"):
        st.subheader("Release Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            release_year = st.slider("Year", 2021, 2025, 2024)
            track_popularity = st.slider("Popularity (0-100)", 0, 100, 50)
            is_explicit = st.checkbox("Explicit Content")
            
        with col2:
            album_type = st.selectbox("Album Type", ["single", "album"])
            available_markets = st.slider("Available Markets", 1, 200, 185)
            
        if st.form_submit_button("Predict Streams"):
            input_data = pd.DataFrame({
                'artist_followers': [7937],
                'artist_popularity': [41],
                'release_year': [release_year],
                'total_tracks_in_album': [1],
                'available_markets_count': [available_markets],
                'track_popularity': [track_popularity],
                'release_month': [6],  # Default to June
                'is_single': [1 if album_type == "single" else 0],
                'is_explicit': [1 if is_explicit else 0],
                'album_type': [album_type]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Streams: {int(prediction):,}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
else:
    st.warning("Model not loaded - check error messages above")
