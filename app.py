# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('gwamz_streams_predictor.pkl')

# Set up the app
st.set_page_config(page_title="Gwamz Song Performance Predictor", layout="wide")

# App title and description
st.title("🎵 Gwamz Song Performance Predictor")
st.markdown("""
This app predicts the expected streams for Gwamz's new songs based on historical data.
Adjust the parameters below to see how different factors affect predicted performance.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Calculate days since first release (2021-04-29)
first_release_date = datetime(2021, 4, 29)

# User input fields
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        release_date = st.date_input("Release Date", datetime(2025, 6, 1))
        total_tracks_in_album = st.number_input("Total Tracks in Album", min_value=1, max_value=10, value=1)
        available_markets_count = st.number_input("Available Markets Count", min_value=1, max_value=200, value=185)
        track_number = st.number_input("Track Number", min_value=1, max_value=10, value=1)
        
    with col2:
        is_explicit = st.checkbox("Explicit Content", value=True)
        track_popularity = st.slider("Track Popularity (1-100)", min_value=1, max_value=100, value=45)
        track_version = st.selectbox("Track Version", 
                                   ["original", "sped up", "jersey club", "jersey edit", "instrumental", "new gen remix"])
    
    submit_button = st.form_submit_button("Predict Streams")

# When the user clicks the predict button
if submit_button:
    # Prepare the input data
    release_datetime = datetime.combine(release_date, datetime.min.time())
    days_since_first_release = (release_datetime - first_release_date).days
    
    input_data = pd.DataFrame({
        'release_year': [release_date.year],
        'release_month': [release_date.month],
        'release_day': [release_date.day],
        'release_day_of_week': [release_date.weekday()],
        'days_since_first_release': [days_since_first_release],
        'total_tracks_in_album': [total_tracks_in_album],
        'available_markets_count': [available_markets_count],
        'track_number': [track_number],
        'is_explicit': [1 if is_explicit else 0],
        'track_popularity': [track_popularity],
        'is_first_track': [1 if track_number == 1 else 0],
        'track_version': [track_version]
    })
    
    # Make prediction
    predicted_streams = model.predict(input_data)[0]
    
    # Display results
    st.success(f"### Predicted Streams: {int(predicted_streams):,}")
    
    # Add some interpretation
    st.subheader("Performance Insights")
    
    if predicted_streams > 2000000:
        st.markdown("🔥 **Excellent Potential!** This track is predicted to perform among Gwamz's top songs.")
    elif predicted_streams > 1000000:
        st.markdown("💪 **Strong Performance!** This track is expected to perform well above average.")
    elif predicted_streams > 500000:
        st.markdown("👍 **Good Potential!** This track should perform decently based on current parameters.")
    else:
        st.markdown("🤔 **Moderate Performance.** Consider optimizing release strategy or track features.")
    
    # Add some feature impact analysis
    st.subheader("Key Factors Affecting Prediction")
    
    factors = {
        'Track Popularity': f"{track_popularity}/100",
        'Release Date': release_date.strftime("%B %Y"),
        'Track Version': track_version,
        'Explicit Content': "Yes" if is_explicit else "No",
        'Album Position': f"Track {track_number} of {total_tracks_in_album}"
    }
    
    st.table(pd.DataFrame.from_dict(factors, orient='index', columns=['Value']))

# Add historical data visualization
st.markdown("---")
st.subheader("Historical Performance Data")

# You would need to load your actual historical data here
# For demonstration, we'll show some sample metrics
col1, col2, col3 = st.columns(3)
col1.metric("Highest Streams", "2,951,075", "Last Night (Original)")
col2.metric("Average Streams", "787,234", "All Tracks")
col3.metric("Lowest Streams", "8,473", "Last Night (Jersey Edit)")

# Add some tips for improving performance
st.markdown("---")
st.subheader("Tips to Improve Streams")
st.markdown("""
- 🎧 **Release original versions first**: Original tracks tend to perform better than remixes
- 📅 **Time your release**: Weekend releases (especially Friday) often perform better
- 🌍 **Maximize market availability**: More available markets = more potential streams
- 🔞 **Consider explicit content**: Explicit tracks often perform better for this artist
- 🎯 **Focus on track popularity**: Promote tracks to increase popularity score before release
""")
