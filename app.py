# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('gwamz_stream_predictor.pkl')

model = load_model()

# App title and description
st.set_page_config(page_title="Gwamz Song Performance Predictor", layout="wide")
st.title("🎵 Gwamz Song Performance Predictor")
st.markdown("""
This app predicts the expected streams for Gwamz's new songs based on historical data.
Adjust the parameters below to simulate different release scenarios.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Create input fields
def user_input_features():
    # Basic info
    release_year = st.sidebar.slider('Release Year', 2021, 2025, 2024)
    release_month = st.sidebar.slider('Release Month', 1, 12, 6)
    track_popularity = st.sidebar.slider('Track Popularity (0-100)', 0, 100, 50)
    is_explicit = st.sidebar.selectbox('Explicit Content', ('Yes', 'No'))
    
    # Album info
    album_type = st.sidebar.selectbox('Album Type', ('single', 'album', 'compilation'))
    total_tracks_in_album = st.sidebar.slider('Total Tracks in Album', 1, 20, 1)
    track_number = st.sidebar.slider('Track Number in Album', 1, total_tracks_in_album, 1)
    disc_number = st.sidebar.slider('Disc Number', 1, 5, 1)
    
    # Version info
    version_type = st.sidebar.selectbox('Version Type', 
                                       ('original', 'sped_up', 'remix', 'instrumental', 'jersey'))
    
    # Market info
    available_markets_count = st.sidebar.slider('Available Markets Count', 1, 200, 185)
    
    # Create a DataFrame from inputs
    data = {
        'artist_followers': 7937,
        'artist_popularity': 41,
        'release_year': release_year,
        'total_tracks_in_album': total_tracks_in_album,
        'available_markets_count': available_markets_count,
        'track_number': track_number,
        'disc_number': disc_number,
        'explicit': is_explicit == 'Yes',
        'track_popularity': track_popularity,
        'album_type': album_type,
        'version_type': version_type,
        'release_month': release_month,
        'days_since_first_release': (pd.to_datetime(f'{release_year}-{release_month}-01') - pd.to_datetime('2021-04-29')).days,
        'is_single': int(album_type == 'single'),
        'is_explicit': int(is_explicit == 'Yes')
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Make prediction
if st.button('Predict Streams'):
    prediction = model.predict(input_df)
    
    st.subheader('Prediction')
    st.markdown(f"### Predicted Streams: **{int(prediction[0]):,}**")
    
    # Add context with historical data comparison
    st.subheader('Historical Context')
    
    # Load the original data for comparison
    original_data = pd.read_csv('gwamz_data.csv')
    
    # Calculate percentiles
    percentile = np.percentile(original_data['streams'], 
                              (prediction[0] / original_data['streams'].max()) * 100)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minimum Streams in History", f"{original_data['streams'].min():,}")
    with col2:
        st.metric("Average Streams in History", f"{original_data['streams'].mean():,.0f}")
    with col3:
        st.metric("Maximum Streams in History", f"{original_data['streams'].max():,}")
    
    # Visual comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(original_data['streams'], bins=20, alpha=0.7, label='Historical Streams')
    ax.axvline(prediction[0], color='red', linestyle='--', linewidth=2, 
               label='Predicted Streams')
    ax.set_xlabel('Streams')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction vs Historical Performance')
    ax.legend()
    st.pyplot(fig)
    
    # Feature importance explanation
    st.subheader('Key Factors Influencing Prediction')
    
    # Get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    feature_names = numerical_features + list(cat_features)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
    
    st.write("Top 5 factors affecting this prediction:")
    for idx, row in importance_df.iterrows():
        feature_name = row['Feature']
        importance = row['Importance']
        
        # Human-readable explanations
        if feature_name == 'track_popularity':
            explanation = "Higher popularity scores strongly predict more streams."
        elif feature_name == 'release_year':
            explanation = "Newer releases tend to perform better, possibly due to growing fanbase."
        elif feature_name == 'version_type_original':
            explanation = "Original versions typically outperform remixes or sped-up versions."
        elif feature_name == 'available_markets_count':
            explanation = "More available markets means wider distribution potential."
        elif feature_name == 'is_explicit':
            explanation = "Explicit content can affect audience reach and streaming numbers."
        else:
            explanation = "Significant but complex relationship with streams."
        
        st.markdown(f"- **{feature_name}** (Impact: {importance:.1%}): {explanation}")

# Add some analytics
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")
show_historical = st.sidebar.checkbox("Show Historical Data Analysis")

if show_historical:
    st.subheader("Historical Data Analysis")
    
    # Load and prepare data
    historical_data = pd.read_csv('gwamz_data.csv')
    historical_data['release_date'] = pd.to_datetime(historical_data['release_date'], format='%d/%m/%Y')
    
    # Time series of streams
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    historical_data.groupby('release_date')['streams'].sum().plot(ax=ax1)
    ax1.set_title('Total Streams Over Time')
    ax1.set_xlabel('Release Date')
    ax1.set_ylabel('Streams')
    st.pyplot(fig1)
    
    # Version type comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    historical_data['version_type'] = historical_data['track_name'].apply(
        lambda x: 'original' if not any(kw in x for kw in ['Sped Up', 'Remix', 'Instrumental', 'Jersey']) 
        else next(kw.lower() for kw in ['Sped Up', 'Remix', 'Instrumental', 'Jersey'] if kw in x))
    historical_data.groupby('version_type')['streams'].mean().sort_values().plot(kind='barh', ax=ax2)
    ax2.set_title('Average Streams by Version Type')
    ax2.set_xlabel('Average Streams')
    st.pyplot(fig2)
    
    # Release year comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    historical_data.groupby('release_year')['streams'].sum().plot(kind='bar', ax=ax3)
    ax3.set_title('Total Streams by Release Year')
    ax3.set_xlabel('Release Year')
    ax3.set_ylabel('Total Streams')
    st.pyplot(fig3)

# Add footer
st.markdown("---")
st.markdown("""
**About This App**:  
This predictive model was trained on Gwamz's historical release data using a Random Forest algorithm.
Predictions are estimates based on patterns in past performance and may not account for all real-world factors.
""")