import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from src.feature_engineering import create_features

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return joblib.load('model.pkl')
    except FileNotFoundError:
        st.error("Model not found! Please run train.py first.")
        return None

def predict_price(model, input_data):
    """Your prediction pipeline"""
    # Create features using same pipeline as training
    features_df = create_features(input_data)
    
    # Select same features as training
    feature_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude',
        'rooms_per_household', 'bedrooms_per_room', 
        'population_per_household', 'lat_long_interaction'
    ]
    
    features_selected = features_df[feature_cols]
    return model.predict(features_selected)[0]

def main():
    st.title("🏠 California House Price Predictor")
    st.markdown("### Predict house prices using your ML pipeline!")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create input form
    st.header("Step 1: Enter House Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Economic Factors")
        med_inc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 5.0, 0.1)
        
        st.subheader("House Details")  
        house_age = st.slider("House Age (years)", 1, 52, 20)
        ave_rooms = st.slider("Average Rooms", 1.0, 20.0, 6.0, 0.1)
        ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
    
    with col2:
        st.subheader("Population Details")
        population = st.slider("Population", 3, 35682, 3000)
        ave_occup = st.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)
    
    with col3:
        st.subheader("Location")
        latitude = st.slider("Latitude", 32.54, 41.95, 34.0, 0.01)
        longitude = st.slider("Longitude", -124.35, -114.31, -118.0, 0.01)
    
    # Create prediction button
    if st.button("🔮 Predict House Price", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'MedInc': [med_inc],
            'HouseAge': [house_age], 
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        try:
            # Make prediction
            predicted_price = predict_price(model, input_data)
            
            # Display results
            st.header("Step 2: Prediction Results")
            st.success(f"🏠 **Predicted House Price: ${predicted_price:,.2f}**")
            
            # Show input summary
            with st.expander("📊 Input Summary"):
                st.write(input_data)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check that your feature engineering pipeline matches the training pipeline.")

# This is the missing piece - add this at the very end of your file!
if __name__ == "__main__":
    main()