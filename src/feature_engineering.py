import pandas as pd
import numpy as np

def create_features(df):
    """Step 4: Feature engineering - your modular approach"""
    return (df
        .assign(
            # Create meaningful features
            rooms_per_household=lambda x: x.AveRooms,
            bedrooms_per_room=lambda x: x.AveBedrms / x.AveRooms,
            population_per_room=lambda x: x.Population / x.AveRooms,
            price_per_room=lambda x: x.target / x.AveRooms,
            # Location features
            lat_long_interaction=lambda x: x.Latitude * x.Longitude,
            # Income per room (economic density)
            income_per_room=lambda x: x.MedInc / x.AveRooms
        )
    )

def select_features(df):
    """Step 5: Feature selection"""
    feature_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude',
        'rooms_per_household', 'bedrooms_per_room', 
        'population_per_room', 'lat_long_interaction', 'income_per_room'
    ]
    return df[feature_cols + ['target']]

def feature_pipeline(df):
    """Complete feature engineering pipeline"""
    return (df
        .pipe(create_features)
        .pipe(select_features)
    )