# profit_model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

print("--- Starting Profit Model Training ---")

# --- 1. Load the base dataset ---
# We use the original crop recommendation data as a foundation.
df = pd.read_csv('Crop_recommendation.csv')

# --- 2. Use Real-World Yield and Profit Data ---
# This data is based on averages from Indian agricultural reports (e.g., NABARD, AgStats).
# Yield is in tons per hectare.
crop_yield_data = {
    # Grains & Pulses (Kharif & Rabi crops common in Maharashtra)
    'rice': 3.8,           # Avg. for Maharashtra is lower than national avg.
    'maize': 3.5,          # Common in Maharashtra
    'chickpea': 0.9,       # Also known as Gram, a major Rabi crop
    'kidneybeans': 1.2,    # Rajma, grown in specific regions
    'pigeonpeas': 0.8,     # Tur dal, very important crop in Maharashtra
    'mothbeans': 0.5,      # Matki, drought-resistant and common
    'mungbean': 0.6,       # Moong dal
    'blackgram': 0.7,      # Urad dal
    'lentil': 0.8,         # Masoor dal

    # Commercial Crops
    'cotton': 1.5,         # Major cash crop in Vidarbha region
    'jute': 2.2,           # Less common in MH, but we'll use national avg.
    'coffee': 0.7,         # Grown in specific hilly regions of India

    # Fruits (Horticulture is strong in Maharashtra)
    'pomegranate': 10.5,   # Maharashtra is a leading producer
    'banana': 60.0,        # Jalgaon region is famous for this
    'mango': 7.5,          # Alphonso from Konkan region
    'grapes': 22.0,        # Nashik is the "Grape Capital of India"
    'orange': 15.0,        # Nagpur is the "Orange City"
    'papaya': 40.0,
    'coconut': 9.0,        # Common in coastal Konkan region
    'apple': 8.0,          # Not typically grown in MH, using national avg.
    'watermelon': 25.0,
    'muskmelon': 20.0,
}

# This function applies the base yield and adds some random noise to make the data more realistic.
def synthesize_yield(row):
    crop = row['label']
    base_yield = crop_yield_data.get(crop, 0) # Use .get for safety
    # Add random variation of +/- 15%
    yield_variation = base_yield * np.random.uniform(-0.15, 0.15)
    return round(base_yield + yield_variation, 2)

df['yield'] = df.apply(synthesize_yield, axis=1)
print("Step 1: Synthesized dataset created successfully.")

# --- 3. Prepare Data for Yield Prediction Model ---
# The 'features' (X) are the environmental conditions.
# The 'target' (y) is the numerical yield we want to predict.
X = df.drop(['label', 'yield'], axis=1)
y = df['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Step 2: Data prepared for training.")

# --- 4. Train the Gradient Boosting Regressor Model ---
# This is a powerful algorithm for regression tasks (predicting a continuous number).
profit_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
print("Step 3: Training the model...")
profit_model.fit(X_train, y_train)
print("         ...Training complete.")

# --- 5. Evaluate and Save the Model ---
score = profit_model.score(X_test, y_test)
print(f"Step 4: Model evaluation R^2 Score: {score:.2f}")

model_filename = 'models/profit_forecaster.pkl'
joblib.dump(profit_model, model_filename)
print(f"Step 5: Model successfully saved to '{model_filename}'")
print("--- Profit Model Training Finished ---")