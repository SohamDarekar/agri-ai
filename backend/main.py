from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List
import joblib
import numpy as np
import json
from PIL import Image
import io
import tensorflow as tf
from weather import get_seasonal_weather_averages
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Agriculture AI Backend")

# Add CORS middleware for both localhost and public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"  # This allows all origins, you can replace with your public domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Helper function to load TFLite models ---
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# --- Load All Final Models and Artifacts ---
try:
    # --- Crop Recommendation Models ---
    crop_interpreter = load_tflite_model('models/crop_recommender_float16.tflite')
    crop_scaler = joblib.load('models/crop_data_scaler.pkl')
    crop_soil_encoder = joblib.load('models/crop_soil_encoder.pkl')
    crop_label_encoder = joblib.load('models/crop_label_encoder.pkl')

    # --- Yield Prediction Models ---
    yield_interpreter = load_tflite_model('models/yield_predictor_float16.tflite')
    yield_scaler = joblib.load('models/yield_scaler.pkl')
    yield_crop_encoder = joblib.load('models/yield_crop_encoder.pkl')

    # --- Disease Detection Models ---
    disease_interpreter = load_tflite_model('models/disease_detector_float32.tflite')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    disease_class_names = {v: k for k, v in class_indices.items()}

    # --- Load Calculator Knowledge Base ---
    with open('crop_data.json', 'r') as f:
        crop_details = json.load(f)

    print("✅ All TFLite models and artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models or artifacts: {e}")
    crop_interpreter = yield_interpreter = disease_interpreter = crop_details = None

# --- Pydantic Models ---
class EnrichedSoilData(BaseModel):
    nitrogen: float; phosphorus: float; potassium: float; ph: float; season: str; soil_type: str

class YieldPredictionData(BaseModel):
    nitrogen: float; phosphorus: float; potassium: float; ph: float; season: str; crop: str

class CalculatorData(BaseModel):
    nitrogen: float; phosphorus: float; potassium: float; ph: float; season: str; crop: str; state: str; district: str

# --- Helper Function for Seasonally-Aware Fallback Price API ---
async def get_recent_average_price(state: str, district: str, api_crop_names: List[str]):
    api_key = os.getenv("API_GOV_KEY")
    resource_id = "42823128-a8ed-4434-86a7-a931346a3625"
    
    for crop_name in api_crop_names:
        print(f"INFO: [Fallback] Attempting seasonal average for '{crop_name}'...")
        # Fetch a larger dataset to span multiple years
        api_url = (f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&limit=3000&" 
                   f"filters[State]={state}&filters[Commodity]={crop_name}")
        
        try:
            response = requests.get(api_url, timeout=20)
            response.raise_for_status()
            records = response.json().get('records', [])

            if not records:
                print(f"INFO: [Fallback] No historical data found for '{crop_name}' in {state}.")
                continue

            df = pd.DataFrame(records)
            df['Price_Date'] = pd.to_datetime(df['Price_Date'], errors='coerce')
            df.dropna(subset=['Price_Date'], inplace=True)

            # --- SEASONAL FILTERING LOGIC ---
            current_month = datetime.now().month
            # Create a window of 3 months (current, previous, next) to capture the season
            seasonal_months = [(current_month - 2) % 12 + 1, current_month, (current_month % 12) + 1]
            
            seasonal_df = df[df['Price_Date'].dt.month.isin(seasonal_months)]

            if seasonal_df.empty:
                print(f"INFO: [Fallback] No data found in the current season for '{crop_name}'.")
                continue

            prices = pd.to_numeric(seasonal_df['Modal_Price'], errors='coerce')
            prices = prices[prices > 0]
            
            if not prices.empty:
                average_price = int(prices.mean())
                print(f"INFO: [Fallback] Success! Calculated seasonal average price for '{crop_name}': {average_price}")
                return {'minPrice': average_price, 'maxPrice': average_price}

        except requests.exceptions.RequestException as e:
            print(f"WARN: [Fallback] API request for '{crop_name}' failed: {e}")
            continue
            
    print("INFO: [Fallback] All API names failed for seasonal average.")
    return None

# --- API Endpoints ---
@app.get("/api/crops")
def get_available_crops():
    """Return list of all available crops from crop_data.json"""
    if not crop_details:
        raise HTTPException(status_code=500, detail="Crop data not loaded.")
    
    crops = []
    for crop_key, crop_data in crop_details.items():
        if crop_key != 'Default':  # Exclude the default entry
            crops.append({
                'value': crop_key,
                'label': crop_key.replace('_', ' ').title(),
                'api_names': crop_data.get('api_names', []),
                'cost_per_hectare': crop_data.get('estimated_cost_per_hectare', 0),
                'yield_range': crop_data.get('yield_tons_per_hectare_range', [0, 0]),
                'sustainability': crop_data.get('sustainability', {})
            })
    
    return {'crops': sorted(crops, key=lambda x: x['label'])}

@app.get("/api/prices")
async def get_mandi_prices(state: str, district: str, crop: str):
    crop_key = crop.lower()
    crop_info = crop_details.get(crop_key, crop_details['Default'])
    api_crop_names = crop_info.get('api_names', [crop])

    api_key = os.getenv("API_GOV_KEY")
    if not api_key: raise HTTPException(status_code=500, detail="API key is not configured.")
    
    # LEVEL 1: LIVE API
    source = "Live API"
    resource_id_live = '9ef84268-d588-465a-a308-a864a43d0070'
    records = []

    for crop_name in api_crop_names:
        print(f"INFO: [Live] Checking for '{crop_name}'...")
        api_url_live = (f"https://api.data.gov.in/resource/{resource_id_live}?api-key={api_key}&format=json&"
                        f"filters[state]={state}&filters[district]={district}&filters[commodity]={crop_name}")
        try:
            response = requests.get(api_url_live, timeout=10)
            response.raise_for_status()
            live_records = response.json().get('records', [])
            if live_records:
                records = live_records
                print(f"INFO: [Live] Success! Found live price for '{crop_name}'")
                break
        except requests.exceptions.RequestException:
            continue

    # LEVEL 2: SEASONALLY-AWARE RECENT AVERAGE API
    if not records:
        source = "Recent Average API"
        price_data = await get_recent_average_price(state, district, api_crop_names)
        if price_data:
             records = [{'min_price': price_data['minPrice'], 'max_price': price_data['maxPrice']}]

    if not records:
        return {"message": f"No market price data found for {crop} in {state}."}

    min_price = int(pd.to_numeric(records[0].get('min_price'), errors='coerce'))
    max_price = int(pd.to_numeric(records[0].get('max_price'), errors='coerce'))

    return {'crop': crop, 'state': state, 'district': district, 'minPrice': min_price, 'maxPrice': max_price, 'market': records[0].get('market', 'N/A'), 'dataSource': source}

@app.post("/recommend-crop")
def recommend_crop(data: EnrichedSoilData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not all([crop_interpreter, crop_scaler, crop_soil_encoder, crop_label_encoder]):
        raise HTTPException(status_code=500, detail="Crop model components are not loaded.")
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        numerical_data = pd.DataFrame([{'N': data.nitrogen, 'P': data.phosphorus, 'K': data.potassium, 'temperature': weather_data['temperature'], 'humidity': weather_data['humidity'], 'ph': data.ph, 'rainfall': weather_data['rainfall']}])
        categorical_data = pd.DataFrame({'soil_type': [data.soil_type]})
        scaled_numerical = crop_scaler.transform(numerical_data)
        encoded_categorical = crop_soil_encoder.transform(categorical_data)
        processed_input = np.hstack([scaled_numerical, encoded_categorical]).astype(np.float32)
        input_details = crop_interpreter.get_input_details()
        output_details = crop_interpreter.get_output_details()
        crop_interpreter.set_tensor(input_details[0]['index'], processed_input)
        crop_interpreter.invoke()
        probabilities = crop_interpreter.get_tensor(output_details[0]['index'])[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = []
        for index in top_3_indices:
            crop_name = crop_label_encoder.inverse_transform([index])[0]
            confidence = probabilities[index]
            recommendations.append({"crop": str(crop_name), "confidence": round(float(confidence), 4)})
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-yield")
def predict_yield(data: YieldPredictionData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not all([yield_interpreter, yield_scaler, yield_crop_encoder]):
        raise HTTPException(status_code=500, detail="Yield model components are not loaded.")
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        numerical_data = pd.DataFrame([{'N': data.nitrogen, 'P': data.phosphorus, 'K': data.potassium, 'temperature': weather_data['temperature'], 'humidity': weather_data['humidity'], 'ph': data.ph, 'rainfall': weather_data['rainfall']}])
        categorical_data = pd.DataFrame({'label': [data.crop]})
        scaled_numerical = yield_scaler.transform(numerical_data)
        encoded_categorical = yield_crop_encoder.transform(categorical_data)
        processed_input = np.hstack([scaled_numerical, encoded_categorical]).astype(np.float32)
        input_details = yield_interpreter.get_input_details()
        output_details = yield_interpreter.get_output_details()
        yield_interpreter.set_tensor(input_details[0]['index'], processed_input)
        yield_interpreter.invoke()
        prediction = yield_interpreter.get_tensor(output_details[0]['index'])[0]
        return {"predicted_crop": data.crop, "estimated_yield_tons_per_hectare": round(float(prediction[0]), 2), "weather_data_used": weather_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if not disease_interpreter:
        raise HTTPException(status_code=500, detail="Disease detection model not loaded.")
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        input_details = disease_interpreter.get_input_details()
        output_details = disease_interpreter.get_output_details()
        disease_interpreter.set_tensor(input_details[0]['index'], img_array)
        disease_interpreter.invoke()
        prediction = disease_interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = disease_class_names.get(int(predicted_class_index), "Unknown")
        confidence = float(np.max(prediction))
        return {"predicted_disease": predicted_class_name.replace("_", " "), "confidence": f"{confidence:.2f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-profit-sustainability")
async def calculate_metrics(data: CalculatorData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not all([yield_interpreter, crop_details]):
        raise HTTPException(status_code=500, detail="Model or crop data not loaded.")
    try:
        # --- 1. PREDICT YIELD ---
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        numerical_data = pd.DataFrame([{'N': data.nitrogen, 'P': data.phosphorus, 'K': data.potassium, 'temperature': weather_data['temperature'], 'humidity': weather_data['humidity'], 'ph': data.ph, 'rainfall': weather_data['rainfall']}])
        categorical_data = pd.DataFrame({'label': [data.crop]})
        scaled_numerical = yield_scaler.transform(numerical_data)
        encoded_categorical = yield_crop_encoder.transform(categorical_data)
        processed_input = np.hstack([scaled_numerical, encoded_categorical]).astype(np.float32)
        input_details = yield_interpreter.get_input_details()
        output_details = yield_interpreter.get_output_details()
        yield_interpreter.set_tensor(input_details[0]['index'], processed_input)
        yield_interpreter.invoke()
        prediction = yield_interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_yield_tons_ha = round(float(prediction[0]), 2)

        # --- 2. GET MARKET PRICE ---
        price_response = await get_mandi_prices(data.state, data.district, data.crop)
        
        if 'message' in price_response:
            return {"crop": data.crop, "predicted_yield_tons_per_hectare": predicted_yield_tons_ha, "message": price_response['message']}
        
        avg_market_price_quintal = (price_response['minPrice'] + price_response['maxPrice']) / 2

        # --- 3. GET COSTS & SUSTAINABILITY DATA ---
        crop_key = data.crop.lower()
        details = crop_details.get(crop_key, crop_details.get('Default'))
        estimated_cost_ha = details['estimated_cost_per_hectare']
        sustainability_ratings = details['sustainability']

        # --- 4. CALCULATE PROFIT ---
        total_revenue_ha = predicted_yield_tons_ha * 10 * avg_market_price_quintal
        net_profit_ha = total_revenue_ha - estimated_cost_ha
        
        # --- 5. CALCULATE SUSTAINABILITY SCORE ---
        water = (10 - sustainability_ratings['water_usage_rating']) * 0.4
        pesticide = (10 - sustainability_ratings['pesticide_rating']) * 0.4
        soil = (10 - sustainability_ratings['soil_health_impact']) * 0.2
        sustainability_score = round(water + pesticide + soil, 1)

        return {
            "crop": data.crop,
            "predicted_yield_tons_per_hectare": predicted_yield_tons_ha,
            "avg_market_price_per_quintal": avg_market_price_quintal,
            "estimated_total_revenue_per_hectare": round(total_revenue_ha),
            "estimated_input_cost_per_hectare": estimated_cost_ha,
            "estimated_net_profit_per_hectare": round(net_profit_ha),
            "sustainability_score_out_of_10": sustainability_score,
            "priceDataSource": price_response.get('dataSource', 'Unknown')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during calculation: {str(e)}")