from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import joblib
import numpy as np
import json
from PIL import Image
import io
import tensorflow as tf
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Agriculture AI Backend",
    description="AI-powered agriculture recommendations and predictions",
    version="1.0.0"
)

# Add CORS middleware for both localhost and public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://krishi-sahayak.duckdns.org",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Global Variables for Models ---
crop_interpreter = None
crop_scaler = None
crop_soil_encoder = None
crop_label_encoder = None
yield_interpreter = None
yield_scaler = None
yield_crop_encoder = None
disease_interpreter = None
disease_class_names = None
crop_details = None

# --- Helper Functions ---
def get_seasonal_weather_averages(lat: float, lon: float, season: str):
    """
    Fallback weather data when actual weather service is unavailable.
    In production, this would call a real weather API.
    """
    logger.info(f"Getting weather data for lat: {lat}, lon: {lon}, season: {season}")
    
    # Default weather values based on season
    weather_defaults = {
        "kharif": {"temperature": 28.0, "humidity": 75.0, "rainfall": 150.0},
        "rabi": {"temperature": 22.0, "humidity": 60.0, "rainfall": 50.0},
        "zaid": {"temperature": 35.0, "humidity": 45.0, "rainfall": 25.0}
    }
    
    return weather_defaults.get(season.lower(), weather_defaults["kharif"])

def load_tflite_model(model_path):
    """Helper function to load TFLite models safely"""
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

def create_default_crop_data():
    """Create default crop data when crop_data.json is missing"""
    return {
        "Default": {
            "api_names": ["rice", "wheat", "maize"],
            "estimated_cost_per_hectare": 50000,
            "yield_tons_per_hectare_range": [2.0, 4.0],
            "sustainability": {
                "water_usage_rating": 5,
                "pesticide_rating": 5,
                "soil_health_impact": 5
            }
        },
        "rice": {
            "api_names": ["rice", "paddy"],
            "estimated_cost_per_hectare": 45000,
            "yield_tons_per_hectare_range": [3.0, 6.0],
            "sustainability": {
                "water_usage_rating": 8,
                "pesticide_rating": 4,
                "soil_health_impact": 3
            }
        },
        "wheat": {
            "api_names": ["wheat"],
            "estimated_cost_per_hectare": 40000,
            "yield_tons_per_hectare_range": [2.5, 5.0],
            "sustainability": {
                "water_usage_rating": 6,
                "pesticide_rating": 5,
                "soil_health_impact": 4
            }
        },
        "maize": {
            "api_names": ["maize", "corn"],
            "estimated_cost_per_hectare": 35000,
            "yield_tons_per_hectare_range": [4.0, 8.0],
            "sustainability": {
                "water_usage_rating": 5,
                "pesticide_rating": 6,
                "soil_health_impact": 5
            }
        }
    }

# --- Load All Models and Data ---
def initialize_models_and_data():
    """Initialize all models and data with proper error handling"""
    global crop_interpreter, crop_scaler, crop_soil_encoder, crop_label_encoder
    global yield_interpreter, yield_scaler, yield_crop_encoder
    global disease_interpreter, disease_class_names, crop_details
    
    try:
        # --- Crop Recommendation Models ---
        if os.path.exists('models/crop_recommender_float16.tflite'):
            crop_interpreter = load_tflite_model('models/crop_recommender_float16.tflite')
            if os.path.exists('models/crop_data_scaler.pkl'):
                crop_scaler = joblib.load('models/crop_data_scaler.pkl')
            if os.path.exists('models/crop_soil_encoder.pkl'):
                crop_soil_encoder = joblib.load('models/crop_soil_encoder.pkl')
            if os.path.exists('models/crop_label_encoder.pkl'):
                crop_label_encoder = joblib.load('models/crop_label_encoder.pkl')
            logger.info("✅ Crop recommendation models loaded")
        else:
            logger.warning("❌ Crop recommendation models not found")

        # --- Yield Prediction Models ---
        if os.path.exists('models/yield_predictor_float16.tflite'):
            yield_interpreter = load_tflite_model('models/yield_predictor_float16.tflite')
            if os.path.exists('models/yield_scaler.pkl'):
                yield_scaler = joblib.load('models/yield_scaler.pkl')
            if os.path.exists('models/yield_crop_encoder.pkl'):
                yield_crop_encoder = joblib.load('models/yield_crop_encoder.pkl')
            logger.info("✅ Yield prediction models loaded")
        else:
            logger.warning("❌ Yield prediction models not found")

        # --- Disease Detection Models ---
        if os.path.exists('models/disease_detector_float32.tflite'):
            disease_interpreter = load_tflite_model('models/disease_detector_float32.tflite')
            if os.path.exists('models/class_indices.json'):
                with open('models/class_indices.json', 'r') as f:
                    class_indices = json.load(f)
                disease_class_names = {v: k for k, v in class_indices.items()}
            logger.info("✅ Disease detection model loaded")
        else:
            logger.warning("❌ Disease detection model not found")

        # --- Load Crop Data ---
        if os.path.exists('crop_data.json'):
            with open('crop_data.json', 'r') as f:
                crop_details = json.load(f)
            logger.info("✅ Crop data loaded from file")
        else:
            crop_details = create_default_crop_data()
            logger.info("✅ Using default crop data")

        logger.info("✅ Model initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Error during model initialization: {e}")
        # Ensure crop_details is always available
        if crop_details is None:
            crop_details = create_default_crop_data()

# Initialize on startup
initialize_models_and_data()

# --- Pydantic Models ---
class EnrichedSoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    season: str
    soil_type: str

class YieldPredictionData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    season: str
    crop: str

class CalculatorData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    season: str
    crop: str
    state: str
    district: str

# --- Health Check Endpoints ---
@app.get("/")
async def root():
    """Root endpoint for basic health check"""
    return {
        "status": "healthy",
        "service": "Agriculture AI Backend",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "ok",
        "service": "Agriculture AI Backend",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "crop_recommendation": crop_interpreter is not None,
            "yield_prediction": yield_interpreter is not None,
            "disease_detection": disease_interpreter is not None,
            "crop_data": crop_details is not None
        },
        "environment": {
            "python_version": "3.12",
            "tensorflow_version": tf.__version__,
            "api_key_configured": bool(os.getenv("API_GOV_KEY"))
        }
    }

@app.get("/status")
async def status():
    """Simple status endpoint for load balancers"""
    return {"status": "up"}

# --- Helper Function for Price API ---
async def get_recent_average_price(state: str, district: str, api_crop_names: List[str]):
    """Get recent average prices with better error handling"""
    api_key = os.getenv("API_GOV_KEY")
    if not api_key:
        logger.warning("API_GOV_KEY not configured")
        return None
    
    resource_id = "42823128-a8ed-4434-86a7-a931346a3625"
    
    for crop_name in api_crop_names:
        logger.info(f"[Fallback] Attempting seasonal average for '{crop_name}'...")
        
        api_url = (
            f"https://api.data.gov.in/resource/{resource_id}?"
            f"api-key={api_key}&format=json&limit=1000&"
            f"filters[State]={state}&filters[Commodity]={crop_name}"
        )
        
        try:
            response = requests.get(api_url, timeout=15)
            response.raise_for_status()
            records = response.json().get('records', [])

            if not records:
                logger.info(f"[Fallback] No data found for '{crop_name}' in {state}")
                continue

            df = pd.DataFrame(records)
            df['Price_Date'] = pd.to_datetime(df['Price_Date'], errors='coerce')
            df.dropna(subset=['Price_Date'], inplace=True)

            # Seasonal filtering
            current_month = datetime.now().month
            seasonal_months = [
                (current_month - 2) % 12 + 1 if (current_month - 2) % 12 + 1 != 0 else 12,
                current_month,
                (current_month % 12) + 1
            ]
            
            seasonal_df = df[df['Price_Date'].dt.month.isin(seasonal_months)]

            if seasonal_df.empty:
                logger.info(f"[Fallback] No seasonal data for '{crop_name}'")
                continue

            prices = pd.to_numeric(seasonal_df['Modal_Price'], errors='coerce')
            prices = prices[prices > 0]
            
            if not prices.empty:
                average_price = int(prices.mean())
                logger.info(f"[Fallback] Success! Average price for '{crop_name}': {average_price}")
                return {'minPrice': average_price, 'maxPrice': average_price}

        except requests.exceptions.RequestException as e:
            logger.warning(f"[Fallback] API request failed for '{crop_name}': {e}")
            continue
            
    logger.info("[Fallback] No price data found")
    return None

# --- API Endpoints ---
@app.get("/api/crops")
def get_available_crops():
    """Return list of all available crops"""
    if not crop_details:
        # This should not happen due to initialization, but safety first
        return {"crops": [], "message": "Crop data not available"}
    
    crops = []
    for crop_key, crop_data in crop_details.items():
        if crop_key != 'Default':
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
    """Get market prices for crops"""
    if not crop_details:
        return {"message": "Crop data not available"}
        
    crop_key = crop.lower()
    crop_info = crop_details.get(crop_key, crop_details.get('Default', {}))
    api_crop_names = crop_info.get('api_names', [crop])

    api_key = os.getenv("API_GOV_KEY")
    if not api_key:
        # Return mock data when API key is not available
        return {
            "crop": crop,
            "state": state,
            "district": district,
            "minPrice": 2000,
            "maxPrice": 2500,
            "market": "Mock Data",
            "dataSource": "Default Prices (API Key Not Configured)"
        }
    
    # Try live API first
    source = "Live API"
    resource_id_live = '9ef84268-d588-465a-a308-a864a43d0070'
    records = []

    for crop_name in api_crop_names:
        logger.info(f"[Live] Checking for '{crop_name}'...")
        api_url_live = (
            f"https://api.data.gov.in/resource/{resource_id_live}?"
            f"api-key={api_key}&format=json&"
            f"filters[state]={state}&filters[district]={district}&filters[commodity]={crop_name}"
        )
        
        try:
            response = requests.get(api_url_live, timeout=10)
            response.raise_for_status()
            live_records = response.json().get('records', [])
            if live_records:
                records = live_records
                logger.info(f"[Live] Success! Found live price for '{crop_name}'")
                break
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Live] API failed for '{crop_name}': {e}")
            continue

    # Try fallback API if live API failed
    if not records:
        source = "Recent Average API"
        price_data = await get_recent_average_price(state, district, api_crop_names)
        if price_data:
            records = [{'min_price': price_data['minPrice'], 'max_price': price_data['maxPrice']}]

    # Return default prices if no API data found
    if not records:
        return {
            "crop": crop,
            "state": state,
            "district": district,
            "minPrice": 1800,
            "maxPrice": 2200,
            "market": "Default Estimate",
            "dataSource": "No API Data Available"
        }

    min_price = int(pd.to_numeric(records[0].get('min_price', 0), errors='coerce') or 0)
    max_price = int(pd.to_numeric(records[0].get('max_price', 0), errors='coerce') or 0)

    return {
        'crop': crop,
        'state': state,
        'district': district,
        'minPrice': min_price,
        'maxPrice': max_price,
        'market': records[0].get('market', 'N/A'),
        'dataSource': source
    }

@app.post("/recommend-crop")
def recommend_crop(data: EnrichedSoilData, lat: float = Query(19.07), lon: float = Query(72.87)):
    """Recommend crops based on soil and weather data"""
    if not all([crop_interpreter, crop_scaler, crop_soil_encoder, crop_label_encoder]):
        return {
            "message": "Crop recommendation models not available",
            "fallback_recommendations": [
                {"crop": "rice", "confidence": 0.85},
                {"crop": "wheat", "confidence": 0.75},
                {"crop": "maize", "confidence": 0.65}
            ]
        }
    
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        
        numerical_data = pd.DataFrame([{
            'N': data.nitrogen,
            'P': data.phosphorus,
            'K': data.potassium,
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity'],
            'ph': data.ph,
            'rainfall': weather_data['rainfall']
        }])
        
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
            recommendations.append({
                "crop": str(crop_name),
                "confidence": round(float(confidence), 4)
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Crop recommendation failed: {str(e)}")

@app.post("/predict-yield")
def predict_yield(data: YieldPredictionData, lat: float = Query(19.07), lon: float = Query(72.87)):
    """Predict crop yield based on soil and weather conditions"""
    if not all([yield_interpreter, yield_scaler, yield_crop_encoder]):
        # Return mock prediction when models are not available
        return {
            "predicted_crop": data.crop,
            "estimated_yield_tons_per_hectare": 3.5,
            "weather_data_used": get_seasonal_weather_averages(lat, lon, data.season),
            "message": "Using default yield estimate (models not loaded)"
        }
    
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        
        numerical_data = pd.DataFrame([{
            'N': data.nitrogen,
            'P': data.phosphorus,
            'K': data.potassium,
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity'],
            'ph': data.ph,
            'rainfall': weather_data['rainfall']
        }])
        
        categorical_data = pd.DataFrame({'label': [data.crop]})
        
        scaled_numerical = yield_scaler.transform(numerical_data)
        encoded_categorical = yield_crop_encoder.transform(categorical_data)
        processed_input = np.hstack([scaled_numerical, encoded_categorical]).astype(np.float32)
        
        input_details = yield_interpreter.get_input_details()
        output_details = yield_interpreter.get_output_details()
        
        yield_interpreter.set_tensor(input_details[0]['index'], processed_input)
        yield_interpreter.invoke()
        
        prediction = yield_interpreter.get_tensor(output_details[0]['index'])[0]
        
        return {
            "predicted_crop": data.crop,
            "estimated_yield_tons_per_hectare": round(float(prediction[0]), 2),
            "weather_data_used": weather_data
        }
        
    except Exception as e:
        logger.error(f"Error in yield prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    """Detect plant diseases from uploaded images"""
    if not disease_interpreter or not disease_class_names:
        return {
            "predicted_disease": "Model Not Available",
            "confidence": "0.00",
            "message": "Disease detection model not loaded"
        }
    
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
        
        return {
            "predicted_disease": predicted_class_name.replace("_", " "),
            "confidence": f"{confidence:.2f}"
        }
        
    except Exception as e:
        logger.error(f"Error in disease detection: {e}")
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")

@app.post("/calculate-profit-sustainability")
async def calculate_metrics(data: CalculatorData, lat: float = Query(19.07), lon: float = Query(72.87)):
    """Calculate profit and sustainability metrics"""
    if not crop_details:
        return {"message": "Crop data not available for calculations"}
    
    try:
        # Use mock yield if models not available
        if not all([yield_interpreter, yield_scaler, yield_crop_encoder]):
            predicted_yield_tons_ha = 3.5  # Mock yield
            logger.info("Using mock yield prediction")
        else:
            # Real yield prediction
            weather_data = get_seasonal_weather_averages(lat, lon, data.season)
            numerical_data = pd.DataFrame([{
                'N': data.nitrogen,
                'P': data.phosphorus,
                'K': data.potassium,
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'ph': data.ph,
                'rainfall': weather_data['rainfall']
            }])
            
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

        # Get market price
        price_response = await get_mandi_prices(data.state, data.district, data.crop)
        
        if 'message' in price_response and 'not found' in price_response['message'].lower():
            return {
                "crop": data.crop,
                "predicted_yield_tons_per_hectare": predicted_yield_tons_ha,
                "message": price_response['message']
            }
        
        avg_market_price_quintal = (price_response['minPrice'] + price_response['maxPrice']) / 2

        # Get costs & sustainability data
        crop_key = data.crop.lower()
        details = crop_details.get(crop_key, crop_details.get('Default', {}))
        estimated_cost_ha = details.get('estimated_cost_per_hectare', 50000)
        sustainability_ratings = details.get('sustainability', {
            'water_usage_rating': 5,
            'pesticide_rating': 5,
            'soil_health_impact': 5
        })

        # Calculate profit
        total_revenue_ha = predicted_yield_tons_ha * 10 * avg_market_price_quintal
        net_profit_ha = total_revenue_ha - estimated_cost_ha
        
        # Calculate sustainability score
        water = (10 - sustainability_ratings.get('water_usage_rating', 5)) * 0.4
        pesticide = (10 - sustainability_ratings.get('pesticide_rating', 5)) * 0.4
        soil = (10 - sustainability_ratings.get('soil_health_impact', 5)) * 0.2
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
        logger.error(f"Error in profit calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)