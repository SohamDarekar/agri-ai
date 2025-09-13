from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from typing import List
import joblib
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
from weather import get_seasonal_weather_averages

app = FastAPI(title="Agriculture Project API")

# --- Load Models and Artifacts ---
try:
    crop_model = joblib.load('models/crop_recommender_xgb.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    yield_model = joblib.load('models/yield_predictor_v2.pkl')
    crop_encoder = joblib.load('models/crop_label_encoder.pkl')
    disease_model = tf.keras.models.load_model('models/disease_detector.h5')

    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    disease_class_names = {v: k for k, v in class_indices.items()}
    print("✅ All models and artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    crop_model = yield_model = disease_model = label_encoder = crop_encoder = None
    disease_class_names = {}

# --- Pydantic Models ---
class EnrichedSoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    season: str
    soil_type: str
    market_price: float

class CropRecommendation(BaseModel):
    crop: str
    confidence: float

class YieldPredictionData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    season: str
    crop: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Agriculture Project API"}

@app.post("/recommend-crop", response_model=List[CropRecommendation])
def recommend_crop(data: EnrichedSoilData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not crop_model or not label_encoder:
        raise HTTPException(status_code=500, detail="Crop model or encoder not loaded.")
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        input_data = {
            'N': [data.nitrogen], 'P': [data.phosphorus], 'K': [data.potassium],
            'temperature': [weather_data['temperature']], 'humidity': [weather_data['humidity']],
            'ph': [data.ph], 'rainfall': [weather_data['rainfall']],
            'market_price': [data.market_price], 'soil_type': [data.soil_type]
        }
        input_df = pd.DataFrame(input_data)
        probabilities = crop_model.predict_proba(input_df)
        class_indices = crop_model.classes_
        prob_list = sorted(zip(class_indices, probabilities[0]), key=lambda x: x[1], reverse=True)
        top_3_recommendations = []
        for class_index, prob in prob_list[:3]:
            crop_name = label_encoder.inverse_transform([class_index])[0]
            top_3_recommendations.append({"crop": str(crop_name), "confidence": round(float(prob), 4)})
        return top_3_recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-yield")
def predict_yield(data: YieldPredictionData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not yield_model or not crop_encoder:
        raise HTTPException(status_code=500, detail="Yield model or its encoder not loaded.")
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        
        input_df = pd.DataFrame([{
            'N': data.nitrogen, 'P': data.phosphorus, 'K': data.potassium,
            'temperature': weather_data['temperature'], 'humidity': weather_data['humidity'],
            'ph': data.ph, 'rainfall': weather_data['rainfall']
        }])

        crop_df = pd.DataFrame({'label': [data.crop]})
        encoded_crop = crop_encoder.transform(crop_df[['label']])
        
        # --- FIX IS HERE ---
        # Call get_feature_names_out() without arguments to match the training script
        encoded_crop_df = pd.DataFrame(encoded_crop, columns=crop_encoder.get_feature_names_out())

        final_df = pd.concat([input_df.reset_index(drop=True), encoded_crop_df.reset_index(drop=True)], axis=1)

        prediction = yield_model.predict(final_df)
        estimated_yield = round(float(prediction[0]), 2)
        
        return {
            "predicted_crop": data.crop,
            "estimated_yield_tons_per_hectare": estimated_yield,
            "weather_data_used": weather_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if not disease_model:
        raise HTTPException(status_code=500, detail="Disease detection model not loaded.")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = disease_model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = disease_class_names.get(predicted_class_index, "Unknown")
        confidence = float(np.max(prediction[0]))
        return {
            "predicted_disease": predicted_class_name.replace("_", " "),
            "confidence": f"{confidence:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))