from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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

app = FastAPI(title="Agriculture AI Backend")

# --- Helper function to load TFLite models ---
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# --- Load All Final Models and Artifacts ---
try:
    # --- Crop Recommendation Models (FIXED LOADING) ---
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

    print("✅ All TFLite models and artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    crop_interpreter = yield_interpreter = disease_interpreter = None

# --- Pydantic Models ---
class EnrichedSoilData(BaseModel):
    nitrogen: float; phosphorus: float; potassium: float; ph: float; season: str; soil_type: str

class CropRecommendation(BaseModel):
    crop: str; confidence: float

class YieldPredictionData(BaseModel):
    nitrogen: float; phosphorus: float; potassium: float; ph: float; season: str; crop: str

# --- API Endpoints ---
@app.post("/recommend-crop")
def recommend_crop(data: EnrichedSoilData, lat: float = Query(..., example=19.07), lon: float = Query(..., example=72.87)):
    if not all([crop_interpreter, crop_scaler, crop_soil_encoder, crop_label_encoder]):
        raise HTTPException(status_code=500, detail="Crop model components are not loaded.")
    try:
        weather_data = get_seasonal_weather_averages(lat, lon, data.season)
        
        # --- PREPROCESSING LOGIC (THE FIX) ---
        # Manually preprocess the data using the separate, robust files
        numerical_data = pd.DataFrame([{
            'N': data.nitrogen, 'P': data.phosphorus, 'K': data.potassium,
            'temperature': weather_data['temperature'], 'humidity': weather_data['humidity'],
            'ph': data.ph, 'rainfall': weather_data['rainfall']
        }])
        
        categorical_data = pd.DataFrame({'soil_type': [data.soil_type]})
        
        # Scale the numerical data
        scaled_numerical = crop_scaler.transform(numerical_data)
        
        # One-hot encode the soil type
        encoded_categorical = crop_soil_encoder.transform(categorical_data)
        
        # Combine into a single array with 12 features
        processed_input = np.hstack([scaled_numerical, encoded_categorical]).astype(np.float32)

        # --- INFERENCE LOGIC (REMAINS THE SAME) ---
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

# ... (the rest of your main.py remains the same) ...

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