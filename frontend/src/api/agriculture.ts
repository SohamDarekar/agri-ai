import { apiClient } from './client';
import {
  CropRecommendationResponse,
  YieldPredictionResponse,
  DiseaseDetectionResponse,
  MandiPriceResponse,
  ProfitCalculatorResponse,
  EnrichedSoilData,
  YieldPredictionData,
  ProfitCalculatorData,
  MandiPriceData,
  Location,
  CropsResponse,
} from './types';

export class AgricultureAPI {
  // Crop Recommendation
  static async recommendCrop(
    data: EnrichedSoilData,
    location: Location
  ): Promise<CropRecommendationResponse> {
    return apiClient.post('/recommend-crop', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
  }

  // Yield Prediction
  static async predictYield(
    data: YieldPredictionData,
    location: Location
  ): Promise<YieldPredictionResponse> {
    return apiClient.post('/predict-yield', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
  }

  // Disease Detection
  static async detectDisease(imageFile: File): Promise<DiseaseDetectionResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    return apiClient.postFormData('/detect-disease', formData);
  }

  // Mandi Prices
  static async getMandiPrices(data: MandiPriceData): Promise<MandiPriceResponse> {
    return apiClient.get('/api/prices', {
      state: data.state,
      district: data.district,
      crop: data.crop,
    });
  }

  // Get Available Crops
  static async getAvailableCrops(): Promise<CropsResponse> {
    return apiClient.get('/api/crops');
  }

  // Profit & Sustainability Calculator
  static async calculateProfitSustainability(
    data: ProfitCalculatorData,
    location: Location
  ): Promise<ProfitCalculatorResponse> {
    return apiClient.post('/calculate-profit-sustainability', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
  }

  // Health check
  static async healthCheck(): Promise<{ status: string }> {
    return apiClient.healthCheck();
  }
}

// Re-export types for convenience
export * from './types';