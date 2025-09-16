import apiClient from './apiClient';
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
    const response = await apiClient.post('/recommend-crop', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
    return response.data;
  }

  // Yield Prediction
  static async predictYield(
    data: YieldPredictionData,
    location: Location
  ): Promise<YieldPredictionResponse> {
    const response = await apiClient.post('/predict-yield', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
    return response.data;
  }

  // Disease Detection
  static async detectDisease(imageFile: File): Promise<DiseaseDetectionResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await apiClient.post('/detect-disease', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Mandi Prices
  static async getMandiPrices(data: MandiPriceData): Promise<MandiPriceResponse> {
    const response = await apiClient.get('/api/prices', {
      params: {
        state: data.state,
        district: data.district,
        crop: data.crop,
      },
    });
    return response.data;
  }

  // Get Available Crops
  static async getAvailableCrops(): Promise<CropsResponse> {
    const response = await apiClient.get('/api/crops');
    return response.data;
  }

  // Profit & Sustainability Calculator
  static async calculateProfitSustainability(
    data: ProfitCalculatorData,
    location: Location
  ): Promise<ProfitCalculatorResponse> {
    const response = await apiClient.post('/calculate-profit-sustainability', data, {
      params: {
        lat: location.lat,
        lon: location.lon,
      },
    });
    return response.data;
  }

  // Health check
  static async healthCheck(): Promise<{ status: string }> {
    try {
      await apiClient.get('/docs');
      return { status: 'ok' };
    } catch (error) {
      throw error;
    }
  }
}

// Re-export types for convenience
export * from './types';