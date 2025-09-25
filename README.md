# AgriAI - Smart Agriculture Platform

![AgriAI Logo](https://img.shields.io/badge/KrishiSahayak-Smart%20Agriculture-green)
![Version](https://img.shields.io/badge/version-1.3.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

AgriAI is an AI-powered agriculture platform that helps farmers make data-driven decisions for improved crop yields, disease management, and profit optimization.

## Features

- **Crop Recommendation**: Get personalized crop recommendations based on soil and climate data
- **Disease Detection**: Identify plant diseases from images with AI-powered recognition
- **Yield Prediction**: Forecast crop yields based on various parameters
- **Market Insights**: Get the latest agricultural news and market trends
- **Profit Calculator**: Estimate potential profits for different crops
- **Offline Support**: Core features work even without an internet connection
- **Multi-language Support**: Use the platform in your preferred language

## Technology Stack

### Frontend
- React.js + TypeScript
- Redux Toolkit for state management
- TensorFlow.js for client-side ML inference
- React Router for navigation
- Tailwind CSS for styling
- Vite as the build tool

### Backend
- FastAPI (Python)
- TensorFlow Lite for optimized model inference
- XGBoost for machine learning models
- Docker for containerization
- Weather API integration

## Project Structure

```
.
├── backend/
│   ├── models/           # ML models (TFLite and others)
│   ├── notebooks/        # Jupyter notebooks for model training
│   ├── main.py           # FastAPI application
│   ├── weather.py        # Weather data integration
│   └── requirements.txt  # Python dependencies
│
└── frontend/
    ├── public/           # Static assets
    └── src/
        ├── api/          # API client and types
        ├── components/   # Reusable UI components
        ├── contexts/     # React contexts
        ├── data/         # Static data
        ├── hooks/        # Custom React hooks
        ├── pages/        # Application pages
        ├── services/     # Client-side services
        ├── store/        # Redux store
        ├── styles/       # CSS and styling
        └── translations/ # Internationalization
```

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.10+)
- Docker (optional, for containerized deployment)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 7860
   ```

The backend API will be available at http://localhost:7860.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at http://localhost:5173.

## Docker Deployment

### Backend

1. Build the Docker image:
   ```bash
   cd backend
   docker build -t agri-ai-backend .
   ```

2. Run the container:
   ```bash
   docker run -d -p 7860:7860 --name agri-ai-backend agri-ai-backend
   ```

### Frontend

1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. You can deploy the built files (in the `dist` directory) to any static hosting service.

## API Documentation

Once the backend is running, you can access the API documentation at:
- Swagger UI: http://localhost:7860/docs
- ReDoc: http://localhost:7860/redoc

## Environment Variables

### Backend

Create a `.env` file in the backend directory with the following variables:

```
WEATHER_API_KEY=your_weather_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional, for enhanced recommendations
API_GOV_KEY=your_api_gov_key  # Required for government data APIs
```

### Obtaining API Keys

- **WEATHER_API_KEY**: Register at [OpenWeatherMap](https://openweathermap.org/api) to obtain an API key
- **API_GOV_KEY**: Sign up at [data.gov.in](https://data.gov.in/signup) to receive an API key for accessing Indian government agricultural data

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Weather data provided by OpenWeatherMap API
- Disease detection model trained on the PlantVillage dataset
- Crop recommendation system based on agricultural research data