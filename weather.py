import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime

def get_seasonal_weather_averages(lat: float, lon: float, season: str):
    """
    Fetches historical weather data based on the agricultural season.
    Kharif (Monsoon): June to September
    Rabi (Winter): October to March
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Determine date range based on season
    current_year = datetime.now().year
    if season.lower() == 'kharif':
        start_date = f"{current_year-1}-06-01"
        end_date = f"{current_year-1}-09-30"
    elif season.lower() == 'rabi':
        start_date = f"{current_year-1}-10-01"
        end_date = f"{current_year}-03-31"
    else:
        # Default to last 3 months if season is invalid
        start_date = (datetime.now() - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {
        "date": pd.to_datetime(daily.Time(), unit="s"),
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m_mean": daily.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(2).ValuesAsNumpy()
    }

    daily_df = pd.DataFrame(data=daily_data)
    daily_df.dropna(inplace=True)
    
    # Calculate averages over the entire seasonal period
    seasonal_averages = daily_df.mean()

    return {
        "temperature": float(seasonal_averages['temperature_2m_mean']),
        "humidity": float(seasonal_averages['relative_humidity_2m_mean']),
        "rainfall": float(seasonal_averages['precipitation_sum'] * 30) # Approximate monthly total
    }