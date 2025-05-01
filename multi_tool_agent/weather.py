import python_weather
from python_weather.forecast import Forecast

async def get_weather_at(city: str) -> Forecast:
    async with python_weather.Client() as client:
        weatherReport = await client.get(city)
    
    return weatherReport

if __name__ == "__main__":
    get_weather_at('new york')