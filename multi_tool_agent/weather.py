import openmeteo_requests
from openmeteo_sdk.Variable import Variable

om = openmeteo_requests.Client()

codes = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "freezing light rain", 67: "freezing heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight showers", 81: "moderate showers", 82: "violent showers",
    85: "slight snow shower", 86: "heavy snow shower",
    95: "slight or moderate thunderstorm",
    96: "thunderstorm slight hail", 99: "thunderstorm heavy hail"
}

def get_weather_at(lat: float, long: float):
    params = {
        "latitude": lat,
        "longitude": long,
        "current": ["temperature_2m", "weather_code"]
    }

    responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Current values
    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_weather_code = current.Variables(1).Value()

    print(f"Current time {current.Time()}")
    print(f"Current temperature_2m {current_temperature_2m}")
    print(f"Current weather {codes[current_weather_code]}")

    return current_temperature_2m, codes[current_weather_code]

if __name__ == "__main__":
    get_weather_at(52.54, 13.41)