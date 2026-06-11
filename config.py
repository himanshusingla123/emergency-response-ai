from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP
    google_cloud_project: str
    google_cloud_location: str = "us-central1"
    # google_application_credentials: str = ""

    # APIs
    google_maps_api_key: str
    google_places_api_key: str

    # Twilio
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_number: str

    # Phoenix
    phoenix_collector_endpoint: str
    phoenix_api_key: str

    # Thresholds
    criticality_extreme_threshold: int = 81
    criticality_high_threshold: int = 61
    criticality_medium_threshold: int = 31

    # Gemini — also exported as GOOGLE_API_KEY for the google-genai SDK
    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"

settings = Settings()

# Set env vars so google-genai SDK authenticates directly (no VertexAI)
os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "false")
os.environ.setdefault("PHOENIX_API_KEY", settings.phoenix_api_key)
os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", settings.phoenix_collector_endpoint)