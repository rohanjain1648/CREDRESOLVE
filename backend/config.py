from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    google_api_key: str = ""
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    database_url: str = "sqlite:///./credsolve.db"
    chroma_persist_dir: str = "./chroma_db"
    prometheus_port: int = 9090

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    agent_name: str = "Arya"
    agent_company: str = "CredResolve"
    default_language: str = "hi"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
