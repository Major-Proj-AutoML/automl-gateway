from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_service_url: str = "http://localhost:8001"
    metafeatures_service_url: str = "http://localhost:8002"
    generation_service_url: str = "http://localhost:8003"
    analysis_service_url: str = "http://localhost:8004"
    service_port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: str = "http://localhost:3000,http://localhost:5173"
    http_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
