from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "EarningsOS"
    log_level: str = "INFO"

    database_url: str = ""

    redis_url: str = ""

    market_data_provider: str = ""
    market_data_api_key: str = ""

    news_provider: str = ""
    news_api_key: str = ""

    llm_provider: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
