from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    AWS_NEPTUNE_ENDPOINT: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()