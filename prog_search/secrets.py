from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class SecretsStore(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    huggingface_token: SecretStr


# export
values = SecretsStore()  # type: ignore
