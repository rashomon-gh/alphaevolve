from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class SecretsStore(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    huggingface_token: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None


# export
values = SecretsStore()  # type: ignore
