from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT=Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=ROOT/'.env',extra='ignore')
    database_url:str='postgresql+psycopg://localhost/kneews'
    data_root:Path=ROOT/'data/rsna-knee-abnormality-detection'
    artifact_root:Path=ROOT/'artifacts'
    local_llm_url:str='http://127.0.0.1:11434'
    local_llm_model:str='gemma3:4b'
    cors_origin:str='http://127.0.0.1:3200'
settings=Settings()
