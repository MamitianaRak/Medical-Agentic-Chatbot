import os
import certifi
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    google_api_key: str = ''
    cloudflare_account_id: str = ''
    cloudflare_api_token: str = ''
    pinecone_api_key: str = ''
    tavily_api_key: str = ''
    pinecone_index: str = Field('medical-agent-gemini-768', validation_alias=AliasChoices('PINECONE_INDEX_NAME', 'PINECONE_INDEX'))
    pinecone_namespace: str = Field('', validation_alias=AliasChoices('HR_NAMESPACE', 'PINECONE_NAMESPACE'))
    gemini_chat_model: str = 'gemini-3.5-flash-lite'
    embed_model: str = 'llama-text-embed-v2'
    embed_dimensions: int = 768
    cf_stt_model: str = '@cf/openai/whisper-large-v3-turbo'
    cf_vision_model: str = '@cf/meta/llama-3.2-11b-vision-instruct'
    cf_tts_model: str = '@cf/myshell-ai/melotts'
    tts_lang: str = 'en'
    api_token: str = ''
    allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def configured(self):
        return bool(self.google_api_key and self.pinecone_api_key and self.cloudflare_account_id and self.cloudflare_api_token)



settings = Settings()
