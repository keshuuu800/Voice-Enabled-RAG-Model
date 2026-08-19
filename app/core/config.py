"""
Central configuration using Pydantic Settings.
All environment variables are loaded once and cached.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ───────────────────────────────────────────────────────────────
    sarvam_api_key: str = ""
    llm_provider: str = "openrouter"   # "openrouter" | "openai" | "gemini"
    llm_api_key: str = ""
    openrouter_api_key: str = ""

    # ── LLM Models ─────────────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-flash"
    openrouter_model: str = "openrouter/free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Embedding ──────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 2000
    chunk_overlap: int = 300

    # ── Retrieval ──────────────────────────────────────────────────────────────
    semantic_top_k: int = 10
    bm25_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    min_relevance_score: float = 0.005

    # ── Storage ────────────────────────────────────────────────────────────────
    chroma_path: str = "./storage/chroma"
    bm25_path: str = "./storage/bm25"
    chroma_collection_name: str = "documents"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton. Call this everywhere instead of Settings()."""
    return Settings()
