"""Configuration management for DocuChat using Pydantic settings."""

from typing import Literal
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", env="CHAT_MODEL")
    
    # Gemini Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    gemini_chat_model: str = Field(default="gemini-1.5-flash", env="GEMINI_CHAT_MODEL")
    gemini_embedding_model: str = Field(default="models/text-embedding-004", env="GEMINI_EMBEDDING_MODEL")
    
    # Vector Store Configuration
    vector_store: Literal["faiss", "chroma"] = Field(default="faiss", env="VECTOR_STORE")
    index_dir: str = Field(default="./storage", env="INDEX_DIR")
    
    # RAG Parameters
    chunk_size: int = Field(default=1200, env="CHUNK_SIZE", ge=100, le=8000)
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP", ge=0, le=1000)
    top_k: int = Field(default=4, env="TOP_K", ge=1, le=20)
    max_tokens_answer: int = Field(default=600, env="MAX_TOKENS_ANSWER", ge=50, le=4000)
    temperature: float = Field(default=0.2, env="TEMPERATURE", ge=0.0, le=2.0)
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = values.get("chunk_size", 1200)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @validator("vector_store")
    def validate_vector_store(cls, v):
        """Ensure vector store is supported."""
        if v not in ["faiss", "chroma"]:
            raise ValueError("vector_store must be 'faiss' or 'chroma'")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
