from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None
    # groq_model: str = "llama-3.1-8b-instant"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_classifier_model: str | None = None
    groq_judge_model: str | None = None
    allowed_origins: str = "http://localhost,http://localhost:8080"
    rate_limit_per_minute: int = 30
    max_request_bytes: int = 16_384
    follow_up_database_path: str = "storage/followups/followups.db"
    follow_up_retention_days: int = 90
    # Required for access to stored contact details through the admin endpoint.
    follow_up_admin_token: str | None = None
    retrieval_min_score: float = 0.50
    index_path: str = "storage/index.json"
    chroma_path: str = "storage/chroma"
    chroma_collection: str = "matrix_media"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_candidate_count: int = 18
    retrieval_top_k: int = 5
    embedding_cache_path: str = "storage/fastembed"
    # The model is downloaded at build/setup time; serving must not depend on internet access.
    embedding_local_files_only: bool = True
    escalation_message: str = (
        "This is best handled by a Matrix Media specialist. Contact us at contact@matrixnmedia.com or +91-33-4849 0807 so we can help directly."
    )
    support_message: str = (
        "For help with an existing Matrix Media engagement, contact our support team at contact@matrixnmedia.com or +91-33-4849 0807 so they can access the right details."
    )
    # Keep this configurable because a CMS permalink can change without a code release.
    career_page_url: str = "https://matrixmedia.betatesting.net/career/"
    career_message: str = (
        "Thanks for your interest in joining Matrix Media. View current job openings and apply here: "
        "[View Matrix Media job openings]({career_page_url})"
    )
    client_greeting_message: str = (
        "Hello! Welcome to Matrix Media. How can we help with your digital, technology, or growth goals today?"
    )

    @field_validator("retrieval_top_k")
    @classmethod
    def validate_retrieval_top_k(cls, value: int) -> int:
        if value not in {5, 6}:
            raise ValueError("RETRIEVAL_TOP_K must be 5 or 6.")
        return value

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
