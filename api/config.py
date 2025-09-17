"""Runtime configuration helpers for the Doc_GPT API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
ENV_FILES: tuple[Path, ...] = (ROOT / ".env", ROOT / ".env.local")


def _resolve_path(base: Path, value: Path) -> Path:
    """Return an absolute path, interpreting relative values from the project root."""
    if value.is_absolute():
        return value
    return (base / value).resolve()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and `.env` files."""

    model_config = SettingsConfigDict(
        env_file=[str(path) for path in ENV_FILES],
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    flat_index_dir: Path = Field(default=ROOT / "data" / "flatindex", alias="FLAT_INDEX_DIR")
    web_dir: Path = Field(default=ROOT / "web", alias="WEB_DIR")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    hybrid_alpha: float = Field(default=0.40, alias="HYBRID_ALPHA")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    rerank_top_m: int = Field(default=12, alias="RERANK_TOP_M")
    openai_base_url: str = Field(default="http://127.0.0.1:1234/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="lm-studio", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="meta-llama-3.1-8b-instruct", alias="LLM_MODEL")
    show_retrieved: bool = Field(default=True, alias="SHOW_RETRIEVED")

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - trivial post processing
        # Resolve any relative paths against the project root so downstream code can rely on
        # canonical locations regardless of the working directory used to launch the app.
        self.flat_index_dir = _resolve_path(ROOT, self.flat_index_dir)
        self.web_dir = _resolve_path(ROOT, self.web_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance so configuration is parsed only once."""

    return Settings()


def iter_existing_env_files() -> Iterable[Path]:
    """Yield `.env`-style files that are present on disk."""

    for path in ENV_FILES:
        if path.exists():
            yield path
