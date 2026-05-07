from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field


@dataclass
class Settings:
    app_name: str = "Safeguard-AI Lite API"
    app_version: str = "1.0.0"
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    host: str = os.environ.get("API_HOST", "127.0.0.1")
    port: int = int(os.environ.get("API_PORT", "8000"))
    allowed_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )
    model_bundle_path: str = os.environ.get(
        "MODEL_BUNDLE_PATH",
        "models/trained_multiclass_smoke/best_model.pkl",
    )
    database_path: str = os.environ.get("SAFEGUARD_DB_PATH", "safeguard_ai.db")
    max_explanations: int = int(os.environ.get("MAX_EXPLANATIONS", "10"))
    max_upload_rows: int = int(os.environ.get("MAX_UPLOAD_ROWS", "5000"))
    log_file_path: str = os.environ.get("BACKEND_LOG_FILE", "logs/backend.log")
    prediction_cache_dir: str = os.environ.get(
        "PREDICTION_CACHE_DIR", "models/cache/predictions"
    )
    use_jax_inference: bool = (
        os.environ.get("USE_JAX_INFERENCE", "true").lower() == "true"
    )
    jwt_secret_key: str = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )


settings = Settings()
