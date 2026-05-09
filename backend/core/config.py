from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


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
    rules_path: str = os.environ.get("RULES_PATH", "rules")
    abuseipdb_api_key: str = os.environ.get("ABUSEIPDB_API_KEY", "")
    virustotal_api_key: str = os.environ.get("VIRUSTOTAL_API_KEY", "")
    greynoise_api_key: str = os.environ.get("GREYNOISE_API_KEY", "")
    otx_api_key: str = os.environ.get("OTX_API_KEY", "")
    threat_intel_cache_dir: str = os.environ.get("THREAT_INTEL_CACHE_DIR", "cache/intel")
    threat_intel_cache_ttl: int = int(os.environ.get("THREAT_INTEL_CACHE_TTL", "900"))
    rules_path: str = os.environ.get("RULES_PATH", "rules")
    groq_api_url: str = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_rate_limit_per_minute: int = int(
        os.environ.get("GROQ_RATE_LIMIT_PER_MINUTE", "15")
    )
    groq_max_retries: int = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
    groq_cache_dir: str = os.environ.get("GROQ_CACHE_DIR", "cache/groq")
    groq_cache_ttl: int = int(os.environ.get("GROQ_CACHE_TTL", "900"))
    use_jax_inference: bool = (
        os.environ.get("USE_JAX_INFERENCE", "true").lower() == "true"
    )
    jwt_secret_key: str = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )


settings = Settings()
