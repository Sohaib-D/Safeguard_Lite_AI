from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from backend.core.config import settings
from backend.core.logging_config import configure_logger
from backend.db.postgres_store import PostgresStore

class AuthenticationError(ValueError):
    """Raised when authentication or authorization fails."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class AuthService:
    """JWT authentication and admin user management."""

    def __init__(self, db: Session):
        self.store = PostgresStore(db)
        self.logger = configure_logger("safeguard.backend.auth", settings.log_file_path)

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except (ValueError, TypeError):
            return False

    def admin_exists(self) -> bool:
        return self.store.admin_exists()

    def create_admin_user(self, username: str, password: str) -> dict[str, Any]:
        existing = self.store.get_user_by_username(username)
        if existing:
            self.logger.warning(f"Admin creation blocked: {username} exists.")
            raise AuthenticationError("Username already exists.")

        password_hash = self._hash_password(password)
        user = self.store.create_user(username=username, password_hash=password_hash, is_admin=True)
        return user

    def authenticate_user(self, username: str, password: str) -> dict[str, Any]:
        user = self.store.get_user_by_username(username)
        if user is None or not self._verify_password(password, user["password_hash"]):
            raise AuthenticationError("Invalid username or password.")
        return user

    def create_access_token(self, user: dict[str, Any]) -> tuple[str, int]:
        expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
        expire = datetime.now(timezone.utc) + expires_delta
        payload = {"sub": user["username"], "is_admin": bool(user["is_admin"]), "exp": expire}
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        return token, int(expires_delta.total_seconds())

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        except JWTError as exc:
            raise AuthenticationError("Invalid token.") from exc

        username = payload.get("sub")
        if not username:
            raise AuthenticationError("Invalid token payload.")

        user = self.store.get_user_by_username(username)
        if user is None:
            raise AuthenticationError("User no longer exists.")
        return user
