from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from backend.core.config import settings
from backend.core.logging_config import configure_logger
from backend.db.sqlite_store import SQLiteStore


class AuthenticationError(ValueError):
    """Raised when authentication or authorization fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthService:
    """JWT authentication and admin user management."""

    def __init__(self, database_path: str):
        self.store = SQLiteStore(database_path)
        self.logger = configure_logger("safeguard.backend.auth", settings.log_file_path)

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), password_hash.encode("utf-8")
            )
        except ValueError:
            return False

    def admin_exists(self) -> bool:
        return self.store.admin_exists()

    def create_admin_user(self, username: str, password: str) -> dict[str, Any]:
        existing = self.store.get_user_by_username(username)
        if existing:
            self.logger.warning(
                "Admin creation blocked because username already exists.",
                extra={"event_type": "create_admin_conflict", "username": username},
            )
            raise AuthenticationError("Username already exists.")

        password_hash = self._hash_password(password)
        user = self.store.create_user(
            username=username, password_hash=password_hash, is_admin=True
        )
        self.logger.info(
            "Admin user created.",
            extra={"event_type": "create_admin_success", "username": username},
        )
        return user

    def authenticate_user(self, username: str, password: str) -> dict[str, Any]:
        row = self.store.get_user_by_username(username)
        if row is None or not self._verify_password(password, row["password_hash"]):
            self.logger.warning(
                "Login attempt failed.",
                extra={"event_type": "login_failed", "username": username},
            )
            raise AuthenticationError("Invalid username or password.")

        user = {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "is_admin": bool(row["is_admin"]),
            "created_at": (
                row["created_at"]
                if isinstance(row["created_at"], datetime)
                else datetime.fromisoformat(row["created_at"])
            ),
        }
        self.logger.info(
            "Login attempt succeeded.",
            extra={"event_type": "login_success", "username": username},
        )
        return user

    def create_access_token(self, user: dict[str, Any]) -> tuple[str, int]:
        expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
        expire = datetime.now(timezone.utc) + expires_delta
        payload = {
            "sub": user["username"],
            "is_admin": bool(user["is_admin"]),
            "exp": expire,
        }
        token = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        return token, int(expires_delta.total_seconds())

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
            )
        except JWTError as exc:
            self.logger.warning(
                "Token decode failed.",
                extra={"event_type": "token_decode_failed"},
            )
            raise AuthenticationError(
                "Invalid or expired authentication token."
            ) from exc

        username = payload.get("sub")
        if not username:
            raise AuthenticationError("Invalid authentication token payload.")

        row = self.store.get_user_by_username(username)
        if row is None:
            raise AuthenticationError("Authenticated user no longer exists.")

        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "is_admin": bool(row["is_admin"]),
            "created_at": (
                row["created_at"]
                if isinstance(row["created_at"], datetime)
                else datetime.fromisoformat(row["created_at"])
            ),
        }
