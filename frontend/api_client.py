from __future__ import annotations

from typing import Any

import requests


class APIClientError(RuntimeError):
    """Raised when the backend API returns an error."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.errors = errors or []
        self.status_code = status_code


class SafeguardAPIClient:
    """Small requests-based client for the FastAPI backend."""

    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _handle_response(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or "Unexpected response from server."}

        if response.ok:
            return payload

        raise APIClientError(
            message=str(payload.get("detail", "Request failed.")),
            errors=list(payload.get("errors", [])),
            status_code=response.status_code,
        )

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/health", headers=self._headers(), timeout=20
        )
        return self._handle_response(response)

    def login(self, username: str, password: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
            headers=self._headers(),
            timeout=20,
        )
        return self._handle_response(response)

    def create_admin(self, username: str, password: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/auth/create-admin",
            json={"username": username, "password": password},
            headers=self._headers(),
            timeout=20,
        )
        return self._handle_response(response)

    def model_info(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/model_info", headers=self._headers(), timeout=20
        )
        return self._handle_response(response)

    def stats(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/stats", headers=self._headers(), timeout=20
        )
        return self._handle_response(response)

    def upload_csv(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/upload",
            headers=self._headers(),
            files={"file": (file_name, file_bytes, "text/csv")},
            timeout=60,
        )
        return self._handle_response(response)

    def predict_csv(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict",
            headers=self._headers(),
            files={"file": (file_name, file_bytes, "text/csv")},
            timeout=120,
        )
        return self._handle_response(response)

    def predict_records(
        self,
        records: list[dict[str, Any]],
        include_explanations: bool = True,
        explanation_top_k: int = 5,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "records": records,
                "include_explanations": include_explanations,
                "explanation_top_k": explanation_top_k,
            },
            timeout=120,
        )
        return self._handle_response(response)

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str, comment: str | None = None) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"acknowledged_by": acknowledged_by, "comment": comment},
            timeout=20,
        )
        return self._handle_response(response)

    def analyze_soc(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/soc/analyze",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        return self._handle_response(response)

    def active_scan(self, target: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/scan",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"target": target},
            timeout=120,
        )
        return self._handle_response(response)
