from __future__ import annotations

import importlib
import random
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "trained_multiclass_smoke" / "best_model.pkl"
)


@pytest.fixture(autouse=True)
def fixed_random_seed():
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def api_module(tmp_path, monkeypatch):
    db_path = tmp_path / "test_api.db"
    log_path = tmp_path / "backend.log"

    monkeypatch.setenv("SAFEGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BACKEND_LOG_FILE", str(log_path))
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(SMOKE_MODEL_PATH))

    import backend.core.config as config_module
    import backend.api.main as main_module

    importlib.reload(config_module)
    reloaded = importlib.reload(main_module)
    return reloaded


@pytest.fixture
def client(api_module):
    return TestClient(api_module.app)


@pytest.fixture
def auth_headers(client):
    create_resp = client.post(
        "/auth/create-admin",
        json={"username": "admin_user", "password": "StrongPass123"},
    )
    assert create_resp.status_code == 200

    login_resp = client.post(
        "/auth/login",
        json={"username": "admin_user", "password": "StrongPass123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def fake_prediction_result(row_count: int = 1) -> dict:
    predictions = []
    for idx in range(row_count):
        predictions.append(
            {
                "row_index": idx,
                "predicted_label": "DDoS" if idx % 2 == 0 else "Normal",
                "predicted_index": 1 if idx % 2 == 0 else 2,
                "confidence": 0.98 if idx % 2 == 0 else 0.73,
                "class_probabilities": {
                    "BruteForce": 0.01,
                    "DDoS": 0.98 if idx % 2 == 0 else 0.05,
                    "Normal": 0.01 if idx % 2 == 0 else 0.73,
                    "PortScan": 0.0 if idx % 2 == 0 else 0.22,
                },
                "top_contributions": [
                    {
                        "feature": "num__f0",
                        "feature_value": 1.23,
                        "shap_value": 0.42,
                        "abs_shap_value": 0.42,
                    }
                ],
                "recommendation_severity": "critical" if idx % 2 == 0 else "info",
                "recommendations": (
                    [
                        "Block offending IPs or upstream sources.",
                        "Enable rate limiting and SYN flood protection.",
                    ]
                    if idx % 2 == 0
                    else ["No immediate containment action required."]
                ),
            }
        )

    return {
        "model_name": "random_forest",
        "predictions": predictions,
        "summary": {
            "prediction_count": row_count,
            "labels": {
                "DDoS": sum(
                    1 for item in predictions if item["predicted_label"] == "DDoS"
                ),
                "Normal": sum(
                    1 for item in predictions if item["predicted_label"] == "Normal"
                ),
            },
            "global_feature_importance": [
                {"feature": "num__f0", "mean_abs_shap": 0.42},
                {"feature": "num__f1", "mean_abs_shap": 0.21},
            ],
            "recommended_actions": ["Block offending IPs or upstream sources."],
        },
    }
