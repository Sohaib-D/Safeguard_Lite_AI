from __future__ import annotations

import pandas as pd

from tests.conftest import fake_prediction_result


def test_protected_endpoint_requires_auth(client):
    response = client.get("/model_info")
    assert response.status_code == 401


def test_auth_flow_create_admin_and_login(client):
    create_resp = client.post(
        "/auth/create-admin",
        json={"username": "admin_user", "password": "StrongPass123"},
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["username"] == "admin_user"

    login_resp = client.post(
        "/auth/login",
        json={"username": "admin_user", "password": "StrongPass123"},
    )
    assert login_resp.status_code == 200
    body = login_resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_predict_json_endpoint_returns_expected_shape(
    client, auth_headers, api_module, monkeypatch
):
    monkeypatch.setattr(
        api_module.model_service,
        "predict",
        lambda **kwargs: fake_prediction_result(row_count=2),
    )

    payload = {
        "records": [
            {
                "f0": 0.1,
                "f1": 0.2,
                "f2": 0.3,
                "f3": 0.4,
                "f4": 0.5,
                "f5": 0.6,
                "f6": 0.7,
                "f7": 0.8,
                "f8": 0.9,
                "f9": 1.0,
                "service": "http",
            },
            {
                "f0": 1.1,
                "f1": 1.2,
                "f2": 1.3,
                "f3": 1.4,
                "f4": 1.5,
                "f5": 1.6,
                "f6": 1.7,
                "f7": 1.8,
                "f8": 1.9,
                "f9": 2.0,
                "service": "dns",
            },
        ],
        "include_explanations": True,
        "explanation_top_k": 3,
    }

    response = client.post("/predict", json=payload, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "random_forest"
    assert len(body["predictions"]) == 2
    assert {"predicted_label", "recommendations", "top_contributions"}.issubset(
        body["predictions"][0].keys()
    )
    assert "recommended_actions" in body["summary"]


def test_predict_csv_missing_columns_returns_clear_error(client, auth_headers):
    bad_df = pd.DataFrame([{"f0": 1, "f1": 2, "service": "http"}])
    file_bytes = bad_df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict",
        headers=auth_headers,
        files={"file": ("bad.csv", file_bytes, "text/csv")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Column validation failed."
    assert body["errors"]


def test_upload_endpoint_logs_activity(client, auth_headers):
    df = pd.DataFrame(
        [
            {
                "f0": 0.1,
                "f1": 0.2,
                "f2": 0.3,
                "f3": 0.4,
                "f4": 0.5,
                "f5": 0.6,
                "f6": 0.7,
                "f7": 0.8,
                "f8": 0.9,
                "f9": 1.0,
                "service": "http",
            }
        ]
    )
    file_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("sample.csv", file_bytes, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rows_logged"] == 1
    assert body["source_type"] == "csv"
