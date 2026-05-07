from __future__ import annotations

import pandas as pd

from tests.conftest import fake_prediction_result


def test_end_to_end_upload_csv_prediction_output_format(
    client, auth_headers, api_module, monkeypatch
):
    monkeypatch.setattr(
        api_module.model_service,
        "predict",
        lambda **kwargs: fake_prediction_result(row_count=1),
    )

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
            },
        ]
    )
    file_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict",
        headers=auth_headers,
        files={"file": ("events.csv", file_bytes, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"model_name", "predictions", "summary", "timestamp"}
    assert isinstance(body["predictions"], list)
    assert body["predictions"][0]["predicted_label"] == "DDoS"
    assert isinstance(body["predictions"][0]["recommendations"], list)
    assert isinstance(body["summary"]["global_feature_importance"], list)


def test_end_to_end_predict_rejects_bad_numeric_value(client, auth_headers):
    df = pd.DataFrame(
        [
            {
                "f0": "oops",
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
        ]
    )
    file_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict",
        headers=auth_headers,
        files={"file": ("events.csv", file_bytes, "text/csv")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Data type validation failed."
    assert "must be numeric" in body["errors"][0]
