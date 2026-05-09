"""Unit tests for the modular detection engine."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from backend.schemas.alert import AlertSeverity
from backend.services.alert_service import AlertService
from backend.services.detection_engine import (
    DetectionPipeline,
    ThresholdDetector,
    SignatureDetector,
    RuleBasedDetector,
    BehavioralDetector,
    DetectionResult,
)
from backend.services.rule_parser import RuleParser
from backend.services.websocket_manager import WebSocketManager
from backend.db.sqlite_store import SQLiteStore


class DummyModelService:
    def predict(self, raw_df, include_explanations=False):
        return {"summary": {"confidence": 0.95}}


@pytest.fixture
def temp_db(tmp_path: Path):
    db_path = tmp_path / "test_rules.db"
    return str(db_path)


@pytest.mark.asyncio
async def test_signature_detector_matches():
    detector = SignatureDetector(
        name="suspicious_dns",
        description="Suspicious DNS requests",
        matcher=lambda f: f.get("dns_query_count", 0) > 5,
        severity=AlertSeverity.MEDIUM,
    )

    features = {"dns_query_count": 8}
    result = await detector.detect(features)
    assert result is not None
    assert result.alert_type == "suspicious_dns"
    assert result.severity == AlertSeverity.MEDIUM


@pytest.mark.asyncio
async def test_threshold_detector_triggers():
    detector = ThresholdDetector(
        name="rapid_port_scan",
        description="Rapid port scan",
        threshold=10,
        severity=AlertSeverity.HIGH,
        field_name="unique_ports",
    )
    result = await detector.detect({"unique_ports": 12})
    assert result is not None
    assert result.description.startswith("Rapid port scan")


@pytest.mark.asyncio
async def test_behavioral_detector():
    detector = BehavioralDetector(
        name="failed_logins",
        description="Failed login anomaly",
        detector_fn=lambda f: f.get("failed_login_count", 0) >= 5,
        severity=AlertSeverity.HIGH,
    )
    result = await detector.detect({"failed_login_count": 5})
    assert result is not None


def test_rule_parser(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    yaml_path = rules_dir / "failed_login.yaml"
    yaml_path.write_text(
        """
name: failed_login_threshold
description: Excessive failed login attempts
enabled: true
severity: high
score: 0.9
conditions:
  - field: failed_login_count
    op: greater_than
    value: 5
"""
    )

    parser = RuleParser(str(rules_dir))
    matched = parser.evaluate({"failed_login_count": 6})
    assert len(matched) == 1
    assert matched[0]["name"] == "failed_login_threshold"


@pytest.mark.asyncio
async def test_detection_pipeline_creates_alert(temp_db):
    websocket_manager = WebSocketManager()
    alert_service = AlertService(temp_db, websocket_manager)
    rule_parser = RuleParser("./does_not_exist")
    model_service = DummyModelService()
    pipeline = DetectionPipeline(alert_service, rule_parser, model_service=model_service)

    features = {
        "src_ip": "10.0.0.1",
        "unique_ports": 20,
        "bytes_per_minute": 1_500_000,
        "failed_login_count": 0,
        "dns_query_count": 1,
    }

    alerts = await pipeline.process(features)
    assert isinstance(alerts, list)
    assert len(alerts) >= 1
    assert alerts[0].severity in {AlertSeverity.HIGH, AlertSeverity.MEDIUM}
