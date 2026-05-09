from __future__ import annotations

import abc
import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from shap import Explainer

from backend.schemas.alert import AlertSeverity, DetectionAlert
from backend.services.rule_parser import RuleParser
from backend.services.alert_service import AlertService
from backend.services.model_service import ModelService

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    alert_type: str
    threat_type: str
    description: str
    severity: AlertSeverity
    score: float
    confidence: float
    explanation: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    event_context: dict[str, Any] = field(default_factory=dict)


class BaseDetector(abc.ABC):
    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled

    @abc.abstractmethod
    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        raise NotImplementedError

    async def explain(self, features: dict[str, Any]) -> dict[str, Any]:
        return {}


class ThresholdDetector(BaseDetector):
    def __init__(self, name: str, description: str, threshold: float, severity: AlertSeverity, field_name: str):
        super().__init__(name, description)
        self.threshold = threshold
        self.severity = severity
        self.field_name = field_name

    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        value = features.get(self.field_name, 0)
        if value is not None and value > self.threshold:
            return DetectionResult(
                alert_type=self.name,
                threat_type=self.description,
                description=f"{self.description}: {self.field_name}={value} exceeded threshold {self.threshold}",
                severity=self.severity,
                score=min(float(value) / (self.threshold * 2), 1.0),
                confidence=min(float(value) / (self.threshold * 1.5), 1.0),
                event_context={self.field_name: value},
            )
        return None


class SignatureDetector(BaseDetector):
    def __init__(self, name: str, description: str, matcher: Callable[[dict[str, Any]], bool], severity: AlertSeverity, score: float = 0.7):
        super().__init__(name, description)
        self.matcher = matcher
        self.severity = severity
        self.score = score

    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        if self.matcher(features):
            return DetectionResult(
                alert_type=self.name,
                threat_type=self.description,
                description=f"Signature rule triggered: {self.description}",
                severity=self.severity,
                score=self.score,
                confidence=0.8,
                event_context=features.copy(),
            )
        return None


class RuleBasedDetector(BaseDetector):
    def __init__(self, rule_parser: RuleParser):
        super().__init__("custom_rule", "Custom YAML rule match")
        self.rule_parser = rule_parser

    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        matched = self.rule_parser.evaluate(features)
        if not matched:
            return None
        rule = matched[0]
        return DetectionResult(
            alert_type=rule.get("name", "yaml_rule"),
            threat_type=rule.get("description", "Custom rule match"),
            description=f"YAML rule matched: {rule.get('name')}",
            severity=AlertSeverity(rule.get("severity", AlertSeverity.LOW)),
            score=float(rule.get("score", 0.6)),
            confidence=0.75,
            explanation={"matched_rule": rule},
            event_context=features.copy(),
        )


class BehavioralDetector(BaseDetector):
    def __init__(self, name: str, description: str, detector_fn: Callable[[dict[str, Any]], bool], severity: AlertSeverity, score: float = 0.7):
        super().__init__(name, description)
        self.detector_fn = detector_fn
        self.severity = severity
        self.score = score

    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        if self.detector_fn(features):
            return DetectionResult(
                alert_type=self.name,
                threat_type=self.description,
                description=f"Behavioral anomaly detected: {self.description}",
                severity=self.severity,
                score=self.score,
                confidence=0.75,
                event_context=features.copy(),
            )
        return None


class MLAnomalyDetector(BaseDetector):
    def __init__(self, model_service: ModelService, shap_explainer: Explainer | None = None):
        super().__init__("ml_anomaly", "ML anomaly detection")
        self.model_service = model_service
        self.shap_explainer = shap_explainer
        self.feature_cache: deque[dict[str, Any]] = deque(maxlen=1000)

    async def detect(self, features: dict[str, Any]) -> Optional[DetectionResult]:
        self.feature_cache.append(features)
        try:
            df = pd.DataFrame([features])
            prediction = self.model_service.predict(raw_df=df, include_explanations=False)
            anomaly_score = float(prediction.get("summary", {}).get("confidence", 0.2))
            explanation = {}
            if self.shap_explainer is not None:
                explanation = await asyncio.to_thread(self._explain, df)
            if anomaly_score >= 0.6:
                return DetectionResult(
                    alert_type=self.name,
                    threat_type=self.description,
                    description="ML anomaly threshold exceeded",
                    severity=AlertSeverity.MEDIUM,
                    score=anomaly_score,
                    confidence=anomaly_score,
                    explanation=explanation,
                    event_context=features.copy(),
                )
        except Exception as exc:
            logger.error(f"ML anomaly detector failed: {exc}")
        return None

    def _explain(self, df: pd.DataFrame) -> dict[str, Any]:
        try:
            shap_values = self.shap_explainer(df)
            return {"shap_values": shap_values.values.tolist()}
        except Exception:
            return {}


class DetectionPipeline:
    def __init__(
        self,
        alert_service: AlertService,
        rule_parser: RuleParser,
        model_service: ModelService | None = None,
        shap_explainer: Explainer | None = None,
    ):
        self.alert_service = alert_service
        self.rule_parser = rule_parser
        self.detectors: list[BaseDetector] = []
        self.model_service = model_service
        self.shap_explainer = shap_explainer
        self._load_default_detectors()

    def _load_default_detectors(self) -> None:
        self.detectors.append(
            ThresholdDetector(
                name="rapid_port_scan",
                description="Rapid port scan detected",
                threshold=12,
                severity=AlertSeverity.HIGH,
                field_name="unique_ports",
            )
        )
        self.detectors.append(
            ThresholdDetector(
                name="abnormal_bandwidth",
                description="Abnormal bandwidth spike",
                threshold=1000000,
                severity=AlertSeverity.MEDIUM,
                field_name="bytes_per_minute",
            )
        )
        self.detectors.append(
            BehavioralDetector(
                name="failed_logins",
                description="Excessive failed logins",
                detector_fn=lambda f: f.get("failed_login_count", 0) >= 5,
                severity=AlertSeverity.HIGH,
                score=0.9,
            )
        )
        self.detectors.append(
            BehavioralDetector(
                name="credential_stuffing",
                description="Credential stuffing behavior",
                detector_fn=lambda f: f.get("login_attempts", 0) >= 15 and f.get("unique_usernames", 0) >= 4,
                severity=AlertSeverity.CRITICAL,
                score=0.95,
            )
        )
        self.detectors.append(
            SignatureDetector(
                name="suspicious_dns",
                description="Suspicious DNS request pattern",
                matcher=lambda f: f.get("dns_query_count", 0) > 20 or f.get("dns_query_length", 0) > 512,
                severity=AlertSeverity.MEDIUM,
                score=0.8,
            )
        )
        self.detectors.append(RuleBasedDetector(self.rule_parser))
        if self.model_service is not None:
            self.detectors.append(MLAnomalyDetector(self.model_service, self.shap_explainer))

    def register_detector(self, detector: BaseDetector) -> None:
        self.detectors.append(detector)

    async def process(self, features: dict[str, Any]) -> list[DetectionAlert]:
        alerts: list[DetectionAlert] = []
        for detector in self.detectors:
            if not detector.enabled:
                continue
            try:
                result = await detector.detect(features)
                if result is not None:
                    alert = DetectionAlert(
                        alert_type=result.alert_type,
                        threat_type=result.threat_type,
                        description=result.description,
                        severity=result.severity,
                        score=result.score,
                        confidence=result.confidence,
                        explanation=result.explanation,
                        event_context={**result.event_context, **result.details},
                    )
                    saved_alert = await self.alert_service.create_alert(alert)
                    alerts.append(saved_alert)
            except Exception as exc:
                logger.error(f"Detector {detector.name} failed: {exc}")
        return alerts


class DetectionEngine:
    def __init__(
        self,
        alert_service: AlertService,
        rule_parser: RuleParser,
        model_service: ModelService | None = None,
        shap_explainer: Explainer | None = None,
    ):
        self.pipeline = DetectionPipeline(
            alert_service=alert_service,
            rule_parser=rule_parser,
            model_service=model_service,
            shap_explainer=shap_explainer,
        )

    async def detect(self, features: dict[str, Any]) -> list[DetectionAlert]:
        return await self.pipeline.process(features)

    def register_detector(self, detector: BaseDetector) -> None:
        self.pipeline.register_detector(detector)
