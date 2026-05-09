from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from backend.schemas.groq import GroqAssistantRequest, GroqAssistantResponse


class GroqRateLimiter:
    def __init__(self, calls_per_minute: int = 20):
        self.calls_per_minute = max(1, calls_per_minute)
        self.interval = 60.0 / self.calls_per_minute
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delta = self.interval - (now - self._last_call)
            if delta > 0:
                await asyncio.sleep(delta)
            self._last_call = time.monotonic()


class GroqAssistant:
    """Async Groq SOC analyst wrapper with privacy-safe prompts, retries, and cache."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "groq-alpha",
        rate_limit_per_minute: int = 15,
        max_retries: int = 3,
        cache_dir: str = "cache/groq",
        cache_ttl_seconds: int = 900,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.rate_limiter = GroqRateLimiter(rate_limit_per_minute)
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

    async def analyze(self, payload: GroqAssistantRequest) -> GroqAssistantResponse:
        payload_dict = payload.model_dump(mode="json")
        cache_key = self._build_cache_key(payload_dict)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return GroqAssistantResponse(**cached)

        prompt = self._build_prompt(payload_dict)
        response_data = await self._invoke_groq(prompt)
        analysis = self._parse_response(response_data)
        self._save_cache(cache_key, analysis)
        return GroqAssistantResponse(**analysis)

    def _build_cache_key(self, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            timestamp = datetime.fromisoformat(raw.get("cached_at"))
            if datetime.utcnow() - timestamp > self.cache_ttl:
                return None
            return raw.get("analysis")
        except Exception:
            return None

    def _save_cache(self, cache_key: str, analysis: dict[str, Any]) -> None:
        path = self._cache_path(cache_key)
        payload = {
            "cached_at": datetime.utcnow().isoformat(),
            "analysis": analysis,
        }
        path.write_text(json.dumps(payload, indent=2))

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        packet_metadata = json.dumps(payload.get("packet_metadata", {}), indent=2)
        detection_result = json.dumps(payload.get("detection_result", {}), indent=2)
        threat_intel = json.dumps(payload.get("threat_intelligence", []), indent=2)
        shap_explanations = json.dumps(payload.get("shap_explanations", {}), indent=2)
        historical_events = json.dumps(payload.get("historical_events", []), indent=2)
        system_metrics = json.dumps(payload.get("system_metrics", {}), indent=2)
        analyst_notes = payload.get("analyst_notes") or ""

        return f"""
You are Safeguard-AI Lite, an educational cybersecurity assistant. Your job is to explain network security alerts to users who have absolutely zero technical knowledge.

Use only the structured information below. Do not invent any new packet fields, user identities, or confidential values.
Redact or omit any sensitive fields. Return only valid JSON with the keys listed in the output schema.

Input:
packet_metadata: {packet_metadata}
detection_result: {detection_result}
threat_intelligence: {threat_intel}
shap_explanations: {shap_explanations}
historical_events: {historical_events}
system_metrics: {system_metrics}
analyst_notes: {analyst_notes}

Output format:
{{
  "threat_summary": "...",
  "risk_assessment": "...",
  "remediation_recommendations": ["..."],
  "incident_timeline": [{{"timestamp": "...", "event": "...", "detail": "..."}}],
  "false_positive_analysis": "...",
  "correlated_events": [{{"event_id": "...", "correlation_reason": "...", "shared_indicator": "..."}}],
  "shap_explanation": "...",
  "incident_report": "..."
}}

Instructions:
- Use extremely simple, beginner-friendly English (Plain English).
- NEVER use technical jargon without explaining it immediately with an analogy. (e.g., "Port 22 is like a secure back door").
- Always explain WHAT happened, WHY it matters, the RISK level, and what they should DO.
- Keep answers short, structured, and reassuring. If a threat is a false positive or low risk, explicitly tell them not to worry.
- Provide actionable, easy-to-follow steps for remediation.
- Do not return markdown, tables or any wrapper text outside the JSON object.


If any field is missing from the input, infer the most likely answer using available context.
"""

    async def _invoke_groq(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("Groq API key is not configured.")

        await self.rate_limiter.wait()
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        self.api_url,
                        json=body,
                        headers=headers,
                    )
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                retry_delay = min(2 ** (attempt - 1) + random.random(), 15)
                if attempt < self.max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                raise RuntimeError(
                    "Groq API request failed after retries"
                ) from exc
        raise RuntimeError("Groq API request failed") from last_error

    def _parse_response(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        choices = response_payload.get("choices", [])
        if choices and "message" in choices[0]:
            raw_output = choices[0]["message"].get("content", "")
        else:
            raw_output = response_payload.get("output") or response_payload.get("result") or response_payload
        if isinstance(raw_output, dict):
            return self._normalize_response(raw_output)

        if isinstance(raw_output, str):
            text = raw_output.strip()
            structured = self._extract_json_from_text(text)
            if structured is not None:
                return self._normalize_response(structured)
            return self._fallback_response(text, response_payload)

        return self._fallback_response(json.dumps(raw_output), response_payload)

    def _extract_json_from_text(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    def _normalize_response(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "threat_summary": str(data.get("threat_summary", "No summary generated.")),
            "risk_assessment": str(data.get("risk_assessment", "Unable to assess risk.")),
            "remediation_recommendations": list(data.get("remediation_recommendations", [])) if data.get("remediation_recommendations") is not None else [],
            "incident_timeline": list(data.get("incident_timeline", [])) if data.get("incident_timeline") is not None else [],
            "false_positive_analysis": str(data.get("false_positive_analysis", "Unable to determine false positive likelihood.")),
            "correlated_events": list(data.get("correlated_events", [])) if data.get("correlated_events") is not None else [],
            "shap_explanation": str(data.get("shap_explanation", "No SHAP explanation available.")),
            "incident_report": str(data.get("incident_report", "No incident report generated.")),
            "raw_response": data,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _fallback_response(self, text: str, raw_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "threat_summary": text[:1024],
            "risk_assessment": "Unable to parse structured response; review raw output.",
            "remediation_recommendations": [],
            "incident_timeline": [],
            "false_positive_analysis": "Review required.",
            "correlated_events": [],
            "shap_explanation": "Review raw model output.",
            "incident_report": text[:2048],
            "raw_response": raw_payload,
            "generated_at": datetime.utcnow().isoformat(),
        }
