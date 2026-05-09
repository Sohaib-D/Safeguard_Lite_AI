# Groq SOC Analyst Assistant Integration

## Overview
This integration adds an AI-powered SOC analyst assistant to the Safeguard IDS platform using the Groq API.
It is designed to consume structured security context and produce:
- threat summaries
- risk assessments
- remediation recommendations
- incident timelines
- event correlation
- false positive analysis
- incident reports
- SHAP explanation translation into analyst-readable language

## Architecture

Components:
- `backend/services/groq_service.py`
  - async Groq API wrapper
  - rate limiting
  - retry/backoff logic
  - privacy-preserving prompt building
  - structured JSON response normalization
  - cache support for repeated queries
- `backend/schemas/groq.py`
  - request and response models for the SOC assistant
- `backend/api/main.py`
  - new endpoint: `POST /api/v1/soc/analyze`
- `frontend/api_client.py`
  - client support for the `soc/analyze` endpoint
- `frontend/app.py`
  - new `SOC Assistant` tab in Streamlit
  - allows analysts to submit detection context and view structured AI output

## Prompt Engineering Strategy

The assistant prompt is built to:
- preserve privacy by using only structured context fields
- explicitly avoid inventing new identities or payload values
- require a strict JSON return schema
- keep output analyst readable
- explain SHAP output in plain English
- treat false positive analysis as a first-class output
- correlate suspicious events with threat intelligence

The prompt includes:
- packet metadata
- detection result
- threat intelligence
- SHAP explanations
- historical events
- system metrics
- optional analyst notes

Example payload sections:
- `packet_metadata`
- `detection_result`
- `threat_intelligence`
- `shap_explanations`
- `historical_events`
- `system_metrics`

## Structured JSON Output Schema

The assistant is expected to return a JSON object with the following keys:
- `threat_summary`
- `risk_assessment`
- `remediation_recommendations`
- `incident_timeline`
- `false_positive_analysis`
- `correlated_events`
- `shap_explanation`
- `incident_report`

## Caching Strategy

The wrapper caches responses in `cache/groq` using a SHA-256 hash derived from the request payload.
- cache TTL is configurable via `GROQ_CACHE_TTL`
- repeated identical requests return cached structured output
- cache avoids extra Groq calls for transient repeated analysis

## Rate Limiting and Retry Logic

The service uses a simple rate limiter to space Groq requests across the configured limit.
Retry behavior includes exponential backoff and jitter for transient HTTP errors.

Configurable settings:
- `GROQ_API_URL`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_RATE_LIMIT_PER_MINUTE`
- `GROQ_MAX_RETRIES`
- `GROQ_CACHE_DIR`
- `GROQ_CACHE_TTL`

## FastAPI Integration

New endpoint:
- `POST /api/v1/soc/analyze`

Input is validated by `GroqAssistantRequest`.
Output is returned as `GroqAssistantResponse`.

## Streamlit Analyst Interface

The new `SOC Assistant` tab:
- accepts a structured JSON request payload
- pre-populates the payload from the latest prediction result when available
- submits to `/api/v1/soc/analyze`
- renders:
  - threat summary
  - risk assessment
  - remediation recommendations
  - incident timeline
  - false positive analysis
  - correlated events
  - SHAP explanation
  - incident report
  - raw AI response for auditing

## Example Prompt

```text
You are an AI SOC analyst assistant.
Use only the structured information below. Do not invent any new packet fields, user identities, or confidential values.
Redact or omit any sensitive fields. Return only valid JSON with the keys listed in the output schema.

Input:
packet_metadata: {...}
detection_result: {...}
threat_intelligence: [...]
shap_explanations: {...}
historical_events: [...]
system_metrics: {...}
analyst_notes: ...

Output format:
{
  "threat_summary": "...",
  "risk_assessment": "...",
  "remediation_recommendations": ["..."],
  "incident_timeline": [{"timestamp": "...", "event": "...", "detail": "..."}],
  "false_positive_analysis": "...",
  "correlated_events": [{"event_id": "...", "correlation_reason": "...", "shared_indicator": "..."}],
  "shap_explanation": "...",
  "incident_report": "..."
}
```

## Deployment Notes

- Set `GROQ_API_KEY` in environment variables.
- Adjust `GROQ_RATE_LIMIT_PER_MINUTE` to match your Groq plan.
- Ensure the backend can reach the Groq endpoint from production.
- Use the Streamlit SOC Assistant tab for analyst triage and tabletop incident generation.
