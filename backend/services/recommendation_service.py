from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationRule:
    """Maps a predicted label to actionable response guidance."""

    labels: tuple[str, ...]
    suggestions: tuple[str, ...]
    severity: str = "info"


class RecommendationService:
    """Rule-based recommendation engine for predicted attack classes."""

    def __init__(self) -> None:
        self.rules: list[RecommendationRule] = [
            RecommendationRule(
                labels=("ddos", "dos"),
                suggestions=(
                    "Block offending IPs or upstream sources.",
                    "Enable rate limiting and SYN flood protection.",
                    "Scale edge filtering or WAF protections temporarily.",
                ),
                severity="critical",
            ),
            RecommendationRule(
                labels=("portscan", "reconnaissance", "scan"),
                suggestions=(
                    "Block or tarp it repeated probing sources.",
                    "Tighten firewall rules and close unnecessary ports.",
                    "Increase monitoring on adjacent hosts and subnets.",
                ),
                severity="high",
            ),
            RecommendationRule(
                labels=(
                    "bruteforce",
                    "bruteforce",
                    "brute_force",
                    "ssh-bruteforce",
                    "ftp-bruteforce",
                ),
                suggestions=(
                    "Lock or challenge the targeted user account.",
                    "Force password reset and enable MFA.",
                    "Apply login throttling and IP-based lockout controls.",
                ),
                severity="high",
            ),
            RecommendationRule(
                labels=("botnet",),
                suggestions=(
                    "Isolate the suspected host from the network.",
                    "Block command-and-control destinations immediately.",
                    "Run endpoint malware and persistence checks.",
                ),
                severity="critical",
            ),
            RecommendationRule(
                labels=("infiltration", "exploit", "webattack", "web attack", "attack"),
                suggestions=(
                    "Quarantine the affected endpoint or service.",
                    "Review recent access logs and privilege changes.",
                    "Patch exposed services and rotate sensitive credentials.",
                ),
                severity="critical",
            ),
            RecommendationRule(
                labels=("normal", "benign"),
                suggestions=(
                    "No immediate containment action required.",
                    "Continue passive monitoring for repeated anomalies.",
                ),
                severity="info",
            ),
        ]
        self.default_rule = RecommendationRule(
            labels=("default",),
            suggestions=(
                "Review the traffic sample manually.",
                "Increase monitoring and validate related hosts.",
            ),
            severity="medium",
        )

    def get_recommendation(self, predicted_label: str) -> dict[str, object]:
        """Return matching response guidance for a predicted label."""
        normalized = str(predicted_label).strip().lower()
        compact = normalized.replace(" ", "").replace("-", "").replace("_", "")

        for rule in self.rules:
            for candidate in rule.labels:
                candidate_normalized = candidate.strip().lower()
                candidate_compact = (
                    candidate_normalized.replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )
                if (
                    normalized == candidate_normalized
                    or compact == candidate_compact
                    or candidate_normalized in normalized
                    or candidate_compact in compact
                ):
                    return {
                        "severity": rule.severity,
                        "suggestions": list(rule.suggestions),
                    }

        return {
            "severity": self.default_rule.severity,
            "suggestions": list(self.default_rule.suggestions),
        }
