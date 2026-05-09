from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class RuleParser:
    def __init__(self, rules_path: str = "rules"):
        self.rules_path = Path(rules_path)
        self.rules: list[dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        self.rules.clear()
        self.rules_path.mkdir(parents=True, exist_ok=True)
        for yaml_file in sorted(self.rules_path.glob("*.yaml")):
            try:
                with yaml_file.open("r", encoding="utf-8") as handle:
                    rule = yaml.safe_load(handle)
                if rule and rule.get("enabled", True):
                    rule["source_file"] = str(yaml_file.name)
                    self.rules.append(rule)
            except Exception:
                continue

    def evaluate(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for rule in self.rules:
            if self._matches_rule(rule, features):
                matches.append(rule)
        return matches

    def _matches_rule(self, rule: dict[str, Any], features: dict[str, Any]) -> bool:
        for condition in rule.get("conditions", []):
            field = condition.get("field")
            op = condition.get("op", "equals")
            value = condition.get("value")
            if field is None:
                return False
            feature_value = features.get(field)
            if not self._evaluate_condition(feature_value, op, value):
                return False
        return True

    def _evaluate_condition(self, feature_value: Any, op: str, value: Any) -> bool:
        if op == "equals":
            return feature_value == value
        if op == "not_equals":
            return feature_value != value
        if op == "greater_than":
            return feature_value is not None and feature_value > value
        if op == "less_than":
            return feature_value is not None and feature_value < value
        if op == "greater_or_equal":
            return feature_value is not None and feature_value >= value
        if op == "less_or_equal":
            return feature_value is not None and feature_value <= value
        if op == "contains":
            return value in (feature_value or "")
        if op == "regex":
            try:
                return bool(re.search(value, str(feature_value or "")))
            except re.error:
                return False
        return False
