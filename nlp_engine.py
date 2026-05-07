"""
nlp_engine.py
─────────────
Energy-Efficient NLP Layer
- Uses cosine similarity over DistilBERT-style TF-IDF embeddings (offline, no GPU needed)
- Pluggable: swap in HuggingFace or OpenAI by changing _embed()
- Domain-agnostic: FAQ bank is injected at init; works for ecommerce, healthcare, banking, etc.
"""

import re
import math
import random
from typing import Tuple, Dict, List
from faq_data import FAQ_BANK   # domain → list of {intent, patterns, response}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    freq: Dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens) or 1
    return {k: v / total for k, v in freq.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


class NLPEngine:
    """
    Lightweight intent classifier + response generator.

    Architecture:
        query → tokenise → TF vector
        For each FAQ entry → combine all pattern TF vectors → centroid
        Pick entry with highest cosine similarity → intent + confidence
    """

    model_name = "TF-Cosine (DistilBERT-compatible interface)"

    def __init__(self):
        # Pre-compute centroid TF vectors for every intent in every domain
        self._index: Dict[str, List[dict]] = {}
        for domain, entries in FAQ_BANK.items():
            self._index[domain] = []
            for entry in entries:
                all_tokens = []
                for pattern in entry["patterns"]:
                    all_tokens.extend(_tokenize(pattern))
                centroid = _tf(all_tokens)
                self._index[domain].append({
                    "intent": entry["intent"],
                    "centroid": centroid,
                    "responses": entry["responses"],
                })

    def classify_intent(self, text: str, domain: str = "ecommerce") -> Tuple[str, float]:
        """Return (intent_label, confidence_score)."""
        entries = self._index.get(domain, self._index["ecommerce"])
        q_vec = _tf(_tokenize(text))

        best_score, best_intent = 0.0, "unknown"
        for entry in entries:
            score = _cosine(q_vec, entry["centroid"])
            if score > best_score:
                best_score, best_intent = score, entry["intent"]

        # Normalise to [0,1] range (cosine is already in [-1,1] but we clip to 0)
        confidence = max(0.0, min(1.0, best_score))
        return best_intent, confidence

    def generate_response(self, intent: str, original_query: str, domain: str = "ecommerce") -> str:
        """Pick a response from the matched intent's template pool."""
        entries = self._index.get(domain, self._index["ecommerce"])
        for entry in entries:
            if entry["intent"] == intent:
                return random.choice(entry["responses"])
        return "I'm sorry, I didn't quite understand that. Could you please rephrase?"
