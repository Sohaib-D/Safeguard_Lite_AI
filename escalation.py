"""
escalation.py
─────────────
Handles escalation logic: ticket generation, agent routing messages.
"""

import uuid
import random


ESCALATION_MESSAGES = [
    "I'm connecting you with a human agent who can better assist you. Please hold.",
    "This query requires specialist attention. A support agent will contact you shortly.",
    "I've raised a support ticket for your query. An agent will respond within 24 hours.",
    "Let me escalate this to our team. You'll receive an email with your ticket reference.",
]


class EscalationService:
    def get_escalation_message(self) -> str:
        return random.choice(ESCALATION_MESSAGES)

    def create_ticket(self, session_id: str, reason: str = "Manual escalation") -> str:
        """Generate a unique ticket ID and (in production) notify agents."""
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        # TODO: integrate email/webhook notification to agent queue
        return ticket_id
