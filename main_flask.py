"""
main_flask.py
─────────────
Flask backend for AI Customer Support Chatbot
Endpoints: /health  /ask  /escalate  /log  /stats
"""

from flask import Flask, request, jsonify
import time, uuid, threading
from datetime import datetime

from nlp_engine import NLPEngine
from database import Database
from escalation import EscalationService

app = Flask(__name__)
nlp = NLPEngine()
db  = Database()
esc = EscalationService()

CONFIDENCE_THRESHOLD = 0.65


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": nlp.model_name,
                    "timestamp": datetime.utcnow().isoformat()})


@app.post("/ask")
def ask():
    body        = request.get_json(force=True)
    user_msg    = body.get("user_message", "")
    domain      = body.get("domain", "ecommerce")
    session_id  = body.get("session_id") or str(uuid.uuid4())

    t0 = time.perf_counter()
    intent, conf = nlp.classify_intent(user_msg, domain=domain)
    response     = nlp.generate_response(intent, user_msg, domain=domain)
    escalated    = conf < CONFIDENCE_THRESHOLD
    if escalated:
        response = esc.get_escalation_message()
    latency_ms   = (time.perf_counter() - t0) * 1000

    entry = dict(session_id=session_id, user_message=user_msg, intent=intent,
                 confidence=round(conf, 4), response=response, escalated=escalated,
                 domain=domain, latency_ms=round(latency_ms, 2),
                 timestamp=datetime.utcnow().isoformat())

    threading.Thread(target=db.log_interaction, args=(entry,), daemon=True).start()

    return jsonify(dict(session_id=session_id, intent=intent,
                        confidence=round(conf, 4), response=response,
                        escalated=escalated, latency_ms=round(latency_ms, 2),
                        timestamp=entry["timestamp"]))


@app.post("/escalate")
def escalate():
    body       = request.get_json(force=True)
    session_id = body.get("session_id", "")
    reason     = body.get("reason", "Manual escalation")
    ticket_id  = esc.create_ticket(session_id, reason)
    db.mark_escalated(session_id, ticket_id)
    return jsonify(dict(ticket_id=ticket_id, session_id=session_id,
                        message="Escalated. An agent will contact you within 24 hours.",
                        timestamp=datetime.utcnow().isoformat()))


@app.get("/log")
def get_log():
    limit  = int(request.args.get("limit", 50))
    domain = request.args.get("domain")
    logs   = db.get_logs(limit=limit, domain=domain)
    return jsonify({"count": len(logs), "logs": logs})


@app.get("/stats")
def stats():
    return jsonify(db.get_stats())


if __name__ == "__main__":
    app.run(debug=False, port=5000)
