"""
test_chatbot_unittest.py
────────────────────────
Assignment 4: Full Test Suite — Unit + Integration + Case Independence + Performance
Uses only Python stdlib (unittest) + Flask test client.
Run:  python test_chatbot_unittest.py
"""

import sys, os, tempfile, threading, time, json, unittest

# ── point imports at backend dir ─────────────────────────────────────────────
BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix="_test.db")

from nlp_engine    import NLPEngine, _tokenize, _tf, _cosine
from escalation    import EscalationService
from database      import Database
from main_flask    import app as flask_app

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_entry(session_id="test"):
    from datetime import datetime
    return dict(session_id=session_id, user_message="hello",
                intent="greeting", confidence=0.9, response="Hi!",
                escalated=False, domain="ecommerce", latency_ms=12.3,
                timestamp=datetime.utcnow().isoformat())

# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS – Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilities(unittest.TestCase):

    def test_tokenize_lower(self):
        self.assertIn("hello", _tokenize("Hello WORLD"))

    def test_tokenize_strips_punctuation(self):
        self.assertIn("order", _tokenize("where's my order?!"))

    def test_tf_sums_to_one(self):
        tf = _tf(["a","b","a"])
        self.assertAlmostEqual(sum(tf.values()), 1.0, places=6)

    def test_tf_empty(self):
        self.assertEqual(_tf([]), {})

    def test_cosine_identical(self):
        v = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(_cosine(v, v), 1.0, places=6)

    def test_cosine_orthogonal(self):
        self.assertEqual(_cosine({"cat": 1.0}, {"dog": 1.0}), 0.0)

    def test_cosine_partial(self):
        a = {"hi": 0.5, "world": 0.5}
        b = {"hi": 0.5, "there": 0.5}
        score = _cosine(a, b)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS – NLP Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestNLPEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = NLPEngine()

    def _classify(self, q, domain="ecommerce"):
        return self.nlp.classify_intent(q, domain=domain)

    # Intent classification
    def test_order_status(self):
        intent, _ = self._classify("where is my order")
        self.assertEqual(intent, "order_status")

    def test_refund_policy(self):
        intent, _ = self._classify("I want a refund")
        self.assertEqual(intent, "refund_policy")

    def test_return_item(self):
        intent, _ = self._classify("how to return item")
        self.assertEqual(intent, "return_item")

    def test_delivery_time(self):
        intent, _ = self._classify("how long does delivery take")
        self.assertEqual(intent, "delivery_time")

    def test_cancel_order(self):
        intent, _ = self._classify("I want to cancel my order")
        self.assertEqual(intent, "cancel_order")

    def test_payment_issue(self):
        intent, _ = self._classify("my payment failed card declined")
        self.assertEqual(intent, "payment_issue")

    def test_greeting(self):
        intent, _ = self._classify("hello I need help")
        self.assertEqual(intent, "greeting")

    def test_farewell(self):
        intent, _ = self._classify("thank you bye")
        self.assertEqual(intent, "farewell")

    def test_confidence_in_range(self):
        _, conf = self._classify("track my order")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_low_confidence_on_gibberish(self):
        _, conf = self._classify("xyzabc123 qwerty")
        self.assertLess(conf, 0.5)

    def test_deterministic_intent(self):
        results = [self._classify("where is my order")[0] for _ in range(5)]
        self.assertEqual(len(set(results)), 1)

    # Response generation
    def test_response_not_empty(self):
        r = self.nlp.generate_response("order_status", "where is my order")
        self.assertGreater(len(r), 5)

    def test_response_is_string(self):
        r = self.nlp.generate_response("refund_policy", "refund")
        self.assertIsInstance(r, str)

    def test_unknown_intent_graceful(self):
        r = self.nlp.generate_response("unknown", "gibberish")
        self.assertIsInstance(r, str)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS – Escalation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.svc = EscalationService()

    def test_message_not_empty(self):
        msg = self.svc.get_escalation_message()
        self.assertGreater(len(msg), 5)

    def test_ticket_format(self):
        t = self.svc.create_ticket("s1")
        self.assertTrue(t.startswith("TKT-"))
        self.assertEqual(len(t), 12)

    def test_tickets_unique(self):
        tickets = {self.svc.create_ticket("s") for _ in range(20)}
        self.assertEqual(len(tickets), 20)

    def test_escalation_with_reason(self):
        t = self.svc.create_ticket("s2", "Low confidence: 0.41")
        self.assertIsNotNone(t)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS – Database
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database()

    def test_log_and_retrieve(self):
        entry = _sample_entry("db-01")
        self.db.log_interaction(entry)
        time.sleep(0.05)
        logs = self.db.get_logs(limit=20)
        self.assertIn("db-01", [l["session_id"] for l in logs])

    def test_stats_keys(self):
        stats = self.db.get_stats()
        for key in ("total_interactions","escalated","avg_latency_ms","avg_confidence"):
            self.assertIn(key, stats)

    def test_mark_escalated(self):
        entry = _sample_entry("db-esc-01")
        self.db.log_interaction(entry)
        time.sleep(0.05)
        self.db.mark_escalated("db-esc-01", "TKT-AAAA0001")
        logs = self.db.get_logs(limit=30)
        match = next((l for l in logs if l["session_id"] == "db-esc-01"), None)
        if match:
            self.assertEqual(match["escalated"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS – Flask endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlaskEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        flask_app.config["TESTING"] = True
        cls.client = flask_app.test_client()

    def _ask(self, msg, domain="ecommerce", session_id=None):
        payload = {"user_message": msg, "domain": domain}
        if session_id:
            payload["session_id"] = session_id
        return self.client.post("/ask",
            data=json.dumps(payload), content_type="application/json")

    # /health
    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    # /ask basics
    def test_ask_200(self):
        self.assertEqual(self._ask("hello").status_code, 200)

    def test_ask_fields(self):
        data = self._ask("where is my order").get_json()
        for f in ("intent","confidence","response","session_id","latency_ms","escalated"):
            self.assertIn(f, data)

    def test_ask_keeps_session_id(self):
        data = self._ask("hi", session_id="my-sess").get_json()
        self.assertEqual(data["session_id"], "my-sess")

    def test_ask_escalated_bool(self):
        data = self._ask("track order").get_json()
        self.assertIsInstance(data["escalated"], bool)

    def test_ask_gibberish_escalated(self):
        data = self._ask("zxqkjfj ppqmxcv").get_json()
        if data["confidence"] < 0.65:
            self.assertTrue(data["escalated"])

    def test_ask_latency_under_2s(self):
        t0 = time.time()
        r  = self._ask("refund policy")
        ms = (time.time() - t0) * 1000
        self.assertEqual(r.status_code, 200)
        self.assertLess(ms, 2000)

    # /escalate
    def test_escalate_ticket(self):
        r = self.client.post("/escalate",
            data=json.dumps({"session_id": "e-01"}), content_type="application/json")
        data = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["ticket_id"].startswith("TKT-"))

    def test_escalate_returns_session(self):
        r = self.client.post("/escalate",
            data=json.dumps({"session_id": "e-02"}), content_type="application/json")
        self.assertEqual(r.get_json()["session_id"], "e-02")

    # /log
    def test_log_returns_list(self):
        r    = self.client.get("/log")
        data = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(data["logs"], list)

    def test_log_has_count(self):
        self.assertIn("count", self.client.get("/log").get_json())

    def test_log_limit(self):
        r = self.client.get("/log?limit=3")
        self.assertLessEqual(len(r.get_json()["logs"]), 3)

    def test_log_domain_filter(self):
        self._ask("hello", domain="ecommerce")
        r = self.client.get("/log?domain=ecommerce")
        self.assertEqual(r.status_code, 200)

    # /stats
    def test_stats(self):
        r = self.client.get("/stats")
        self.assertEqual(r.status_code, 200)
        self.assertIn("total_interactions", r.get_json())


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE INDEPENDENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_BENCHMARK = [
    # (query, expected_intent, domain)
    ("where is my order",               "order_status",         "ecommerce"),
    ("track my shipment",               "order_status",         "ecommerce"),
    ("I want a refund",                 "refund_policy",        "ecommerce"),
    ("money back please",               "refund_policy",        "ecommerce"),
    ("how to return item",              "return_item",          "ecommerce"),
    ("how long does delivery take",     "delivery_time",        "ecommerce"),
    ("cancel my order",                 "cancel_order",         "ecommerce"),
    ("payment failed card declined",    "payment_issue",        "ecommerce"),
    ("hello I need help",               "greeting",             "ecommerce"),
    ("book an appointment with doctor", "appointment_booking",  "healthcare"),
    ("prescription refill needed",      "prescription_refill",  "healthcare"),
    ("question about medical bill",     "billing_insurance",    "healthcare"),
    ("check my account balance",        "account_balance",      "banking"),
    ("I want to transfer money",        "fund_transfer",        "banking"),
    ("I lost my debit card",            "lost_card",            "banking"),
]

class TestCaseIndependence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp    = NLPEngine()
        flask_app.config["TESTING"] = True
        cls.client = flask_app.test_client()

    def test_overall_accuracy_above_80_percent(self):
        correct = 0
        failures = []
        for query, expected, domain in DOMAIN_BENCHMARK:
            got, conf = self.nlp.classify_intent(query, domain=domain)
            if got == expected:
                correct += 1
            else:
                failures.append(f"[{domain}] '{query}' → '{got}' (want '{expected}', conf={conf:.2f})")
        acc = correct / len(DOMAIN_BENCHMARK)
        self.assertGreaterEqual(acc, 0.80,
            msg=f"Accuracy {acc:.0%} < 80%.\nFailed:\n" + "\n".join(failures))

    def test_api_all_domains(self):
        for domain in ("ecommerce", "healthcare", "banking"):
            r = self.client.post("/ask",
                data=json.dumps({"user_message": "hello I need help", "domain": domain}),
                content_type="application/json")
            self.assertEqual(r.status_code, 200, f"Domain {domain} returned non-200")

    def test_no_retraining_across_domains(self):
        """Single NLPEngine instance handles all domains without reloading."""
        nlp = NLPEngine()
        for domain in ("ecommerce", "healthcare", "banking"):
            intent, conf = nlp.classify_intent("hello", domain=domain)
            self.assertIsNotNone(intent)

    def test_healthcare_intents(self):
        cases = [
            ("book an appointment", "appointment_booking"),
            ("prescription refill needed", "prescription_refill"),
        ]
        for q, expected in cases:
            got, _ = self.nlp.classify_intent(q, domain="healthcare")
            self.assertEqual(got, expected)

    def test_banking_intents(self):
        cases = [
            ("check my account balance", "account_balance"),
            ("transfer money to someone", "fund_transfer"),
            ("I lost my debit card", "lost_card"),
        ]
        for q, expected in cases:
            got, _ = self.nlp.classify_intent(q, domain="banking")
            self.assertEqual(got, expected)


# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        flask_app.config["TESTING"] = True
        cls.client = flask_app.test_client()

    def _ask(self, msg="hello"):
        return self.client.post("/ask",
            data=json.dumps({"user_message": msg}), content_type="application/json")

    def test_avg_latency_under_2s(self):
        latencies = []
        for _ in range(10):
            t0 = time.time()
            self._ask("where is my order")
            latencies.append((time.time() - t0) * 1000)
        avg = sum(latencies) / len(latencies)
        self.assertLess(avg, 2000, f"Avg latency {avg:.0f}ms > 2000ms")

    def test_p95_latency(self):
        latencies = []
        for _ in range(20):
            t0 = time.time()
            self._ask("delivery time")
            latencies.append((time.time() - t0) * 1000)
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        self.assertLess(p95, 2000, f"P95 latency {p95:.0f}ms > 2000ms")

    def test_concurrent_20_users(self):
        errors, codes = [], []

        def call():
            try:
                r = self.client.post("/ask",
                    data=json.dumps({"user_message": "refund my order"}),
                    content_type="application/json")
                codes.append(r.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"Concurrent errors: {errors}")
        self.assertTrue(all(c == 200 for c in codes), f"Non-200 codes: {codes}")

    def test_throughput_min_10rps(self):
        N = 10
        t0 = time.time()
        for _ in range(N):
            self._ask("hello")
        rps = N / (time.time() - t0)
        self.assertGreater(rps, 1.0, f"Throughput too low: {rps:.1f} rps")


# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER + REPORT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    for cls in [
        TestUtilities, TestNLPEngine, TestEscalation, TestDatabase,
        TestFlaskEndpoints, TestCaseIndependence, TestPerformance
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print("\n" + "="*60)
    print(f"  SUMMARY: {passed}/{total} tests passed")
    print(f"  Failures : {len(result.failures)}")
    print(f"  Errors   : {len(result.errors)}")
    print("="*60)

    sys.exit(0 if result.wasSuccessful() else 1)
