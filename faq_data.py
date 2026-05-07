"""
faq_data.py
───────────
Synthetic, case-independent FAQ bank.
Supports three domains out-of-the-box:
  - ecommerce   (order status, refund, delivery, returns)
  - healthcare  (appointments, prescriptions, billing)
  - banking     (account balance, transfers, lost card)

Each entry has:
  intent    : machine label
  patterns  : training phrases (used to build cosine centroid)
  responses : randomly sampled reply templates
"""

FAQ_BANK = {
    # ─── E-COMMERCE ────────────────────────────────────────────────────────────
    "ecommerce": [
        {
            "intent": "order_status",
            "patterns": [
                "where is my order", "track my order", "order status",
                "when will my package arrive", "shipping status", "delivery update",
                "has my order shipped", "order tracking number", "check my order"
            ],
            "responses": [
                "You can track your order at our website under 'My Orders'. Typically orders ship within 2 business days.",
                "Your order status is available in the 'My Orders' section. Please allow 2–5 business days for delivery.",
                "To track your package, visit the tracking page and enter your order ID. Need help finding it?"
            ]
        },
        {
            "intent": "refund_policy",
            "patterns": [
                "how do I get a refund", "refund policy", "money back", "return my money",
                "claim refund", "refund request", "get my money back", "refund status",
                "can I return and get refund"
            ],
            "responses": [
                "Refunds are processed within 5–7 business days after we receive your return. You'll get a confirmation email.",
                "Our refund policy allows returns within 30 days. Once the item is received, the refund posts in 5–7 days.",
                "To initiate a refund, go to 'My Orders' → select item → 'Request Refund'. Processing takes 5–7 days."
            ]
        },
        {
            "intent": "return_item",
            "patterns": [
                "how to return", "return policy", "send item back", "exchange product",
                "return my purchase", "I want to return", "return request", "wrong item received"
            ],
            "responses": [
                "Returns are free within 30 days of delivery. Visit 'My Orders', select the item, and click 'Return'.",
                "You can return items within 30 days. Print the prepaid label from your account and drop it at any courier.",
                "To return an item: My Orders → Return Item → Print label. No charges for returns within 30 days."
            ]
        },
        {
            "intent": "delivery_time",
            "patterns": [
                "how long does delivery take", "estimated delivery", "shipping time",
                "when will I receive", "delivery days", "express shipping", "overnight delivery"
            ],
            "responses": [
                "Standard delivery takes 3–5 business days. Express (1–2 days) and overnight options are also available.",
                "Delivery times: Standard 3–5 days, Express 1–2 days, Overnight by next business day.",
                "Most orders arrive within 3–5 business days. Express shipping is available at checkout."
            ]
        },
        {
            "intent": "cancel_order",
            "patterns": [
                "cancel my order", "I want to cancel", "stop my order", "cancel purchase",
                "order cancellation", "did not want this"
            ],
            "responses": [
                "Orders can be cancelled within 1 hour of placement. Go to 'My Orders' → 'Cancel Order'.",
                "If your order hasn't shipped, you can cancel it from 'My Orders'. After shipping, please use our returns process.",
                "To cancel: My Orders → Select Order → Cancel. If shipped, we'll help you return it instead."
            ]
        },
        {
            "intent": "payment_issue",
            "patterns": [
                "payment failed", "card declined", "payment not processed", "billing error",
                "charge issue", "double charged", "payment problem", "incorrect charge"
            ],
            "responses": [
                "If your payment failed, please check your card details and try again. Contact your bank if the issue persists.",
                "Double charges are automatically reversed within 3–5 days. If not, please share your order ID.",
                "For payment issues, verify card details or try a different payment method. We support UPI, cards, and wallets."
            ]
        },
        {
            "intent": "account_help",
            "patterns": [
                "reset password", "login problem", "forgot password", "can't access account",
                "account locked", "create account", "change email", "update profile"
            ],
            "responses": [
                "To reset your password, click 'Forgot Password' on the login page. A reset link will be emailed to you.",
                "If your account is locked after multiple attempts, wait 30 minutes or use the password reset link.",
                "For account issues, use the 'Help' section or contact support with your registered email address."
            ]
        },
        {
            "intent": "greeting",
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon", "help me",
                "support", "I need help", "start"
            ],
            "responses": [
                "Hello! 👋 I'm your support assistant. How can I help you today?",
                "Hi there! I can help with orders, refunds, delivery, and more. What do you need?",
                "Welcome to support! Ask me about order status, returns, refunds, or any other query."
            ]
        },
        {
            "intent": "farewell",
            "patterns": [
                "bye", "goodbye", "thanks", "thank you", "that's all", "done", "exit", "quit"
            ],
            "responses": [
                "Thank you for contacting us! Have a great day. 😊",
                "Happy to help! Don't hesitate to reach out anytime.",
                "Goodbye! Your feedback helps us improve. Rated us? 🌟"
            ]
        }
    ],

    # ─── HEALTHCARE ─────────────────────────────────────────────────────────────
    "healthcare": [
        {
            "intent": "appointment_booking",
            "patterns": [
                "book appointment", "schedule doctor", "see a doctor", "make appointment",
                "available slots", "when can I see", "appointment booking", "consult doctor"
            ],
            "responses": [
                "To book an appointment, visit the Patient Portal or call us at our helpline. Slots are available Monday–Saturday.",
                "You can schedule an appointment online. Select your preferred doctor, date, and time in the portal.",
                "Appointments are available within 24–48 hours for most specialties. Book via the portal or our app."
            ]
        },
        {
            "intent": "prescription_refill",
            "patterns": [
                "refill prescription", "renew medicine", "medication refill", "need prescription",
                "prescription renewal", "order medicine again"
            ],
            "responses": [
                "Prescription refills can be requested through the portal under 'My Prescriptions'. Allow 2 business days.",
                "To renew your prescription, contact your assigned doctor via the portal's messaging feature.",
                "Automatic refill enrollment is available for chronic medications. Ask your care team to enable it."
            ]
        },
        {
            "intent": "billing_insurance",
            "patterns": [
                "billing question", "insurance claim", "cost of treatment", "medical bill",
                "insurance coverage", "payment plan", "health insurance"
            ],
            "responses": [
                "For billing queries, contact our billing department. We accept most major insurance plans.",
                "Insurance claims are processed within 30 days. For itemised bills, request from the billing desk.",
                "We offer flexible payment plans for uninsured patients. Speak to our financial counsellor."
            ]
        },
        {
            "intent": "greeting",
            "patterns": ["hello", "hi", "help", "support", "start"],
            "responses": [
                "Hello! I'm your healthcare support assistant. How can I assist you today?",
                "Hi! I can help with appointments, prescriptions, billing, and more."
            ]
        }
    ],

    # ─── BANKING ────────────────────────────────────────────────────────────────
    "banking": [
        {
            "intent": "account_balance",
            "patterns": [
                "check balance", "account balance", "how much money", "available funds",
                "current balance", "bank balance", "savings balance"
            ],
            "responses": [
                "Check your balance via our mobile app, internet banking, or by calling our 24/7 helpline.",
                "Your account balance is accessible anytime through our app under 'My Accounts'.",
                "For real-time balance, log in to internet banking or use our ATM network."
            ]
        },
        {
            "intent": "fund_transfer",
            "patterns": [
                "transfer money", "send money", "NEFT", "RTGS", "wire transfer",
                "transfer funds", "pay someone", "bank transfer"
            ],
            "responses": [
                "Transfers can be made via the app (Instant/NEFT/RTGS). Add beneficiary first, then initiate transfer.",
                "For NEFT/RTGS transfers, log in to internet banking, add the payee, and follow the transfer steps.",
                "Instant transfers (IMPS) are 24/7. NEFT and RTGS have processing windows; check our website for times."
            ]
        },
        {
            "intent": "lost_card",
            "patterns": [
                "lost my card", "stolen card", "block card", "card stolen", "card missing",
                "debit card lost", "credit card lost", "freeze card"
            ],
            "responses": [
                "Block your card immediately via the app (Card Services → Block Card) or call our 24/7 helpline.",
                "To block a lost/stolen card: App → Cards → Block. A replacement will arrive within 5–7 days.",
                "Call our emergency helpline immediately to block your card. We'll courier a replacement within a week."
            ]
        },
        {
            "intent": "greeting",
            "patterns": ["hello", "hi", "help", "support", "start"],
            "responses": [
                "Hello! Welcome to banking support. How can I help you?",
                "Hi! I can assist with account balance, transfers, cards, and more."
            ]
        }
    ]
}
