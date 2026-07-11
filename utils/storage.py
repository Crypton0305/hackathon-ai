"""
storage.py

Stores every human review decision so we have a record of:
- What the AI recommended
- What the human decided (approve / reject / modify)
- Any notes the human added

For the hackathon demo we use a simple local JSON file as the database.
This keeps the app easy to run anywhere with zero setup. In a production
version, this file could be swapped for Firebase or any real database
without changing the rest of the app, since only this file would need
to change.
"""

import json
import os
from datetime import datetime

DECISIONS_FILE = os.path.join("data", "human_decisions.json")


def _load_all_decisions():
    if not os.path.exists(DECISIONS_FILE):
        return []
    with open(DECISIONS_FILE, "r") as file:
        return json.load(file)


def save_decision(claim_fields, damage_result, risk_result, ai_recommendation,
                   human_decision, human_notes):
    """Appends one new decision record to the local JSON database."""
    all_decisions = _load_all_decisions()

    new_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "claim_number": claim_fields.get("claim_number", "N/A"),
        "policy_number": claim_fields.get("policy_number", "N/A"),
        "damage_label": damage_result["label"],
        "damage_confidence": damage_result["confidence"],
        "risk_label": risk_result["label"],
        "risk_confidence": risk_result["confidence"],
        "ai_recommendation": ai_recommendation,
        "human_decision": human_decision,
        "human_notes": human_notes,
    }
    all_decisions.append(new_record)

    os.makedirs("data", exist_ok=True)
    with open(DECISIONS_FILE, "w") as file:
        json.dump(all_decisions, file, indent=2)

    return new_record


def get_all_decisions():
    """Returns all past decisions, most recent first."""
    decisions = _load_all_decisions()
    return list(reversed(decisions))
