"""
copilot.py

This is the "LLM Co-Pilot" part of the app.

IMPORTANT: the LLM is NEVER used to predict damage or risk.
Those predictions come only from the CNN and ANN models.
The LLM's only job here is to read the results and write a
plain-language summary and suggested action for the human reviewer.

If the user provides a Groq API key in the sidebar, we call the real
Groq API (fast, free-tier friendly). If no key is provided, we fall
back to a simple template-based summary so the app still works for a
live demo without any API key.
"""

import requests

GROQ_MODEL = "llama-3.3-70b-versatile"


def generate_recommendation(claim_fields, damage_result, risk_result, api_key=None):
    """
    Returns a short text summary + recommended action for the officer.
    Uses the Groq API if an api_key is given, otherwise uses a
    simple built-in template (no external call needed).
    """
    if api_key:
        try:
            return _call_groq_api(claim_fields, damage_result, risk_result, api_key)
        except Exception as error:
            # If the API call fails for any reason, fall back safely
            return _template_recommendation(claim_fields, damage_result, risk_result) + \
                f"\n\n(Note: Groq API call failed, showing template summary instead. Error: {error})"

    return _template_recommendation(claim_fields, damage_result, risk_result)


def _call_groq_api(claim_fields, damage_result, risk_result, api_key):
    """Calls the real Groq API to generate a natural-language summary."""
    prompt = f"""You are an assistant helping an insurance claims officer.
Summarize this claim in 3-4 short sentences and suggest one clear next action
(approve, reject, or send for manual inspection). Do not invent facts that
are not given below.

Claim details: {claim_fields}
Damage detection result: {damage_result}
Risk prediction result: {risk_result}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _template_recommendation(claim_fields, damage_result, risk_result):
    """A simple rule-based summary used when no API key is provided."""
    customer_name = claim_fields.get("customer_name", "the customer")
    policy_number = claim_fields.get("policy_number", "N/A")

    damage_label = damage_result["label"]
    risk_label = risk_result["label"]

    if risk_label == "High" or damage_label == "Damaged" and risk_result["confidence"] > 0.7:
        suggested_action = "Send for manual inspection before approval."
    elif risk_label == "Low" and damage_label == "Damaged":
        suggested_action = "Likely safe to approve, but confirm damage photos match the claim."
    elif damage_label == "No Damage Detected":
        suggested_action = "Request additional photo evidence before approving."
    else:
        suggested_action = "Review manually — mixed signals from damage and risk models."

    summary = (
        f"Claim for policy {policy_number} (customer: {customer_name}) shows "
        f"'{damage_label}' from the image analysis and a '{risk_label}' risk level "
        f"from the historical claims model. "
        f"Suggested action: {suggested_action}"
    )
    return summary