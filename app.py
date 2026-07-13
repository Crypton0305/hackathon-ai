"""
app.py

Main Streamlit app: AI Insurance Claim Co-Pilot

This app walks an insurance officer through:
1. Uploading a claim (image + PDF + tabular info)
2. Seeing the AI's damage detection and risk prediction
3. Seeing WHY the AI made those predictions (explainability)
4. Approving / rejecting / modifying the AI's recommendation (human-in-the-loop)
5. Downloading a full PDF report of everything

The code is organized as one function per page, and a sidebar to move
between them. All data for the current claim is kept in st.session_state
so it carries over as the user moves between pages.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import pandas as pd

from utils.pdf_extract import extract_text_from_pdf, extract_claim_fields
from utils.explainability import make_gradcam_heatmap, overlay_heatmap_on_image, get_feature_importance
from utils.copilot import generate_recommendation
from utils.report_generator import build_claim_report
from utils.storage import save_decision, get_all_decisions

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Insurance Claim Co-Pilot", layout="wide")

IMG_SIZE = 128  # must match the input size the CNN was trained on (train_models.py IMG_SIZE)
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# -------------------------------------------------------------------
# Load models once and cache them
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("models/cnn_damage_model.keras")
    ann_model = tf.keras.models.load_model("models/ann_risk_model.keras")
    scaler = joblib.load("models/ann_scaler.pkl")
    feature_columns = joblib.load("models/ann_feature_columns.pkl")
    return cnn_model, ann_model, scaler, feature_columns


cnn_model, ann_model, scaler, feature_columns = load_models()


# -------------------------------------------------------------------
# Session state defaults (holds the data for the claim being reviewed)
# -------------------------------------------------------------------
def init_session_state():
    defaults = {
        "image_array": None,
        "original_image": None,
        "claim_fields": None,
        "tabular_inputs": None,
        "damage_result": None,
        "risk_result": None,
        "feature_importance": None,
        "ai_recommendation": None,
        "human_decision": None,
        "human_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# -------------------------------------------------------------------
# PAGE: Dashboard
# -------------------------------------------------------------------
def page_dashboard():
    st.title("AI Insurance Claim Co-Pilot")
    st.write(
        "An AI assistant that helps insurance officers evaluate vehicle damage "
        "claims faster, using image analysis, document extraction, and risk prediction — "
        "while keeping a human in charge of the final decision."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Modalities Used", "3", "Image, PDF, Tabular")
    col2.metric("Deep Learning Models", "2", "CNN + ANN")
    col3.metric("Human Oversight", "Required", "Every claim")

    st.subheader("How it works")
    st.markdown(
        """
        1. **Upload Claim** — upload a damage photo, the claim PDF, and enter claim details.
        2. **Damage Analysis** — a CNN model detects whether the vehicle is damaged.
        3. **Risk Analysis** — an ANN model predicts the claim's risk level.
        4. **Explainability** — see exactly why the AI made each prediction.
        5. **Human Review** — approve, reject, or modify the AI's recommendation.
        6. **Report** — download a full PDF report of the whole decision.
        """
    )
    st.info("Use the sidebar to move through each step in order.")


# -------------------------------------------------------------------
# PAGE: Claim Upload (Image + PDF + Tabular)
# -------------------------------------------------------------------
def page_claim_upload():
    st.title("Step 1: Upload Claim")

    # --- Modality 1: Image ---
    st.subheader("Vehicle Damage Image")
    uploaded_image = st.file_uploader("Upload a photo of the vehicle", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        resized_image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(resized_image, dtype=np.float32) / 255.0

        st.session_state.original_image = image_array
        st.session_state.image_array = np.expand_dims(image_array, axis=0)
        st.image(image, caption="Uploaded vehicle image", width=300)

    st.divider()

    # --- Modality 2: PDF ---
    st.subheader("Claim Document (PDF)")
    uploaded_pdf = st.file_uploader("Upload the claim PDF", type=["pdf"])
    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        claim_fields = extract_claim_fields(pdf_text)
        st.session_state.claim_fields = claim_fields

        st.write("**Extracted fields:**")
        st.json(claim_fields)

    st.divider()

    # --- Modality 3: Tabular data ---
    st.subheader("Claim & Customer Details")
    st.write("Enter the historical/tabular details used for risk prediction.")

    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, value=5.0, step=1.0)
    with col2:
        previous_claims = st.number_input("Previous Claims by Customer", min_value=0, value=0, step=1)
        driver_age = st.number_input("Driver Age", min_value=18, value=35, step=1)

    if st.button("Save Claim Details"):
        st.session_state.tabular_inputs = {
            "claim_amount": claim_amount,
            "vehicle_age": vehicle_age,
            "previous_claims": previous_claims,
            "driver_age": driver_age,
        }
        st.success("Claim details saved. Go to 'Damage Analysis' next.")


# -------------------------------------------------------------------
# PAGE: Damage Analysis (CNN)
# -------------------------------------------------------------------
def page_damage_analysis():
    st.title("Step 2: Damage Analysis (CNN)")

    if st.session_state.image_array is None:
        st.warning("Please upload a vehicle image on the 'Claim Upload' page first.")
        return

    prediction = cnn_model.predict(st.session_state.image_array, verbose=0)[0][0]
    is_damaged = prediction >= 0.5
    confidence = float(prediction if is_damaged else 1 - prediction)

    damage_result = {
        "label": "Damaged" if is_damaged else "No Damage Detected",
        "confidence": confidence,
        "raw_score": float(prediction),
    }
    st.session_state.damage_result = damage_result

    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.original_image, caption="Vehicle Image", width=300)
    with col2:
        st.metric("Prediction", damage_result["label"])
        st.metric("Confidence Score", f"{damage_result['confidence']:.1%}")
        st.progress(damage_result["confidence"])

    st.info("This prediction comes from the CNN deep learning model, not an LLM.")


# -------------------------------------------------------------------
# PAGE: Risk Analysis (ANN)
# -------------------------------------------------------------------
def page_risk_analysis():
    st.title("Step 3: Claim Risk Prediction (ANN)")

    if st.session_state.tabular_inputs is None:
        st.warning("Please fill in and save claim details on the 'Claim Upload' page first.")
        return
    if st.session_state.damage_result is None:
        st.warning("Please run 'Damage Analysis' first (its result feeds into this model).")
        return

    tabular = st.session_state.tabular_inputs
    damage_severity = st.session_state.damage_result["raw_score"]

    input_values = [
        tabular["claim_amount"],
        tabular["vehicle_age"],
        tabular["previous_claims"],
        tabular["driver_age"],
        damage_severity,
    ]

    scaled_input = scaler.transform([input_values])
    prediction_probs = ann_model.predict(scaled_input, verbose=0)[0]
    predicted_class = int(np.argmax(prediction_probs))
    confidence = float(prediction_probs[predicted_class])

    risk_result = {
        "label": RISK_LABELS[predicted_class],
        "confidence": confidence,
        "class_index": predicted_class,
        "input_values": input_values,
    }
    st.session_state.risk_result = risk_result

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Risk Level", risk_result["label"])
        st.metric("Confidence Score", f"{risk_result['confidence']:.1%}")
    with col2:
        chart_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Probability": prediction_probs,
        })
        st.bar_chart(chart_data.set_index("Risk Level"))

    st.info("This prediction comes from the ANN deep learning model, trained on historical claims data.")


# -------------------------------------------------------------------
# PAGE: Explainability
# -------------------------------------------------------------------
def page_explainability():
    st.title("Step 4: Explainability")

    if st.session_state.damage_result is None or st.session_state.risk_result is None:
        st.warning("Please complete Damage Analysis and Risk Analysis first.")
        return

    st.subheader("CNN Explainability — Grad-CAM Heatmap")
    st.write("This shows which part of the image most influenced the damage prediction.")

    heatmap = make_gradcam_heatmap(st.session_state.image_array, cnn_model, last_conv_layer_name="conv2")
    overlay = overlay_heatmap_on_image(st.session_state.original_image, heatmap)

    col1, col2 = st.columns(2)
    col1.image(st.session_state.original_image, caption="Original Image", width=300)
    col2.image(overlay, caption="Grad-CAM Heatmap (red = high influence)", width=300)

    st.divider()

    st.subheader("ANN Explainability — Feature Importance")
    st.write("This shows which claim detail most influenced the risk prediction.")

    importance_scores, predicted_class, base_confidence = get_feature_importance(
        ann_model, scaler, feature_columns, st.session_state.risk_result["input_values"]
    )
    st.session_state.feature_importance = importance_scores

    importance_df = pd.DataFrame({
        "Feature": list(importance_scores.keys()),
        "Importance": list(importance_scores.values()),
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))
    top_feature = importance_df.iloc[0]["Feature"]
    st.success(f"Most influential feature: **{top_feature}**")


# -------------------------------------------------------------------
# PAGE: Human Review (Human-in-the-Loop)
# -------------------------------------------------------------------
def page_human_review():
    st.title("Step 5: Human Review")

    if st.session_state.damage_result is None or st.session_state.risk_result is None:
        st.warning("Please complete Damage Analysis and Risk Analysis first.")
        return

    st.subheader("AI Co-Pilot Recommendation")

    api_key = st.session_state.get("groq_api_key", "")
    if st.button("Generate AI Recommendation") or st.session_state.ai_recommendation is None:
        recommendation = generate_recommendation(
            st.session_state.claim_fields or {},
            st.session_state.damage_result,
            st.session_state.risk_result,
            api_key=api_key if api_key else None,
        )
        st.session_state.ai_recommendation = recommendation

    st.write(st.session_state.ai_recommendation)

    st.divider()
    st.subheader("Your Decision")
    st.write("The AI never makes the final call — you do. Please review and decide.")

    decision = st.radio("Choose an action:", ["Approve", "Reject", "Modify"], horizontal=True)
    notes = st.text_area("Reviewer notes (optional, required if modifying)")

    if st.button("Submit Final Decision"):
        st.session_state.human_decision = decision
        st.session_state.human_notes = notes

        save_decision(
            st.session_state.claim_fields or {},
            st.session_state.damage_result,
            st.session_state.risk_result,
            st.session_state.ai_recommendation,
            decision,
            notes,
        )
        st.success(f"Decision recorded: {decision}. You can now download the report.")


# -------------------------------------------------------------------
# PAGE: Report
# -------------------------------------------------------------------
def page_report():
    st.title("Step 6: Download Report")

    # claim_fields (from PDF upload) is optional -- damage_result, risk_result,
    # and human_decision are the only steps that are actually required.
    required = [
        st.session_state.damage_result,
        st.session_state.risk_result,
        st.session_state.human_decision,
    ]
    if any(item is None for item in required):
        st.warning("Please complete Damage Analysis, Risk Analysis, and Human Review first.")
        return

    if st.session_state.claim_fields is None:
        st.info("No claim PDF was uploaded — the report will be generated without extracted claim fields.")

    explanation_text = (
        f"Damage detection confidence: {st.session_state.damage_result['confidence']:.1%}\n"
        f"Risk prediction confidence: {st.session_state.risk_result['confidence']:.1%}\n"
    )
    if st.session_state.feature_importance:
        top_feature = max(st.session_state.feature_importance, key=st.session_state.feature_importance.get)
        explanation_text += f"Most influential factor in risk score: {top_feature}"

    pdf_bytes = build_claim_report(
        st.session_state.claim_fields or {},
        st.session_state.damage_result,
        st.session_state.risk_result,
        explanation_text,
        st.session_state.ai_recommendation or "N/A",
        st.session_state.human_decision,
        st.session_state.human_notes,
    )

    st.success("Report is ready.")
    st.download_button(
        "Download Claim Report (PDF)",
        data=pdf_bytes,
        file_name="claim_report.pdf",
        mime="application/pdf",
    )


# -------------------------------------------------------------------
# PAGE: Business Model
# -------------------------------------------------------------------
def page_business_model():
    st.title("Business Model")

    st.markdown(
        """
### Problem
Manual review of vehicle damage claims is slow and inconsistent. Officers spend a lot
of time cross-checking photos, documents, and customer history by hand.

### Solution
AI Insurance Claim Co-Pilot speeds this up by automatically analyzing the damage photo,
extracting claim documents, and predicting risk — while always keeping a human reviewer
in control of the final decision.

### Target Market
- Mid-size and large insurance companies handling vehicle claims
- Third-party claim assessment firms
- Insurance aggregator platforms

### Revenue Strategy
- **SaaS subscription**: monthly fee per insurance company, tiered by claim volume
- **Per-claim pricing**: pay-as-you-go for smaller firms
- **Enterprise licensing**: on-premise deployment for large insurers with strict data policies

### Pricing (example)
| Plan | Price | Claims/month |
|------|-------|---------------|
| Starter | $299/mo | up to 500 |
| Growth | $999/mo | up to 3,000 |
| Enterprise | Custom | Unlimited |

### Competitive Advantage
- Combines 3 data modalities (image, document, tabular) in one workflow
- Explainable AI builds trust with regulators and reviewers
- Human-in-the-loop keeps compliance and accountability intact
- Modular design — models can be retrained on a company's own claims data

### Scalability Plan
- Start with vehicle claims, expand to property and health claims
- Add more languages for document extraction
- Offer an API so insurers can plug this into their existing claim systems

### Go-To-Market Plan
1. Pilot with 1-2 mid-size insurers, refine the models on real data
2. Case study + demo video showing time saved per claim
3. Outreach to insurance tech conferences and industry partners
4. Expand via reseller partnerships with insurance software vendors
        """
    )


# -------------------------------------------------------------------
# PAGE: Past Decisions
# -------------------------------------------------------------------
def page_past_decisions():
    st.title("Past Human Decisions")

    decisions = get_all_decisions()
    if not decisions:
        st.info("No decisions recorded yet.")
        return

    st.dataframe(pd.DataFrame(decisions))


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.title("Navigation")

with st.sidebar.expander("Optional: Groq API Key (for LLM summary)"):
    st.session_state.groq_api_key = st.text_input(
        "Groq API Key", type="password",
        help="If left empty, a built-in template summary is used instead."
    )

page = st.sidebar.radio(
    "Go to:",
    [
        "Dashboard",
        "1. Claim Upload",
        "2. Damage Analysis",
        "3. Risk Analysis",
        "4. Explainability",
        "5. Human Review",
        "6. Report",
        "Business Model",
        "Past Decisions",
    ],
)

if page == "Dashboard":
    page_dashboard()
elif page == "1. Claim Upload":
    page_claim_upload()
elif page == "2. Damage Analysis":
    page_damage_analysis()
elif page == "3. Risk Analysis":
    page_risk_analysis()
elif page == "4. Explainability":
    page_explainability()
elif page == "5. Human Review":
    page_human_review()
elif page == "6. Report":
    page_report()
elif page == "Business Model":
    page_business_model()
elif page == "Past Decisions":
    page_past_decisions()