"""
report_generator.py

Builds the final downloadable claim report (PDF) using reportlab.
The report includes everything required by the hackathon spec:
- Claim information (from the PDF extraction step)
- Image analysis result (CNN damage detection)
- Risk prediction (ANN)
- Explainability (confidence + top influencing factor)
- AI recommendation (from the copilot text)
- Human decision (approve / reject / modify)
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io


def build_claim_report(claim_fields, damage_result, risk_result, explanation_text,
                        ai_recommendation, human_decision, human_notes):
    """
    Creates the PDF report in memory and returns the raw bytes,
    so it can be offered directly as a Streamlit download button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )

    story = []

    # Title
    story.append(Paragraph("AI Insurance Claim Co-Pilot — Claim Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # 1. Claim information
    story.append(Paragraph("1. Claim Information", heading_style))
    claim_table_data = [["Field", "Value"]] + [
        [key.replace("_", " ").title(), str(value)] for key, value in claim_fields.items()
    ]
    story.append(_make_table(claim_table_data))

    # 2. Image analysis (CNN)
    story.append(Paragraph("2. Damage Detection (CNN Model)", heading_style))
    story.append(Paragraph(
        f"Prediction: <b>{damage_result['label']}</b><br/>"
        f"Confidence: <b>{damage_result['confidence']:.1%}</b>",
        styles["Normal"]
    ))

    # 3. Risk prediction (ANN)
    story.append(Paragraph("3. Claim Risk Prediction (ANN Model)", heading_style))
    story.append(Paragraph(
        f"Risk Level: <b>{risk_result['label']}</b><br/>"
        f"Confidence: <b>{risk_result['confidence']:.1%}</b>",
        styles["Normal"]
    ))

    # 4. Explainability
    story.append(Paragraph("4. Explainability", heading_style))
    story.append(Paragraph(explanation_text.replace("\n", "<br/>"), styles["Normal"]))

    # 5. AI Recommendation
    story.append(Paragraph("5. AI Co-Pilot Recommendation", heading_style))
    story.append(Paragraph(ai_recommendation.replace("\n", "<br/>"), styles["Normal"]))

    # 6. Human decision
    story.append(Paragraph("6. Human Review Decision", heading_style))
    story.append(Paragraph(
        f"Final Decision: <b>{human_decision}</b><br/>"
        f"Reviewer Notes: {human_notes if human_notes else 'None'}",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _make_table(data):
    """Small helper to build a consistently styled table."""
    table = Table(data, colWidths=[160, 320])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    return table
