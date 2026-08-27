import io
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable

def generate_pdf_report(
    patient_id: str,
    eye: str,
    doctor_notes: str,
    top_predictions: list,  # List of tuples: [("Optic Disc Cupping", 0.93), ...]
    original_img_pil: Image.Image,
    gradcam_img_pil: Image.Image,
) -> bytes:
    """Generates a downloadable PDF report buffer for OcuNet predictions."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#003366'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        spaceAfter=15
    )
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#003366'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)

    # 1. Header & Title
    story.append(Paragraph("OcuNet Diagnostics — Retinal Assessment Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Platform: Research Build v1.0", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#003366'), spaceAfter=15))

    # 2. Patient & Examination Metadata
    meta_data = [
        [Paragraph(f"<b>Patient ID:</b> {patient_id or 'N/A'}", body_style),
         Paragraph(f"<b>Eye Evaluated:</b> {eye}", body_style)],
        [Paragraph("<b>Assessment Type:</b> Automated Multi-Label AI Screen", body_style),
         Paragraph("<b>Model Backbone:</b> EfficientNet-B0 (OcuNet)", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F6F9')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.HexColor('#DDDDDD')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Top Findings / Predictions (Top 8 to fill report layout cleanly)
    story.append(Paragraph("Diagnostic Findings (Top Probability Ranking)", header_style))
    
    pred_data = [["Rank", "Pathology / Condition", "Confidence Score", "Status Flag"]]
    for idx, (label, prob) in enumerate(top_predictions[:8], 1):
        status = "HIGH RISK" if prob >= 0.65 else ("SUSPECT" if prob >= 0.30 else "LOW RISK")
        pred_data.append([
            str(idx),
            label,
            f"{prob * 100:.1f}%",
            status
        ])

    pred_table = Table(pred_data, colWidths=[40, 260, 120, 120])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')])
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 15))

    # 4. Visual Evidence Side-by-Side (Original vs Grad-CAM Overlay)
    story.append(Paragraph("Visual Interpretability (Grad-CAM Attention Overlay)", header_style))
    
    # Resize PIL images to memory buffers for ReportLab
    img_width, img_height = 240, 200
    
    buf_orig = io.BytesIO()
    original_img_pil.save(buf_orig, format='PNG')
    buf_orig.seek(0)
    
    buf_grad = io.BytesIO()
    gradcam_img_pil.save(buf_grad, format='PNG')
    buf_grad.seek(0)

    rl_orig = RLImage(buf_orig, width=img_width, height=img_height)
    rl_grad = RLImage(buf_grad, width=img_width, height=img_height)

    img_table = Table([
        [rl_orig, rl_grad],
        [Paragraph("<b>Original Fundus Image</b>", body_style), Paragraph("<b>Grad-CAM Disease Localization</b>", body_style)]
    ], colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    # 5. Doctor's Clinical Notes
    story.append(Paragraph("Clinician Notes & Observations", header_style))
    notes_text = doctor_notes.strip() if doctor_notes.strip() else "No additional notes entered by the attending clinician."
    notes_table = Table([[Paragraph(notes_text, body_style)]], colWidths=[540])
    notes_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFF9E6')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#FFE0B2')),
    ]))
    story.append(notes_table)
    story.append(Spacer(1, 20))

    # 6. Medical Disclaimer Footer
    disclaimer_text = (
        "<b>Disclaimer:</b> OcuNet is an AI-assisted decision support system designed for retinal screening[cite: 1]. "
        "This report is automatically generated and must be reviewed and verified by a licensed ophthalmologist or clinician "
        "prior to diagnostic or therapeutic action."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=7, textColor=colors.HexColor('#777777'))))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()