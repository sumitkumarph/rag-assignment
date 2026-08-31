import json
import sys
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    PROJECT_ROOT /
    "data" /
    "ragas_results.json"
)

PDF_FILE = (
    PROJECT_ROOT /
    "data" /
    "RAGAS_Evaluation_Report.pdf"
)


def create_pdf():

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    # -----------------------------------------------------
    # Document
    # -----------------------------------------------------

    document = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        leading=10
    )

    story = []

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "RAGAS Evaluation Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Retrieval-Augmented Generation System",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"Evaluation Date: "
            f"{results['evaluation_date']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Total Questions: "
            f"{results['total_questions']}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    # -----------------------------------------------------
    # Overall result
    # -----------------------------------------------------

    overall = results["overall_result"]

    overall_text = (
        f"<b>OVERALL RESULT: {overall}</b>"
    )

    story.append(
        Paragraph(
            overall_text,
            heading_style
        )
    )

    # -----------------------------------------------------
    # Score table
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Evaluation Summary",
            heading_style
        )
    )

    table_data = [
        [
            "Metric",
            "Score",
            "Required",
            "Status"
        ]
    ]

    for metric, score in results["scores"].items():

        threshold = results["thresholds"][metric]

        status = results[
            "metric_status"
        ][metric]["passed"]

        table_data.append([
            metric.replace("_", " ").title(),
            f"{score:.2f}%",
            f"> {threshold:.0f}%",
            "PASS" if status else "FAIL"
        ])

    table = Table(
        table_data,
        colWidths=[
            70 * mm,
            35 * mm,
            35 * mm,
            30 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.grey
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.black
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(table)

    # -----------------------------------------------------
    # Requirements
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Required Minimum Scores",
            heading_style
        )
    )

    requirements = [
        ["Faithfulness", "> 90%"],
        ["Answer Correctness", "> 80%"],
        ["Context Recall", "> 85%"],
        ["Context Precision", "> 80%"]
    ]

    req_table = Table(
        [
            ["Metric", "Minimum"]
        ] + requirements,
        colWidths=[
            90 * mm,
            60 * mm
        ]
    )

    req_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.grey
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.black
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            )
        ])
    )

    story.append(req_table)

    # -----------------------------------------------------
    # Question-level results
    # -----------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Question-Level Evaluation",
            heading_style
        )
    )

    question_results = results[
        "question_results"
    ]

    question_table = [
        [
            "Question",
            "Faith.",
            "Correct.",
            "Recall",
            "Precision"
        ]
    ]

    for index, item in enumerate(
        question_results,
        start=1
    ):

        question = item.get(
            "question",
            ""
        )

        question_text = (
            f"Q{index}: {question}"
        )

        question_table.append([
            Paragraph(
                question_text,
                small_style
            ),

            f"{item.get('faithfulness', 0) * 100:.1f}%",

            f"{item.get('answer_correctness', 0) * 100:.1f}%",

            f"{item.get('context_recall', 0) * 100:.1f}%",

            f"{item.get('context_precision', 0) * 100:.1f}%"
        ])

    q_table = Table(
        question_table,
        colWidths=[
            90 * mm,
            22 * mm,
            22 * mm,
            22 * mm,
            22 * mm
        ],
        repeatRows=1
    )

    q_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.grey
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.black
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),
            (
                "FONTSIZE",
                (1, 1),
                (-1, -1),
                7
            )
        ])
    )

    story.append(q_table)

    # -----------------------------------------------------
    # Build PDF
    # -----------------------------------------------------

    document.build(story)

    print()
    print("=" * 60)
    print("PDF REPORT CREATED")
    print("=" * 60)
    print(PDF_FILE)


if __name__ == "__main__":
    create_pdf()