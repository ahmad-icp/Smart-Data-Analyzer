import base64
import io
from typing import Any, Dict, List, Optional

import pandas as pd
from fpdf import FPDF

from .export_tools import plot_to_image_bytes


def _format_dict(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def generate_markdown_report(
    df: pd.DataFrame,
    overview: Dict[str, Any],
    stats: pd.DataFrame,
    suggestions: List[Dict[str, Any]],
    charts: List[Dict[str, Any]],
) -> str:
    """Build a Markdown report string containing dataset insights."""

    md = ["# Smart Data Analyzer Report\n"]

    md.append("## Dataset Overview\n")
    md.append(_format_dict(overview))
    md.append("\n## Top Rows\n")
    md.append(df.head(10).to_markdown(index=False))

    if not stats.empty:
        md.append("\n## Descriptive Statistics\n")
        md.append(stats.to_markdown())

    if suggestions:
        md.append("\n## Data Quality Suggestions\n")
        for s in suggestions:
            md.append(f"- {s.get('suggestion')}\n")

    if charts:
        md.append("\n## Charts\n")
        for idx, chart in enumerate(charts, start=1):
            md.append(f"### Chart {idx}: {chart.get('title', 'Chart')}\n")
            md.append(f"![chart_{idx}](data:image/png;base64,{chart.get('image_base64')})\n")

    return "\n".join(md)


def markdown_to_html(markdown_text: str) -> str:
    """Convert markdown text to simple HTML."""
    try:
        import markdown

        return markdown.markdown(markdown_text, extensions=["tables"])
    except ImportError:
        # Fallback simple transformation
        html = markdown_text.replace("\n", "<br>\n")
        return f"<html><body>{html}</body></html>"


def generate_pdf_report(
    markdown_text: str,
    charts: List[Dict[str, Any]],
    title: str = "Smart Data Analyzer Report",
) -> bytes:
    """Generate a PDF report from markdown and embedded chart images."""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=10)

    lines = markdown_text.splitlines()
    for line in lines:
        if line.startswith("# "):
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, line.replace("# ", ""), ln=True)
            pdf.set_font("Arial", size=10)
        elif line.startswith("## "):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, line.replace("## ", ""), ln=True)
            pdf.set_font("Arial", size=10)
        elif line.startswith("### "):
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, line.replace("### ", ""), ln=True)
            pdf.set_font("Arial", size=10)
        else:
            pdf.multi_cell(0, 5, line)

    for chart in charts:
        image_b64 = chart.get("image_base64")
        if not image_b64:
            continue
        try:
            image_bytes = base64.b64decode(image_b64)
            img_buffer = io.BytesIO(image_bytes)
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, chart.get("title", "Chart"), ln=True)
            pdf.image(img_buffer, w=180)
        except Exception:
            continue

    return pdf.output(dest="S").encode("utf-8")


def build_chart_image_base64(fig) -> Optional[str]:
    try:
        png_bytes = plot_to_image_bytes(fig)
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception:
        return None
