"""Export docs/FINAL_REPORT.md to reports/Task9_10_Report.pdf."""
import re
from pathlib import Path

import markdown
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "docs" / "FINAL_REPORT.md").read_text()
text = text.replace("≈", "~").replace("≥", ">=").replace("≤", "<=").replace("±", "+/-").replace("—", "-").replace("→", "->").replace("’", "'")
html = markdown.markdown(text, extensions=["tables"])
html = re.sub(r'src="\.\./([^"]+)"', lambda m: f'width="170" src="{ROOT / m.group(1)}"', html)
html = re.sub(r"</?(code|strong|em)>", "", html)
html = re.sub(r"</?blockquote>", "", html)

pdf = FPDF(); pdf.set_auto_page_break(True, 15); pdf.add_page(); pdf.set_font("Helvetica", size=10)
pdf.write_html(html, table_line_separators=True)
out = ROOT / "reports" / "Task9_10_Report.pdf"
pdf.output(str(out)); print("wrote", out, f"({pdf.page} pages)")
