"""Capture a screenshot of every app page (app must be running: streamlit run app.py --server.port 8599)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_batch.csv"
OUT.mkdir(parents=True, exist_ok=True)


def settle(page, ms=4000):
    page.wait_for_timeout(ms)
    page.wait_for_selector("[data-testid=stApp]")
    page.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]')", timeout=120000)


def shot(page, name):
    page.screenshot(path=str(OUT / name), full_page=True)
    print("saved", name)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 2400})
    page.goto(BASE); settle(page, 6000); shot(page, "01_home.png")
    page.goto(f"{BASE}/Score_Transaction"); settle(page, 8000)
    page.get_by_role("button", name="Predict").click(); settle(page, 5000); shot(page, "02_score_prediction.png")
    page.get_by_label("Amount ($)").fill("-250"); page.keyboard.press("Enter")
    page.get_by_role("button", name="Predict").click(); settle(page, 4000); shot(page, "03_score_rejected.png")
    page.goto(f"{BASE}/Batch_Scoring"); settle(page, 6000)
    page.set_input_files("input[type=file]", str(SAMPLE)); settle(page, 8000); shot(page, "04_batch.png")
    for path, name in [("Model_Performance", "05_performance.png"), ("Responsible_AI", "06_responsible_ai.png"),
                       ("Monitoring_Readiness", "07_monitoring.png"), ("Documentation", "08_documentation.png")]:
        page.goto(f"{BASE}/{path}"); settle(page, 7000); shot(page, name)
    b.close()
