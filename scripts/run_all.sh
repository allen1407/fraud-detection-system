#!/usr/bin/env bash
# One command to reproduce everything: data -> train -> evaluate -> tests -> notebook -> docs PDF.
# Usage: bash scripts/run_all.sh [path/to/creditcard.csv]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-$(ls data/raw/creditcard*.csv data/raw/*.zip 2>/dev/null | head -1 || true)}"
[ -n "$DATA" ] || { echo "Place creditcard.csv in data/raw/ (or pass its path)"; exit 1; }

echo "== 1/6 prepare data";   python scripts/prepare_data.py "$DATA"
echo "== 2/6 train";          python -m src.train
echo "== 3/6 evaluate";       python -m src.evaluate
echo "== 4/6 tests";          python -m pytest -q
python -m src.evaluate --readiness > /dev/null     # re-run readiness now that test results exist
echo "== 5/6 notebook";       jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_training.ipynb
echo "== 6/6 docs + final report PDF"; python scripts/build_docs.py; python scripts/build_report_pdf.py
echo "Done. Start the app with: streamlit run app.py"
