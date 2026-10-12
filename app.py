import streamlit as st

from src import config as C
from src.ui import read_json, require_artifacts

st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")
try:
    require_artifacts()
    m = read_json("metrics.json")
    import json
    data = json.loads((C.PROCESSED_DIR / "data_summary.json").read_text())
    meta = json.loads(C.METADATA_PATH.read_text())

    st.title("🛡️ Intelligent Fraud Detection System")
    st.caption("Computational Intelligence Lab · Task 9 (System Integration & Deployment) and Task 10 (Evaluation, Responsible AI & Documentation)")
    st.write("A decision-support system that scores banking transactions with a Keras DNN and a logistic-regression baseline, "
             "explains each decision with SHAP, measures how familiar the transaction is (trust score), and routes uncertain "
             "cases to a human analyst.")

    st.subheader("Dataset")
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{data['rows']:,}"); b.metric("Fraud rate", f"{data['fraud_rate']:.1%}")
    c.metric("Date range", f"{data['date_min'][:10]} → {data['date_max'][:10]}"); d.metric("Features after engineering", len(meta["feature_list"]))
    st.caption("Source: Kaggle banking-transactions fraud dataset (USA, 2023-2024). A ~49% fraud rate is not realistic for live traffic.")

    st.subheader("Architecture")
    st.markdown("**Input** → **Validate & engineer features** → **Score (DNN + LR) & explain (SHAP) & trust score** → "
                "**Decision support** (AUTO-BLOCK / AUTO-APPROVE / MANUAL REVIEW)")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#2563eb"];
        Input -> "Validate &\\nengineer features" -> "Score & explain\\n(DNN, LR, SHAP, trust)" -> "Decision support\\n(route + report)" }""")

    st.subheader("Headline results (test set)")
    cols = st.columns(5)
    cols[0].metric("DNN ROC-AUC", f"{m['dnn']['roc_auc']:.3f}"); cols[1].metric("LR ROC-AUC", f"{m['lr']['roc_auc']:.3f}")
    cols[2].metric("DNN F1", f"{m['dnn']['f1']:.3f}"); cols[3].metric("DNN precision", f"{m['dnn']['precision']:.3f}")
    cols[4].metric("DNN recall", f"{m['dnn']['recall']:.3f}")
    if m["dnn"]["roc_auc"] < 0.6:
        st.warning("Both models score close to chance (ROC-AUC ≈ 0.5). This dataset shows no learnable fraud signal, so the "
                   "system demonstrates integration, explainability and governance, **not** real fraud-detection skill. "
                   "See the Responsible AI and Documentation pages.")
    st.info("Use the sidebar to score a transaction, run a batch, inspect performance, fairness, monitoring and documentation.")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("The home page could not be loaded. Make sure the models and reports exist (run `bash scripts/run_all.sh`).")
