import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config as C
from src.ui import read_csv, read_json, require_artifacts

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
try:
    require_artifacts()
    m, pred, lat = read_json("metrics.json"), read_csv("test_predictions.csv"), read_json("latency.json")
    st.caption(f"Held-out test set: {m['n_test']} transactions, {m['test_fraud_rate']:.1%} fraud.")
    if m["dnn"]["roc_auc"] < 0.6:
        st.warning("ROC-AUC near 0.5 means the models perform at chance level on this dataset.")
    cols = st.columns(6)
    for col, (k, lab) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("accuracy", "Accuracy")]):
        col.metric(f"DNN {lab}", f"{m['dnn'][k]:.3f}", help=f"LR: {m['lr'][k]:.3f}")

    st.subheader("DNN vs LR")
    rows = [{"metric": k, "DNN": m["dnn"][k], "LR": m["lr"][k]} for k in ["threshold", "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "flag_rate"]]
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)

    a, b = st.columns(2)
    y = pred.y_true.values
    fig = go.Figure()
    for name, c in [("dnn_prob", "#2563eb"), ("lr_prob", "#ea580c")]:
        f, t, _ = roc_curve(y, pred[name]); fig.add_scatter(x=f, y=t, name=name.split("_")[0].upper(), line_color=c)
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color="grey"))
    fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", height=380); a.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    for name, c in [("dnn_prob", "#2563eb"), ("lr_prob", "#ea580c")]:
        p, r, _ = precision_recall_curve(y, pred[name]); fig.add_scatter(x=r, y=p, name=name.split("_")[0].upper(), line_color=c)
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", height=380); b.plotly_chart(fig, use_container_width=True)

    st.subheader("Interactive threshold (DNN, from saved test-set probabilities)")
    t = st.slider("Decision threshold", 0.01, 0.99, float(m["thresholds"]["dnn_operating"]), 0.01)
    flag = pred.dnn_prob.values >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    cost = fn * C.COST_FN + fp * C.COST_FP
    c1, c2 = st.columns([1, 2])
    fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"], color_continuous_scale="Blues")
    fig.update_layout(height=320, coloraxis_showscale=False); c1.plotly_chart(fig, use_container_width=True)
    k = c2.columns(4)
    k[0].metric("Precision", f"{tp / max(tp + fp, 1):.3f}"); k[1].metric("Recall", f"{tp / max(tp + fn, 1):.3f}")
    k[2].metric("Flag rate", f"{flag.mean():.1%}"); k[3].metric("Expected cost", f"${cost:,.0f}")
    c2.caption(f"Cost = {fn} missed frauds × \\${C.COST_FN:.0f} + {fp} false alarms × \\${C.COST_FP:.0f}. "
               f"Operating threshold {m['thresholds']['dnn_operating']:.2f}; unconstrained F1-optimal "
               f"{m['thresholds']['dnn_f1_unconstrained']:.2f}; cost-optimal {m['thresholds']['dnn_cost_optimal']:.2f}.")
    if t <= 0.05:
        c2.warning("At very low thresholds the model flags nearly everything. This is cheapest under the cost model only because the model has no real signal.")
    st.image(str(C.FIGURES_DIR / "cost_threshold_analysis.png"))

    st.subheader("Latency")
    a, b, c = st.columns(3)
    a.metric("Single row, no SHAP (p50 / p95)", f"{lat['single_no_explanation_ms']['p50']:.1f} / {lat['single_no_explanation_ms']['p95']:.1f} ms")
    b.metric("Single row, with SHAP (p50 / p95)", f"{lat['single_with_shap_ms']['p50']:.0f} / {lat['single_with_shap_ms']['p95']:.0f} ms")
    c.metric("Batch throughput", f"{lat['batch']['rows_per_second']:,.0f} rows/s")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("Performance results could not be displayed. Run `bash scripts/run_all.sh` to regenerate reports.")
