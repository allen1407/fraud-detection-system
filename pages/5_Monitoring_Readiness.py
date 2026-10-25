import plotly.graph_objects as go
import streamlit as st

from src.ui import read_csv, require_artifacts

st.set_page_config(page_title="Monitoring & Readiness", page_icon="🩺", layout="wide")
st.title("🩺 Monitoring & Readiness")
try:
    require_artifacts()
    d = read_csv("drift_baseline.csv")
    fig = go.Figure(go.Bar(x=d.psi, y=d.feature, orientation="h", marker_color="#2563eb"))
    fig.add_vline(x=0.25, line_color="#dc2626", line_dash="dash", annotation_text="retrain trigger 0.25")
    fig.add_vline(x=0.10, line_color="#d97706", line_dash="dot", annotation_text="watch 0.10", annotation_position="bottom")
    fig.update_layout(title="Population Stability Index, train vs test (baseline)", height=720, yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Deployment readiness checklist (measured values)")
    r = read_csv("deployment_readiness.csv")
    st.dataframe(r.style.map(lambda v: "background-color:#dcfce7" if v == "PASS" else ("background-color:#fee2e2" if v == "FAIL" else ""), subset=["status"]),
                 hide_index=True, use_container_width=True, height=(len(r) + 1) * 35 + 3)
    fails = r[r.status == "FAIL"]
    if len(fails):
        st.error(f"{len(fails)} of {len(r)} checks FAIL: {', '.join(fails.check)}. Resolve or formally accept these before production use.")
    else:
        st.success("All checks pass.")

    st.subheader("Retraining & rollback policy")
    st.markdown("""
- **Monitoring:** compute PSI per numeric feature monthly against the training distribution.
- **Retrain trigger:** any feature PSI > 0.25, or a quarterly schedule (whichever comes first).
- **Rollback:** keep the previous `fraud_*_model` + `preprocessor.pkl` + `metadata.json` set as a pair in a dated folder; if a new pair fails the readiness checks or live metrics degrade, restore the previous pair together (never mix a model with a different preprocessor).
- **Recalibration:** the data covers two days in September 2013; fraud patterns change, so thresholds and the cost trade-off must be re-validated on recent traffic before real use.""")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("Monitoring results could not be displayed. Run `bash scripts/run_all.sh` to regenerate reports.")
