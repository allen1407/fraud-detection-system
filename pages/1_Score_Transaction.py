import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.ui import badge, get_pipeline, presets, require_artifacts

st.set_page_config(page_title="Score Transaction", page_icon="🔎", layout="centered")
st.title("🔎 Score Transaction")
st.caption("Pick an example or edit the values, then press Predict.")
try:
    require_artifacts()
    pipe = get_pipeline()
    examples = presets()
    name = st.selectbox("Load example (real test-set row)", [k for k in examples if k != "Invalid Input"])
    ex = examples[name]

    with st.form("txn"):
        a, b = st.columns(2)
        time = a.number_input("Time (seconds since window start, 0-172800)", value=float(ex[C.TIME_COL]), step=60.0,
                              key=f"time_{name}")
        amount = b.number_input("Amount ($)", value=float(ex["Amount"]), min_value=-1e9, step=1.0, key=f"amount_{name}")
        with st.expander("PCA components V1-V28 (anonymised by the data provider)", expanded=False):
            cols = st.columns(4)
            pcs = {v: cols[i % 4].number_input(v, value=float(ex[v]), format="%.4f", key=f"{v}_{name}")
                   for i, v in enumerate(C.PCA_INPUTS)}
        go_btn = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if go_btn:
        rec = {C.TIME_COL: time, **pcs, "Amount": amount}
        res = pipe.assess(rec)
        badge(res["verdict"]); st.write("")
        if not res["valid"]:
            st.error("Please fix the following:")
            for e in res["errors"]:
                st.markdown(f"- {e}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("DNN fraud probability", f"{res['dnn_prob']:.2%}")
            c2.metric("LR fraud probability", f"{res['lr_prob']:.2%}")
            c3.metric("Trust score", f"{res['trust']:.2f}")
            st.write(f"**Recommendation:** {res['recommendation']}")
            f = res["factors"][::-1]
            fig = go.Figure(go.Bar(x=[x["shap"] for x in f], y=[x["label"] for x in f], orientation="h",
                                   marker_color=["#dc2626" if x["shap"] > 0 else "#16a34a" for x in f]))
            fig.update_layout(title="Top 5 factors (red raises fraud risk)", height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Full report"):
                st.code(pipe.generate_report(res), language=None)
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("Something went wrong while scoring this transaction. Please check the inputs and try again.")
