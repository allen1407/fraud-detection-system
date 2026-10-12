import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.ui import badge, gauge, get_pipeline, presets, report_pdf, require_artifacts

st.set_page_config(page_title="Score Transaction", page_icon="🔎", layout="wide")
st.title("🔎 Score Transaction")
try:
    require_artifacts()
    pipe = get_pipeline()
    P = presets()
    cats = pipe.categories

    DEFAULT = P["Clear Legit"]
    def load(rec):
        d = pd.to_datetime(rec[C.DATE_COL])
        st.session_state.update(f_date=d.date(), f_time=d.time(), f_amount=float(rec["Transaction_Amount"]),
                                f_age=int(rec["Customer_Age"]), f_income=float(rec["Customer_Income"]),
                                f_balance=float(rec["Account_Balance"]), f_type=rec["Transaction_Type"],
                                f_cat=rec["Category"], f_pay=rec["Payment_Method"], f_gender=rec["Customer_Gender"],
                                f_occ=rec["Customer_Occupation"], f_merch=rec["Merchant_Name"], run=True)
    if "f_amount" not in st.session_state:
        load(DEFAULT); st.session_state.run = False

    st.caption("Presets are real rows from the held-out test set.")
    cols = st.columns(4)
    for col, name in zip(cols, P):
        col.button(name, use_container_width=True, on_click=load, args=(P[name],))

    with st.form("txn"):
        a, b, c = st.columns(3)
        a.date_input("Transaction date", key="f_date"); a.time_input("Transaction time", key="f_time")
        a.number_input("Transaction amount ($)", key="f_amount", min_value=-1e9, step=10.0)
        b.number_input("Customer age", key="f_age", min_value=-10, max_value=1000, step=1)
        b.number_input("Customer income ($)", key="f_income", min_value=-1e9, step=1000.0)
        b.number_input("Account balance ($)", key="f_balance", min_value=-1e9, step=100.0)
        c.selectbox("Transaction type", cats["Transaction_Type"], key="f_type")
        c.selectbox("Merchant category", cats["Category"], key="f_cat")
        c.selectbox("Payment method", cats["Payment_Method"], key="f_pay")
        c.selectbox("Gender", cats["Customer_Gender"], key="f_gender")
        d, e = st.columns(2)
        d.text_input("Occupation", key="f_occ"); e.text_input("Merchant name", key="f_merch")
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted or st.session_state.get("run"):
        s = st.session_state
        rec = {C.DATE_COL: dt.datetime.combine(s.f_date, s.f_time).isoformat(sep=" "),
               "Transaction_Amount": s.f_amount, "Customer_Age": s.f_age, "Customer_Income": s.f_income,
               "Account_Balance": s.f_balance, "Transaction_Type": s.f_type, "Category": s.f_cat,
               "Payment_Method": s.f_pay, "Customer_Gender": s.f_gender,
               "Customer_Occupation": s.f_occ, "Merchant_Name": s.f_merch}
        res = pipe.assess(rec)
        s.run = False
        badge(res["verdict"]); st.write("")
        if not res["valid"]:
            st.error("This transaction was rejected because of invalid input:")
            for e in res["errors"]:
                st.markdown(f"- {e}")
        else:
            g1, g2, g3 = st.columns(3)
            g1.plotly_chart(gauge("DNN fraud probability", res["dnn_prob"], pipe.threshold), use_container_width=True)
            g2.plotly_chart(gauge("LR fraud probability", res["lr_prob"], pipe.lr_threshold), use_container_width=True)
            g3.metric("Model agreement", "Agree" if res["models_agree"] else "Disagree")
            g3.metric("Trust score", f"{res['trust']:.2f}", help="≥ 1.0 means the transaction is as familiar as a typical training transaction")
            st.write(f"**Recommendation:** {res['recommendation']}")
            f = res["factors"][::-1]
            fig = go.Figure(go.Bar(x=[x["shap"] for x in f], y=[f"{x['label']} = {x['value'] if isinstance(x['value'], str) else format(x['value'], '.4g')}" for x in f],
                                   orientation="h", marker_color=["#dc2626" if x["shap"] > 0 else "#16a34a" for x in f]))
            fig.update_layout(title="Top 5 factors (SHAP, red = raises fraud risk)", height=320, margin=dict(l=10, r=10, t=50, b=10),
                              xaxis_title="SHAP value (change in DNN fraud probability)")
            st.plotly_chart(fig, use_container_width=True)
        text = pipe.generate_report(res)
        with st.expander("Full report", expanded=True):
            st.code(text, language=None)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (.txt)", text, "fraud_report.txt", "text/plain")
        d2.download_button("Download report (.pdf)", report_pdf(text), "fraud_report.pdf", "application/pdf")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("Something went wrong while scoring this transaction. Please check the inputs and try again.")
