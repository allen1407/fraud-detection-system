import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.ui import badge, get_pipeline, require_artifacts

st.set_page_config(page_title="Score Transaction", page_icon="🔎", layout="centered")
st.title("🔎 Score Transaction")
st.caption("Enter the transaction details and press Predict.")
try:
    require_artifacts()
    pipe = get_pipeline()
    cats = pipe.categories

    with st.form("txn"):
        a, b = st.columns(2)
        date = a.date_input("Date", dt.date(2024, 6, 1))
        time = b.time_input("Time", dt.time(12, 0))
        amount = a.number_input("Amount ($)", value=1000.0, min_value=-1e9, step=10.0)
        age = b.number_input("Customer age", value=40, min_value=-10, max_value=1000, step=1)
        income = a.number_input("Customer income ($)", value=60000.0, min_value=-1e9, step=1000.0)
        balance = b.number_input("Account balance ($)", value=8000.0, min_value=-1e9, step=100.0)
        ttype = a.selectbox("Transaction type", cats["Transaction_Type"])
        category = b.selectbox("Merchant category", cats["Category"])
        pay = a.selectbox("Payment method", cats["Payment_Method"])
        gender = b.selectbox("Gender", cats["Customer_Gender"])
        occupation = a.text_input("Occupation", "Teacher")
        merchant = b.text_input("Merchant name", "Example Store")
        go_btn = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if go_btn:
        rec = {C.DATE_COL: dt.datetime.combine(date, time).isoformat(sep=" "), "Transaction_Amount": amount,
               "Customer_Age": age, "Customer_Income": income, "Account_Balance": balance,
               "Transaction_Type": ttype, "Category": category, "Payment_Method": pay, "Customer_Gender": gender,
               "Customer_Occupation": occupation, "Merchant_Name": merchant}
        res = pipe.assess(rec)
        badge(res["verdict"]); st.write("")
        if not res["valid"]:
            st.error("Please fix the following:")
            for e in res["errors"]:
                st.markdown(f"- {e}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("DNN fraud probability", f"{res['dnn_prob']:.1%}")
            c2.metric("LR fraud probability", f"{res['lr_prob']:.1%}")
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
