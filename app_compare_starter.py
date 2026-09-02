# app_compare_starter.py — Week 2: encodings side by side
# Run with:  streamlit run app_compare_starter.py
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title("One Question, Two Encodings")
st.caption("TODO: state your question here")

df = px.data.tips()     # TODO: your dataset

left, right = st.columns(2)

with left:
    st.subheader("Encoding A")
    fig, ax = plt.subplots()
    # TODO: your STRONG encoding
    sns.scatterplot(data=df, x="total_bill", y="tip", ax=ax)
    st.pyplot(fig)
    st.caption("Channel: ___ · ranking position: ___")

with right:
    st.subheader("Encoding B")
    fig, ax = plt.subplots()
    # TODO: your second encoding (the weak one is more instructive!)
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="day", ax=ax)
    st.pyplot(fig)
    st.caption("Channel: ___ · ranking position: ___")
