# demo_2_widgets_layout.py — Week 5, Part 2: widgets, layout, state
# Run with:  streamlit run demo_2_widgets_layout.py
import plotly.express as px
import streamlit as st

st.title("Tips Dashboard")
df = px.data.tips()

# --- Sidebar: controls live here ---------------------------------------------
# `key=` puts each widget's value in st.session_state under that name, so you can
# read it anywhere instead of passing the variable around. Keys must be unique.
days = st.sidebar.multiselect("Days", sorted(df["day"].unique()),
                              default=list(df["day"].unique()), key="days")
smoker = st.sidebar.radio("Smoker", ["All", "Yes", "No"], key="smoker")

sub = df[df["day"].isin(st.session_state.days)]
if st.session_state.smoker != "All":
    sub = sub[sub["smoker"] == st.session_state.smoker]

# --- Layout: metrics in columns ----------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Parties", len(sub))
c2.metric("Avg bill", f"${sub['total_bill'].mean():.2f}")
c3.metric("Avg tip %", f"{(sub['tip']/sub['total_bill']).mean()*100:.1f}%")

tab1, tab2 = st.tabs(["Chart", "Data"])
with tab1:
    fig = px.scatter(sub, x="total_bill", y="tip", color="time")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.dataframe(sub)

# --- A form: several inputs, one re-run --------------------------------------
# Outside a form, each of these would re-run the whole script on every change --
# and the text box would re-run it on every keystroke.
st.divider()
st.subheader("Annotate a view")
with st.form("annotation"):
    label = st.text_input("Label for this view")
    size = st.slider("Marker size", 4, 20, 8)
    submitted = st.form_submit_button("Apply")

if submitted:
    fig = px.scatter(sub, x="total_bill", y="tip", color="time", title=label)
    fig.update_traces(marker=dict(size=size))
    st.plotly_chart(fig, use_container_width=True)

# --- A fragment: re-run one function, not the page ---------------------------
# Changing this selectbox redraws only the fragment. Everything above it,
# including the filtering and the metrics, is left alone.
st.divider()


@st.fragment
def quick_look(data):
    st.subheader("Fragment: only this block re-runs")
    col = st.selectbox("Histogram of", ["total_bill", "tip", "size"], key="hist_col")
    st.plotly_chart(px.histogram(data, x=col), use_container_width=True)


quick_look(sub)

# --- session_state: variables do NOT survive re-runs; state does -------------
st.divider()
if "clicks" not in st.session_state:
    st.session_state.clicks = 0
if st.button("I clicked the chart insight button"):
    st.session_state.clicks += 1
st.caption(f"Button clicked {st.session_state.clicks} times this session "
           "(a plain variable would reset to 0 on every interaction).")
