# app.py — DATS 6401 course demo app
#
# A small, shared demo for the course site. It is embedded in several chapters
# through dats6401/_live_app_embed.qmd, so it is built for that setting:
#
#   * every interaction is a click-then-redraw, with no long-running loops --
#     Streamlit Community Cloud's free tier handles that workload well
#   * the only data is the gapminder table bundled with plotly, so there is no
#     download at startup and no data files to ship
#   * four dependencies, so a cold start after the app sleeps stays short
#   * laid out for a ~600px iframe: controls above the chart, no sidebar
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="DATS 6401 demo", layout="wide")


@st.cache_data
def load():
    return px.data.gapminder()


gap = load()
NUMERIC = ["lifeExp", "pop", "gdpPercap"]

st.markdown("#### DATS 6401 — three ideas from the course, in one app")

tab1, tab2, tab3 = st.tabs(["Encodings", "Order a heatmap", "Honest axes"])

# --------------------------------------------------------------- Week 2 -----
with tab1:
    st.caption("Week 2 — the same data says different things depending on which "
               "variable you put on which channel.")
    year = st.select_slider("Year", sorted(gap["year"].unique()), value=2007)
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X (position)", NUMERIC, index=2)
    y = c2.selectbox("Y (position)", NUMERIC, index=0)
    size = c3.selectbox("Size (area)", ["none"] + NUMERIC, index=2)

    sub = gap[gap["year"] == year]
    fig = px.scatter(
        sub, x=x, y=y,
        size=None if size == "none" else size,
        color="continent", hover_name="country",
        log_x=(x == "gdpPercap"), size_max=40, height=380,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Position is read most accurately, area least. Move a variable "
               "from Size to an axis and watch how much easier it is to compare.")

# --------------------------------------------------------------- Week 3 -----
with tab2:
    st.caption("Week 3 — a correlation matrix in arbitrary column order hides the "
               "blocks that are already in the data.")
    # Correlate the CHANGE between periods, not the level. Life expectancy rose
    # almost everywhere, so correlating levels gives ~0.9 for every pair and the
    # matrix is a uniform red wall -- the spurious-correlation trap from Week 4.
    # Differencing first strips the shared trend and leaves real co-movement.
    wide = (gap.pivot_table(index="year", columns="country", values="lifeExp")
               .diff().dropna())
    region = st.selectbox("Continent", sorted(gap["continent"].unique()), index=0)
    members = sorted(gap.loc[gap["continent"] == region, "country"].unique())[:18]
    corr = wide[members].corr()

    if st.checkbox("Order rows and columns by similarity", value=True):
        # A tiny greedy seriation: start from the most-correlated pair, then keep
        # appending whichever column is closest to the one just placed. Enough to
        # make the point without pulling in scipy's clustering.
        cols, remaining = list(corr.columns), None
        a, b = np.unravel_index(np.argmax(corr.abs().values - np.eye(len(cols))),
                                corr.shape)
        order = [cols[a], cols[b]]
        remaining = [c for c in cols if c not in order]
        while remaining:
            last = order[-1]
            nxt = max(remaining, key=lambda c: abs(corr.loc[last, c]))
            order.append(nxt)
            remaining.remove(nxt)
        corr = corr.loc[order, order]

    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    height=430, aspect="auto")
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlations are between five-yearly **changes** in life expectancy, "
               "not the levels — correlating the levels would return about 0.9 for "
               "every pair, because almost every country rose. Red is positive, blue "
               "negative. Untick the box: the numbers are identical, only the order "
               "changes, and the blocks stop being visible.")

# --------------------------------------------------------------- Week 4 -----
with tab3:
    st.caption("Week 4 — a truncated axis is not always wrong, but it changes the "
               "story and has to be disclosed.")
    c1, c2 = st.columns([2, 1])
    country = c1.selectbox("Country", sorted(gap["country"].unique()),
                           index=sorted(gap["country"].unique()).index("Japan"))
    metric = c2.selectbox("Metric", NUMERIC, index=0)
    zero = st.checkbox("Start the y axis at zero", value=False)

    one = gap[gap["country"] == country]
    fig = px.line(one, x="year", y=metric, markers=True, height=360)
    fig.update_traces(line_color="#2E6E8E")
    if zero:
        fig.update_yaxes(range=[0, one[metric].max() * 1.1])
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tick and untick the box. Same numbers, and a very different "
               "impression of how much changed.")

st.divider()
st.caption("Built for DATS 6401 · Visualization of Complex Data · GWU — "
           "data: gapminder, bundled with plotly.")
