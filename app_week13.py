# app_week13.py — DATS 6401 Week 13: visualizing ML models
#
# Embedded in the Week 13 chapter of the course site through
# dats6401/_live_app_week13.qmd in the gaimsds/dataviz repository.
#
# Two tabs, each one slide from the lecture made adjustable:
#
#   1. The threshold is a choice. A random forest scores 143 test tumors.
#      Moving one slider changes who is called malignant, and the confusion
#      matrix, the ROC point and the error counts all follow.
#   2. The parameters are part of the result. DBSCAN groups 2,254 Supercharger
#      sites. Moving the radius changes how many clusters there are.
#
# Built for the embed: controls on top, no sidebar, laid out for a ~800px
# iframe. It needs only scikit-learn and altair (altair ships with streamlit),
# and the models train once and are cached.
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Week 13 — visualizing ML models", layout="wide")

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------- tab 1 data
@st.cache_data(show_spinner="Training the model…")
def score_test_tumors():
    """Train the lecture's random forest and score the held-back tumors.

    Same split and settings as the slides, so the numbers match: 143 test
    tumors, 53 of them malignant. Returns one row per test tumor.
    """
    bc = load_breast_cancer(as_frame=True)
    X, y = bc.data, bc.target                      # y: 0 = malignant, 1 = benign
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # Column 0 of predict_proba is class 0, which is malignant.
    return pd.DataFrame({
        "p_malignant": model.predict_proba(X_test)[:, 0],
        "is_malignant": (y_test == 0).to_numpy(),
        "worst radius": X_test["worst radius"].to_numpy(),
    })


# ---------------------------------------------------------------- tab 2 data
@st.cache_data
def load_sites():
    return pd.read_csv("data/superchargers_week13.csv")


@st.cache_data(max_entries=200)
def cluster_sites(radius_km, min_sites):
    """DBSCAN on latitude and longitude. Returns one label per site; -1 is noise.

    The haversine metric measures distance along the Earth's surface. It wants
    radians, and a radius expressed as a fraction of the Earth's radius.
    """
    sites = load_sites()
    return DBSCAN(
        eps=radius_km / EARTH_RADIUS_KM, min_samples=min_sites,
        algorithm="ball_tree", metric="haversine",
    ).fit_predict(np.radians(sites[["lat", "lon"]]))


st.subheader("Week 13: two choices the model does not make for you")

tab_threshold, tab_dbscan = st.tabs(
    ["The threshold is a choice", "The parameters are part of the result"])

# ======================================================================
# Tab 1: threshold
# ======================================================================
with tab_threshold:
    tumors = score_test_tumors()
    n_malignant = int(tumors["is_malignant"].sum())
    n_benign = len(tumors) - n_malignant

    st.markdown(
        f"A random forest gives each of {len(tumors)} test tumors a probability "
        "of being malignant. The model does not decide where to draw the line. "
        "You do.")

    threshold = st.slider(
        "Call a tumor malignant when P(malignant) is at least",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05)

    called = tumors["p_malignant"] >= threshold
    truth = tumors["is_malignant"]
    caught = int((truth & called).sum())
    missed = int((truth & ~called).sum())
    alarms = int((~truth & called).sum())
    cleared = int((~truth & ~called).sum())

    # The four outcomes, named the way the slide names them.
    tumors = tumors.assign(outcome=np.select(
        [truth & called, truth & ~called, ~truth & called],
        ["cancer, caught", "cancer, missed", "false alarm"],
        default="healthy, cleared"))

    with st.container(horizontal=True):
        st.metric("Cancers missed", missed, border=True,
                  help="Malignant tumors the model called benign.")
        st.metric("False alarms", alarms, border=True,
                  help="Benign tumors the model called malignant.")
        st.metric("Recall", f"{caught / n_malignant:.3f}", border=True,
                  help="Of the malignant tumors, the share the model caught.")
        st.metric("Precision",
                  f"{caught / (caught + alarms):.3f}" if caught + alarms else "n/a",
                  border=True,
                  help="Of the tumors called malignant, the share that are.")

    # ---- every tumor as one dot, with the line you drew
    outcome_color = alt.Color(
        "outcome:N", title=None,
        scale=alt.Scale(
            domain=["cancer, caught", "cancer, missed",
                    "false alarm", "healthy, cleared"],
            range=["steelblue", "red", "orange", "lightgray"]),
        legend=alt.Legend(orient="top"))

    dots = alt.Chart(tumors).mark_circle(size=70, opacity=0.8).encode(
        x=alt.X("p_malignant:Q", title="P(malignant) from the model",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("worst radius:Q", title="Worst radius (mm)",
                scale=alt.Scale(zero=False)),
        color=outcome_color,
        tooltip=[alt.Tooltip("p_malignant:Q", format=".2f", title="P(malignant)"),
                 alt.Tooltip("worst radius:Q", format=".1f"),
                 "outcome:N"])
    line = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color="black", strokeDash=[6, 4], size=2).encode(x="t:Q")

    # ---- the confusion matrix, oriented like the slide
    matrix = pd.DataFrame({
        "actual": ["Actually malignant"] * 2 + ["Actually benign"] * 2,
        "called": ["Called malignant", "Called benign"] * 2,
        "count": [caught, missed, alarms, cleared],
        "outcome": ["cancer, caught", "cancer, missed",
                    "false alarm", "healthy, cleared"],
    })
    cells = alt.Chart(matrix).encode(
        x=alt.X("called:N", title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=200),
                sort=["Called malignant", "Called benign"]),
        y=alt.Y("actual:N", title=None, axis=alt.Axis(labelLimit=200),
                sort=["Actually malignant", "Actually benign"]))
    matrix_chart = (
        cells.mark_rect(opacity=0.45).encode(color=outcome_color.legend(None))
        + cells.mark_text(fontSize=30, fontWeight="bold").encode(text="count:Q")
        + cells.mark_text(fontSize=13, dy=30).encode(text="outcome:N")
    ).properties(height=alt.Step(95), title="Confusion matrix at this threshold")

    # ---- the ROC curve, with the point this threshold lands on
    fpr, tpr, _ = roc_curve(truth, tumors["p_malignant"])
    curve = alt.Chart(pd.DataFrame({"fpr": fpr, "tpr": tpr})).mark_line(
        color="steelblue", interpolate="step-after").encode(
        x=alt.X("fpr:Q", title="False alarm rate"),
        y=alt.Y("tpr:Q", title="Recall"),
        order="fpr:Q")
    here = alt.Chart(pd.DataFrame({
        "fpr": [alarms / n_benign], "tpr": [caught / n_malignant],
        "label": [f"threshold {threshold:.2f}"],
    })).encode(x="fpr:Q", y="tpr:Q")
    roc_chart = (
        curve
        + here.mark_point(color="red", size=220, filled=True)
        + here.mark_text(align="left", dx=12, dy=12, fontSize=13).encode(
            text="label:N")
    ).properties(height=220, title="Where this threshold sits on the ROC curve")

    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(
            (dots + line).properties(
                height=500, title="Each dot is one tumor. The dashed line is "
                                  "your threshold."))
    with right:
        st.altair_chart(matrix_chart)
        st.altair_chart(roc_chart)

    st.caption(
        "Try this: lower the threshold until no cancer is missed, and count the "
        "false alarms that cost. Then ask which error is worse for a patient.")

# ======================================================================
# Tab 2: DBSCAN
# ======================================================================
with tab_dbscan:
    sites = load_sites()

    st.markdown(
        f"DBSCAN groups {len(sites):,} Tesla Supercharger sites in the United "
        "States. It has no setting for the number of clusters. It has a radius "
        "and a minimum count, and the clusters follow from those two numbers.")

    with st.container(horizontal=True):
        radius_km = st.select_slider(
            "Radius: sites closer than this are neighbors (km)",
            options=[5, 10, 15, 25, 35, 50, 75, 100, 150], value=25)
        min_sites = st.slider(
            "Minimum sites within the radius to start a cluster",
            min_value=3, max_value=15, value=5)

    labels = cluster_sites(radius_km, min_sites)
    sites = sites.assign(cluster=labels)
    clustered = sites[sites["cluster"] >= 0]
    sizes = clustered["cluster"].value_counts()
    n_noise = int((labels == -1).sum())

    # Name each cluster by the state most of its sites are in.
    names = clustered.groupby("cluster")["State"].agg(lambda s: s.mode().iat[0])
    sites["group"] = np.where(
        sites["cluster"] >= 0,
        sites["cluster"].map(lambda c: f"cluster {c}, mostly {names.get(c, '')}"),
        "noise")
    sites["sites in cluster"] = sites["cluster"].map(sizes).fillna(0).astype(int)

    with st.container(horizontal=True):
        st.metric("Clusters", len(sizes), border=True)
        st.metric("Sites labeled noise", f"{n_noise:,}", border=True,
                  help="Sites that belong to no cluster at these settings.")
        st.metric("Largest cluster",
                  f"{int(sizes.iloc[0]):,} sites" if len(sizes) else "none",
                  border=True)
        st.metric("Share of sites in a cluster",
                  f"{1 - n_noise / len(sites):.0%}", border=True)

    # State outlines come from the vega-datasets CDN, fetched by the browser.
    # The Albers USA projection draws Alaska and Hawaii as insets at lower left.
    states = alt.Chart(
        alt.Data(url="https://cdn.jsdelivr.net/npm/vega-datasets@2.11.0/data/us-10m.json",
                 format=alt.TopoDataFormat(type="topojson", feature="states"))
    ).mark_geoshape(fill="whitesmoke", stroke="darkgray", strokeWidth=0.6)

    noise_layer = alt.Chart(sites[sites["cluster"] < 0]).mark_circle(
        size=14, color="gray", opacity=0.45).encode(
        longitude="lon:Q", latitude="lat:Q",
        tooltip=["Supercharger:N", "group:N"])
    cluster_layer = alt.Chart(sites[sites["cluster"] >= 0]).mark_circle(
        size=26, opacity=0.9).encode(
        longitude="lon:Q", latitude="lat:Q",
        color=alt.Color("cluster:N", legend=None,
                        scale=alt.Scale(scheme="category20")),
        tooltip=["Supercharger:N", "group:N", "sites in cluster:Q"])

    left, right = st.columns([3, 1])
    with left:
        st.altair_chart(
            (states + noise_layer + cluster_layer).project("albersUsa").properties(
                height=480, title="Colored sites are in a cluster. Gray sites "
                                  "are noise."))
    with right:
        st.markdown("**Largest clusters**")
        top = (sizes.head(8).rename("sites").to_frame()
               .assign(where=lambda d: d.index.map(names))
               .reset_index(drop=True)[["where", "sites"]])
        st.dataframe(top, hide_index=True,
                     column_config={"where": "Mostly in", "sites": "Sites"})

    st.caption(
        "Try this: move the radius from 10 km to 100 km. The cluster count rises "
        "and then falls, because nearby clusters merge. Colors repeat after 20 "
        "clusters, so hover over a site to see which cluster it is in.")
