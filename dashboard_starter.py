# dashboard_starter.py — Week 10: one dataset, three ways to ask about it
# Run with:  streamlit run dashboard_starter.py
# Requires:  pip install streamlit-folium plotly scikit-learn
#
# The three tabs are the three Block 2 ideas, each in the form you would
# actually ship: a map you can pan, a map that answers a question when you
# point at it, and a map that compares places rather than showing them.
#
# No geopandas here, on purpose. The spatial join in Block 2 is what produced
# the `state` column in this CSV, and once a point carries its state the
# grouping is an ordinary pandas groupby. Leaving geopandas out keeps the
# deployed install small -- see the note at the top of app_map_starter.py.
import json
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium

st.set_page_config(page_title="Supercharger dashboard", layout="wide")
st.title("Supercharger dashboard")

HERE = Path(__file__).resolve().parent
EARTH_KM = 6371.0088
DROP = ["Alaska", "Hawaii", "Puerto Rico"]

GW_BLUE = "#033c5a"
GW_FILL = "#ebf0f2"
ACCENT = "#d9534f"


@st.cache_data
def load_sites():
    df = pd.read_csv(HERE / "data" / "superchargers_us.csv")
    # Contiguous states only, so the map is not mostly ocean. This is a choice
    # and it belongs in the caption.
    return df[(df.lat.between(24, 50)) & (df.lon.between(-125, -66))].copy()


@st.cache_data
def load_states():
    """The contiguous state polygons, as a plain list of GeoJSON features.

    A list, not the FeatureCollection: handed the whole collection with
    format={"property": "features"}, Altair lifts it into a named dataset and
    the property is never applied, so the layer silently draws nothing.
    """
    gj = json.loads((HERE / "data" / "states.geojson").read_text())
    return [f for f in gj["features"]
            if f["properties"]["state"] not in DROP]


@st.cache_data
def nearest_links(df):
    """Each site, and the straight line to the nearest other site.

    BallTree with metric="haversine" measures across the surface of a sphere.
    It wants RADIANS, and what it returns is an ANGLE until you multiply by
    the Earth's radius. k=2 because the nearest point to any point is itself,
    so column 1 is the first real neighbor.
    """
    pts = np.radians(df[["lat", "lon"]].values)
    dist, idx = BallTree(pts, metric="haversine").query(pts, k=2)
    return pd.DataFrame({
        "name": df.name.values,
        "state": df.state.values,
        "lat": df.lat.values, "lon": df.lon.values,
        "lat2": df.lat.values[idx[:, 1]], "lon2": df.lon.values[idx[:, 1]],
        "nearest": df.name.values[idx[:, 1]],
        "nearest_km": (dist[:, 1] * EARTH_KM).round(1),
    })


sites = load_sites()

# --- the controls, in one place ---------------------------------------------
# One filter feeding several views is the pattern from the end of the deck.
# Minimum stalls applies everywhere; the state picker only moves the map,
# which is why it says so.
st.sidebar.header("Filters")
min_stalls = st.sidebar.slider("Minimum stalls", 1, int(sites.stalls.max()), 1)
state_names = ["All"] + sorted(sites.state.unique())
state = st.sidebar.selectbox("State", state_names,
                             help="Applies to the map tab only. The other two "
                                  "tabs compare every state.")

shown = sites[sites.stalls >= min_stalls]
if shown.empty:
    st.warning("No sites have that many stalls.")
    st.stop()

st.caption(f"{len(shown):,} of {len(sites):,} contiguous-US sites, "
           f"{int(shown.stalls.sum()):,} stalls. Alaska, Hawaii and Puerto "
           f"Rico are excluded, which is a choice, not a fact about the data.")

tab_map, tab_near, tab_states = st.tabs(
    ["Where they are", "What is nearby", "How states compare"])

# --- 1. the map you pan -----------------------------------------------------
with tab_map:
    on_map = shown if state == "All" else shown[shown.state == state]
    if on_map.empty:
        st.info("No sites in that state at this stall threshold.")
    else:
        def as_geojson(df):
            """One GeoJson layer rather than one marker per point.

            The deck measured this: 2,254 points are about 1,481 KB as
            CircleMarkers and about 401 KB as a single GeoJson layer. The map
            looks the same.
            """
            return {"type": "FeatureCollection", "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                 "properties": {"name": r["name"], "stalls": int(r.stalls)}}
                for _, r in df.iterrows()]}

        m = folium.Map(location=[on_map.lat.mean(), on_map.lon.mean()],
                       zoom_start=4 if state == "All" else 6,
                       tiles="OpenStreetMap", scrollWheelZoom=False)
        folium.GeoJson(
            as_geojson(on_map),
            name="Supercharger sites",
            marker=folium.CircleMarker(radius=4, fill=True, fill_opacity=0.7,
                                       color=ACCENT, fill_color=ACCENT),
            tooltip=folium.GeoJsonTooltip(fields=["name", "stalls"],
                                          aliases=["", "stalls:"]),
        ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=900, height=460, returned_objects=[])
        st.caption(f"{len(on_map):,} sites drawn. Leaflet tiles, so you can "
                   "zoom into a parking lot — which is the thing this view is "
                   "good for, and the thing the other two tabs cannot do.")

# --- 2. the map that answers when you point at it ---------------------------
with tab_near:
    links = nearest_links(shown)
    pick = alt.selection_point(fields=["state"], on="pointerover", empty=False)

    # A white stroke on this fill is invisible against a white page: the
    # states merge into one blob and there is nothing to aim at.
    base = alt.Chart(alt.Data(values=load_states())).mark_geoshape(
        fill=GW_FILL, stroke="#9ab2bd", strokeWidth=0.6)

    edges = alt.Chart(links).mark_rule(color=ACCENT, strokeWidth=1).encode(
        longitude="lon:Q", latitude="lat:Q",
        longitude2="lon2:Q", latitude2="lat2:Q",
        # 0.15 at rest, not 0.04. The point of the layer is that every link is
        # already drawn and the selection only changes what stands out; at 0.04
        # the resting state is blank and that point is lost.
        opacity=alt.condition(pick, alt.value(0.85), alt.value(0.15)))

    dots = alt.Chart(links).mark_circle(color=GW_BLUE).encode(
        longitude="lon:Q", latitude="lat:Q",
        # size is an AREA in Altair, so 6 is a dot under 3px across -- too
        # small to point at. 18 is still small enough not to blot out the
        # cities, where sites sit almost on top of each other.
        size=alt.condition(pick, alt.value(70), alt.value(18)),
        tooltip=["name:N", "state:N", "nearest:N", "nearest_km:Q"],
    ).add_params(pick)

    # use_container_width rather than width="stretch": Streamlit only grew the
    # width= argument on charts in 1.46, and the lab machines are on 1.45. The
    # old argument still works on current Streamlit, where it maps to the same
    # thing and logs the deprecation to the server rather than the page.
    st.altair_chart(
        (base + edges + dots).project(type="albersUsa")
        .properties(width="container", height=430),
        use_container_width=True)
    st.caption("Point at any site to light up every site in its state, and "
               "the line from each of them to its nearest neighbor. The lines "
               "are all there the whole time; the selection only changes what "
               "is opaque, which is why nothing has to be recomputed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median gap to nearest site", f"{links.nearest_km.median():.1f} km")
    c2.metric("90th percentile", f"{links.nearest_km.quantile(.9):.0f} km")
    c3.metric("Most isolated site", f"{links.nearest_km.max():.0f} km")
    st.caption(f"Most isolated: **{links.loc[links.nearest_km.idxmax(), 'name']}**. "
               "Distances are haversine, measured across the sphere. The "
               "median sits well below the mean because the distribution has "
               "a long right tail, which is why the median is the one quoted.")

# --- 3. the map that compares --------------------------------------------
with tab_states:
    per_state = (shown.groupby("state")
                 .agg(sites=("name", "size"), stalls=("stalls", "sum"))
                 .reset_index())

    left, right = st.columns(2)
    for col, metric, label in [(left, "sites", "Sites per state"),
                               (right, "stalls", "Stalls per state")]:
        fig = px.choropleth(
            per_state, locations="state", locationmode="USA-states",
            scope="usa", color=metric, color_continuous_scale="OrRd",
            hover_name="state",
            hover_data={"sites": True, "stalls": True, "state": False},
            labels={metric: label.split(" per ")[0].lower()})
        fig.update_layout(margin=dict(l=0, r=0, t=28, b=0), height=330,
                          title=dict(text=label, x=0.5, font=dict(size=14)),
                          coloraxis_colorbar=dict(thickness=10, len=0.8))
        col.plotly_chart(fig, use_container_width=True)

    st.caption("`locationmode=\"USA-states\"` takes the two-letter code and "
               "supplies the geometry, so this tab needs no boundary file at "
               "all. `scope=\"usa\"` is also what moves Alaska and Hawaii into "
               "the corner — the composite Albers projection from the deck, "
               "for free. Both sit empty, because the data above them was "
               "trimmed to the contiguous states; empty is the honest "
               "rendering of absent, and it is not the same as zero.")

    st.markdown("**The two maps are not the same map.** A site with 40 stalls "
                "is not one site's worth of charging, so the counts disagree "
                "wherever stations are unusually large or small.")
    st.dataframe(
        per_state.sort_values("stalls", ascending=False)
        .assign(stalls_per_site=lambda d: (d.stalls / d.sites).round(1))
        .head(10).rename(columns={"state": "State", "sites": "Sites",
                                  "stalls": "Stalls",
                                  "stalls_per_site": "Stalls per site"}),
        hide_index=True, use_container_width=True)

# ------------------------------------------------------------------
# TODO 1: a coordinated chart.
#         A map cannot rank: position and area are already spent on geography.
#         Add a bar chart of the ten cities with the most stalls IN THE
#         CURRENT SELECTION, so it moves when the filters do.
#
# TODO 2: the same question, measured wrongly.
#         Tab 2 measures with haversine. Recompute `nearest_km` with plain
#         Euclidean distance on raw degrees, and compare the two lists of the
#         ten most isolated sites. Which states move, and in which direction?
#         The deck says what to expect; check whether this data agrees.
# ------------------------------------------------------------------
