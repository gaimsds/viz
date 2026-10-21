# app_map_starter.py — Week 9 starter: a choropleth you can change
# Run with:  streamlit run app_map_starter.py
#
# Uses folium (the Python wrapper around Leaflet) but NOT geopandas: a
# choropleth only needs the GeoJSON and a table, and leaving geopandas out
# keeps the install small. geopandas' own .explore() does the same thing in one
# line when you already have a GeoDataFrame -- see the Block 2 demo.
import json
from pathlib import Path

import bisect
import copy

import folium
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Week 9 — State Map Explorer", layout="wide")
st.title("State Map Explorer")
st.caption("2024 American Community Survey, 5-year estimates. "
           "Boundaries from the US Census cartographic files.")

HERE = Path(__file__).resolve().parent
DROP = ["Alaska", "Hawaii", "Puerto Rico"]   # far from the rest; see the caption

MEASURES = {
    "People aged 65+ (count)": ("pop_65plus", "people"),
    "Share aged 65+ (rate, %)": ("pct_65plus", "% of the state"),
    "Total population (count)": ("pop_total", "people"),
    "People per square mile (rate)": ("pop_per_sqmi", "people / sq mi"),
}


# Every name here works in both folium (which draws the map) and matplotlib
# (which draws the legend swatches), so the two always agree.
RAMPS = {
    "OrRd": "sequential", "YlOrRd": "sequential", "YlGnBu": "sequential",
    "BuPu": "sequential", "Blues": "sequential", "Greens": "sequential",
    "Purples": "sequential", "viridis": "sequential",
    "RdBu": "diverging", "RdYlBu": "diverging", "PuOr": "diverging",
    "Spectral": "diverging",
}


@st.cache_data
def load():
    gj = json.loads((HERE / "data" / "states.geojson").read_text())
    df = pd.read_csv(HERE / "data" / "state_stats.csv", dtype={"fips": str})
    keep = {f["properties"]["fips"] for f in gj["features"]
            if f["properties"]["state"] not in DROP}
    gj["features"] = [f for f in gj["features"] if f["properties"]["fips"] in keep]
    return gj, df[df["fips"].isin(keep)].copy()


geojson, stats = load()

c1, c2 = st.columns([2, 1])
label = c1.selectbox("What to map", list(MEASURES))
k = c2.slider("Number of color classes", 3, 8, 5)
ramp = c2.selectbox("Color ramp", list(RAMPS),
                    format_func=lambda r: f"{r}  ({RAMPS[r]})")
column, unit = MEASURES[label]

scheme = st.radio(
    "How to cut the classes",
    ["Quantiles — equal number of states per class",
     "Equal interval — equal-width value ranges",
     "Natural breaks (Jenks) — cut where the data has gaps"],
    horizontal=False)

def class_breaks(col, k, scheme):
    """The class boundaries for one measure, under the chosen scheme."""
    v = stats[col]
    if scheme.startswith("Quantiles"):
        edges = list(v.quantile([i / k for i in range(k + 1)]))
    elif scheme.startswith("Equal interval"):
        lo, hi = v.min(), v.max()
        edges = [lo + (hi - lo) * i / k for i in range(k + 1)]
    else:
        # Jenks looks for the cuts that make each class as internally similar
        # as possible. Unlike the other two it reads the actual values, so it
        # is the only one of the three that reacts to the distribution's shape.
        import mapclassify
        edges = [float(v.min())] + [float(b) for b in
                                    mapclassify.NaturalBreaks(v.values, k=k).bins]

    # Round the edges so the legend reads cleanly, then clamp the outer two
    # back out: rounding can move them inside the data, and a state in no
    # class has no color.
    span = v.max() - v.min()
    step = 10 ** max(0, len(str(int(span))) - 2)        # 2 significant figures
    edges = sorted({round(b / step) * step if step > 1 else round(b, 1)
                    for b in edges})
    edges[0] = min(edges[0], float(v.min()))
    edges[-1] = max(edges[-1], float(v.max()))
    return edges if len(edges) >= 2 else [float(v.min()), float(v.max())]


# --- which measures are currently on the map --------------------------------
# Streamlit re-runs this file top to bottom on every click, so anything that
# should survive a click has to live in st.session_state.
if "layers" not in st.session_state:
    st.session_state["layers"] = []
if label not in st.session_state["layers"]:
    st.session_state["layers"].append(label)
layers = st.session_state["layers"]

bins = class_breaks(column, k, scheme)

if len(layers) > 1:
    _b1, _b2 = st.columns([3, 1])
    _b1.caption(f"{len(layers)} measures on the map: " + ", ".join(layers) +
                ". Switch between them with the radio buttons on the map.")
    if _b2.button(f"Keep only this one", use_container_width=True):
        st.session_state["layers"] = [label]
        st.rerun()

# The basemap is added with control=False so it stays on permanently and does
# not appear in the layer list. That leaves the list holding only the measures.
# scrollWheelZoom=False: inside a page a Leaflet map swallows the scroll wheel
# and zooms itself, so the reader cannot scroll past it. The +/- buttons work.
m = folium.Map(location=[39, -98], zoom_start=4, tiles=None,
               scrollWheelZoom=False)
folium.TileLayer("OpenStreetMap", control=False).add_to(m)


def add_measure(fmap, measure_label, active):
    """Add one measure as a BASE layer, so the control shows radio buttons.

    Not overlays. These layers are opaque and cover exactly the same 49
    polygons, so two of them switched on at once would show only the top one
    with nothing to say which. Radio buttons match what is actually possible:
    one at a time.
    """
    col, _ = MEASURES[measure_label]
    edges = class_breaks(col, k, scheme)
    n = len(edges) - 1
    colors = [to_hex(matplotlib.colormaps[ramp](i / max(n - 1, 1)))
              for i in range(n)]
    value_of = dict(zip(stats["fips"], stats[col]))

    def style(feature):
        v = value_of.get(feature["properties"]["fips"])
        if v is None:
            return {"fillColor": "#cccccc", "color": "white", "weight": 0.5,
                    "fillOpacity": 0.8}
        i = min(max(bisect.bisect_left(edges, v) - 1, 0), n - 1)
        return {"fillColor": colors[i], "color": "white", "weight": 0.5,
                "fillOpacity": 0.8}

    # A fresh copy per layer: the tooltip reads a property off the feature, and
    # the layers hold different numbers.
    data = copy.deepcopy(geojson)
    for f in data["features"]:
        v = value_of.get(f["properties"]["fips"])
        f["properties"]["reading"] = "no data" if v is None else f"{v:,.1f}"

    folium.GeoJson(
        data,
        name=measure_label,
        overlay=False,               # base layer -> radio button
        show=active,                 # the one currently chosen above
        style_function=style,
        highlight_function=lambda _: {"weight": 2, "color": "#333333"},
        tooltip=folium.GeoJsonTooltip(fields=["state", "reading"],
                                      aliases=["", f"{measure_label}:"]),
    ).add_to(fmap)


for _lab in layers:
    add_measure(m, _lab, active=(_lab == label))

folium.LayerControl(collapsed=False).add_to(m)

components.html(m._repr_html_(), height=520)


def fmt(v):
    """Short, readable numbers: 1.6M, 24k, 18.2."""
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if abs(v) >= 10_000:    return f"{v/1_000:.0f}k"
    if abs(v) >= 100:       return f"{v:,.0f}"
    return f"{v:.1f}"


def draw_legend(bins, ramp, title):
    """The legend, drawn with matplotlib.

    This could be hand-written HTML passed to st.markdown with
    unsafe_allow_html=True, and plenty of Streamlit examples do exactly that.
    Drawing it instead keeps the app in one language: it is the same
    matplotlib used for every other figure in this course, it is readable, and
    there is no markup to get wrong. Reach for unsafe_allow_html only when
    there is no other way.
    """
    n = len(bins) - 1
    fig, ax = plt.subplots(figsize=(9, 0.9))
    fig.patch.set_alpha(0)                      # sit on the app's own background
    for i in range(n):
        ax.add_patch(Rectangle(
            (i, 0), 0.92, 1,
            facecolor=matplotlib.colormaps[ramp](i / max(n - 1, 1)),
            edgecolor="#888888", linewidth=0.6))
        ax.text(i + 0.46, -0.3, f"{fmt(bins[i])} - {fmt(bins[i + 1])}",
                ha="center", va="top", fontsize=8, color="#888888")
    ax.set_title(title, fontsize=9, loc="left", color="#888888")
    ax.set_xlim(-0.08, n); ax.set_ylim(-1.3, 1.2); ax.axis("off")
    return fig


# One legend per layer on the map. Leaflet switches base layers in the
# browser, so this script has no way of knowing which one the reader is
# looking at -- showing every key is the honest answer, and it also lets you
# compare the class boundaries between measures without switching.
for _lab in layers:
    _col, _unit = MEASURES[_lab]
    _fig = draw_legend(class_breaks(_col, k, scheme), ramp, f"{_lab}  ({_unit})")
    st.pyplot(_fig)
    plt.close(_fig)

# Not .lower(): "Jenks" is a surname (George Jenks), so it keeps its capital.
_scheme_name = scheme.split(" — ")[0]
st.caption(f"**{label}**, {_scheme_name} with {len(bins) - 1} "
           f"classes, {ramp} color ramp. Alaska, Hawaii and Puerto Rico are not "
           f"shown. Every one of those four statements is a choice someone "
           f"made, which is why they belong under the map.")

if RAMPS[ramp] == "diverging":
    st.warning(
        f"**{ramp} is a diverging ramp.** It runs from one hue through a pale "
        "middle to a different hue, which tells the reader there is a "
        "meaningful midpoint — a zero, an average, a target. None of the "
        "measures here has one, so the pale band is falling at whatever value "
        "the class breaks happen to put it. Diverging ramps are for things "
        "like change since last year, or distance from the national rate."
    )

left, right = st.columns(2)
with left:
    st.markdown("**Highest five**")
    st.dataframe(stats.nlargest(5, column)[["state", column]],
                 hide_index=True, use_container_width=True)
with right:
    st.markdown("**Lowest five**")
    st.dataframe(stats.nsmallest(5, column)[["state", column]],
                 hide_index=True, use_container_width=True)

st.info(
    "**Switch between a count and a rate and watch the map redraw.**\n\n"
    "The two count measures agree almost perfectly with each other — their "
    "rank correlation is **+0.99**, and they share four of their top five "
    "states. That is the point: a count map is largely a map of where people "
    "live, whatever you are counting.\n\n"
    "Count against share is **-0.17**, near enough to no relationship. The "
    "rate is showing you something the count cannot."
)

# ------------------------------------------------------------------
# TODO (pick one):
# 1. Add a second map beside the first so two measures can be compared at once
# 2. Color by rank instead of value, and say what that hides
# 3. Add a search box that zooms to a state
# 4. Swap the tiles (folium.Map(tiles=...)) and judge whether the basemap helps
#    or competes with the data
# ------------------------------------------------------------------
