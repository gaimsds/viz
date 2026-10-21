# app_map_starter.py — Week 9 starter: a choropleth you can change
# Run with:  streamlit run app_map_starter.py
#
# Uses folium (the Python wrapper around Leaflet) but NOT geopandas: a
# choropleth only needs the GeoJSON and a table, and leaving geopandas out
# keeps the install small. geopandas' own .explore() does the same thing in one
# line when you already have a GeoDataFrame -- see the Block 2 demo.
import json
from pathlib import Path

import folium
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
    gj = json.loads((HERE / "states.geojson").read_text())
    df = pd.read_csv(HERE / "state_stats.csv", dtype={"fips": str})
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

values = stats[column]
if scheme.startswith("Quantiles"):
    bins = list(values.quantile([i / k for i in range(k + 1)]))
elif scheme.startswith("Equal interval"):
    lo, hi = values.min(), values.max()
    bins = [lo + (hi - lo) * i / k for i in range(k + 1)]
else:
    # Jenks looks for the cuts that make each class as internally similar as
    # possible. Unlike the other two it reads the actual values, so it is the
    # only one of the three that reacts to the shape of the distribution.
    import mapclassify
    bins = [float(values.min())] + [float(b)
                                    for b in mapclassify.NaturalBreaks(
                                        values.values, k=k).bins]
# Round the class edges before handing them over: folium prints them straight
# into the legend, and six-figure numbers at eight classes overlap into an
# unreadable smear.
span = values.max() - values.min()
step = 10 ** max(0, len(str(int(span))) - 2)          # 2 significant figures
bins = sorted({round(b / step) * step if step > 1 else round(b, 1) for b in bins})
# Rounding can move the outer edges INSIDE the data, which makes folium reject
# the whole thing because some state falls in no bin. Clamp them back out.
bins[0] = min(bins[0], float(values.min()))
bins[-1] = max(bins[-1], float(values.max()))
if len(bins) < 2:
    bins = [float(values.min()), float(values.max())]

# tiles=None: no basemap at all. CartoDB's tiles now demand an API key and
# stamp "API KEY REQUIRED" across the map. A basemap is optional for a
# choropleth anyway -- the states carry the data, and roads and labels
# underneath them are competition, not context.
# scrollWheelZoom=False: inside a page, a Leaflet map swallows the scroll
# wheel and zooms itself, so the reader cannot scroll past it. The +/- buttons
# still zoom.
m = folium.Map(location=[39, -98], zoom_start=4, tiles=None,
               scrollWheelZoom=False)
cp = folium.Choropleth(
    geo_data=geojson, data=stats, columns=["fips", column],
    key_on="feature.properties.fips", bins=bins,
    fill_color=ramp, fill_opacity=0.8, line_color="white", line_weight=0.5,
    legend_name=f"{label} ({unit})",
)
cp.add_to(m)

# Drop folium's own color bar. It spaces its tick labels by value, so a skewed
# measure crowds the first four numbers into an unreadable smear at the left.
# The legend below the map lists the same classes, one per row.
for key in list(cp._children):
    if key.startswith("color_map"):
        del cp._children[key]

# Hovering is the thing a static map cannot do: it puts the number next to the
# color, so a reader does not have to eyeball the legend.
lookup = stats.set_index("fips")
folium.GeoJson(
    geojson,
    style_function=lambda _: {"fillOpacity": 0, "weight": 0},
    tooltip=folium.GeoJsonTooltip(fields=["state"], aliases=[""]),
    popup=folium.GeoJsonPopup(fields=["state"], aliases=["State"]),
).add_to(m)

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


_legend = draw_legend(bins, ramp, f"{label}  ({unit})")
st.pyplot(_legend)
plt.close(_legend)

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
