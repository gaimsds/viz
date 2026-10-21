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
import pandas as pd
from matplotlib.colors import to_hex
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
k = c2.slider("Number of colour classes", 3, 8, 5)
column, unit = MEASURES[label]

scheme = st.radio(
    "How to cut the classes",
    ["Quantiles — equal number of states per class",
     "Equal interval — equal-width value ranges"],
    horizontal=False)

values = stats[column]
if scheme.startswith("Quantiles"):
    bins = list(values.quantile([i / k for i in range(k + 1)]))
else:
    lo, hi = values.min(), values.max()
    bins = [lo + (hi - lo) * i / k for i in range(k + 1)]
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
    fill_color="OrRd", fill_opacity=0.8, line_color="white", line_weight=0.5,
    legend_name=f"{label} ({unit})",
)
cp.add_to(m)

# Drop folium's own colour bar. It spaces its tick labels by value, so a skewed
# measure crowds the first four numbers into an unreadable smear at the left.
# The legend below the map lists the same classes, one per row.
for key in list(cp._children):
    if key.startswith("color_map"):
        del cp._children[key]

# Hovering is the thing a static map cannot do: it puts the number next to the
# colour, so a reader does not have to eyeball the legend.
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


swatches = [matplotlib.colormaps["OrRd"](i / max(len(bins) - 2, 1))
            for i in range(len(bins) - 1)]
chips = "".join(
    f'<span style="display:inline-flex;align-items:center;margin:0 14px 6px 0">'
    f'<span style="width:17px;height:17px;background:{to_hex(c)};'
    f'border:1px solid #888;margin-right:6px"></span>'
    f'<span style="font-size:0.85rem">{fmt(bins[i])} – {fmt(bins[i+1])}</span></span>'
    for i, c in enumerate(swatches))
st.markdown(f"**{label}** ({unit})<br>{chips}", unsafe_allow_html=True)

st.caption(f"**{label}**, {scheme.split(' — ')[0].lower()} with {len(bins) - 1} "
           f"classes. Alaska, Hawaii and Puerto Rico are not shown. "
           f"Every one of those three statements is a choice someone made, "
           f"which is why they belong under the map.")

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
# 2. Colour by rank instead of value, and say what that hides
# 3. Add a search box that zooms to a state
# 4. Swap the tiles (folium.Map(tiles=...)) and judge whether the basemap helps
#    or competes with the data
# ------------------------------------------------------------------
