# app_api_client_wk11.py — DATS 6401 Week 11: consuming an API
#
# Embedded in the Week 11 chapter of the course site through
# dats6401/_live_app_week11.qmd in the gaimsds/dataviz repository.
#
# Built for that setting:
#
#   * every interaction is a click-then-redraw, with no long-running loops --
#     Streamlit Community Cloud's free tier handles that workload well
#   * no bundled data files: the whole point is that the data arrives over HTTP
#   * laid out for a ~700px iframe -- controls on top, no sidebar
#   * the only import beyond streamlit/pandas is requests, which streamlit
#     already depends on, so this app adds nothing to the shared cold start
#
# It is Block 2 made clickable. Four things are on display: query parameters as
# the interface, the three checks on a response, caching keyed on arguments,
# and what the page does when the call fails.
import time

import pandas as pd
import requests
import streamlit as st

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Each mode points the same code at something that breaks differently. The
# behavior of all four is what the three checks below have to survive.
MODES = {
    "normal": (USGS, None),
    "bad parameter → 400": (USGS, {"minmagnitude": "abc"}),
    "200 that lies → HTML": ("https://earthquake.usgs.gov/", {}),
    "unreachable host": ("https://not-a-real-endpoint.invalid/", {}),
}

st.set_page_config(page_title="Week 11 — consuming an API", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def call_api(url, params, timeout=20):
    """One GET, reported honestly.

    Cached on its ARGUMENTS: repeat a combination and the answer comes back
    without touching the network; change any one of them and it refetches.
    A cached function that read the widgets from global scope instead of
    taking them as arguments would return its first answer forever.

    Returns a dict rather than raising, so the page can show what happened at
    each check instead of losing it inside a traceback.
    """
    out = {"url": url, "status": None, "ctype": None, "elapsed": None,
           "payload": None, "error": None, "checks": []}
    try:
        t0 = time.perf_counter()
        r = requests.get(url, params=params, timeout=timeout)
        out["elapsed"] = time.perf_counter() - t0
    except requests.RequestException as e:
        # It never answered. Nothing below can run.
        out["error"] = (f"Could not reach the endpoint — {type(e).__name__}. "
                        "A network problem, not a data problem.")
        return out

    out["url"] = r.url
    out["status"] = r.status_code
    out["ctype"] = r.headers.get("content-type", "").split(";")[0]

    # 1. status -- what raise_for_status() checks, and all it checks
    out["checks"].append(("status is 2xx", r.ok, str(r.status_code)))
    if not r.ok:
        out["error"] = f"The server answered {r.status_code}: {r.text[:140]}"
        return out

    # 2. content type -- a 200 does not promise JSON
    is_json = "json" in out["ctype"]
    out["checks"].append(("content type is JSON", is_json, out["ctype"] or "(none)"))
    if not is_json:
        out["error"] = (f"200, but the body is {out['ctype']}, not JSON: "
                        f"{r.text[:140]}")
        return out

    # 3. shape -- valid JSON is not the same as the JSON this code parses
    payload = r.json()
    has_features = isinstance(payload, dict) and "features" in payload
    out["checks"].append(("shape has 'features'", has_features,
                          type(payload).__name__))
    if not has_features:
        out["error"] = ("200, valid JSON, wrong shape. This is the failure that "
                        "surfaces later as a TypeError on a line that has been "
                        "correct for years.")
        return out

    out["payload"] = payload
    return out


def to_frame(payload):
    """GeoJSON features into a tidy table.

    The trap: coordinates arrive as [longitude, latitude, depth]. Longitude
    first, and a third value most code does not expect. Read them positionally
    in the wrong order and every point lands in the ocean.
    """
    rows = []
    for f in payload["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        rows.append({
            "place": p.get("place"),
            "magnitude": p.get("mag"),
            # USGS timestamps are epoch MILLIseconds, not seconds
            "when": pd.to_datetime(p.get("time"), unit="ms", utc=True),
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(rows)


st.subheader("Consuming an API, live")
st.caption("Every control below changes a **query parameter**. That is the "
           "interface — you are not filtering a table you downloaded, you are "
           "asking a different question.")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
minmag = c1.slider("Minimum magnitude", 1.0, 7.0, 4.5, 0.5)
days = c2.slider("Days back", 1, 30, 7)
limit = c3.slider("Max results", 10, 500, 200, 10)
mode = c4.selectbox("Break it on purpose", list(MODES), index=0,
                    help="The same code, pointed at something that fails "
                         "differently each time.")

base, override = MODES[mode]
params = {
    "format": "geojson",
    "minmagnitude": minmag,
    "starttime": (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
    "limit": limit,
    "orderby": "time",
}
if override is not None:
    params = {**params, **override} if override else None

outer = time.perf_counter()
res = call_api(base, params)
outer = time.perf_counter() - outer
from_cache = res["elapsed"] is not None and outer < res["elapsed"] / 2

st.code(res["url"] or base, language="text")

# The three checks, as a row of pass/fail. They stay on screen whatever
# happened, because the point is which one caught the problem.
cols = st.columns(3)
labels = ["status is 2xx", "content type is JSON", "shape has 'features'"]
done = {name: (passed, detail) for name, passed, detail in res["checks"]}
for col, label in zip(cols, labels):
    # The detail goes in a caption, not in metric's delta slot: delta draws an
    # up arrow, and there is no direction here to point in.
    if label in done:
        passed, detail = done[label]
        col.metric(label, "pass" if passed else "FAIL")
        col.caption(detail)
    else:
        col.metric(label, "—")
        col.caption("not reached")

if res["error"]:
    st.error(res["error"])
    st.caption("The app reported the problem and kept running. A client that "
               "shows a traceback here fails the homework.")
    st.stop()

df = to_frame(res["payload"])
m1, m2, m3 = st.columns(3)
m1.metric("Earthquakes returned", len(df))
m2.metric("Round trip", f"{res['elapsed']:.2f} s")
m3.metric("This render", "from cache" if from_cache else "fresh request")

if df.empty:
    st.info("A valid, successful, empty answer. Raise the day range or lower "
            "the magnitude.")
    st.stop()

st.map(df[["lat", "lon"]], size=20)
st.dataframe(
    df.sort_values("magnitude", ascending=False)
      .assign(when=lambda d: d["when"].dt.strftime("%Y-%m-%d %H:%M UTC")),
    width="stretch", hide_index=True,
)
st.caption("Source: USGS Earthquake Catalog, a public endpoint that needs no "
           "key. Results cached for 5 minutes, keyed on the controls above.")
