# app_starter.py — Week 6 starter: interactive network explorer
# Run with:  streamlit run app_starter.py
#
# The graph below is a live PyVis/vis.js network, not a picture: drag a node and
# its neighbours follow, hover for its stats, scroll to zoom.
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from networkx.algorithms.community import greedy_modularity_communities
from pyvis.network import Network

st.set_page_config(page_title="Network Explorer", layout="wide")
st.title("Network Explorer — Les Misérables")
st.caption("Week 6 starter app. Extend it for your homework.")

PALETTE = ["#2E6E8E", "#d9534f", "#6aa84f", "#b07aa1", "#e6a23c", "#5a6672"]

# --- Sidebar controls --------------------------------------------------------
layout = st.sidebar.selectbox("Layout", ["force-directed", "circular"])
min_degree = st.sidebar.slider("Hide characters with degree below", 1, 10, 1)
show_labels = st.sidebar.checkbox("Show names", value=True)


# --- Load, filter, measure ---------------------------------------------------
@st.cache_data
def load_graph():
    return nx.les_miserables_graph()


G = load_graph()
H = G.subgraph([n for n, d in G.degree() if d >= min_degree]).copy()

if H.number_of_nodes() == 0:
    st.warning("No characters left at that threshold — lower the slider.")
    st.stop()

degree = dict(H.degree())
betweenness = nx.betweenness_centrality(H)
community = {n: i for i, c in enumerate(greedy_modularity_communities(H)) for n in c}

c1, c2, c3 = st.columns(3)
c1.metric("Characters", H.number_of_nodes())
c2.metric("Co-appearances", H.number_of_edges())
c3.metric("Communities", len(set(community.values())))

# --- Build the interactive network -------------------------------------------
# Encode the computed structure the way the Block 1 deck did: the layout takes
# x and y, so degree goes to size and community goes to colour.
net = Network(height="600px", width="100%", bgcolor="#ffffff",
              font_color="#222222", cdn_resources="in_line", notebook=False)

for n in H.nodes():
    net.add_node(
        n,
        label=n if show_labels else " ",
        size=8 + degree[n] * 1.4,
        color=PALETTE[community[n] % len(PALETTE)],
        title=(f"{n}\ndegree {degree[n]}\n"
               f"betweenness {betweenness[n]:.3f}\ncommunity {community[n]}"),
    )
for u, v in H.edges():
    net.add_edge(u, v, color="#d5d5d5")

if layout == "circular":
    # Turning physics off only freezes nodes wherever vis.js first placed them,
    # which is not a circular layout. Compute the positions and hand them over.
    pos = nx.circular_layout(H, scale=420)
    for node in net.nodes:
        x, y = pos[node["id"]]
        node["x"], node["y"] = float(x), float(y)
        node["physics"] = False
    net.toggle_physics(False)

components.html(net.generate_html(notebook=False), height=620)

st.caption("Size = degree · colour = detected community · hover a node for its "
           "betweenness. Drag a node to see the force layout re-settle.")

# ------------------------------------------------------------------
# TODO (in class, then extend for homework):
# 1. Add a selectbox that switches node SIZE between degree, betweenness and
#    closeness centrality. Scale each one so the range stays readable —
#    betweenness values are small, so something like v * 2000 + 10.
# 2. Add a text input that highlights one character (colour it differently and
#    enlarge it) so a reader can find someone by name.
# 3. Betweenness is recomputed on every interaction and is the slow step here.
#    Move it into its own @st.cache_data function keyed on the degree threshold.
# ------------------------------------------------------------------
