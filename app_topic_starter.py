# app_topic_starter.py — Week 7 starter: text as data, in three views
# Run with:  streamlit run app_topic_starter.py
#
# Tab 1 reads a corpus and counts it.  Tab 2 projects and clusters it live.
# Tab 3 shows BERTopic output that was computed offline and committed as JSON.
#
# Why tab 3 is precomputed: BERTopic depends on sentence-transformers, which
# depends on torch.  Streamlit Community Cloud gives an app between 690 MB and
# 2.7 GB of memory, and requirements.txt here is shared by every app in this
# repo, so installing torch would slow down all of them.  The expensive half
# therefore runs on a laptop and only the results travel.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

st.set_page_config(page_title="Week 7 — Text Explorer", layout="wide")

INK = "#2E6E8E"
GREY = "#cccccc"
ARTIFACT = Path(__file__).resolve().parent / "week7_bertopic.json"

DEFAULT_CORPUS = """the model trains on data and the data shapes what the model can say
data visualization turns numbers into pictures that people can read quickly
a topic model finds themes hiding in a large corpus of text
embeddings place similar text near similar text in a shared space
people read pictures faster than they read tables of numbers
text needs a transform before any visualization can begin at all
a word cloud sizes each word by how often it appears in the text
the sorted bar chart puts frequency on position and length
stop words are removed because they appear in every document
tf idf asks which words are frequent here and rare everywhere else
a data scientist reads the chart and then reads the raw text
machine learning models need clean data more than clever algorithms
the corpus is small so every topic model result deserves suspicion
good visualization design starts from the question not from the data
social media text is noisy and full of repeated boilerplate phrases
a language model can summarize text but it can also invent detail
counting words is easy and interpreting the counts is the hard part
data science borrows from statistics computing and visual design
the same corpus gives different topics under different random seeds
clean text first because preprocessing choices change every later chart
a chart of word frequency hides the sentences the words came from
readers trust a picture of data more than a table of the same data"""


# --------------------------------------------------------------- helpers -----
@st.cache_data
def term_counts(docs, drop_stopwords, ngram_max):
    cv = CountVectorizer(stop_words="english" if drop_stopwords else None,
                         ngram_range=(1, ngram_max))
    X = cv.fit_transform(docs)
    return pd.Series(np.asarray(X.sum(axis=0)).ravel(),
                     index=cv.get_feature_names_out()).sort_values(ascending=False)


@st.cache_data
def tfidf(docs, drop_stopwords, ngram_max):
    tv = TfidfVectorizer(stop_words="english" if drop_stopwords else None,
                         ngram_range=(1, ngram_max))
    return tv.fit_transform(docs).toarray(), tv.get_feature_names_out()


@st.cache_data
def project_and_cluster(docs, n_neighbors, min_dist, min_cluster_size, n_svd):
    """TF-IDF -> SVD -> UMAP(2) -> HDBSCAN.  No transformer embeddings here."""
    from umap import UMAP          # imported lazily: it drags in numba

    X, _ = tfidf(docs, True, 1)
    k = min(n_svd, min(X.shape) - 1)
    Z = TruncatedSVD(n_components=max(k, 2), random_state=0).fit_transform(X)
    xy = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
              metric="cosine", random_state=0).fit_transform(Z)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(xy)
    return xy, labels


@st.cache_data
def load_artifact():
    if not ARTIFACT.exists():
        return None
    return json.loads(ARTIFACT.read_text())


def read_upload(upload):
    """Return a list of documents, or None if the file could not be used."""
    if upload.name.lower().endswith(".csv"):
        df = pd.read_csv(upload)
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            st.error("No text column found in that CSV.")
            return None
        col = st.selectbox("Which column holds the text?", text_cols)
        return [str(v).strip() for v in df[col].dropna() if str(v).strip()]
    raw = upload.read().decode("utf-8", errors="replace")
    return [d.strip() for d in raw.splitlines() if d.strip()]


# ------------------------------------------------------------------ app ------
st.title("Week 7 — Text Explorer")
st.caption("Text has no x and no y. Each tab is a picture of a different "
           "transform of the same words.")

tab1, tab2, tab3 = st.tabs(
    ["1 · Corpus and counts", "2 · Project and cluster", "3 · BERTopic (precomputed)"])

# =========================================================== Tab 1 ===========
with tab1:
    st.subheader("Load a corpus")
    source = st.radio("Source", ["Use the built-in corpus", "Upload a file"],
                      horizontal=True, label_visibility="collapsed")

    docs = None
    if source == "Upload a file":
        upload = st.file_uploader("A .txt file with one document per line, "
                                  "or a .csv with a text column",
                                  type=["txt", "csv"])
        if upload is not None:
            docs = read_upload(upload)
    else:
        raw = st.text_area("Corpus (one document per line)",
                           DEFAULT_CORPUS, height=180)
        docs = [d.strip() for d in raw.split("\n") if d.strip()]

    if not docs or len(docs) < 2:
        st.info("Load at least 2 documents to continue.")
        st.stop()

    st.session_state["docs"] = tuple(docs)      # tuple: hashable, so cacheable
    st.success(f"{len(docs)} documents, "
               f"{sum(len(d.split()) for d in docs):,} words in total.")

    c1, c2, c3 = st.columns(3)
    drop_sw = c1.checkbox("Remove stop words", value=True)
    ngram_max = c2.slider("Longest phrase (n-gram)", 1, 3, 1)
    n_terms = c3.slider("How many terms to show", 10, 60, 25, 5)

    counts = term_counts(tuple(docs), drop_sw, ngram_max)
    top = counts.head(n_terms)

    st.subheader("The same counts, drawn two ways")
    st.caption(f"Both pictures below show the same {len(top)} terms. Only the "
               f"encoding differs.")
    left, right = st.columns(2)

    with left:
        fig = px.bar(x=top.values[::-1], y=top.index[::-1], orientation="h",
                     labels={"x": "count", "y": ""},
                     color_discrete_sequence=[INK], height=460)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Sorted bars.** Frequency is on position and length, the "
                   "two channels Week 2 ranked most accurate. You can read off "
                   "whether the third term beats the fifth.")

    with right:
        wc = WordCloud(width=800, height=460, background_color="white",
                       prefer_horizontal=0.9,
                       color_func=lambda *a, **k: INK   # one colour: see caption
                       ).generate_from_frequencies(top.to_dict())
        fig2, ax = plt.subplots(figsize=(8, 4.6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig2.tight_layout(pad=0)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.caption("**Word cloud.** Frequency is on area, which Cleveland and "
                   "McGill ranked near the bottom, and position is whatever the "
                   "packing algorithm chose. A long word also takes more space "
                   "at equal frequency. Every word is drawn in one colour here, "
                   "because colour in a word cloud usually encodes nothing.")

    st.divider()
    st.subheader("What makes one document different: TF–IDF")
    T, terms = tfidf(tuple(docs), drop_sw, ngram_max)
    which = st.selectbox("Document", range(len(docs)),
                         format_func=lambda i: f"doc {i + 1}: {docs[i][:60]}…")
    row = pd.Series(T[which], index=terms)
    best = row[row > 0].sort_values().tail(8)
    if best.empty:
        st.info("Every term in this document appears in all the others, so "
                "TF–IDF gives it no distinctive terms.")
    else:
        fig3 = px.bar(x=best.values, y=best.index, orientation="h",
                      labels={"x": "TF–IDF weight", "y": ""},
                      color_discrete_sequence=["#d9534f"], height=300)
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Counts answer *what is frequent here*. TF–IDF answers "
                   "*what is frequent here and rare everywhere else*, which is "
                   "usually the more useful question.")

# =========================================================== Tab 2 ===========
with tab2:
    st.subheader("Project, then cluster")
    docs = st.session_state.get("docs")
    if not docs:
        st.info("Load a corpus in tab 1 first.")
        st.stop()

    st.markdown(
        "The pipeline here is **TF–IDF → SVD → UMAP → HDBSCAN**. That is the "
        "same shape as BERTopic's middle two stages, with one difference worth "
        "naming: the input is word counts, not sentence embeddings. So two "
        "documents land near each other when they *use the same words*, not "
        "when they *mean the same thing*. Tab 3 shows the other case."
    )

    if len(docs) < 15:
        st.warning(f"UMAP needs more than a handful of documents and you have "
                   f"{len(docs)}. Upload a larger corpus in tab 1, or read the "
                   f"result below as a demonstration rather than a finding.")

    c1, c2, c3 = st.columns(3)
    n_neighbors = c1.slider("UMAP n_neighbors", 2, 50,
                            min(15, max(2, len(docs) - 1)),
                            help="How much of the neighbourhood each point sees. "
                                 "Low values keep local detail; high values keep "
                                 "the global shape.")
    min_dist = c2.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01,
                         help="How tightly points may pack together.")
    min_cluster_size = c3.slider("HDBSCAN min_cluster_size", 2,
                                 max(3, len(docs) // 3), max(2, len(docs) // 10),
                                 help="The smallest group that counts as a cluster.")

    n_neighbors = min(n_neighbors, len(docs) - 1)

    try:
        xy, labels = project_and_cluster(tuple(docs), n_neighbors, min_dist,
                                         min_cluster_size, 50)
    except Exception as exc:                       # UMAP is fussy on tiny inputs
        st.error(f"The projection failed on this corpus: {exc}")
        st.stop()

    n_found = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())

    df = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "cluster": ["unassigned" if v == -1 else f"cluster {v}" for v in labels],
        "document": [d[:90] + ("…" if len(d) > 90 else "") for d in docs],
    })
    # Sort by cluster number, so "cluster 2" precedes "cluster 10".
    order = ([f"cluster {v}" for v in sorted(set(labels.tolist()) - {-1})]
             + (["unassigned"] if n_noise else []))
    fig = px.scatter(df, x="x", y="y", color="cluster", hover_name="document",
                     category_orders={"cluster": order},
                     color_discrete_map={"unassigned": GREY}, height=520)
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.update_layout(xaxis_title=None, yaxis_title=None,
                      xaxis_showticklabels=False, yaxis_showticklabels=False,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"UMAP of TF–IDF vectors reduced with SVD "
               f"(n_neighbors={n_neighbors}, min_dist={min_dist}); "
               f"HDBSCAN with min_cluster_size={min_cluster_size}. "
               f"**{n_found} clusters, {n_noise} documents unassigned.** "
               f"A caption like this is the honest kind: it names the transform "
               f"and the parameters, so a reader can tell what they are seeing.")

    st.info("**Move one slider at a time and watch the number of clusters "
            "change.** Nothing about the corpus changed. That is the point: "
            "the parameters are part of the result, so a topic count is never "
            "just a property of the data.")

    if n_found:
        st.subheader("What is in each cluster?")
        pick = st.selectbox("Cluster", sorted(set(labels.tolist()) - {-1}),
                            format_func=lambda v: f"cluster {v} "
                                                  f"({int((labels == v).sum())} docs)")
        members = [d for d, v in zip(docs, labels) if v == pick]
        sub_counts = term_counts(tuple(members), True, 1).head(8)
        st.write("**Most frequent terms:** " + ", ".join(sub_counts.index))
        st.dataframe(pd.DataFrame({"document": members[:15]}),
                     use_container_width=True, hide_index=True)
        st.caption("Reading the documents is how you check a cluster. A label "
                   "built from frequent terms is a summary, not evidence.")

# =========================================================== Tab 3 ===========
with tab3:
    art = load_artifact()
    if art is None:
        st.warning(
            "`week7_bertopic.json` is missing, so this tab has nothing to draw. "
            "Generate it with the script in the Week 7 materials and commit it "
            "next to this app."
        )
        st.stop()

    p, corpus = art["params"], art["corpus"]
    st.subheader("BERTopic on a real corpus")
    st.markdown(
        f"These are real BERTopic results, computed once offline and stored as "
        f"a small JSON file. They do **not** react to the corpus in tab 1 — "
        f"running BERTopic live needs sentence-transformers and torch, which do "
        f"not fit on a free Streamlit server. The live BERTopic cells are in the "
        f"Block 2 demo notebook, to run in class.\n\n"
        f"**Corpus:** {corpus['name']} — {corpus['n_docs']:,} documents. "
        f"**Pipeline:** `{p['embedding']}` embeddings → UMAP "
        f"(n_neighbors={p['umap']['n_neighbors']}, "
        f"n_components={p['umap']['n_components']}) → HDBSCAN "
        f"(min_cluster_size={p['hdbscan']['min_cluster_size']}) → c-TF-IDF. "
        f"**Result:** {p['n_topics']} topics, {p['n_unassigned']:,} documents "
        f"left unassigned."
    )

    topics = [t for t in art["topics"] if t["topic"] != -1]

    st.markdown("#### 1. What each topic is made of")
    st.caption("c-TF-IDF treats all the documents in a topic as one long "
               "document, then asks which terms are distinctive to it. These "
               "bars are that score — they are not word counts.")
    pick = st.selectbox(
        "Topic", [t["topic"] for t in topics],
        format_func=lambda i: f"Topic {i}: "
                              f"{next(t['label'] for t in topics if t['topic'] == i)} "
                              f"({next(t['count'] for t in topics if t['topic'] == i)} docs)")
    chosen = next(t for t in topics if t["topic"] == pick)
    wdf = pd.DataFrame(chosen["words"], columns=["term", "score"])
    figb = px.bar(wdf.iloc[::-1], x="score", y="term", orientation="h",
                  color_discrete_sequence=[INK], height=380,
                  labels={"score": "c-TF-IDF score", "term": ""})
    figb.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figb, use_container_width=True)

    st.markdown("#### 2. Where the topics sit relative to each other")
    tdf = pd.DataFrame(art["topic_coords"])
    tdf["name"] = "Topic " + tdf["topic"].astype(str) + ": " + tdf["label"]
    figt = px.scatter(tdf, x="x", y="y", size="count", hover_name="name",
                      text="topic", size_max=60, height=520,
                      color_discrete_sequence=[INK])
    figt.update_traces(textposition="middle center",
                       textfont=dict(color="white", size=11))
    figt.update_layout(xaxis_title=None, yaxis_title=None,
                       xaxis_showticklabels=False, yaxis_showticklabels=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figt, use_container_width=True)
    st.caption("Each circle is a topic, placed at the average position of its "
               "documents and sized by how many it has. Distance here is a "
               "projection of embedding space, so treat near and far as a hint, "
               "not a measurement.")

    st.markdown("#### 3. Every document, coloured by topic")
    ddf = pd.DataFrame(art["docs"])
    labels_by_id = {t["topic"]: f"Topic {t['topic']}: {t['label']}" for t in topics}
    ddf["Topic"] = ddf["topic"].map(labels_by_id).fillna("unassigned (−1)")
    show_noise = st.checkbox("Show the unassigned documents", value=True)
    plot_df = ddf if show_noise else ddf[ddf["topic"] != -1]
    # Order the legend by topic number, not by the label text, so Topic 2 comes
    # before Topic 10.
    legend_order = [labels_by_id[t] for t in sorted(labels_by_id)
                    if labels_by_id[t] in set(plot_df["Topic"])]
    if show_noise:
        legend_order.append("unassigned (−1)")
    figd = px.scatter(plot_df, x="x", y="y", color="Topic", hover_name="snippet",
                      category_orders={"Topic": legend_order},
                      color_discrete_map={"unassigned (−1)": GREY}, height=560,
                      opacity=0.75)
    figd.update_traces(marker=dict(size=5))
    figd.update_layout(xaxis_title=None, yaxis_title=None,
                       xaxis_showticklabels=False, yaxis_showticklabels=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figd, use_container_width=True)
    st.caption(f"UMAP of {p['embedding']} embeddings, 2 components. The grey "
               f"points are BERTopic's topic −1: the documents HDBSCAN declined "
               f"to assign. Toggle them off and the picture looks much tidier, "
               f"which is exactly why leaving them in is the honest choice.")

# ------------------------------------------------------------------
# TODO (pick one):
# 1. Tab 1: add a toggle for the minimum document frequency (min_df) and watch
#    rare terms leave the word cloud.
# 2. Tab 2: colour the points by a metadata column from your uploaded CSV
#    instead of by cluster, and see whether the clusters agree with it.
# 3. Tab 2: add a second HDBSCAN run at a different min_cluster_size and show
#    the two side by side, the way the Block 2 demo does.
# ------------------------------------------------------------------
