#!/usr/bin/env python3
"""Run BERTopic once, offline, and export a small JSON the Streamlit app can draw.

The deployed app must not depend on bertopic/sentence-transformers/torch -- see
the note in viz/README. So the expensive half runs here, on a laptop, and only
the results travel: topic term weights, topic coordinates, document coordinates.

    python3 make_bertopic_artifact.py OUT.json
"""
import json
import sys

import numpy as np
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = [
    "rec.sport.baseball",
    "sci.space",
    "comp.graphics",
    "sci.med",
    "rec.autos",
    "talk.politics.mideast",
]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_KW = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
               random_state=42)
# BERTopic's own defaults, which is what a student reproducing this will get.
HDBSCAN_KW = dict(min_cluster_size=25, metric="euclidean",
                  cluster_selection_method="eom")
# c-TF-IDF runs on THIS vectorizer. Leaving it at the default means no stop-word
# removal, and then every topic is labelled "the, to, and" -- which is Task 1 of
# the hands-on, demonstrated by accident.
# token_pattern keeps letters only, so bare numbers such as "92" cannot become a
# topic term; the extra stop words are Usenet plumbing that survives sklearn's
# remove=("headers","footers","quotes") and otherwise shows up in topic labels.
NEWSGROUP_BOILERPLATE = ["edu", "com", "writes", "article", "subject", "lines",
                         "organization", "ax", "nntp", "posting", "host", "reply"]
VECTORIZER_KW = dict(
    stop_words="english",
    min_df=5,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)


def main(out_path):
    print("loading 20 newsgroups ...", flush=True)
    data = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=42)
    import re
    bar = re.compile(r"-{4,}[^\n]*-{4,}")        # "----Original message----"
    docs = []
    for d in data.data:
        d = " ".join(bar.sub(" ", d).split())
        # Keep only documents with real prose in them.
        if len(d) > 200 and sum(c.isalpha() for c in d) / len(d) > 0.6:
            docs.append(d)
    print(f"{len(docs)} documents after dropping stubs and boilerplate", flush=True)

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import (
        ENGLISH_STOP_WORDS, CountVectorizer)
    from umap import UMAP

    print(f"embedding with {EMBEDDING_MODEL} ...", flush=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    emb = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    print("fitting BERTopic ...", flush=True)
    model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(**UMAP_KW),
        hdbscan_model=HDBSCAN(**HDBSCAN_KW),
        vectorizer_model=CountVectorizer(
            **{**VECTORIZER_KW,
               "stop_words": list(ENGLISH_STOP_WORDS) + NEWSGROUP_BOILERPLATE}),
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs, embeddings=emb)
    topics = np.asarray(topics)

    info = model.get_topic_info()
    print(info.head(12).to_string(), flush=True)

    # --- topic term weights (this is the c-TF-IDF output) --------------------
    topic_rows = []
    for tid in sorted(set(topics.tolist())):
        words = model.get_topic(tid) or []
        topic_rows.append({
            "topic": int(tid),
            "count": int((topics == tid).sum()),
            "label": ", ".join(w for w, _ in words[:3]) if tid != -1 else "unassigned",
            "words": [[w, round(float(s), 5)] for w, s in words[:10]],
        })

    # --- 2-D coordinates --------------------------------------------------
    # Documents: a fresh 2-component UMAP of the same embeddings, which is what
    # BERTopic's own document plot does.
    print("projecting documents to 2-D ...", flush=True)
    doc_xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.0,
                  metric="cosine", random_state=42).fit_transform(emb)

    # Topics: mean position of their documents, so the two plots agree.
    topic_xy = {}
    for tid in sorted(set(topics.tolist())):
        sel = topics == tid
        topic_xy[tid] = doc_xy[sel].mean(axis=0)

    payload = {
        "corpus": {
            "name": "20 Newsgroups (6 groups)",
            "categories": CATEGORIES,
            "n_docs": len(docs),
            "source": "sklearn.datasets.fetch_20newsgroups, headers/footers/quotes removed",
        },
        "params": {
            "embedding": EMBEDDING_MODEL,
            "umap": {k: v for k, v in UMAP_KW.items()},
            "hdbscan": HDBSCAN_KW,
            "vectorizer": {**VECTORIZER_KW,
                           "stop_words": "english + newsgroup boilerplate",
                           "extra_stop_words": NEWSGROUP_BOILERPLATE},
            "n_topics": int(len(set(topics.tolist())) - (1 if -1 in topics else 0)),
            "n_unassigned": int((topics == -1).sum()),
        },
        "topics": topic_rows,
        "topic_coords": [
            {"topic": int(t), "x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3),
             "count": int((topics == t).sum()),
             "label": next(r["label"] for r in topic_rows if r["topic"] == t)}
            for t, xy in topic_xy.items() if t != -1
        ],
        "docs": [
            {"x": round(float(doc_xy[i, 0]), 3), "y": round(float(doc_xy[i, 1]), 3),
             "topic": int(topics[i]),
             "snippet": " ".join(docs[i].split())[:90]}
            for i in range(len(docs))
        ],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    import os
    print(f"\nwrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"{payload['params']['n_topics']} topics, "
          f"{payload['params']['n_unassigned']} unassigned of {len(docs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "week7_bertopic.json")
