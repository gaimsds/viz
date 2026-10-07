# app_topic_starter.py — Week 7: TF-IDF corpus explorer (fully offline)
# Run with:  streamlit run app_topic_starter.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Distinctive-Terms Explorer")
st.caption("Paste documents (one per line). TF-IDF surfaces what makes each one different.")

default = "the model trains on data\nvisualization turns data into pictures\nembeddings place similar text near similar text"
raw = st.text_area("Corpus (one document per line)", default, height=160)
docs = [d.strip() for d in raw.split("\n") if d.strip()]

if len(docs) >= 2:
    tfv = TfidfVectorizer(stop_words="english")
    T = tfv.fit_transform(docs)
    terms = tfv.get_feature_names_out()

    which = st.selectbox("Document", range(len(docs)), format_func=lambda i: f"doc {i+1}: {docs[i][:40]}…")
    row = pd.Series(T[which].toarray()[0], index=terms)
    top = row[row > 0].sort_values().tail(8)
    st.bar_chart(top)
else:
    st.info("Need at least 2 documents.")

# ------------------------------------------------------------------
# TODO (pick one):
# 1. Add an n-grams toggle: TfidfVectorizer(ngram_range=(1,2))
# 2. Add a similarity view: cosine_similarity(T) heat table via st.dataframe
# 3. Add a stop-words on/off toggle and watch the bars change
# ------------------------------------------------------------------
