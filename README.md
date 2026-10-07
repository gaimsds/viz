# viz

Streamlit apps for **DATS 6401 — Visualization of Complex Data**, deployed on
Streamlit Community Cloud and linked from the course site.

| File | Used by |
| --- | --- |
| `app.py` | the course demo app embedded in several chapters |
| `app_topic_starter.py` | Week 7 — text explorer (counts, clustering, BERTopic) |
| `app_starter_wk6.py` | Week 6 — interactive network explorer |
| `demo_2_widgets_layout.py` | Week 5 — widgets, layout, state |
| `app_compare_starter.py` | Week 2 — one question, two encodings |
| `streamlit_app.py` | the stock Streamlit geo example |

## requirements.txt is shared

Community Cloud reads one `requirements.txt` per repo, so **every line is
installed for every app in this repo** and lengthens each one's cold start.
Keep it short, and check whether a dependency is really needed by more than one
app before adding it.

In particular, do not add `bertopic`, `sentence-transformers` or `torch`. An app
here gets [between 690 MB and 2.7 GB of
memory](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app),
and on Linux a plain `pip install torch` pulls the CUDA build and its
`nvidia-*` wheels — several GB before the app has done anything.

## Week 7: how the BERTopic tab works

Tab 3 of `app_topic_starter.py` shows real BERTopic output without BERTopic
being installed. The expensive half runs on a laptop and only the results are
committed:

```bash
pip install bertopic sentence-transformers umap-learn hdbscan   # locally only
python3 make_bertopic_artifact.py week7_bertopic.json
```

That writes a ~600 KB JSON holding the topic term weights, the topic
coordinates and one record per document. The app reads it with `json.loads` and
draws the three standard BERTopic views in plotly. Nothing in the deployed app
imports `bertopic`.

The trade-off is that tab 3 cannot run on a corpus someone uploads in tab 1, and
the tab says so. Live BERTopic belongs in the Block 2 demo notebook, run in
class.

## Apps sleep

A free app sleeps after a period of inactivity, and the first visitor then gets
a "Zzzz" wake-up screen and waits roughly a minute. These apps are iframed into
chapters, so open each one before class.
