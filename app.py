"""OSINT dashboard web app (Flask). Reads pre-computed pipeline output (run.py -> dashboard_data/), so it is light enough
for a free 512 MB host. Data folder is auto-detected: dashboard_data/, output/, or the repo root."""
import io, json, os
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
_cache = {}


def data_dir():
    cands = [os.environ.get("DATA_DIR", ""), "dashboard_data", "output", "."]
    for d in cands:
        if d and os.path.exists(os.path.join(BASE, d, "analysed_posts.csv")):
            return os.path.join(BASE, d)
    return None


def _json(name, default):
    d = data_dir()
    p = os.path.join(d, name) if d else None
    if p and os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return default


def load():
    d = data_dir()
    if not d:
        return None
    p = os.path.join(d, "analysed_posts.csv"); m = os.path.getmtime(p)
    if _cache.get("key") != (p, m):
        df = pd.read_csv(p, parse_dates=["timestamp"])
        for c, v in [("country", "Unknown"), ("lang", "en"), ("platform", "X"), ("label", "neutral"), ("state", ""), ("text", "")]:
            if c not in df: df[c] = v
            df[c] = df[c].fillna(v).astype(str)
        for c in ["misinfo_flag", "coordinated"]:
            df[c] = df[c].astype(str).str.lower().eq("true") if c in df else False
        for c, v in [("sentiment", 0.0), ("misinfo_score", 0.0), ("shares", 0), ("followers", 0), ("narrative", -1)]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(v) if c in df else v
        _cache.update(key=(p, m), df=df)
    return _cache["df"]


def filt(df):
    a = request.args
    for k in ["country", "lang", "label", "platform"]:
        v = a.get(k)
        if v and v != "all": df = df[df[k] == v]
    if a.get("start"): df = df[df.timestamp >= pd.to_datetime(a["start"])]
    if a.get("end"): df = df[df.timestamp < pd.to_datetime(a["end"]) + pd.Timedelta(days=1)]
    if a.get("q"): df = df[df.text.str.contains(a["q"], case=False, regex=False)]
    if a.get("flagged") == "1": df = df[df.misinfo_flag | df.coordinated]
    return df


def recs(df):
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


def timeline(df, col_fn, freq):
    idx = pd.date_range(df.timestamp.min().floor(freq), df.timestamp.max().floor(freq), freq=freq)
    return idx


@app.route("/")
def index():
    return Response(open(os.path.join(BASE, "dashboard.html"), encoding="utf-8").read(), mimetype="text/html")


@app.route("/health")
def health():
    return "ok"


@app.route("/api/meta")
def meta():
    df = load()
    if df is None:
        return jsonify(error="No data found. Run run.py and commit dashboard_data/."), 404
    m = _json("meta.json", {})
    return jsonify(total=len(df), start=str(df.timestamp.min().date()), end=str(df.timestamp.max().date()),
                   countries=sorted(df.country.unique()), langs=sorted(df.lang.unique()), platforms=sorted(df.platform.unique()),
                   synthetic=m.get("synthetic", False), generated_at=m.get("generated_at", ""), source=m.get("source", ""))


@app.route("/api/dashboard")
def dashboard():
    df0 = load()
    if df0 is None:
        return jsonify(error="No data"), 404
    df = filt(df0)
    out = dict(alerts=_json("alerts.json", []), coordination=_json("coordination.json", []))
    n = len(df)
    out["kpis"] = dict(posts=n, authors=int(df.author.nunique()) if n else 0,
                       mean_sentiment=round(float(df.sentiment.mean()), 2) if n else 0,
                       pct_negative=round(100 * float((df.label == "negative").mean()), 1) if n else 0,
                       flagged=int(df.misinfo_flag.sum()), coordinated=int(df.coordinated.sum()))
    if n == 0:
        out.update(timeline=dict(labels=[], series=[]), volume=dict(labels=[], total=[], flagged=[]), labels={}, narratives=[], states=[])
        return jsonify(out)
    span = (df.timestamp.max() - df.timestamp.min()).days
    f1 = "D" if span >= 3 else "h"; fmt1 = "%d %b" if f1 == "D" else "%d %b %H:%M"
    idx = pd.date_range(df.timestamp.min().floor(f1), df.timestamp.max().floor(f1), freq=f1)
    g = df.set_index("timestamp").groupby("country")["sentiment"].resample(f1).agg(["mean", "count"]).reset_index()
    series = []
    for c, s in g.groupby("country"):
        s = s.set_index("timestamp").reindex(idx)
        ok = s["count"].fillna(0) >= (3 if f1 == "D" else 2)
        series.append(dict(name=c, data=[round(float(v), 3) if k else None for v, k in zip(s["mean"], ok)], posts=int(s["count"].sum())))
    out["timeline"] = dict(labels=[t.strftime(fmt1) for t in idx], series=sorted(series, key=lambda x: -x["posts"])[:6])
    f2 = "h" if span < 7 else "D"; fmt2 = "%d %b %H:%M" if f2 == "h" else "%d %b"
    idx2 = pd.date_range(df.timestamp.min().floor(f2), df.timestamp.max().floor(f2), freq=f2)
    tot = df.set_index("timestamp").resample(f2).size().reindex(idx2, fill_value=0)
    fl = df[df.misinfo_flag | df.coordinated].set_index("timestamp").resample(f2).size().reindex(idx2, fill_value=0)
    out["volume"] = dict(labels=[t.strftime(fmt2) for t in idx2], total=[int(v) for v in tot], flagged=[int(v) for v in fl])
    out["labels"] = {k: int(v) for k, v in df.label.value_counts().items()}
    nl = {x["narrative"]: x["label"] for x in _json("narratives.json", [])}
    nar = df.groupby("narrative").agg(posts=("post_id", "count"), sentiment=("sentiment", "mean")).reset_index().sort_values("posts", ascending=False)
    out["narratives"] = [dict(label=nl.get(int(r.narrative), f"Narrative {int(r.narrative)}"), posts=int(r.posts), sentiment=round(float(r.sentiment), 2)) for r in nar.itertuples()]
    st = df[df.state != ""].groupby("state").agg(posts=("post_id", "count"), sentiment=("sentiment", "mean")).reset_index().sort_values("posts", ascending=False).head(10)
    out["states"] = [dict(state=r.state, posts=int(r.posts), sentiment=round(float(r.sentiment), 2)) for r in st.itertuples()]
    return jsonify(out)


COLS = ["timestamp", "author", "country", "state", "platform", "lang", "label", "sentiment", "misinfo_score", "coordinated", "shares", "followers", "text"]


@app.route("/api/posts")
def posts():
    df = load()
    if df is None:
        return jsonify([])
    df = filt(df)
    df = df.sort_values("misinfo_score", ascending=False) if request.args.get("flagged") == "1" else df.sort_values("timestamp", ascending=False)
    df = df.head(60)[[c for c in COLS if c in df.columns]].copy()
    df["text"] = df["text"].str.slice(0, 280)
    return jsonify(recs(df))


@app.route("/api/export.csv")
def export():
    df = load()
    df = filt(df)[[c for c in COLS if c in df.columns]]
    return Response(df.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=osint_export.csv"})
