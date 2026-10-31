import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel

TOK = re.compile(r"[^\s.,!?;:#\"'।،()]+")
URL = re.compile(r"https?://\S+|@\w+")

def clean(t):
    return re.sub(r"\s+", " ", URL.sub("", str(t)).lower()).strip()

def detect_script(t):
    """Lightweight script-based language hint (no external deps). Swap for fastText/langdetect in production."""
    if re.search(r"[\u0900-\u097F]", t): return "hi"
    if re.search(r"[\u0600-\u06FF]", t): return "ur/ar"
    if re.search(r"[\u0B80-\u0BFF]", t): return "ta"
    if re.search(r"[\u0D00-\u0D7F]", t): return "ml"
    if re.search(r"[\u4E00-\u9FFF]", t): return "zh"
    return "en"

POS = {"proud", "salute", "brave", "success", "peace", "safe", "hero", "positive", "strong", "restored", "forward",
       "गर्व", "सलाम", "शांति", "सफल", "امن", "کامیاب"}
NEG = {"attack", "terror", "killed", "fear", "condemned", "violation", "scared", "shameful", "failure", "war", "angry",
       "हमला", "डरे", "निंदा", "حملہ", "خوفزدہ", "جنگ"}

def lexicon_sentiment(t):
    toks = TOK.findall(t.lower())
    p = sum(w in POS for w in toks); n = sum(w in NEG for w in toks)
    return 0.0 if p + n == 0 else (p - n) / (p + n)

_hf = None
def transformer_sentiment(texts):
    """Optional: multilingual transformer (en/hi/ur...). pip install transformers torch"""
    global _hf
    from transformers import pipeline
    if _hf is None:
        import torch
        _hf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", top_k=None,
                       device=0 if torch.cuda.is_available() else -1)
    out = []
    for res in _hf(list(texts), truncation=True, max_length=128, batch_size=32):
        d = {r["label"].lower(): r["score"] for r in res}
        out.append(d.get("positive", 0) - d.get("negative", 0))
    return out

def add_sentiment(df, use_transformer=False):
    df["clean"] = df["text"].map(clean)
    df["lang"] = df["text"].map(detect_script)
    df["sentiment"] = transformer_sentiment(df["clean"]) if use_transformer else df["clean"].map(lexicon_sentiment)
    df["label"] = pd.cut(df["sentiment"], [-1.01, -0.2, 0.2, 1.01], labels=["negative", "neutral", "positive"])
    return df

def find_narratives(df, kmin=3, kmax=7):
    vec = TfidfVectorizer(token_pattern=r"[^\s.,!?;:#\"'।،()]{2,}", min_df=3, max_df=0.6, ngram_range=(1, 2))
    X = vec.fit_transform(df["clean"])
    best, best_s = None, -1
    for k in range(kmin, kmax + 1):
        km = KMeans(k, n_init=5, random_state=0).fit(X)
        s = silhouette_score(X, km.labels_, sample_size=min(5000, X.shape[0]), random_state=0)
        if s > best_s: best, best_s = km, s
    df["narrative"] = best.labels_
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for c in range(best.n_clusters):
        top = terms[best.cluster_centers_[c].argsort()[::-1][:6]]
        sub = df[df.narrative == c]
        rows.append(dict(narrative=c, label=", ".join(top[:4]), posts=len(sub), mean_sentiment=round(sub.sentiment.mean(), 2),
                         countries=", ".join(f"{k}:{v}" for k, v in sub.country.value_counts().head(3).items()),
                         example=sub.text.iloc[0][:110]))
    return df, pd.DataFrame(rows).sort_values("posts", ascending=False)

KNOWN_FALSE = ["army base destroyed 500 soldiers killed media hiding truth",
               "ceasefire declared fake news viral message forwarded",
               "city captured by enemy troops viral video"]
SENSATIONAL = re.compile(r"breaking|!!+|share before|forward this|media (is )?hiding|they don.?t want you|deleted", re.I)

def score_misinformation(df, coord_authors=frozenset()):
    vec = TfidfVectorizer().fit(list(df["clean"]) + KNOWN_FALSE)
    sim = linear_kernel(vec.transform(df["clean"]), vec.transform(KNOWN_FALSE)).max(axis=1)
    sens = df["text"].map(lambda t: 1.0 if SENSATIONAL.search(str(t)) else 0.0)
    nosrc = (~df["has_source_link"].fillna(False).astype(bool)).astype(float)
    young = (df["account_age_days"].fillna(999) < 30).astype(float)
    viral = (df["shares"] > df["shares"].quantile(0.95)).astype(float)
    coord = df["author"].isin(coord_authors).astype(float)
    df["misinfo_score"] = (0.35 * np.clip(sim * 2, 0, 1) + 0.2 * sens + 0.1 * nosrc + 0.1 * young + 0.1 * viral + 0.15 * coord).round(2)
    df["misinfo_flag"] = df["misinfo_score"] >= 0.5
    return df

def detect_coordination(df, sim_thr=0.7, win_min=90, min_authors=4, max_span_min=180, block=500):
    """Near-duplicate text from DIFFERENT accounts within win_min minutes. Time-windowed blocks => scales to 100k+ posts."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    d = df.sort_values("timestamp").reset_index(drop=True)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(d["clean"])
    ts = ((d["timestamp"] - d["timestamp"].min()).dt.total_seconds() / 60).values
    au = pd.factorize(d["author"])[0]
    n = len(d); rows, cols = [], []
    for a in range(0, n, block):
        b = min(a + block, n)
        lo = np.searchsorted(ts, ts[a] - win_min, "left"); hi = np.searchsorted(ts, ts[b - 1] + win_min, "right")
        S = linear_kernel(X[a:b], X[lo:hi])
        ii, jj = np.nonzero(S >= sim_thr)
        gi, gj = a + ii, lo + jj
        m = (gi < gj) & (au[gi] != au[gj]) & (np.abs(ts[gi] - ts[gj]) <= win_min)
        rows.append(gi[m]); cols.append(gj[m])
    r = np.concatenate(rows) if rows else np.array([], int); c = np.concatenate(cols) if cols else np.array([], int)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    d["_g"] = lab
    out = []
    for g, sub in d.groupby("_g"):
        if len(sub) < min_authors: continue
        span = (sub.timestamp.max() - sub.timestamp.min()).total_seconds() / 60
        # real campaigns = tight burst; organic repeats of common phrases spread over days
        if sub.author.nunique() >= min_authors and span <= max_span_min:
            age = sub.account_age_days.median()
            out.append(dict(cluster=len(out) + 1, posts=len(sub), distinct_authors=sub.author.nunique(),
                            span_minutes=round(span), median_account_age_days=int(age) if age == age else -1,
                            total_shares=int(sub.shares.sum()), first_seen=sub.timestamp.min(),
                            authors=sorted(sub.author.unique()), sample=sub.text.iloc[0][:120]))
    return pd.DataFrame(out)

def sentiment_shifts(df, drop_thr=0.2):
    daily = df.assign(day=df.timestamp.dt.date).groupby(["day", "country"]).agg(
        posts=("post_id", "count"), sentiment=("sentiment", "mean")).reset_index()
    alerts = []
    for c, sub in daily.groupby("country"):
        sub = sub[sub.posts >= 10].sort_values("day")  # ignore thin/partial days
        for i in range(1, len(sub)):
            base = sub.sentiment.iloc[:i].mean(); cur = sub.sentiment.iloc[i]
            if abs(cur - base) >= drop_thr:
                alerts.append(dict(country=c, day=sub.day.iloc[i], baseline=round(base, 2), current=round(cur, 2),
                                   change=round(cur - base, 2), direction="NEGATIVE shift" if cur < base else "POSITIVE shift"))
    return daily, pd.DataFrame(alerts)

def volume_spikes(df, z_thr=3.5):
    h = df.set_index("timestamp").resample("1h").size()
    med = h.median(); mad = (h - med).abs().median() or 1
    z = 0.6745 * (h - med) / mad
    return h, h[z > z_thr]
