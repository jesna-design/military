"""Loader for the user's real files: tweet sheets (name, location, url, followers, text, datetime, description)
+ an Indian city list (S.No, Name of City, State, Type, Population, Population class)."""
import glob, os, re
import numpy as np, pandas as pd

SEC_KEYWORDS = r"army|military|border|ceasefire|terror|airstrike|\bloc\b|drill|exercise|soldier|jawan|defen[cs]e|missile|attack|" \
               r"war\b|bsf|isi|pulwama|balakot|pahalgam|sindoor|सेना|सीमा|हमला|आतंक|فوج|سرحد|حملہ"
FOREIGN = {"Pakistan": ["pakistan", "karachi", "lahore", "islamabad", "rawalpindi", "peshawar", "quetta", "multan", "faisalabad"],
           "China": ["china", "beijing", "shanghai", "hong kong", "shenzhen", "guangzhou", "tibet", "xinjiang"],
           "Bangladesh": ["bangladesh", "dhaka"], "Nepal": ["nepal", "kathmandu"], "USA": ["usa", "united states", "new york", "washington"],
           "UK": ["united kingdom", "london", "uk"]}


def _read(p):
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p, encoding_errors="replace", on_bad_lines="skip", low_memory=False)


def _ngrams(s, n=3):
    w = re.findall(r"[^\W_]+", s.lower())
    return {" ".join(w[i:i + k]) for k in range(1, n + 1) for i in range(len(w) - k + 1)}


def load_real(folder, filter_security=False, keep_retweets=False):
    files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    tweets, cities = [], None
    for p in files:
        df = _read(p); df.columns = [str(c).strip().lower() for c in df.columns]
        if "text" in df.columns:
            df["source_file"] = os.path.basename(p); tweets.append(df)
        elif any(c.startswith("name of") for c in df.columns):
            cities = df
        print(f"read {os.path.basename(p)}: {len(df)} rows")
    assert tweets, "no tweet file (with a 'text' column) found"
    t = pd.concat(tweets, ignore_index=True)
    n0 = len(t)
    t["timestamp"] = pd.to_datetime(t["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    t = t.dropna(subset=["timestamp", "text"]).copy()
    t["text"] = t["text"].astype(str)
    t = t.drop_duplicates(subset=["name", "text", "timestamp"])
    if not keep_retweets:
        t = t[~t["text"].str.match(r"^\s*RT\s+@", case=False)]
    if filter_security:
        t = t[t["text"].str.contains(SEC_KEYWORDS, case=False, regex=True)]
    print(f"tweets: {n0} -> {len(t)} after cleaning")

    # ---- geo: map free-text location -> country/state using the city list
    gz = {}
    if cities is not None:
        ncol = next(c for c in cities.columns if c.startswith("name of"))
        scol = "state" if "state" in cities.columns else None
        for _, r in cities.iterrows():
            gz[str(r[ncol]).strip().lower()] = str(r[scol]).strip() if scol else ""
        states = {v.lower(): v for v in gz.values() if v}
    else:
        states = {}

    def locate(loc):
        if pd.isna(loc) or not str(loc).strip():
            return "Unknown", ""
        s = str(loc).lower(); grams = _ngrams(s)
        for ctry, keys in FOREIGN.items():
            if any(k in grams for k in keys):
                return ctry, ""
        if "india" in grams or "bharat" in grams:
            st = next((states[g] for g in grams if g in states), "")
            return "India", st
        for g in grams:
            if g in gz:
                return "India", gz[g]
            if g in states:
                return "India", states[g]
        return "Other", ""

    uniq = {l: locate(l) for l in t["location"].unique()}
    t["country"] = t["location"].map(lambda l: uniq[l][0]); t["state"] = t["location"].map(lambda l: uniq[l][1])
    out = pd.DataFrame(dict(
        post_id=np.arange(len(t)), author=t["name"].astype(str).values, timestamp=t["timestamp"].values, country=t["country"].values,
        platform="X", text=t["text"].values, likes=0, shares=0, account_age_days=np.nan,
        has_source_link=t["text"].str.contains("http", case=False).values,
        followers=pd.to_numeric(t["followers"], errors="coerce").fillna(0).values, state=t["state"].values,
        profile_description=t.get("description", pd.Series([""] * len(t))).astype(str).values))
    print(out["country"].value_counts().to_string())
    return out.reset_index(drop=True)
