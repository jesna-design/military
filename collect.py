"""Collectors. All return a DataFrame with the common schema:
post_id, author, timestamp, country, platform, text, likes, shares, account_age_days, has_source_link
Only PUBLIC content is collected, respecting each platform's ToS / API terms."""
import pandas as pd

SCHEMA = ["post_id", "author", "timestamp", "country", "platform", "text", "likes", "shares",
          "account_age_days", "has_source_link"]

KEYWORDS = ["army", "military", "border", "ceasefire", "terror attack", "airstrike", "LoC", "drill", "exercise",
            "सेना", "सीमा", "हमला", "فوج", "سرحد", "حملہ"]


def from_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = None
    return df[SCHEMA]


def from_reddit_public(subreddits=("india", "pakistan", "worldnews"), limit=100):
    """Public Reddit JSON endpoint (no key). Requires internet + polite rate limiting."""
    import requests, time
    rows = []
    for sub in subreddits:
        r = requests.get(f"https://www.reddit.com/r/{sub}/new.json?limit={limit}",
                         headers={"User-Agent": "osint-research-prototype/0.1"}, timeout=20)
        for p in r.json()["data"]["children"]:
            d = p["data"]
            text = f'{d.get("title", "")} {d.get("selftext", "")}'.strip()
            if any(k.lower() in text.lower() for k in KEYWORDS):
                rows.append(dict(post_id=d["id"], author=d["author"], timestamp=pd.to_datetime(d["created_utc"], unit="s"),
                                 country={"india": "India", "pakistan": "Pakistan"}.get(sub, "Other"), platform="Reddit",
                                 text=text, likes=d["score"], shares=d["num_comments"], account_age_days=None,
                                 has_source_link=not d.get("is_self", True)))
        time.sleep(2)
    return pd.DataFrame(rows, columns=SCHEMA)


def from_x_api(bearer_token, query, max_results=100):
    """X/Twitter API v2 recent search (needs a developer bearer token). Plug in here."""
    raise NotImplementedError("Add X API v2 call: GET /2/tweets/search/recent with expansions=author_id")
