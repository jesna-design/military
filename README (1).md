# Automated OSINT System – Military & Security Social-Media Monitoring

## Problem
Monitor publicly available social media on military/security incidents in India and adversary countries; analyse
sentiment and emerging narratives; detect misinformation and coordinated information campaigns; generate timely reports.

## Architecture
Collectors (Reddit public JSON / X API / CSV)  →  normalise to common schema  →  language ID  →  multilingual sentiment
→  narrative clustering (TF-IDF + KMeans, auto-k by silhouette)  →  coordinated-behaviour detection
→  misinformation scoring  →  trend/shift + volume-anomaly alerts  →  HTML report + CSV.

## Modules
| Module | Purpose |
|---|---|
| osint/collect.py | Pluggable collectors, public data only |
| osint/analyze.py | Language ID, sentiment (lexicon or XLM-R), narratives, coordination, misinformation, shifts |
| osint/report.py | Self-contained HTML brief with charts and alerts |
| osint/synthetic.py | Fake demo data with injected event + bot campaign |

## Techniques
* **Sentiment**: multilingual lexicon (en/hi/ur) by default; `--transformer` uses cardiffnlp XLM-R.
* **Narratives**: TF-IDF (1-2 grams) + KMeans, k chosen by silhouette; clusters labelled by top terms.
* **Coordination**: char n-gram cosine similarity ≥0.7 between *different* accounts within 90 min, union-find
  clustering; flagged when ≥4 distinct accounts in a ≤3 h burst. Reports account age, shares, timing.
* **Misinformation**: weighted score = similarity to known-debunked claims + sensational language + no source link
  + new account + virality + membership in coordinated cluster. Output is a triage score, not a verdict.
* **Shifts**: per-country daily sentiment vs running baseline; robust (MAD) hourly volume-spike detection.

## Run
```
pip install -r requirements.txt
python run.py                      # demo on synthetic data -> output/report.html
python run.py --source csv --csv mydata.csv
python run.py --source reddit      # needs internet
```

## Demo result (synthetic)
Recovered the injected 18-account campaign (1 cluster, 18 posts flagged, no false positives), detected the negative
sentiment shift from 28 Oct and volume spikes around the incident.

## Ethics, limits, next steps
Public data only, aggregate analysis, no individual profiling; respect platform ToS. Limits: lexicon sentiment is
coarse, script-based language ID, small known-claims list. Next: fact-check API (e.g. Google Fact Check) integration,
translation + LaBSE embeddings, image/video hash matching, network graph analysis, streaming + dashboard, geolocation.
