"""Train a sentiment classifier on YOUR tweets.
Your files have no labels, so this does weak supervision (teacher -> student):
  1. teacher = pretrained multilingual XLM-R (--transformer) or the lexicon (fallback) labels every tweet
  2. student = TF-IDF (word+char) + Logistic Regression trained on those labels (fast, yours, saved to disk)
  3. evaluation on a small HAND-LABELLED file (--gold) is the only honest accuracy number.
Usage:
  python train.py --folder data/raw --transformer --filter
  (open output/labelling_sample.csv, fill the 'label' column with positive/neutral/negative, save as gold.csv)
  python train.py --folder data/raw --transformer --filter --gold gold.csv
"""
import argparse, joblib
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from osint import real_data, analyze

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="data/raw"); ap.add_argument("--filter", action="store_true")
ap.add_argument("--transformer", action="store_true"); ap.add_argument("--gold"); ap.add_argument("--max_rows", type=int, default=50000)
a = ap.parse_args()

df = real_data.load_real(a.folder, a.filter)
if len(df) > a.max_rows: df = df.sample(a.max_rows, random_state=0)
df = analyze.add_sentiment(df, a.transformer)           # teacher labels
y = df["label"].astype(str)
print("\nTeacher label distribution:\n", y.value_counts(normalize=True).round(3).to_string())

def build():
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[^\s.,!?;:#\"'।،()]{2,}"),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True))
    return make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced", C=3))

Xtr, Xte, ytr, yte = train_test_split(df["clean"], y, test_size=0.2, stratify=y, random_state=0)
model = build().fit(Xtr, ytr)
print("\nStudent vs teacher labels (agreement, NOT true accuracy):\n", classification_report(yte, model.predict(Xte), zero_division=0))
model = build().fit(df["clean"], y)                       # refit on everything
joblib.dump(model, "output/sentiment_model.joblib")

if a.gold:
    g = pd.read_csv(a.gold).dropna(subset=["label"]); g["clean"] = g["text"].map(analyze.clean)
    g = analyze.add_sentiment(g.rename(columns={"label": "gold"}), a.transformer)
    print("\n=== TEACHER vs HAND LABELS ===\n", classification_report(g["gold"], g["label"].astype(str), zero_division=0))
    print("=== STUDENT vs HAND LABELS ===\n", classification_report(g["gold"], model.predict(g["clean"]), zero_division=0))
else:
    n = min(200, len(df))
    df.sample(n, random_state=1)[["post_id", "text", "label"]].assign(label="").to_csv("output/labelling_sample.csv", index=False)
    print("\nWrote output/labelling_sample.csv -> fill 'label' (positive/neutral/negative) by hand, then rerun with --gold")
print("Saved model -> output/sentiment_model.joblib")
