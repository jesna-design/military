import argparse
from osint import synthetic, collect, analyze, report, real_data

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "csv", "reddit", "real"], default="synthetic")
ap.add_argument("--csv"); ap.add_argument("--folder", default="data/raw"); ap.add_argument("--filter", action="store_true"); ap.add_argument("--transformer", action="store_true")
a = ap.parse_args()

df = {"synthetic": synthetic.generate, "csv": lambda: collect.from_csv(a.csv), "reddit": collect.from_reddit_public,
      "real": lambda: real_data.load_real(a.folder, a.filter)}[a.source]()
df = analyze.add_sentiment(df, a.transformer)
coord = analyze.detect_coordination(df)
cauth = set(a_ for l in coord.get("authors", []) for a_ in l)
df = analyze.score_misinformation(df, cauth)
df, narr = analyze.find_narratives(df)
daily, shifts = analyze.sentiment_shifts(df)
hourly, spikes = analyze.volume_spikes(df)
df.drop(columns=["clean"]).to_csv("output/analysed_posts.csv", index=False)
report.build(df, narr, coord, daily, shifts, hourly, spikes, "output/report.html", synthetic=a.source == "synthetic")
print(narr[["label", "posts", "mean_sentiment"]].to_string(index=False))
print("\nCoordination clusters:", len(coord), "| misinfo flagged:", int(df.misinfo_flag.sum()))
print(shifts.to_string(index=False)); print("Volume spikes:", list(spikes.index))
