import base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _img(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return '<img src="data:image/png;base64,%s"/>' % base64.b64encode(b.getvalue()).decode()

def build(df, narratives, coord, daily, shifts, hourly, spikes, path, synthetic=True):
    figs = []
    fig, ax = plt.subplots(figsize=(7, 3))
    for c, s in daily.groupby("country"): ax.plot([str(x) for x in s.day], s.sentiment, marker="o", label=c)
    ax.axhline(0, color="gray", lw=.5); ax.set_title("Daily mean sentiment by country"); ax.legend(); figs.append(_img(fig))
    fig, ax = plt.subplots(figsize=(7, 3)); hourly.plot(ax=ax, lw=1)
    for t in spikes.index: ax.axvline(t, color="r", alpha=.3)
    ax.set_title("Post volume per hour (red = anomaly)"); figs.append(_img(fig))
    fig, ax = plt.subplots(figsize=(5, 3)); df.label.value_counts().reindex(["negative", "neutral", "positive"]).plot.bar(ax=ax, color=["#c0392b", "#7f8c8d", "#27ae60"])
    ax.set_title("Overall sentiment"); figs.append(_img(fig))

    flagged = df[df.misinfo_flag]
    t = lambda d, **k: d.to_html(index=False, border=0, classes="t", **k)
    html = f"""<html><head><meta charset="utf-8"><title>OSINT Report</title><style>
    body{{font-family:Arial;max-width:960px;margin:24px auto;color:#222}} h1,h2{{color:#12355b}}
    .t{{border-collapse:collapse;font-size:12px}} .t td,.t th{{border:1px solid #ddd;padding:4px 8px}} .t th{{background:#eef3f8}}
    .alert{{background:#fff3cd;padding:8px 12px;border-left:4px solid #f0ad4e;margin:6px 0}}</style></head><body>
    <h1>Automated OSINT Brief – Military &amp; Security Discourse</h1>
    <p>Window: {df.timestamp.min():%d %b %Y %H:%M} to {df.timestamp.max():%d %b %Y %H:%M} | Posts analysed: {len(df)} |
    Languages: {', '.join(sorted(df.lang.unique()))} | Countries: {', '.join(sorted(df.country.unique()))}</p>
    {'<p><i>Note: demo run on SYNTHETIC data.</i></p>' if synthetic else ''}
    <h2>1. Key alerts</h2>"""
    for _, r in shifts.iterrows():
        html += f'<div class="alert"><b>{r.direction}</b> – {r.country} on {r.day}: sentiment {r.baseline} → {r.current} ({r.change:+})</div>'
    for ts_ in spikes.index:
        html += f'<div class="alert"><b>VOLUME SPIKE</b> – {ts_:%d %b %H:%M}, {spikes[ts_]} posts/hour</div>'
    for _, r in coord.iterrows():
        html += f'<div class="alert"><b>POSSIBLE COORDINATED CAMPAIGN</b> – {r.distinct_authors} accounts posted near-identical text within {r.span_minutes} min (median account age {r.median_account_age_days} days): “{r["sample"]}”</div>'
    html += "<h2>2. Sentiment &amp; volume trends</h2>" + "".join(figs)
    html += "<h2>3. Emerging narratives</h2>" + t(narratives)
    html += "<h2>4. Coordinated activity clusters</h2>" + (t(coord.drop(columns=['authors'])) if len(coord) else "<p>None detected.</p>")
    html += f"<h2>5. Potential misinformation ({len(flagged)} posts flagged)</h2>" + t(
        flagged.sort_values("misinfo_score", ascending=False)[["timestamp", "author", "country", "misinfo_score", "shares", "text"]].head(10))
    html += """<h2>6. Method &amp; limitations</h2><p>Scores are indicators for analyst triage, not verdicts. Language detection is script-based;
    lexicon sentiment can be swapped for XLM-R. Coordination = text similarity + temporal proximity + distinct accounts. Only public,
    aggregate data is analysed; no individual profiling. Findings require human verification.</p></body></html>"""
    open(path, "w", encoding="utf-8").write(html)
