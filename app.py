import os
from flask import Flask, send_file, abort

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["sample_output/report.html", "output/report.html", "report.html", "sample_report.html", "sample_output/sample_report.html"]


def find_report():
    for c in CANDIDATES:
        p = os.path.join(BASE, c)
        if os.path.exists(p):
            return p
    return None


@app.route("/")
def index():
    p = find_report()
    if not p:
        files = sorted(os.listdir(BASE))
        abort(404, "No report found. Upload report.html (or sample_report.html) to the repo root. Files seen: " + ", ".join(files))
    return send_file(p)


@app.route("/health")
def health():
    return "ok"
