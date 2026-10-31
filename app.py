import os
from flask import Flask, send_file, abort

app = Flask(__name__)
REPORT = os.path.join(os.path.dirname(__file__), "sample_output", "report.html")


@app.route("/")
def index():
    if not os.path.exists(REPORT):
        abort(404, "report.html not found. Run the pipeline locally and commit it to sample_output/report.html")
    return send_file(REPORT)


@app.route("/health")
def health():
    return "ok"
