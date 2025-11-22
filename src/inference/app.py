"""
Creating a REST API for predicting phishing probability of URLs.

Exposes POST /predict endpoint:
- Accepts JSON: {"url": "<URL>"}
- Returns JSON: {"phishing": <probability>}

Features:
- Interactive API docs available at /docs (Swagger UI) and /redoc (ReDoc)
- Automatic input validation
- Returns 400 if "url" is missing or invalid
- Returns 500 for unexpected errors

Usage:
1. Run the server:
    uvicorn app:app --reload
2. Open browser to http://127.0.0.1:8000/docs to test endpoints interactively
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from flask import Flask, request, jsonify
from data.tokenizer import url_to_ids

app = Flask(__name__)

# Loading TorchScript Model and setting it to evaluation
model = torch.jit.load("./models/phish_model_ts.pt")
model.eval()

@app.route("/", methods=["GET"])
def home():
    return """
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>PhishGuard API</title>

<style>
    body {
        font-family: system-ui;
        background: #f6f8fa;
        margin: 2rem;
        line-height: 1.6;
    }
    h1 { color: #d32f2f; }
    pre {
        background: #ebebeb;
        padding: 1rem;
        overflow-x: auto;
        border-radius: 6px;
        white-space: pre-wrap;
    }
    code {
        background: #ebebeb;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .step { margin: 1.2rem 0; }
</style>

<h1>PhishGuard API</h1>

<p>
    This server predicts how <em>phishy</em> a URL is.<br>
    Send a POST request to <code>/predict</code> with JSON:
</p>

<pre>{"url": "https://example.com"}</pre>
<p><i>Tip: Replace <code>https://example.com</code> with any URL you want to check.</i></p>

<h2>❶ Test in CMD / Terminal (5 seconds)</h2>
<div class="step">
    <b>Copy → paste → Enter:</b>
    <pre>
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"url\": \"https://example.com\"}"
    </pre>
    <p><i>Replace the URL with the one you want to test.</i></p>
</div>

<h2>② GUI inside VS Code</h2>
<div class="step">
    <b>Using Thunder Client (free):</b><br>
    1. Install "Thunder Client" from Extensions.<br>
    2. New Request → POST<br>
    3. URL: <code>http://127.0.0.1:8000/predict</code><br>
    4. Body → Raw → JSON<br>
    5. Paste:<br>
    <code>{"url": "https://example.com"}</code><br>
    6. Send.
</div>

<h2>③ Postman (any OS)</h2>
<div class="step">
    1. New Request → POST<br>
    2. URL: <code>http://127.0.0.1:8000/predict</code><br>
    3. Body → Raw → JSON<br>
    4. Paste payload → Send.
</div>

<h2>What the number means</h2>
<ul>
    <li>0.0 – 0.5 → likely <strong>safe</strong></li>
    <li>0.5 – 1.0 → likely <strong>phishing</strong></li>
</ul>

<p>That's it — happy testing!</p>

</html>
""", 200

@app.route("/predict", methods = ["POST"])
def predict():
    """Processing POST requests and returning phishing score"""
    try:
        # Getting URL from request JSON safely
        data = request.get_json(force = True)
        url = data.get("url")

        # Converting URL to token IDs
        ids = torch.tensor([url_to_ids(url)], dtype= torch.long)

        #  Running model inference without gradients
        with torch.no_grad():
            score = float(model(ids)[0])

        # Returning phishing probability as JSON
        return jsonify({"phishing" : score})
    
    except Exception as e:
        return jsonify({"error" : str(e)}), 500

if __name__ == "__main__":
    # Running flask server on port 5000
    # print(url_to_ids("http://secure-login-paypal.com.verify-account-update.co/login"))
    app.run(port=5000, debug = False)