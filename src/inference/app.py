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

@app.route("/", methods = ["GET"])
def home():
    return (
        "<h1>PhishGuard API</h1>"
        "<p>POST <code>/predict</code> with JSON:</p>"
        "<pre>{\"url\": \"https://example.com\"}</pre>"
        "<hr>"
        "<h2>Test quickly</h2>"
        "<b>cURL:</b><br>"
        "<pre>curl -X POST http://127.0.0.1:8000/predict -H \"Content-Type: application/json\" -d '{\"url\":\"https://paypal-secure-login.ru\"}'</pre>"
        "<hr>"
        "<b>Thunder Client (VS Code):</b><br>"
        "Method: POST  URL: <code>http://127.0.0.1:8000/predict</code><br>"
        "Body (JSON): <code>{\"url\": \"https://paypal-secure-login.ru\"}</code>"
    ), 200

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