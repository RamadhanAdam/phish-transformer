# Phish-Transformer
CPU-only character-level transformer that flags phishing URLs in Chrome.

## Quick start
1. pip install -r requirements.txt
2. python train.py              # trains & saves phish_model_ts.pt
3. python app.py                 # starts Flask API on :5000
4. Chrome → Extensions → Developer mode → Load unpacked → select extension/
5. pytest                        # run tests
