from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import os
import logging

WHITELIST = [
    "google.com",
    "chrome.com",
    "github.com",
    "microsoft.com",
    "openai.com",
    "edunetfoundation.com"
]

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and tokenizer with error handling
model1 = None
model2 = None
tokenizer = None

def load_resources():
    global model1, model2, tokenizer
    try:
        if os.path.exists("model1_url.h5"):
            model1 = tf.keras.models.load_model("model1_url.h5")
            logger.info("Loaded model1_url.h5")
        else:
            logger.warning("model1_url.h5 not found")

        if os.path.exists("model2_bilstm.h5"):
            model2 = tf.keras.models.load_model("model2_bilstm.h5")
            logger.info("Loaded model2_bilstm.h5")
        else:
            logger.warning("model2_bilstm.h5 not found")

        if os.path.exists("tokenizer.pkl"):
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            logger.info("Loaded tokenizer.pkl")
        else:
            logger.warning("tokenizer.pkl not found")
    except Exception as e:
        logger.exception("Error loading models/tokenizer: %s", e)

# Try loading at startup
load_resources()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    url = data["url"]
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "Invalid URL"}), 400

    url = url.lower().strip()

    # Normalize and prepare for models
    from utils import normalize_url, heuristic_score, tokenize_and_pad
    domain, normalized, _ = normalize_url(url)

    # Whitelist check using exact domain match or subdomain
    for site in WHITELIST:
        if domain == site or domain.endswith("." + site):
            return jsonify({"status": "safe", "confidence": 0.01})

    # Ensure resources are available
    if tokenizer is None or model1 is None or model2 is None:
        logger.error("Models or tokenizer not loaded")
        return jsonify({"error": "Models or tokenizer not loaded"}), 500

    try:
        pad = tokenize_and_pad(tokenizer, normalized, maxlen=100)

        p1 = model1.predict(pad)
        p2 = model2.predict(pad)

        # Safely extract scalar predictions
        p1 = float(p1.ravel()[0])
        p2 = float(p2.ravel()[0])
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": "Prediction failed"}), 500

    model_conf = (p1 + p2) / 2

    heur = heuristic_score(domain, url)

    # Combine model and heuristic: more weight to models but heuristics adjust borderline cases
    final_conf = 0.75 * model_conf + 0.25 * heur
    final_conf = min(max(final_conf, 0.0), 0.99)

    threshold = 0.5
    status = "phishing" if final_conf >= threshold else "safe"

    # Include debug details if requested
    debug = bool(data.get("debug", False))
    resp = {"status": status, "confidence": float(final_conf)}
    if debug:
        resp.update({"model_confidence": float(model_conf), "heuristic": float(heur), "p1": float(p1), "p2": float(p2), "domain": domain})
        logger.info("URL=%s domain=%s p1=%.4f p2=%.4f model=%.4f heur=%.4f final=%.4f", url, domain, p1, p2, model_conf, heur, final_conf)

    return jsonify(resp)


if __name__ == "__main__":
    # Start the Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)

