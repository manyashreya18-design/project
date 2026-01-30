"""Quick evaluation script for the URL phishing models.

Usage:
    python evaluate.py --dataset all_urls_dataset.csv --samples 20000

Outputs a small JSON report and prints summary metrics.
"""
import argparse
import json
import logging
import pickle
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

import api_server
from utils import normalize_url, heuristic_score, tokenize_and_pad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reservoir_sample_csv(filepath, n_samples=5000, url_col="url", label_col="label"):
    # Read in chunks and perform reservoir sampling more efficiently using itertuples
    import pandas as pd
    rng = np.random.default_rng(42)
    reservoir = []  # will hold tuples (url, label)
    total = 0
    for chunk in pd.read_csv(filepath, usecols=[url_col, label_col], chunksize=10000):
        for row in chunk.itertuples(index=False):
            total += 1
            tup = (getattr(row, url_col), int(getattr(row, label_col)))
            if len(reservoir) < n_samples:
                reservoir.append(tup)
            else:
                j = rng.integers(0, total)
                if j < n_samples:
                    reservoir[j] = tup
    df = pd.DataFrame(reservoir, columns=[url_col, label_col])
    logger.info("Sampled %d rows from %s (total seen: %d)", len(df), filepath, total)
    return df


def predict_row(model1, model2, tokenizer, url_text, domain, normalized_text):
    pad = tokenize_and_pad(tokenizer, normalized_text, maxlen=100)
    p1 = float(model1.predict(pad).ravel()[0])
    p2 = float(model2.predict(pad).ravel()[0])
    model_conf = (p1 + p2) / 2
    heur = heuristic_score(domain, url_text)
    final_conf = 0.75 * model_conf + 0.25 * heur
    final_conf = min(max(final_conf, 0.0), 0.99)
    return final_conf, model_conf, heur, p1, p2


def evaluate(dataset=None, samples=20000, label_col="label", url_col="url"):
    # load models/tokenizer
    model1 = api_server.model1
    model2 = api_server.model2
    tokenizer = api_server.tokenizer

    if model1 is None or model2 is None or tokenizer is None:
        logger.error("Models or tokenizer not loaded - call api_server.load_resources() first or check files")
        return

    # Load test data from pickle
    try:
        test_data = pickle.load(open("test_data.pkl", "rb"))
        X_test = test_data["X_test"]
        y_true = test_data["y_test"].tolist()
        urls_test = test_data["urls"]
        logger.info("Loaded test data with %d samples", len(y_true))
    except FileNotFoundError:
        logger.error("test_data.pkl not found. Please run train_model1.py first to generate test data.")
        return

    y_scores = []
    details = []

    for i, (x, label, url) in enumerate(zip(X_test, y_true, urls_test)):
        url_text = str(url)
        domain, normalized, _ = normalize_url(url_text)
        try:
            # Use pre-tokenized x instead of re-tokenizing
            pad = x.reshape(1, -1)  # Reshape to (1, maxlen)
            p1 = float(model1.predict(pad).ravel()[0])
            p2 = float(model2.predict(pad).ravel()[0])
            model_conf = (p1 + p2) / 2
            heur = heuristic_score(domain, url_text)
            final_conf = 0.75 * model_conf + 0.25 * heur
            final_conf = min(max(final_conf, 0.0), 0.99)
        except Exception as e:
            logger.exception("Prediction failed for url %s: %s", url_text, e)
            continue
        y_scores.append(final_conf)
        details.append({"url": url_text, "label": label, "score": final_conf, "model_conf": model_conf, "heur": heur})

    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = None

    report = {
        "samples": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation completed. %s", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all_urls_dataset.csv")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    evaluate(args.dataset, samples=args.samples)
