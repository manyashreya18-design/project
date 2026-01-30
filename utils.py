from urllib.parse import urlparse
import re
import tensorflow as tf

SUSPICIOUS_TLDS = (".tk", ".ml", ".ga", ".cf", ".gq", ".pw")


def normalize_url(input_url: str):
    """Return (domain, normalized_text, original_url).
    Normalized text concatenates domain + path (without scheme, query, fragment) for tokenizer input.
    """
    url = input_url.strip()
    parsed = urlparse(url if "//" in url else "//" + url)
    domain = parsed.netloc or parsed.path
    # remove credentials
    if "@" in domain:
        domain = domain.split("@")[-1]
    if domain.startswith("www."):
        domain = domain[4:]
    path = parsed.path or ""
    normalized = domain + path
    return domain.lower(), normalized.lower(), url


def is_ip(s: str) -> bool:
    if not s:
        return False
    # IPv4 check
    if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", s):
        return True
    # basic IPv6 heuristic
    if ":" in s and all(c in "0123456789abcdefABCDEF:" for c in s):
        return True
    return False


def heuristic_score(domain: str, url: str) -> float:
    score = 0.0
    if len(url) > 75:
        score += 0.25
    if "@" in url:
        score += 0.45
    if "-" in domain:
        score += 0.12
    if domain.count('.') > 3:
        score += 0.12
    if is_ip(domain):
        score += 0.6
    for t in SUSPICIOUS_TLDS:
        if domain.endswith(t):
            score += 0.2
            break
    return min(1.0, score)


def tokenize_and_pad(tokenizer, text, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    return tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
