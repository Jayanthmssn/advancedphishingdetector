import ipaddress
import re
from urllib.parse import urlsplit

import pandas as pd


SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account",
    "signin", "password", "banking", "confirm", "wallet",
    "suspended", "recover", "authenticate", "payment"
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd",
    "ow.ly", "shorturl.at", "buff.ly"
}


def extract_features(url):
    url = str(url).strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        is_ip = 1
    except ValueError:
        is_ip = 0

    try:
        port = parsed.port
        unusual_port = int(port not in (None, 80, 443))
    except ValueError:
        unusual_port = 1

    labels = hostname.split(".") if hostname else []
    registered_domain = ".".join(labels[-2:]) if len(labels) >= 2 else hostname

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "slash_count": url.count("/"),
        "digit_count": sum(c.isdigit() for c in url),
        "subdomain_count": max(0, len(labels) - 2),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": is_ip,
        "has_suspicious_word": int(
            any(word in url.lower() for word in SUSPICIOUS_WORDS)
        ),
        "has_shortener": int(
            hostname in SHORTENERS
        ),
        "has_percent_encoding": int("%" in url),
        "has_double_slash_path": int("//" in path),
        "has_at_symbol": int("@" in url),
        "unusual_port": unusual_port,
    }

    return features


def explain_url(url):
    """Return understandable indicators, not proof of phishing."""
    text = str(url).strip().lower()
    normalized = text

    if not re.match(r"^https?://", normalized):
        normalized = "http://" + normalized

    parsed = urlsplit(normalized)
    hostname = (parsed.hostname or "").lower()

    indicators = []

    if len(text) > 100:
        indicators.append("The URL is unusually long.")

    if "@" in text:
        indicators.append(
            "The URL contains '@', which can obscure the destination."
        )

    if "%" in text:
        indicators.append(
            "The URL contains percent-encoded characters."
        )

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        indicators.append(
            "The hostname is an IP address instead of a conventional domain."
        )
    except ValueError:
        pass

    labels = hostname.split(".") if hostname else []

    if len(labels) >= 5:
        indicators.append(
            "The hostname contains many dot-separated labels."
        )

    if "-" in hostname:
        indicators.append(
            "The hostname contains hyphens. This is not automatically malicious."
        )

    if any(word in text for word in SUSPICIOUS_WORDS):
        indicators.append(
            "The URL contains words often used in account-related lures."
        )

    if hostname in SHORTENERS:
        indicators.append(
            "The URL uses a known link-shortening service."
        )

    if parsed.scheme.lower() != "https":
        indicators.append(
            "The URL does not use HTTPS. This alone does not prove phishing."
        )

    try:
        port = parsed.port
        if port not in (None, 80, 443):
            indicators.append(
                f"The URL specifies a nonstandard web port: {port}."
            )
    except ValueError:
        indicators.append("The URL contains an invalid port specification.")

    if not indicators:
        indicators.append(
            "No obvious warning signs were detected by these basic checks."
        )

    return indicators


def get_feature_frame(url):
    """Keep feature order consistent with model training."""
    return pd.DataFrame([extract_features(url)])
    