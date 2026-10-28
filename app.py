import ipaddress
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import streamlit as st

from url_features import explain_url, get_feature_frame


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ PhishGuard AI")
st.subheader("Advanced URL Risk Analysis")
st.caption(
    "A cybersecurity learning project using URL features "
    "and machine learning."
)

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


def validate_url(raw_url):
    """Basic input validation; this does not establish website safety."""
    value = raw_url.strip()

    if not value:
        return None, "Please enter a URL."

    if any(ord(char) < 32 for char in value):
        return None, "The URL contains invalid control characters."

    if len(value) > 2048:
        return None, "The URL is too long."

    if "://" not in value:
        value = "https://" + value

    parsed = urlsplit(value)

    if parsed.scheme.lower() not in ("http", "https"):
        return None, "Only HTTP and HTTPS URLs are supported."

    if not parsed.hostname:
        return None, "The URL must contain a valid hostname."

    try:
        parsed.port
    except ValueError:
        return None, "The URL contains an invalid port."

    try:
        ipaddress.ip_address(parsed.hostname.strip("[]"))
    except ValueError:
        # Hostnames are permitted; this is not a DNS or ownership check.
        pass

    return value, None


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()

with st.sidebar:
    st.header("Project Information")
    st.write("**Name:** PhishGuard AI")
    st.write("**Analysis:** URL structure")
    st.write("**Prediction:** Machine learning, if trained model exists")

    if model is None:
        st.warning(
            "No trained model found. The application will use "
            "rule-based indicators until you train the model."
        )
    else:
        st.success("Machine-learning model loaded.")

    st.info(
        "Do not enter passwords, private tokens, or other sensitive "
        "information into URLs."
    )


st.markdown("### Analyze a URL")

url_input = st.text_input(
    "Website URL",
    placeholder="https://example.com",
)

analyze = st.button("🔍 Analyze URL", type="primary")

if analyze:
    normalized_url, error = validate_url(url_input)

    if error:
        st.error(error)
    else:
        features = get_feature_frame(normalized_url)
        indicators = explain_url(normalized_url)

        # A transparent heuristic score, not a calibrated probability.
        risk_score = 0
        reasons = []

        if len(normalized_url) > 100:
            risk_score += 10
            reasons.append("Unusually long URL (+10)")

        if "@" in normalized_url:
            risk_score += 20
            reasons.append("Contains @ symbol (+20)")

        if features.loc[0, "has_ip_address"] == 1:
            risk_score += 20
            reasons.append("Uses an IP address as hostname (+20)")

        if features.loc[0, "has_suspicious_word"] == 1:
            risk_score += 10
            reasons.append("Contains suspicious keywords (+10)")

        if features.loc[0, "subdomain_count"] >= 3:
            risk_score += 10
            reasons.append("Many subdomain labels (+10)")

        if features.loc[0, "has_shortener"] == 1:
            risk_score += 10
            reasons.append("Uses a URL shortener (+10)")

        if features.loc[0, "https"] == 0:
            risk_score += 10
            reasons.append("Does not use HTTPS (+10)")

        risk_score = min(risk_score, 100)

        ml_probability = None
        prediction = None

        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = list(model.classes_)

                    if 1 in classes:
                        ml_probability = float(
                            probabilities[classes.index(1)]
                        )
                    else:
                        st.warning(
                            "The loaded model has no phishing class "
                            "labelled 1. Check your training labels."
                        )

                prediction = int(model.predict(features)[0])

            except Exception as exc:
                st.error(
                    "The trained model could not analyze this URL. "
                    "Check the training features and labels."
                )
                st.exception(exc)

        st.divider()
        st.markdown("## Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rule-based risk score", f"{risk_score}/100")

        with col2:
            if ml_probability is not None:
                st.metric(
                    "Model phishing score",
                    f"{ml_probability * 100:.1f}%",
                )
            else:
                st.metric("Model phishing score", "Unavailable")

        with col3:
            if prediction == 1:
                st.error("Model: PHISHING")
            elif prediction == 0:
                st.success("Model: LEGITIMATE")
            else:
                st.info("Model: Not available")

        if risk_score >= 40:
            st.warning(
                "Multiple warning signs were detected. Treat the URL "
                "with caution."
            )
        else:
            st.info(
                "Few warning signs were detected. This does not mean "
                "the website is safe."
            )

        st.markdown("### Detected indicators")

        for indicator in indicators:
            st.write(f"• {indicator}")

        st.markdown("### Rule-based scoring details")

        if reasons:
            for reason in reasons:
                st.write(f"• {reason}")
        else:
            st.write("No rule-based risk points were added.")

        result = {
            "time": datetime.now(timezone.utc).isoformat(),
            "url": normalized_url,
            "risk_score": risk_score,
            "model_prediction": (
                "Phishing" if prediction == 1
                else "Legitimate" if prediction == 0
                else "Unavailable"
            ),
            "model_phishing_score": (
                round(ml_probability * 100, 2)
                if ml_probability is not None else None
            ),
        }

        st.session_state.scan_history.insert(0, result)
        st.session_state.scan_history = (
            st.session_state.scan_history[:50]
        )

        report = {
            **result,
            "indicators": indicators,
            "scoring_reasons": reasons,
        }

        st.download_button(
            "📄 Download Analysis Report",
            data=__import__("json").dumps(
                report, indent=2
            ),
            file_name="phishing_analysis_report.json",
            mime="application/json",
        )


st.divider()
st.markdown("## 📜 Scan History")

if st.session_state.scan_history:
    st.dataframe(
        st.session_state.scan_history,
        use_container_width=True,
    )

    if st.button("Clear Scan History"):
        st.session_state.scan_history = []
        st.rerun()
else:
    st.write("No URLs have been analyzed in this session.")

st.divider()
st.caption(
    "Educational tool only. Neither the rule-based score nor the "
    "machine-learning prediction proves that a website is safe or malicious."
)