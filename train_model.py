from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from url_features import extract_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "urls.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"


def convert_label(value):
    """Convert supported dataset labels to 0 or 1."""
    text = str(value).strip().lower()

    if text in {"1", "phishing", "malicious", "bad"}:
        return 1

    if text in {"0", "legitimate", "benign", "good"}:
        return 0

    raise ValueError(
        f"Unsupported label: {value!r}. "
        "Use 0/1 or phishing/legitimate labels."
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your labelled URLs in data/urls.csv."
        )

    data = pd.read_csv(DATA_PATH)

    required = {"url", "label"}

    if not required.issubset(data.columns):
        raise ValueError(
            "CSV must contain columns named 'url' and 'label'."
        )

    data = data.dropna(subset=["url", "label"]).copy()
    data["url"] = data["url"].astype(str).str.strip()
    data = data[data["url"] != ""]
    data["label"] = data["label"].map(convert_label)

    # Remove exact duplicate URLs to reduce data leakage.
    data["url_key"] = data["url"].str.lower()
    data = data.drop_duplicates(subset="url_key")

    if data["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both phishing and legitimate URLs."
        )

    if data["label"].value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two distinct URLs for splitting."
        )

    print("Dataset rows:", len(data))
    print("Label counts:")
    print(data["label"].value_counts())

    X = pd.DataFrame(
        [extract_features(url) for url in data["url"]]
    )
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Restart Streamlit after training to load the model.")


if __name__ == "__main__":
    main()