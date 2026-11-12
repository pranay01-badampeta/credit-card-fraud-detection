"""Credit card fraud detection on highly imbalanced transaction data.

Trains Logistic Regression, Random Forest and XGBoost, handles class imbalance
with SMOTE (applied to the training split only) and reports precision, recall,
F1-score, ROC-AUC, PR-AUC and the confusion matrix for each model.

Data: the "Credit Card Fraud Detection" dataset (creditcard.csv, 284,807
transactions, 492 frauds) from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Usage:
    python fraud_detection.py --data data/creditcard.csv
    python fraud_detection.py --synthetic          # quick demo without the dataset
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

RANDOM_STATE = 42


def load_data(path: str | None, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        return make_synthetic_data()
    if path is None or not Path(path).exists():
        raise SystemExit(
            "Dataset not found. Download creditcard.csv from Kaggle into data/ "
            "or run with --synthetic for a quick demo."
        )
    return pd.read_csv(path)


def make_synthetic_data(n: int = 20000, fraud_rate: float = 0.0035) -> pd.DataFrame:
    """Small dataset with the same columns as creditcard.csv, for demos and tests."""
    rng = np.random.default_rng(RANDOM_STATE)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    X[y == 1, :6] += rng.normal(2.5, 1.0, size=(y.sum(), 6))  # frauds differ on a few components
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = np.where(y == 1, rng.gamma(2, 120, n), rng.gamma(2, 45, n)).round(2)
    df["Class"] = y
    return df


def prepare(df: pd.DataFrame):
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].values
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("xgboost is not installed: skipping the XGBoost model.")
    return models


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--synthetic", action="store_true", help="use generated demo data")
    parser.add_argument("--no-smote", action="store_true", help="train on the original class balance")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    df = load_data(args.data, args.synthetic)
    print(f"{len(df):,} transactions, {int(df['Class'].sum()):,} frauds "
          f"({df['Class'].mean():.3%})")

    X_train, X_test, y_train, y_test = prepare(df)

    if not args.no_smote:
        if SMOTE is None:
            print("imbalanced-learn is not installed: training without SMOTE.")
        else:
            X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(y_train):,} training rows, {y_train.mean():.1%} fraud")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    models = build_models()
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        r = evaluate(name, model, X_test, y_test)
        results.append(r)
        ConfusionMatrixDisplay(np.array(r["confusion_matrix"]), display_labels=["Legit", "Fraud"]).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)

    table = pd.DataFrame(results).drop(columns=["confusion_matrix"]).set_index("model")
    print("\n" + table.round(4).to_string())
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics and plots to {out}/")


if __name__ == "__main__":
    main()
