"""
04_train_logreg.py
Train Logistic Regression with 5-fold stratified cross-validation
on handcrafted features for both binary and multiclass tasks.

Usage:
    python 04_train_logreg.py
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import FEATURES_DIR, RESULTS_DIR, MODELS_DIR, FOUL_TYPE_NAMES, OFFENCE_NAMES


def load_features(split):
    """Load pre-extracted features and labels."""
    X = np.load(FEATURES_DIR / f"{split}_combined.npy")
    y_bin = np.load(FEATURES_DIR / f"{split}_labels_binary.npy")
    y_multi = np.load(FEATURES_DIR / f"{split}_labels_multi.npy")
    return X, y_bin, y_multi


def filter_binary(X, y_bin):
    """Remove samples with label -1 (ambiguous offence labels)."""
    mask = y_bin >= 0
    return X[mask], y_bin[mask]


def run_task(task, X_train, y_train, X_test, y_test, label_names):
    """Run cross-validated grid search, evaluate on test set."""

    print(f"\n{'='*60}")
    print(f"  LOGISTIC REGRESSION — {task.upper()}")
    print(f"{'='*60}")

    # ── Scale features ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Hyperparameter grid ──
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    grid = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("\n  Running 5-fold CV grid search...")
    grid.fit(X_train_s, y_train)

    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV balanced accuracy: {grid.best_score_:.4f}")

    # ── CV results summary ──
    print(f"\n  CV Results:")
    print(f"  {'C':<10} {'Mean BA':>10} {'Std BA':>10}")
    print(f"  {'-'*32}")
    results = grid.cv_results_
    for i in range(len(results["params"])):
        c = results["params"][i]["C"]
        mean = results["mean_test_score"][i]
        std = results["std_test_score"][i]
        marker = " <-- best" if results["params"][i] == grid.best_params_ else ""
        print(f"  {c:<10} {mean:>10.4f} {std:>10.4f}{marker}")

    # ── Test set evaluation ──
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_s)
    test_ba = balanced_accuracy_score(y_test, y_pred)

    print(f"\n  TEST Balanced Accuracy: {test_ba:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Logistic Regression — {task}\nTest Balanced Acc: {test_ba:.4f}")
    plt.tight_layout()
    fig_path = RESULTS_DIR / f"logreg_{task}_confusion.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {fig_path}")

    # ── Save model + scaler ──
    model_path = MODELS_DIR / f"logreg_{task}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler}, f)
    print(f"  Model saved to {model_path}")

    return test_ba


def main():
    # ── Load data ──
    X_train, y_bin_train, y_multi_train = load_features("train")
    X_test, y_bin_test, y_multi_test = load_features("test")

    print(f"  Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"  Test samples:  {X_test.shape[0]}")

    # ── Task 1: Binary ──
    X_tr_bin, y_tr_bin = filter_binary(X_train, y_bin_train)
    X_te_bin, y_te_bin = filter_binary(X_test, y_bin_test)
    ba_binary = run_task("binary", X_tr_bin, y_tr_bin, X_te_bin, y_te_bin, OFFENCE_NAMES)

    # ── Task 2: Multiclass ──
    ba_multi = run_task("multiclass", X_train, y_multi_train, X_test, y_multi_test, FOUL_TYPE_NAMES)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  LOGISTIC REGRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary (Offence Detection)   Test BA: {ba_binary:.4f}")
    print(f"  Multiclass (Foul Type)       Test BA: {ba_multi:.4f}")


if __name__ == "__main__":
    main()