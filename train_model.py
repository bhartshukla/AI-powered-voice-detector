"""
╔══════════════════════════════════════════════════════════════════╗
║           BhashaKavach – AI Voice Deepfake Detection             ║
║                      train_model.py  v2                          ║
║                                                                  ║
║  Improvements over v1:                                           ║
║   • Shared feature_extractor.py (no train/inference mismatch)   ║
║   • Fast YIN pitch (5-10× faster than pyin)                     ║
║   • 5-Fold Stratified Cross Validation                           ║
║   • RandomizedSearchCV hyper-parameter tuning                    ║
║   • Optional XGBoost with auto model selection                   ║
║   • ROC-AUC in metrics                                           ║
║   • feature_importance.csv + training_report.png                 ║
║   • Structured logging → training.log                            ║
║   • Memory-efficient one-by-one file processing                  ║
╚══════════════════════════════════════════════════════════════════╝

Usage
-----
    python train_model.py

Outputs
-------
    deepfake_model.pkl
    training_metrics.json
    feature_importance.csv
    training_report.png
    training.log
"""

import os
import sys
import json
import pickle
import logging
import datetime
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler, label_binarize
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from feature_extractor import (
    load_and_preprocess,
    extract_features,
    get_feature_names,
    is_valid_audio,
    EXPECTED_DIMS,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
DATASET_DIR      = "dataset"
MODEL_PATH       = "deepfake_model.pkl"
METRICS_PATH     = "training_metrics.json"
IMPORTANCE_PATH  = "feature_importance.csv"
REPORT_PATH      = "training_report.png"
LOG_PATH         = "training.log"
SUPPORTED_EXT    = {".wav", ".mp3"}
LABELS           = {"real": 0, "fake": 1}

# Cross-validation
CV_FOLDS         = 5
CV_SCORING       = ["accuracy", "precision", "recall", "f1"]

# RandomizedSearch space
PARAM_DIST = {
    "clf__n_estimators"     : [200, 300, 400, 500],
    "clf__max_depth"        : [12, 15, 18, 22, None],
    "clf__min_samples_split": [2, 4, 6, 8],
    "clf__min_samples_leaf" : [1, 2, 3, 4],
    "clf__max_features"     : ["sqrt", "log2", 0.3, 0.5],
}
SEARCH_ITERS     = 20   # increase for exhaustive tuning
SEARCH_CV        = 3    # inner CV folds for hyper-param search


# ──────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    log = logging.getLogger("BhashaKavach.Train")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler (INFO only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ──────────────────────────────────────────────────────────────────
# Dataset scanner
# ──────────────────────────────────────────────────────────────────

def scan_dataset(root: str, log: logging.Logger) -> list[tuple[str, int]]:
    """Recursively find all supported audio files under root/real and root/fake."""
    samples = []
    for label_name, label_int in LABELS.items():
        folder = os.path.join(root, label_name)
        if not os.path.isdir(folder):
            log.warning("Folder not found: %s", folder)
            continue
        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT:
                    samples.append((os.path.join(dirpath, fn), label_int))
    log.info("Scan complete – %d candidate files found.", len(samples))
    return samples


# ──────────────────────────────────────────────────────────────────
# Feature extraction loop  (memory-efficient, one file at a time)
# ──────────────────────────────────────────────────────────────────

def build_feature_matrix(
    samples: list[tuple[str, int]],
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Process files one-by-one (generator pattern) to avoid loading
    all audio into RAM simultaneously.

    Returns
    -------
    X          : (n_valid, 151) float32
    y          : (n_valid,)     int32
    skipped    : list of (path, reason)
    """
    X_rows  : list[np.ndarray] = []
    y_rows  : list[int]        = []
    skipped : list[str]        = []

    log.info("Starting feature extraction for %d files …", len(samples))

    for filepath, label in tqdm(
        samples, desc="  Extracting features", unit="file", ncols=76, colour="cyan"
    ):
        try:
            y_audio, sr = load_and_preprocess(filepath)
            if not is_valid_audio(y_audio):
                raise ValueError("Audio failed validity check after preprocessing.")
            feat = extract_features(y_audio, sr)
            X_rows.append(feat)
            y_rows.append(label)
            log.debug("OK  %s  (%d dims)", os.path.basename(filepath), len(feat))

        except Exception as exc:
            reason = str(exc)
            skipped.append(f"{os.path.basename(filepath)}: {reason}")
            log.warning("SKIP  %s  → %s", os.path.basename(filepath), reason)
            tqdm.write(f"  [SKIP] {os.path.basename(filepath)}: {reason}")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows,  dtype=np.int32)
    log.info("Feature matrix: %s  |  Skipped: %d", X.shape, len(skipped))
    return X, y, skipped


# ──────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────

def _build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = 18,
            random_state  = 42,
            n_jobs        = -1,
            class_weight  = "balanced",
        )),
    ])


def _try_xgboost_pipeline():
    """Return an XGBoost pipeline or None if XGBoost is not installed."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators      = 300,
                max_depth         = 8,
                learning_rate     = 0.05,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                use_label_encoder = False,
                eval_metric       = "logloss",
                random_state      = 42,
                n_jobs            = -1,
            )),
        ])
    except ImportError:
        return None


def _run_cv(pipeline, X_train, y_train, log) -> dict:
    """5-Fold Stratified Cross Validation."""
    log.info("Running %d-Fold Stratified Cross Validation …", CV_FOLDS)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv             = skf,
        scoring        = CV_SCORING,
        return_train_score = False,
        n_jobs         = -1,
    )
    summary = {}
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │   {CV_FOLDS}-Fold Cross Validation Results      │")
    print(f"  ├──────────────────────────────────────┤")
    for metric in CV_SCORING:
        scores = cv_results[f"test_{metric}"]
        mean   = float(scores.mean())
        std    = float(scores.std())
        summary[f"cv_{metric}_mean"] = round(mean, 6)
        summary[f"cv_{metric}_std"]  = round(std,  6)
        log.info("CV %-10s  mean=%.4f  std=%.4f", metric, mean, std)
        print(f"  │  {metric:<12}: {mean*100:6.2f}% ± {std*100:.2f}%          │")
    print(f"  └──────────────────────────────────────┘\n")
    return summary


def _tune_hyperparams(pipeline, X_train, y_train, log) -> Pipeline:
    """RandomizedSearchCV over RF hyper-parameters."""
    log.info(
        "Running RandomizedSearchCV (%d iterations, %d-fold inner CV) …",
        SEARCH_ITERS, SEARCH_CV,
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions = PARAM_DIST,
        n_iter              = SEARCH_ITERS,
        cv                  = StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=42),
        scoring             = "f1",
        refit               = True,
        n_jobs              = -1,
        random_state        = 42,
        verbose             = 0,
    )
    with tqdm(total=1, desc="  Hyper-param search", unit="search",
              ncols=76, colour="yellow") as pbar:
        search.fit(X_train, y_train)
        pbar.update(1)

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    log.info("Best parameters: %s", best_params)
    log.info("Best CV F1: %.4f", search.best_score_)
    print(f"\n  Best hyper-parameters found:")
    for k, v in best_params.items():
        print(f"    {k:<22}: {v}")
    print(f"  Best CV F1: {search.best_score_*100:.2f}%\n")
    return search.best_estimator_


# ──────────────────────────────────────────────────────────────────
# Visual training report
# ──────────────────────────────────────────────────────────────────

def _generate_report(
    cm:           np.ndarray,
    importance_df: pd.DataFrame,
    real_count:   int,
    fake_count:   int,
    metrics:      dict,
) -> None:
    """
    Save training_report.png with three panels:
      • Confusion Matrix
      • Top 20 Feature Importances
      • Class Distribution
    """
    fig = plt.figure(figsize=(18, 7), facecolor="#0d1117")
    fig.suptitle(
        "BhashaKavach – Training Report",
        fontsize=18, fontweight="bold",
        color="#58a6ff", y=1.01,
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    accent   = "#58a6ff"
    bg_panel = "#161b22"
    text_col = "#e6edf3"
    danger   = "#f85149"
    success  = "#3fb950"

    # ── Panel 1: Confusion Matrix ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_panel)
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax1)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["REAL", "FAKE"], color=text_col)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["REAL", "FAKE"], color=text_col)
    ax1.set_xlabel("Predicted", color=text_col)
    ax1.set_ylabel("Actual",    color=text_col)
    ax1.set_title("Confusion Matrix", color=accent, fontweight="bold", pad=10)
    for i in range(2):
        for j in range(2):
            ax1.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white", fontsize=16, fontweight="bold",
            )
    ax1.tick_params(colors=text_col)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 2: Top 20 Feature Importances ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_panel)
    top20 = importance_df.head(20).iloc[::-1]  # reverse for horizontal bar
    colors = [accent if "mfcc" in n else
              success if "pitch" in n or "f0" in n or "voiced" in n or "jitter" in n else
              "#d29922" if "contrast" in n else
              "#bc8cff" for n in top20["feature_name"]]
    ax2.barh(top20["feature_name"], top20["importance_score"], color=colors, height=0.7)
    ax2.set_xlabel("Importance Score", color=text_col)
    ax2.set_title("Top 20 Feature Importances", color=accent, fontweight="bold", pad=10)
    ax2.tick_params(axis="y", labelsize=7.5, colors=text_col)
    ax2.tick_params(axis="x", colors=text_col)
    ax2.set_facecolor(bg_panel)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 3: Class Distribution + metrics ────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(bg_panel)
    bars = ax3.bar(
        ["REAL", "FAKE"],
        [real_count, fake_count],
        color=[success, danger],
        width=0.5, edgecolor="#30363d", linewidth=1.2,
    )
    ax3.set_ylabel("Sample Count", color=text_col)
    ax3.set_title("Class Distribution", color=accent, fontweight="bold", pad=10)
    ax3.tick_params(colors=text_col)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")
    for bar, cnt in zip(bars, [real_count, fake_count]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(cnt),
            ha="center", va="bottom",
            color=text_col, fontsize=13, fontweight="bold",
        )
    # Metrics text block
    metrics_text = (
        f"Accuracy : {metrics['accuracy']*100:.2f}%\n"
        f"Precision: {metrics['precision']*100:.2f}%\n"
        f"Recall   : {metrics['recall']*100:.2f}%\n"
        f"F1 Score : {metrics['f1_score']*100:.2f}%\n"
        f"ROC AUC  : {metrics['roc_auc']*100:.2f}%"
    )
    ax3.text(
        0.5, 0.55, metrics_text,
        transform=ax3.transAxes,
        ha="center", va="center",
        fontsize=9.5, color=text_col,
        fontfamily="monospace",
        bbox=dict(facecolor="#21262d", edgecolor="#30363d", boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    fig.savefig(REPORT_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    log = _setup_logging()

    print("\n" + "═" * 66)
    print("  BhashaKavach – AI Voice Deepfake Detection  │  Training v2")
    print("═" * 66)
    log.info("=== BhashaKavach Training Started ===")

    # ── 1. Scan ───────────────────────────────────────────────
    print("\n[1/7]  Scanning dataset …")
    log.info("Scanning dataset at: %s", os.path.abspath(DATASET_DIR))
    samples = scan_dataset(DATASET_DIR, log)
    if not samples:
        log.error("No audio files found. Aborting.")
        raise FileNotFoundError(
            f"No audio files found under '{DATASET_DIR}/real' or '{DATASET_DIR}/fake'."
        )

    # ── 2. Feature extraction ─────────────────────────────────
    print("\n[2/7]  Extracting features  (YIN pitch – fast mode) …")
    X, y, skipped = build_feature_matrix(samples, log)
    if len(X) == 0:
        raise RuntimeError("Feature extraction produced zero valid samples.")

    real_count  = int(np.sum(y == 0))
    fake_count  = int(np.sum(y == 1))
    total_count = int(len(y))

    print(f"\n  ┌───────────────────────────────────┐")
    print(f"  │         Dataset Summary            │")
    print(f"  ├───────────────────────────────────┤")
    print(f"  │  Real Samples  : {real_count:<18}│")
    print(f"  │  Fake Samples  : {fake_count:<18}│")
    print(f"  │  Total Samples : {total_count:<18}│")
    print(f"  │  Skipped Files : {len(skipped):<18}│")
    print(f"  └───────────────────────────────────┘")
    log.info("Dataset – Real: %d  Fake: %d  Total: %d  Skipped: %d",
             real_count, fake_count, total_count, len(skipped))

    # ── 3. Train/test split ───────────────────────────────────
    print("\n[3/7]  Train / Test split (80/20, stratified) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    log.info("Split – Train: %d  Test: %d", len(X_train), len(X_test))

    # ── 4. Cross validation ───────────────────────────────────
    print("\n[4/7]  Running 5-Fold Stratified Cross Validation …")
    rf_base = _build_rf_pipeline()
    cv_summary = _run_cv(rf_base, X_train, y_train, log)

    # ── 5. Hyper-parameter tuning ─────────────────────────────
    print("\n[5/7]  Hyper-parameter optimisation (RandomizedSearchCV) …")
    best_rf = _tune_hyperparams(_build_rf_pipeline(), X_train, y_train, log)

    # ── 6. Optional XGBoost comparison ────────────────────────
    print("\n[6/7]  XGBoost comparison …")
    xgb_pipeline = _try_xgboost_pipeline()
    chosen_pipeline = best_rf
    chosen_name     = "RandomForest"

    if xgb_pipeline is not None:
        log.info("XGBoost available – training and comparing …")
        with tqdm(total=1, desc="  Training XGBoost", unit="model",
                  ncols=76, colour="magenta") as pbar:
            xgb_pipeline.fit(X_train, y_train)
            pbar.update(1)
        xgb_f1 = f1_score(y_test, xgb_pipeline.predict(X_test), zero_division=0)
        rf_f1  = f1_score(y_test, best_rf.predict(X_test),       zero_division=0)
        log.info("RandomForest F1=%.4f  |  XGBoost F1=%.4f", rf_f1, xgb_f1)
        if xgb_f1 > rf_f1:
            chosen_pipeline = xgb_pipeline
            chosen_name     = "XGBoost"
            log.info("XGBoost selected as best model.")
        else:
            log.info("RandomForest selected as best model.")
        print(f"  RandomForest F1: {rf_f1*100:.2f}%  |  XGBoost F1: {xgb_f1*100:.2f}%")
        print(f"  → Selected: {chosen_name}")
    else:
        log.info("XGBoost not installed – using RandomForest only.")
        print("  XGBoost not installed – skipping (pip install xgboost to enable).")
        # Re-fit best_rf on full train set (search already refit, but ensure)
        chosen_pipeline = best_rf

    # ── 7. Evaluate ───────────────────────────────────────────
    print(f"\n[7/7]  Evaluating {chosen_name} …")
    y_pred      = chosen_pipeline.predict(X_test)
    y_prob      = chosen_pipeline.predict_proba(X_test)[:, 1]

    acc   = float(accuracy_score (y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score   (y_test, y_pred, zero_division=0))
    f1    = float(f1_score       (y_test, y_pred, zero_division=0))
    auc   = float(roc_auc_score  (y_test, y_prob))
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │         Final Test Results            │")
    print(f"  ├──────────────────────────────────────┤")
    print(f"  │  Accuracy   : {acc*100:6.2f} %                │")
    print(f"  │  Precision  : {prec*100:6.2f} %                │")
    print(f"  │  Recall     : {rec*100:6.2f} %                │")
    print(f"  │  F1 Score   : {f1*100:6.2f} %                │")
    print(f"  │  ROC AUC    : {auc*100:6.2f} %                │")
    print(f"  └──────────────────────────────────────┘")
    print(f"\n  Classification Report:\n"
          + classification_report(y_test, y_pred,
                                  target_names=["REAL", "FAKE"],
                                  zero_division=0))
    log.info("Test – Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%.4f",
             acc, prec, rec, f1, auc)

    # ── Save model ────────────────────────────────────────────
    print(f"  Saving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(chosen_pipeline, fh)
    log.info("Model saved: %s", MODEL_PATH)

    # ── Feature importance ────────────────────────────────────
    print(f"  Saving feature importance → {IMPORTANCE_PATH}")
    feat_names = get_feature_names()
    # Works for both RF and XGBoost (both expose feature_importances_)
    importances = chosen_pipeline.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature_name"    : feat_names,
        "importance_score": importances,
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    log.info("Feature importance saved: %s", IMPORTANCE_PATH)

    print("\n  Top 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        "model"          : chosen_name,
        "accuracy"       : round(acc,  6),
        "precision"      : round(prec, 6),
        "recall"         : round(rec,  6),
        "f1_score"       : round(f1,   6),
        "roc_auc"        : round(auc,  6),
        "real_samples"   : real_count,
        "fake_samples"   : fake_count,
        "total_samples"  : total_count,
        "train_samples"  : len(X_train),
        "test_samples"   : len(X_test),
        "skipped_files"  : len(skipped),
        "feature_dims"   : EXPECTED_DIMS,
        "training_date"  : datetime.datetime.now().isoformat(),
        **cv_summary,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)
    log.info("Metrics saved: %s", METRICS_PATH)

    # ── Visual report ─────────────────────────────────────────
    print(f"  Generating visual report → {REPORT_PATH}")
    try:
        _generate_report(cm, importance_df, real_count, fake_count, metrics)
        log.info("Training report saved: %s", REPORT_PATH)
    except Exception as exc:
        log.warning("Report generation failed (non-fatal): %s", exc)
        print(f"  [WARN] Report generation failed: {exc}")

    # ── Skipped file summary ──────────────────────────────────
    if skipped:
        print(f"\n  ⚠  {len(skipped)} file(s) skipped:")
        for entry in skipped[:10]:
            print(f"     • {entry}")
        if len(skipped) > 10:
            print(f"     … and {len(skipped)-10} more (see {LOG_PATH})")

    print("\n" + "═" * 66)
    print(f"  Training complete!  Model: {chosen_name}  →  {MODEL_PATH}")
    print("═" * 66 + "\n")
    log.info("=== BhashaKavach Training Finished ===")


if __name__ == "__main__":
    main()
