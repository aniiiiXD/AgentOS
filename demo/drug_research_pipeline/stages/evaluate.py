"""Stage: evaluate model with comprehensive metrics."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ensure_sdk_on_path, log_line, maybe_fail_once, read_json, wait_for_message, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

STAGE_NAME = "evaluate"


def prepare_data(features: List[dict], split: str) -> Tuple[List[List[float]], List[int], List[str]]:
    """Extract features, labels, and IDs for a specific split.

    Args:
        features: List of feature dictionaries
        split: Split name ('train', 'valid', 'test')

    Returns:
        Tuple of (X, y, ids) where X is feature matrix, y is labels, ids is molecule IDs
    """
    X = []
    y = []
    ids = []
    for f in features:
        if f.get("split") == split:
            X.append(f["features"])
            y.append(int(f["label"]))
            ids.append(f["id"])
    return X, y, ids


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary of metric names and values
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        matthews_corrcoef,
        balanced_accuracy_score,
    )

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 4)

    # Precision, Recall, F1
    # Handle edge cases where only one class is present
    try:
        metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
        metrics["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    except Exception:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    # Matthews Correlation Coefficient
    try:
        metrics["mcc"] = round(matthews_corrcoef(y_true, y_pred), 4)
    except Exception:
        metrics["mcc"] = 0.0

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

            # Specificity (True Negative Rate)
            if (tn + fp) > 0:
                metrics["specificity"] = round(tn / (tn + fp), 4)
            else:
                metrics["specificity"] = 0.0
    except Exception:
        metrics["confusion_matrix"] = []

    # AUC-ROC and Average Precision (require probabilities)
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["auc_roc"] = None

        try:
            metrics["average_precision"] = round(average_precision_score(y_true, y_prob), 4)
        except Exception:
            metrics["average_precision"] = None
    else:
        metrics["auc_roc"] = None
        metrics["average_precision"] = None

    # Class distribution
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = sum(y_true)
    metrics["n_negative"] = len(y_true) - sum(y_true)
    metrics["positive_rate"] = round(sum(y_true) / len(y_true), 4) if y_true else 0.0

    return metrics


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[evaluate] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(STAGE_NAME)
        message = wait_for_message(client, expected_type="run_stage", expected_stage=STAGE_NAME)

        run_id = message.get("run_id", "run_000")
        artifacts_dir = Path(message.get("artifacts_dir", "artifacts"))
        logs_dir = Path(message.get("logs_dir", "logs"))
        config = message.get("config", {})
        input_payload = message.get("input", {})
        reply_to = message.get("reply_to", "orchestrator")

        run_dir = artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / run_id / f"{STAGE_NAME}.log"

        log_line(log_path, "stage start")
        maybe_fail_once(run_dir, STAGE_NAME, config)

        # Load model and features
        model_path = Path(input_payload.get("model_path", run_dir / "model.pkl"))
        features_path = Path(input_payload.get("features_path", run_dir / "features.json"))
        dataset_path = Path(input_payload.get("dataset_path", run_dir / "dataset.json"))

        # Load the trained model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features_data = read_json(features_path)
        features = features_data.get("features", [])

        log_line(log_path, f"loaded model from {model_path}")

        # Evaluate on each split
        all_metrics = {}
        all_predictions = {}

        for split in ["train", "valid", "test"]:
            X, y_true, ids = prepare_data(features, split)

            if not X:
                log_line(log_path, f"no data for split: {split}")
                continue

            # Get predictions
            y_pred = model.predict(X)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    # Get probability of positive class (class 1)
                    if proba.shape[1] == 2:
                        y_prob = proba[:, 1].tolist()
                except Exception:
                    pass

            # Compute metrics
            metrics = compute_metrics(y_true, y_pred.tolist(), y_prob)
            all_metrics[split] = metrics

            # Store predictions for analysis
            all_predictions[split] = {
                "ids": ids,
                "y_true": y_true,
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob,
            }

            log_line(log_path, f"{split}: accuracy={metrics['accuracy']}, f1={metrics['f1']}, auc={metrics.get('auc_roc', 'N/A')}")

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, all_metrics)

        # Save predictions for detailed analysis
        predictions_path = run_dir / "predictions.json"
        write_json(predictions_path, all_predictions)

        # Determine primary evaluation metric (test set, fallback to valid, then train)
        primary_split = "test" if "test" in all_metrics else ("valid" if "valid" in all_metrics else "train")
        primary_metrics = all_metrics.get(primary_split, {})

        log_line(log_path, f"primary evaluation on {primary_split} set")

        output = {
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
            "model_path": str(model_path),
            "features_path": str(features_path),
            "dataset_path": str(dataset_path),
            "primary_split": primary_split,
        }
        metadata = {
            "metrics": all_metrics,
            "primary_metrics": primary_metrics,
            "evaluated_splits": list(all_metrics.keys()),
        }
        stage_result = {
            "type": "stage_complete",
            "stage": STAGE_NAME,
            "run_id": run_id,
            "status": "ok",
            "output": output,
            "metadata": metadata,
        }

        write_json(run_dir / f"{STAGE_NAME}.json", stage_result)
        client.store(f"pipeline:{run_id}:{STAGE_NAME}", stage_result, scope="global")
        client.send_message(stage_result, to_name=reply_to)
        log_line(log_path, "stage complete")
    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
