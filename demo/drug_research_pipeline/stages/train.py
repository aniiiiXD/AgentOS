"""Stage: train ML model using scikit-learn."""
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

STAGE_NAME = "train"


def prepare_data(features: List[dict], split: str) -> Tuple[List[List[float]], List[int]]:
    """Extract features and labels for a specific split.

    Args:
        features: List of feature dictionaries
        split: Split name ('train', 'valid', 'test')

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels
    """
    X = []
    y = []
    for f in features:
        if f.get("split") == split:
            X.append(f["features"])
            y.append(int(f["label"]))
    return X, y


def train_random_forest(X_train: List, y_train: List, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Tuple of (trained model, model info dict)
    """
    from sklearn.ensemble import RandomForestClassifier

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", None)
    min_samples_split = config.get("min_samples_split", 2)
    random_state = config.get("random_state", 42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    info = {
        "model_type": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
        "feature_importances_top10": sorted(
            enumerate(model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:10],
    }
    return model, info


def train_gradient_boosting(X_train: List, y_train: List, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Train a Gradient Boosting classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Tuple of (trained model, model info dict)
    """
    from sklearn.ensemble import GradientBoostingClassifier

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", 3)
    learning_rate = config.get("learning_rate", 0.1)
    random_state = config.get("random_state", 42)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    info = {
        "model_type": "GradientBoosting",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
    }
    return model, info


def train_logistic_regression(X_train: List, y_train: List, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Train a Logistic Regression classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Tuple of (trained model, model info dict)
    """
    from sklearn.linear_model import LogisticRegression

    C = config.get("C", 1.0)
    max_iter = config.get("max_iter", 1000)
    random_state = config.get("random_state", 42)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    info = {
        "model_type": "LogisticRegression",
        "C": C,
        "max_iter": max_iter,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
    }
    return model, info


def train_svm(X_train: List, y_train: List, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Train a Support Vector Machine classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Tuple of (trained model, model info dict)
    """
    from sklearn.svm import SVC

    C = config.get("C", 1.0)
    kernel = config.get("kernel", "rbf")
    random_state = config.get("random_state", 42)

    model = SVC(
        C=C,
        kernel=kernel,
        probability=True,  # Enable probability estimates for AUC-ROC
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    info = {
        "model_type": "SVM",
        "C": C,
        "kernel": kernel,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
    }
    return model, info


def train_xgboost(X_train: List, y_train: List, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Train an XGBoost classifier (if available).

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Tuple of (trained model, model info dict)
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", 6)
    learning_rate = config.get("learning_rate", 0.1)
    random_state = config.get("random_state", 42)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    info = {
        "model_type": "XGBoost",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_features": model.n_features_in_,
    }
    return model, info


MODEL_TRAINERS = {
    "random_forest": train_random_forest,
    "gradient_boosting": train_gradient_boosting,
    "logistic_regression": train_logistic_regression,
    "svm": train_svm,
    "xgboost": train_xgboost,
}


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[train] ERROR: Failed to connect to Clove kernel")
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

        # Load features
        features_path = Path(input_payload.get("features_path", run_dir / "features.json"))
        dataset_path = Path(input_payload.get("dataset_path", run_dir / "dataset.json"))
        splits_path = Path(input_payload.get("splits_path", run_dir / "splits.json"))

        features_data = read_json(features_path)
        features = features_data.get("features", [])

        # Prepare training and validation data
        X_train, y_train = prepare_data(features, "train")
        X_valid, y_valid = prepare_data(features, "valid")

        log_line(log_path, f"training data: {len(X_train)} samples, validation: {len(X_valid)} samples")

        # Get model type from config
        model_type = config.get("model", "random_forest")
        if model_type not in MODEL_TRAINERS:
            log_line(log_path, f"unknown model type: {model_type}, falling back to random_forest")
            model_type = "random_forest"

        log_line(log_path, f"training model: {model_type}")

        # Train the model
        trainer = MODEL_TRAINERS[model_type]
        model, model_info = trainer(X_train, y_train, config)

        # Evaluate on validation set
        if X_valid and y_valid:
            valid_score = model.score(X_valid, y_valid)
            model_info["validation_accuracy"] = round(valid_score, 4)
            log_line(log_path, f"validation accuracy: {valid_score:.4f}")

        # Save model as pickle
        model_pickle_path = run_dir / "model.pkl"
        with open(model_pickle_path, "wb") as f:
            pickle.dump(model, f)

        # Save model info as JSON (for inspection without loading pickle)
        model_info_path = run_dir / "model.json"
        # Convert numpy types to Python types for JSON serialization
        serializable_info = {}
        for k, v in model_info.items():
            if hasattr(v, "tolist"):
                serializable_info[k] = v.tolist()
            elif isinstance(v, list) and v and hasattr(v[0], "__iter__"):
                serializable_info[k] = [[int(x[0]), float(x[1])] for x in v]
            else:
                serializable_info[k] = v

        write_json(model_info_path, serializable_info)

        log_line(log_path, f"model saved to {model_pickle_path}")

        output = {
            "model_path": str(model_pickle_path),
            "model_info_path": str(model_info_path),
            "features_path": str(features_path),
            "dataset_path": str(dataset_path),
            "splits_path": str(splits_path),
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
        }
        metadata = {
            "model_type": model_type,
            "model_info": serializable_info,
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
