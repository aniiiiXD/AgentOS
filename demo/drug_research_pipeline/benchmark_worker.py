#!/usr/bin/env python3
"""Benchmark worker - runs a single experiment (dataset + features + model).

This worker is spawned by the benchmark orchestrator and runs independently.
It receives configuration via IPC, then loads data, computes features,
trains a model, evaluates, and reports results.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ensure_sdk_on_path, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient


def load_dataset(dataset_name: str, data_dir: Path) -> Dict[str, Any]:
    """Load a TDC dataset."""
    metadata_path = data_dir / f"{dataset_name.lower()}_metadata.json"

    if not metadata_path.exists():
        # Download using TDC
        from tdc.single_pred import Tox, ADME, HTS

        # Map user-friendly names to TDC internal names
        name_mapping = {
            "BBBP": "BBB_Martins",  # Blood-Brain Barrier Penetration
        }
        tdc_name = name_mapping.get(dataset_name, dataset_name)

        tox_datasets = {"hERG", "AMES", "ClinTox"}
        adme_datasets = {"BBB_Martins", "BBBP", "CYP2D6_Substrate", "Caco2_Wang", "Lipophilicity_AstraZeneca"}
        hts_datasets = {"HIV"}

        if dataset_name in tox_datasets:
            data = Tox(name=tdc_name)
        elif dataset_name in adme_datasets:
            data = ADME(name=tdc_name)
        elif dataset_name in hts_datasets:
            data = HTS(name=tdc_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        split = data.get_split()

        # Save locally
        import pandas as pd
        data_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in split.items():
            df.to_csv(data_dir / f"{dataset_name.lower()}_{split_name}.csv", index=False)

        combined = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
        combined.to_csv(data_dir / f"{dataset_name.lower()}_full.csv", index=False)

        metadata = {
            "dataset_name": dataset_name,
            "total_samples": len(combined),
            "splits": {k: len(v) for k, v in split.items()},
            "files": {
                "train": str(data_dir / f"{dataset_name.lower()}_train.csv"),
                "valid": str(data_dir / f"{dataset_name.lower()}_valid.csv"),
                "test": str(data_dir / f"{dataset_name.lower()}_test.csv"),
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Load from files
    import pandas as pd

    with open(metadata_path) as f:
        metadata = json.load(f)

    result = {"train": [], "valid": [], "test": []}
    for split_name in ["train", "valid", "test"]:
        file_path = Path(metadata["files"][split_name])
        if file_path.exists():
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                result[split_name].append({
                    "smiles": row["Drug"],
                    "label": int(row["Y"]),
                })

    return result


def compute_fingerprint(smiles: str, method: str, radius: int = 2, n_bits: int = 1024) -> Optional[List[int]]:
    """Compute molecular fingerprint."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if method == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return list(fp)
        elif method == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            return list(fp)
        elif method == "descriptors":
            return [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.NumAromaticRings(mol),
                Lipinski.HeavyAtomCount(mol),
                Lipinski.FractionCSP3(mol),
            ]
        else:
            return None
    except Exception:
        return None


def featurize_split(data: List[Dict], method: str, radius: int = 2, n_bits: int = 1024) -> Tuple[List, List]:
    """Featurize a data split."""
    X, y = [], []
    for item in data:
        fp = compute_fingerprint(item["smiles"], method, radius, n_bits)
        if fp is not None:
            X.append(fp)
            y.append(item["label"])
    return X, y


def train_model(X_train: List, y_train: List, model_name: str, config: Dict) -> Any:
    """Train a model."""
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 3),
            learning_rate=config.get("learning_rate", 0.1),
            random_state=42,
        )
    elif model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "svm":
        from sklearn.svm import SVC
        model = SVC(
            C=config.get("C", 1.0),
            kernel=config.get("kernel", "rbf"),
            probability=True,
            random_state=42,
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: List, y: List) -> Dict[str, float]:
    """Evaluate a model."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, balanced_accuracy_score, matthews_corrcoef
    )

    y_pred = model.predict(X)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y, y_pred), 4),
    }

    # AUC-ROC
    if hasattr(model, "predict_proba") and len(set(y)) > 1:
        try:
            y_prob = model.predict_proba(X)[:, 1]
            metrics["auc_roc"] = round(roc_auc_score(y, y_prob), 4)
        except Exception:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    return metrics


def run_experiment(
    experiment_id: str,
    dataset_name: str,
    feature_method: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    model_config: Dict = None,
) -> Dict[str, Any]:
    """Run a single experiment."""
    start_time = time.time()
    model_config = model_config or {}

    result = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "features": feature_method,
        "model": model_name,
        "status": "running",
        "start_time": start_time,
    }

    try:
        # Load data
        data = load_dataset(dataset_name, data_dir)

        # Featurize
        X_train, y_train = featurize_split(data["train"], feature_method)
        X_valid, y_valid = featurize_split(data["valid"], feature_method)
        X_test, y_test = featurize_split(data["test"], feature_method)

        if not X_train or not X_test:
            raise ValueError("No valid molecules after featurization")

        result["train_samples"] = len(X_train)
        result["valid_samples"] = len(X_valid)
        result["test_samples"] = len(X_test)
        result["n_features"] = len(X_train[0])

        # Train
        model = train_model(X_train, y_train, model_name, model_config)

        # Evaluate
        result["train_metrics"] = evaluate_model(model, X_train, y_train)
        if X_valid:
            result["valid_metrics"] = evaluate_model(model, X_valid, y_valid)
        result["test_metrics"] = evaluate_model(model, X_test, y_test)

        # Save model
        model_path = output_dir / f"{experiment_id}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        result["model_path"] = str(model_path)

        result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["end_time"] = time.time()
    result["duration_s"] = round(result["end_time"] - start_time, 2)

    # Save result
    result_path = output_dir / f"{experiment_id}_result.json"
    write_json(result_path, result)

    return result


def wait_for_config(client: CloveClient, timeout: float = 60) -> Optional[Dict]:
    """Wait for experiment configuration from orchestrator."""
    start = time.time()
    while time.time() - start < timeout:
        result = client.recv_messages()
        for msg in result.get("messages", []):
            payload = msg.get("message", {})
            if payload.get("type") == "run_experiment":
                return payload
        time.sleep(0.1)
    return None


def main() -> int:
    """Main entry point for worker."""
    import os

    # Generate unique worker name based on PID
    pid = os.getpid()
    worker_name = f"worker_{pid}"

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    # Connect to Clove
    client = CloveClient(socket_path="/tmp/clove.sock")
    if not client.connect():
        print(f"[{worker_name}] ERROR: Failed to connect to Clove kernel")
        return 1

    # Register with PID-based name initially
    client.register_name(worker_name)

    result = None
    experiment_id = worker_name

    try:
        # Wait for configuration from orchestrator
        config = wait_for_config(client, timeout=30)
        if not config:
            print(f"[{worker_name}] ERROR: No configuration received")
            return 1

        experiment_id = config.get("experiment_id", worker_name)
        dataset_name = config.get("dataset")
        feature_method = config.get("features")
        model_name = config.get("model")
        output_dir = Path(config.get("output_dir", base_dir / "benchmark_results"))

        if not all([dataset_name, feature_method, model_name]):
            print(f"[{experiment_id}] ERROR: Incomplete configuration")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{experiment_id}] Starting: {dataset_name}/{feature_method}/{model_name}")

        # Run the experiment
        result = run_experiment(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            feature_method=feature_method,
            model_name=model_name,
            data_dir=data_dir,
            output_dir=output_dir,
        )

        # Report back to orchestrator
        client.send_message({
            "type": "experiment_complete",
            "experiment_id": experiment_id,
            "result": result,
        }, to_name="benchmark_orchestrator")

        if result["status"] == "completed":
            auc = result.get('test_metrics', {}).get('auc_roc', 'N/A')
            print(f"[{experiment_id}] Completed: AUC={auc}")
        else:
            print(f"[{experiment_id}] Failed: {result.get('error', 'Unknown error')}")

    finally:
        client.disconnect()

    return 0 if (result and result.get("status") == "completed") else 1


if __name__ == "__main__":
    sys.exit(main())
