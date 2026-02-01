"""Stage: generate comprehensive research report."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ensure_sdk_on_path, log_line, maybe_fail_once, read_json, wait_for_message, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

STAGE_NAME = "report"


def format_metrics_table(metrics: dict, split_name: str) -> list:
    """Format metrics as markdown table rows.

    Args:
        metrics: Dictionary of metrics
        split_name: Name of the data split

    Returns:
        List of markdown lines
    """
    lines = [
        f"### {split_name.capitalize()} Set Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    # Primary metrics
    primary_metrics = [
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced Accuracy", metrics.get("balanced_accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall (Sensitivity)", metrics.get("recall")),
        ("Specificity", metrics.get("specificity")),
        ("F1 Score", metrics.get("f1")),
        ("MCC", metrics.get("mcc")),
        ("AUC-ROC", metrics.get("auc_roc")),
        ("Average Precision", metrics.get("average_precision")),
    ]

    for name, value in primary_metrics:
        if value is not None:
            lines.append(f"| {name} | {value:.4f} |")

    # Sample counts
    lines.extend([
        "",
        f"**Samples:** {metrics.get('n_samples', 'N/A')} "
        f"(Positive: {metrics.get('n_positive', 'N/A')}, "
        f"Negative: {metrics.get('n_negative', 'N/A')})",
    ])

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [])
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        lines.extend([
            "",
            "**Confusion Matrix:**",
            "",
            "|  | Predicted Negative | Predicted Positive |",
            "|--|-------------------|-------------------|",
            f"| **Actual Negative** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |",
            f"| **Actual Positive** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |",
        ])

    return lines


def generate_report(
    run_id: str,
    dataset_info: dict,
    model_info: dict,
    metrics: dict,
    featurize_info: dict,
) -> str:
    """Generate a comprehensive markdown report.

    Args:
        run_id: Pipeline run identifier
        dataset_info: Dataset metadata
        model_info: Model configuration and info
        metrics: Evaluation metrics for all splits
        featurize_info: Featurization configuration

    Returns:
        Markdown report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Drug Research Pipeline Report",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
        "## 1. Dataset Summary",
        "",
        f"- **Source:** {dataset_info.get('source', 'TDC')}",
        f"- **Dataset:** {dataset_info.get('dataset_name', 'N/A')}",
        f"- **Task Type:** {dataset_info.get('task_type', 'binary classification')}",
        f"- **Total Molecules:** {dataset_info.get('count', 'N/A')}",
        "",
        "### Data Splits",
        "",
        "| Split | Samples |",
        "|-------|---------|",
        f"| Train | {dataset_info.get('train_count', 'N/A')} |",
        f"| Validation | {dataset_info.get('valid_count', 'N/A')} |",
        f"| Test | {dataset_info.get('test_count', 'N/A')} |",
        "",
        "---",
        "",
        "## 2. Featurization",
        "",
        f"- **Method:** {featurize_info.get('method', 'N/A')}",
        f"- **Feature Dimensions:** {featurize_info.get('n_features', 'N/A')}",
    ]

    # Add fingerprint-specific info
    if featurize_info.get("radius"):
        lines.append(f"- **Morgan Radius:** {featurize_info.get('radius')}")
    if featurize_info.get("n_bits"):
        lines.append(f"- **Fingerprint Bits:** {featurize_info.get('n_bits')}")

    lines.extend([
        f"- **Valid Molecules:** {featurize_info.get('valid_count', 'N/A')}",
        f"- **Invalid SMILES:** {featurize_info.get('invalid_count', 0)}",
        "",
        "---",
        "",
        "## 3. Model",
        "",
    ])

    # Model info
    model_type = model_info.get("model_type", "N/A")
    lines.append(f"- **Algorithm:** {model_type}")

    # Model-specific parameters
    if model_type == "RandomForest":
        lines.extend([
            f"- **Number of Trees:** {model_info.get('n_estimators', 'N/A')}",
            f"- **Max Depth:** {model_info.get('max_depth', 'None (unlimited)')}",
        ])
    elif model_type == "GradientBoosting" or model_type == "XGBoost":
        lines.extend([
            f"- **Number of Estimators:** {model_info.get('n_estimators', 'N/A')}",
            f"- **Learning Rate:** {model_info.get('learning_rate', 'N/A')}",
            f"- **Max Depth:** {model_info.get('max_depth', 'N/A')}",
        ])
    elif model_type == "LogisticRegression":
        lines.extend([
            f"- **Regularization (C):** {model_info.get('C', 'N/A')}",
        ])
    elif model_type == "SVM":
        lines.extend([
            f"- **Kernel:** {model_info.get('kernel', 'N/A')}",
            f"- **C:** {model_info.get('C', 'N/A')}",
        ])

    lines.extend([
        f"- **Input Features:** {model_info.get('n_features', 'N/A')}",
        "",
    ])

    # Feature importances for tree-based models
    feat_imp = model_info.get("feature_importances_top10", [])
    if feat_imp:
        lines.extend([
            "### Top 10 Feature Importances",
            "",
            "| Rank | Feature Index | Importance |",
            "|------|---------------|------------|",
        ])
        for rank, (idx, imp) in enumerate(feat_imp[:10], 1):
            lines.append(f"| {rank} | {idx} | {imp:.4f} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Evaluation Results",
        "",
    ])

    # Add metrics for each split
    for split in ["test", "valid", "train"]:
        if split in metrics:
            lines.extend(format_metrics_table(metrics[split], split))
            lines.append("")

    # Summary
    test_metrics = metrics.get("test", metrics.get("valid", {}))
    lines.extend([
        "---",
        "",
        "## 5. Summary",
        "",
    ])

    if test_metrics:
        auc = test_metrics.get("auc_roc")
        acc = test_metrics.get("accuracy")
        f1 = test_metrics.get("f1")

        lines.append("### Key Performance Indicators")
        lines.append("")

        if auc is not None:
            if auc >= 0.9:
                quality = "Excellent"
            elif auc >= 0.8:
                quality = "Good"
            elif auc >= 0.7:
                quality = "Fair"
            else:
                quality = "Poor"
            lines.append(f"- **AUC-ROC:** {auc:.4f} ({quality})")

        if acc is not None:
            lines.append(f"- **Accuracy:** {acc:.4f}")
        if f1 is not None:
            lines.append(f"- **F1 Score:** {f1:.4f}")

    lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- This report was generated by the Clove Drug Research Pipeline",
        "- Data source: Therapeutics Data Commons (TDC)",
        "- Molecular features computed using RDKit",
        f"- Model trained using scikit-learn",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[report] ERROR: Failed to connect to Clove kernel")
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

        # Load all stage results
        metrics_path = Path(input_payload.get("metrics_path", run_dir / "metrics.json"))
        model_info_path = run_dir / "model.json"
        dataset_path = Path(input_payload.get("dataset_path", run_dir / "dataset.json"))
        features_path = run_dir / "features.json"

        # Read stage outputs
        metrics = read_json(metrics_path)
        model_info = read_json(model_info_path)
        dataset = read_json(dataset_path)
        features_data = read_json(features_path)

        # Get load_data stage result for dataset info
        load_data_result = read_json(run_dir / "load_data.json")
        dataset_info = {
            **load_data_result.get("output", {}),
            **load_data_result.get("metadata", {}),
        }

        # Get featurize stage result
        featurize_result = read_json(run_dir / "featurize.json")
        featurize_info = {
            **featurize_result.get("output", {}),
            **featurize_result.get("metadata", {}),
        }

        log_line(log_path, "generating report")

        # Generate report
        report_content = generate_report(
            run_id=run_id,
            dataset_info=dataset_info,
            model_info=model_info,
            metrics=metrics,
            featurize_info=featurize_info,
        )

        report_path = run_dir / "report.md"
        report_path.write_text(report_content, encoding="utf-8")

        log_line(log_path, f"report saved to {report_path}")

        output = {
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
            "model_path": str(input_payload.get("model_path", run_dir / "model.pkl")),
            "dataset_path": str(dataset_path),
        }
        metadata = {
            "format": "markdown",
            "sections": ["dataset", "featurization", "model", "evaluation", "summary"],
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
