"""Stage: load molecular dataset from TDC (Therapeutics Data Commons)."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ensure_sdk_on_path, log_line, maybe_fail_once, wait_for_message, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient  # noqa: E402

STAGE_NAME = "load_data"

# Default data directory relative to pipeline root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_tdc_dataset(dataset_name: str, data_dir: Path) -> dict:
    """Load a TDC dataset from local files or download if missing.

    Args:
        dataset_name: Name of the dataset (e.g., 'hERG', 'AMES')
        data_dir: Directory containing downloaded data

    Returns:
        Dictionary with train/valid/test splits and metadata
    """
    import json

    metadata_path = data_dir / f"{dataset_name.lower()}_metadata.json"

    # Check if data exists locally
    if not metadata_path.exists():
        print(f"[load_data] Dataset {dataset_name} not found locally, downloading...")
        # Try to download using TDC directly
        try:
            from tdc.single_pred import Tox, ADME, HTS

            # Map dataset to category
            tox_datasets = {"hERG", "AMES", "ClinTox"}
            adme_datasets = {"BBBP", "CYP2D6_Substrate", "Caco2_Wang", "Lipophilicity_AstraZeneca", "Solubility_AqSolDB"}
            hts_datasets = {"HIV"}

            if dataset_name in tox_datasets:
                data = Tox(name=dataset_name)
            elif dataset_name in adme_datasets:
                data = ADME(name=dataset_name)
            elif dataset_name in hts_datasets:
                data = HTS(name=dataset_name)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            split = data.get_split()

            # Save locally for future use
            data_dir.mkdir(parents=True, exist_ok=True)
            for split_name, df in split.items():
                file_path = data_dir / f"{dataset_name.lower()}_{split_name}.csv"
                df.to_csv(file_path, index=False)

            # Create metadata
            import pandas as pd
            combined = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
            combined_path = data_dir / f"{dataset_name.lower()}_full.csv"
            combined.to_csv(combined_path, index=False)

            metadata = {
                "dataset_name": dataset_name,
                "total_samples": len(combined),
                "splits": {k: len(v) for k, v in split.items()},
                "files": {
                    "train": str(data_dir / f"{dataset_name.lower()}_train.csv"),
                    "valid": str(data_dir / f"{dataset_name.lower()}_valid.csv"),
                    "test": str(data_dir / f"{dataset_name.lower()}_test.csv"),
                    "full": str(combined_path),
                },
                "columns": list(combined.columns),
                "label_column": "Y",
                "smiles_column": "Drug",
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except ImportError:
            raise RuntimeError(
                f"Dataset {dataset_name} not found and PyTDC not installed. "
                "Run: pip install PyTDC pandas"
            )

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load CSV files
    import pandas as pd

    result = {
        "metadata": metadata,
        "splits": {},
    }

    for split_name in ["train", "valid", "test"]:
        file_path = Path(metadata["files"][split_name])
        if file_path.exists():
            df = pd.read_csv(file_path)
            result["splits"][split_name] = df.to_dict(orient="records")

    return result


def main() -> int:
    client = CloveClient()
    if not client.connect():
        print("[load_data] ERROR: Failed to connect to Clove kernel")
        return 1

    try:
        client.register_name(STAGE_NAME)
        message = wait_for_message(client, expected_type="run_stage", expected_stage=STAGE_NAME)

        run_id = message.get("run_id", "run_000")
        artifacts_dir = Path(message.get("artifacts_dir", "artifacts"))
        logs_dir = Path(message.get("logs_dir", "logs"))
        config = message.get("config", {})
        reply_to = message.get("reply_to", "orchestrator")

        run_dir = artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / run_id / f"{STAGE_NAME}.log"

        log_line(log_path, "stage start")
        maybe_fail_once(run_dir, STAGE_NAME, config)

        # Get dataset configuration
        dataset_name = config.get("dataset", "hERG")
        data_dir = Path(config.get("data_dir", str(DATA_DIR)))

        log_line(log_path, f"loading dataset: {dataset_name}")

        # Load the dataset
        data = load_tdc_dataset(dataset_name, data_dir)

        # Prepare molecules for downstream stages
        molecules = []
        mol_id = 0

        for split_name, records in data["splits"].items():
            for record in records:
                mol_id += 1
                molecules.append({
                    "id": f"m{mol_id}",
                    "smiles": record.get("Drug", ""),
                    "label": int(record.get("Y", 0)) if data["metadata"].get("task_type", "binary") == "binary" else record.get("Y", 0.0),
                    "split": split_name,
                })

        # Save full dataset
        dataset_path = run_dir / "dataset.json"
        write_json(dataset_path, {
            "molecules": molecules,
            "metadata": data["metadata"],
        })

        # Save split information
        splits_info = {
            "train": [m for m in molecules if m["split"] == "train"],
            "valid": [m for m in molecules if m["split"] == "valid"],
            "test": [m for m in molecules if m["split"] == "test"],
        }
        splits_path = run_dir / "splits.json"
        write_json(splits_path, {
            "train_ids": [m["id"] for m in splits_info["train"]],
            "valid_ids": [m["id"] for m in splits_info["valid"]],
            "test_ids": [m["id"] for m in splits_info["test"]],
            "counts": {k: len(v) for k, v in splits_info.items()},
        })

        log_line(log_path, f"loaded {len(molecules)} molecules")

        output = {
            "dataset_path": str(dataset_path),
            "splits_path": str(splits_path),
            "count": len(molecules),
            "train_count": len(splits_info["train"]),
            "valid_count": len(splits_info["valid"]),
            "test_count": len(splits_info["test"]),
        }
        metadata = {
            "dataset_name": dataset_name,
            "source": "TDC (Therapeutics Data Commons)",
            "label_column": data["metadata"].get("label_column", "Y"),
            "smiles_column": data["metadata"].get("smiles_column", "Drug"),
            "task_type": data["metadata"].get("task_type", "binary"),
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
