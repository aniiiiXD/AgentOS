#!/usr/bin/env python3
"""Download drug research datasets from Therapeutics Data Commons (TDC).

This script downloads real molecular datasets for drug research ML pipelines.
Data is cached locally after first download.

Available datasets:
- hERG: Cardiac toxicity (ion channel blockade)
- AMES: Mutagenicity prediction
- BBBP: Blood-brain barrier penetration
- CYP2D6_Substrate: Drug metabolism
- Caco2_Wang: Cell permeability (intestinal absorption)

Usage:
    python download_data.py                    # Download default (hERG)
    python download_data.py --dataset AMES     # Download specific dataset
    python download_data.py --list             # List available datasets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Available TDC ADMET datasets with descriptions
AVAILABLE_DATASETS = {
    "hERG": {
        "category": "Tox",
        "description": "hERG channel blockade - cardiac toxicity prediction",
        "task": "binary",
    },
    "AMES": {
        "category": "Tox",
        "description": "Ames mutagenicity test - genotoxicity prediction",
        "task": "binary",
    },
    "BBBP": {
        "category": "ADME",
        "description": "Blood-brain barrier penetration",
        "task": "binary",
    },
    "CYP2D6_Substrate": {
        "category": "ADME",
        "description": "CYP2D6 enzyme substrate prediction - drug metabolism",
        "task": "binary",
    },
    "Caco2_Wang": {
        "category": "ADME",
        "description": "Caco-2 cell permeability - intestinal absorption",
        "task": "regression",
    },
    "Lipophilicity_AstraZeneca": {
        "category": "ADME",
        "description": "Lipophilicity (LogD) - membrane permeability indicator",
        "task": "regression",
    },
    "Solubility_AqSolDB": {
        "category": "ADME",
        "description": "Aqueous solubility - drug formulation property",
        "task": "regression",
    },
    "ClinTox": {
        "category": "Tox",
        "description": "Clinical trial toxicity - FDA approval/failure",
        "task": "binary",
    },
    "HIV": {
        "category": "HTS",
        "description": "HIV replication inhibition - antiviral activity",
        "task": "binary",
    },
}

DEFAULT_DATASET = "hERG"


def check_dependencies() -> bool:
    """Check if required packages are installed."""
    missing = []

    try:
        import tdc  # noqa: F401
    except ImportError:
        missing.append("PyTDC")

    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def download_dataset(name: str, output_dir: Path) -> dict:
    """Download a dataset from TDC and save locally.

    Args:
        name: Dataset name (e.g., 'hERG', 'AMES')
        output_dir: Directory to save the data

    Returns:
        Dictionary with dataset info and file paths
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Use --list to see available datasets.")

    dataset_info = AVAILABLE_DATASETS[name]
    category = dataset_info["category"]
    task_type = dataset_info["task"]

    print(f"Downloading {name} dataset from TDC...")
    print(f"  Category: {category}")
    print(f"  Task: {task_type}")
    print(f"  Description: {dataset_info['description']}")

    # Import TDC based on category
    if category == "Tox":
        from tdc.single_pred import Tox
        data = Tox(name=name)
    elif category == "ADME":
        from tdc.single_pred import ADME
        data = ADME(name=name)
    elif category == "HTS":
        from tdc.single_pred import HTS
        data = HTS(name=name)
    else:
        raise ValueError(f"Unknown category: {category}")

    # Get the split
    split = data.get_split()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    saved_files = {}
    total_samples = 0

    for split_name, df in split.items():
        file_path = output_dir / f"{name.lower()}_{split_name}.csv"
        df.to_csv(file_path, index=False)
        saved_files[split_name] = str(file_path)
        total_samples += len(df)
        print(f"  Saved {split_name}: {len(df)} samples -> {file_path.name}")

    # Also save combined dataset
    import pandas as pd
    combined = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    combined_path = output_dir / f"{name.lower()}_full.csv"
    combined.to_csv(combined_path, index=False)
    saved_files["full"] = str(combined_path)

    # Save metadata
    metadata = {
        "dataset_name": name,
        "category": category,
        "task_type": task_type,
        "description": dataset_info["description"],
        "total_samples": total_samples,
        "splits": {
            "train": len(split["train"]),
            "valid": len(split["valid"]),
            "test": len(split["test"]),
        },
        "files": saved_files,
        "columns": list(combined.columns),
        "label_column": "Y",
        "smiles_column": "Drug",
    }

    metadata_path = output_dir / f"{name.lower()}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved -> {metadata_path.name}")

    print(f"\nDataset {name} downloaded successfully!")
    print(f"  Total samples: {total_samples}")
    print(f"  Location: {output_dir}")

    return metadata


def list_datasets() -> None:
    """Print available datasets."""
    print("\nAvailable TDC datasets for drug research:\n")
    print(f"{'Name':<25} {'Task':<12} {'Description'}")
    print("-" * 80)
    for name, info in AVAILABLE_DATASETS.items():
        print(f"{name:<25} {info['task']:<12} {info['description']}")
    print("\nUsage: python download_data.py --dataset <NAME>")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download drug research datasets from TDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", "-d",
        default=DEFAULT_DATASET,
        help=f"Dataset to download (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory (default: current script directory)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return 0

    if not check_dependencies():
        return 1

    try:
        if args.all:
            print("Downloading all available datasets...\n")
            for name in AVAILABLE_DATASETS:
                try:
                    download_dataset(name, args.output)
                    print()
                except Exception as e:
                    print(f"  ERROR downloading {name}: {e}\n")
        else:
            download_dataset(args.dataset, args.output)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
