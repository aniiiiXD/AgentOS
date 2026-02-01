# Drug Research Pipeline Data

This directory contains datasets for the drug research ML pipeline.

## Quick Start

### Prerequisites

```bash
pip install PyTDC pandas rdkit scikit-learn
```

### Download Data

```bash
# Download default dataset (hERG - cardiac toxicity)
python download_data.py

# Download a specific dataset
python download_data.py --dataset AMES

# List all available datasets
python download_data.py --list

# Download all datasets
python download_data.py --all
```

## Available Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| **hERG** | Binary | Cardiac toxicity - hERG ion channel blockade |
| **AMES** | Binary | Mutagenicity - Ames test for genotoxicity |
| **BBBP** | Binary | Blood-brain barrier penetration |
| **CYP2D6_Substrate** | Binary | CYP2D6 enzyme substrate (drug metabolism) |
| **Caco2_Wang** | Regression | Caco-2 cell permeability (intestinal absorption) |
| **Lipophilicity_AstraZeneca** | Regression | Lipophilicity (LogD) |
| **Solubility_AqSolDB** | Regression | Aqueous solubility |
| **ClinTox** | Binary | Clinical trial toxicity |
| **HIV** | Binary | HIV replication inhibition |

## Data Source

All datasets are from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/), a comprehensive resource for AI-ready drug discovery datasets.

## File Structure

After downloading, each dataset will have:

```
data/
├── download_data.py           # Download script
├── README.md                  # This file
├── <dataset>_train.csv        # Training split
├── <dataset>_valid.csv        # Validation split
├── <dataset>_test.csv         # Test split
├── <dataset>_full.csv         # Combined dataset
└── <dataset>_metadata.json    # Dataset metadata
```

## Automatic Download

If data is not present when running the pipeline, the `load_data` stage will automatically download the configured dataset using TDC.

## Citation

If you use TDC datasets, please cite:

```bibtex
@article{Huang2021tdc,
  title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  author={Huang, Kexin and Fu, Tianfan and others},
  journal={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```
