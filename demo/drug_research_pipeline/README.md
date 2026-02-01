# Clove Drug Research Pipeline Demo

A production-grade ML pipeline for drug research using the Clove SDK.
Each stage runs as an isolated OS process, supervised by the Clove kernel.

## Features

- **Real datasets** from Therapeutics Data Commons (TDC)
- **RDKit molecular fingerprints** (Morgan, MACCS, descriptors)
- **scikit-learn models** (Random Forest, Gradient Boosting, SVM, etc.)
- **Comprehensive metrics** (AUC-ROC, F1, precision, recall, MCC)
- **Process isolation** with resource limits via Clove
- **Failure recovery** with configurable retry policies
- **Benchmark suite** for parallel multi-experiment execution

## Two Modes of Operation

### 1. Single Pipeline (`main.py`)
Run a single experiment with one dataset, one feature method, one model.

### 2. Benchmark Suite (`benchmark_orchestrator.py`)
Run **50+ parallel experiments** across multiple datasets, features, and models.
Demonstrates Clove's ability to manage many concurrent isolated processes.

---

## Prerequisites

### Required
```bash
pip install PyTDC pandas rdkit scikit-learn pyyaml
```

### Optional
```bash
pip install xgboost  # For XGBoost model support
```

### Clove Kernel
```bash
./build/clove_kernel  # or with sudo for cgroups
```

---

## Quick Start: Single Pipeline

```bash
# Download data (optional - auto-downloads)
cd demo/drug_research_pipeline/data
python download_data.py --dataset hERG

# Run pipeline
python3 demo/drug_research_pipeline/main.py

# Check results
cat demo/drug_research_pipeline/artifacts/run_*/report.md
```

---

## Quick Start: Benchmark Suite

```bash
# Run full benchmark (4 datasets × 3 features × 4 models = 48 experiments)
python3 demo/drug_research_pipeline/benchmark_orchestrator.py

# Quick test (fewer experiments)
python3 demo/drug_research_pipeline/benchmark_orchestrator.py --quick

# Custom configuration
python3 demo/drug_research_pipeline/benchmark_orchestrator.py \
  --datasets hERG AMES BBBP ClinTox \
  --features morgan maccs descriptors \
  --models random_forest gradient_boosting svm logistic_regression \
  --max-parallel 8
```

### Benchmark Output

```
╔═══════════════════════════════════════════════════════════════════╗
║           CLOVE DRUG DISCOVERY BENCHMARK SUITE                    ║
╚═══════════════════════════════════════════════════════════════════╝

Configuration:
  Datasets:    hERG, AMES, BBBP, ClinTox (4)
  Features:    morgan, maccs, descriptors (3)
  Models:      random_forest, gradient_boosting, logistic_regression, svm (4)
  Total experiments: 48
  Max parallel: 8

Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 24/48 (✓24 ⟳8 ✗0 ○16) [45.2s]

  ✓ hERG/morg/rand: AUC=0.823 Acc=0.840 (12.3s)
  ✓ AMES/macc/grad: AUC=0.791 Acc=0.812 (8.7s)
  ...

BENCHMARK COMPLETE
==================
  Total experiments:  48
  Completed:          48
  Failed:             0
  Wall-clock time:    127.4s

Top 5 Results (by AUC-ROC):
  Rank   Dataset    Features     Model              AUC-ROC    Accuracy
  ------ ---------- ------------ ------------------ ---------- ----------
  1      hERG       morgan       gradient_boosting  0.8543     0.8473
  2      AMES       maccs        random_forest      0.8321     0.8156
  ...
```

---

## Configuration

### Single Pipeline Config (`configs/`)

| File | Purpose |
|------|---------|
| `dataset.yaml` | Dataset selection (hERG, AMES, BBBP, etc.) |
| `featurize.yaml` | Feature method (morgan, maccs, descriptors) |
| `train.yaml` | Model type and hyperparameters |
| `eval.yaml` | Evaluation settings |
| `clove_limits.yaml` | Resource limits and timeouts |

### Benchmark Config

Command line arguments or `configs/benchmark.yaml`:
```yaml
datasets: [hERG, AMES, BBBP, ClinTox]
features: [morgan, maccs, descriptors]
models: [random_forest, gradient_boosting, logistic_regression, svm]
max_parallel: 8
```

---

## Available Datasets

| Dataset | Task | Molecules | Description |
|---------|------|-----------|-------------|
| hERG | Binary | ~650 | Cardiac toxicity (ion channel blockade) |
| AMES | Binary | ~7,200 | Mutagenicity |
| BBBP | Binary | ~2,000 | Blood-brain barrier penetration |
| ClinTox | Binary | ~1,400 | Clinical trial toxicity |
| HIV | Binary | ~40,000 | HIV replication inhibition (large) |

---

## Pipeline Stages

1. **load_data** - Load dataset from TDC (auto-downloads if not present)
2. **featurize** - Compute molecular fingerprints using RDKit
3. **train** - Train ML model using scikit-learn
4. **evaluate** - Compute metrics on train/valid/test splits
5. **report** - Generate comprehensive markdown report
6. **archive** - Create artifact manifest

---

## Clove Features Demonstrated

| Feature | Single Pipeline | Benchmark Suite |
|---------|----------------|-----------------|
| Process spawning | 6 sequential | 48+ parallel |
| Resource isolation | Per-stage limits | Per-worker limits |
| IPC messaging | Stage coordination | Result collection |
| Failure recovery | Retry on timeout | Continue on failure |
| Audit logging | Optional | Full tracking |

---

## Output Files

### Single Pipeline (`artifacts/<run_id>/`)
- `dataset.json`, `features.json`, `model.pkl`, `metrics.json`, `report.md`

### Benchmark Suite (`benchmark_results/<run_id>/`)
- `benchmark_report.md` - Leaderboard and analysis
- `all_results.json` - All experiment results
- `exp_*_result.json` - Individual experiment results
- `exp_*_model.pkl` - Trained models

---

## Failure Injection (Testing)

Test Clove's recovery:

```yaml
# In any stage config
fail_once: true
fail_mode: "memory"  # memory | exception | exit | timeout
```

---

## Example Benchmark Report

```markdown
# Drug Discovery Benchmark Results

## Leaderboard (by Test AUC-ROC)

| Rank | Dataset | Features | Model | AUC-ROC | Accuracy | F1 |
|------|---------|----------|-------|---------|----------|-----|
| 1 | hERG | morgan | gradient_boosting | 0.8543 | 0.8473 | 0.8912 |
| 2 | AMES | maccs | random_forest | 0.8321 | 0.8156 | 0.8234 |
| 3 | BBBP | morgan | xgboost | 0.8156 | 0.7923 | 0.8045 |

## Best Model per Dataset

| Dataset | Features | Model | AUC-ROC |
|---------|----------|-------|---------|
| hERG | morgan | gradient_boosting | 0.8543 |
| AMES | maccs | random_forest | 0.8321 |

## Execution Statistics

- Total experiments: 48
- Completed: 48
- Wall-clock time: 127.4s
- Average per experiment: 15.2s
```
