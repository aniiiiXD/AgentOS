#!/usr/bin/env python3
"""Standalone benchmark - runs experiments WITHOUT Clove using Python multiprocessing.

This provides a baseline for comparison with the Clove-orchestrated benchmark.
No process isolation, no resource limits, no IPC - just raw multiprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


@dataclass
class SystemMetrics:
    """System metrics snapshot."""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    num_processes: int = 0
    num_threads: int = 0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    net_sent_mb: float = 0.0
    net_recv_mb: float = 0.0


def collect_system_metrics() -> SystemMetrics:
    """Collect current system metrics."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()

    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu,
        memory_percent=mem.percent,
        memory_used_mb=mem.used / (1024 * 1024),
        memory_available_mb=mem.available / (1024 * 1024),
        num_processes=len(psutil.pids()),
        num_threads=sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads']),
        disk_read_mb=disk.read_bytes / (1024 * 1024) if disk else 0,
        disk_write_mb=disk.write_bytes / (1024 * 1024) if disk else 0,
        net_sent_mb=net.bytes_sent / (1024 * 1024) if net else 0,
        net_recv_mb=net.bytes_recv / (1024 * 1024) if net else 0,
    )


def metrics_to_dict(m: SystemMetrics) -> Dict:
    return {
        "timestamp": m.timestamp,
        "cpu_percent": m.cpu_percent,
        "memory_percent": m.memory_percent,
        "memory_used_mb": m.memory_used_mb,
        "memory_available_mb": m.memory_available_mb,
        "num_processes": m.num_processes,
        "num_threads": m.num_threads,
        "disk_read_mb": m.disk_read_mb,
        "disk_write_mb": m.disk_write_mb,
        "net_sent_mb": m.net_sent_mb,
        "net_recv_mb": m.net_recv_mb,
    }


def load_dataset(dataset_name: str, data_dir: Path) -> Dict[str, Any]:
    """Load a TDC dataset."""
    metadata_path = data_dir / f"{dataset_name.lower()}_metadata.json"

    if not metadata_path.exists():
        from tdc.single_pred import Tox, ADME, HTS

        name_mapping = {"BBBP": "BBB_Martins"}
        tdc_name = name_mapping.get(dataset_name, dataset_name)

        tox_datasets = {"hERG", "AMES", "ClinTox"}
        adme_datasets = {"BBB_Martins", "BBBP"}
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
        return None
    except Exception:
        return None


def featurize_split(data: List[Dict], method: str) -> Tuple[List, List]:
    """Featurize a data split."""
    X, y = [], []
    for item in data:
        fp = compute_fingerprint(item["smiles"], method)
        if fp is not None:
            X.append(fp)
            y.append(item["label"])
    return X, y


def train_model(X_train: List, y_train: List, model_name: str) -> Any:
    """Train a model."""
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    elif model_name == "svm":
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=42)
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

    if hasattr(model, "predict_proba") and len(set(y)) > 1:
        try:
            y_prob = model.predict_proba(X)[:, 1]
            metrics["auc_roc"] = round(roc_auc_score(y, y_prob), 4)
        except Exception:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    return metrics


def run_single_experiment(args: Tuple) -> Dict[str, Any]:
    """Run a single experiment - called in subprocess."""
    experiment_id, dataset_name, feature_method, model_name, data_dir, output_dir, timeout = args

    start_time = time.time()
    process = psutil.Process()

    # Track per-process metrics
    initial_memory = process.memory_info().rss / (1024 * 1024)
    initial_cpu_times = process.cpu_times()

    result = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "features": feature_method,
        "model": model_name,
        "status": "running",
        "pid": os.getpid(),
        "start_time": start_time,
        "process_metrics": {
            "initial_memory_mb": initial_memory,
        }
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
        model = train_model(X_train, y_train, model_name)

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

    # Final process metrics
    end_time = time.time()
    final_memory = process.memory_info().rss / (1024 * 1024)
    final_cpu_times = process.cpu_times()

    result["end_time"] = end_time
    result["duration_s"] = round(end_time - start_time, 2)
    result["process_metrics"]["peak_memory_mb"] = final_memory
    result["process_metrics"]["memory_delta_mb"] = final_memory - initial_memory
    result["process_metrics"]["user_cpu_time"] = final_cpu_times.user - initial_cpu_times.user
    result["process_metrics"]["system_cpu_time"] = final_cpu_times.system - initial_cpu_times.system

    # Save result
    result_path = output_dir / f"{experiment_id}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_banner():
    banner = f"""
{Colors.YELLOW}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║        STANDALONE BENCHMARK (NO CLOVE)                            ║
║                                                                   ║
║   Pure Python Multiprocessing - No Isolation                      ║
║   - No resource limits (cgroups)                                  ║
║   - No namespace isolation                                        ║
║   - No IPC overhead                                               ║
║   - Direct ProcessPoolExecutor                                    ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)


def print_progress(completed: int, failed: int, running: int, total: int, elapsed: float):
    bar_width = 40
    progress = completed / total if total > 0 else 0
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)

    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.write(
        f"{Colors.BOLD}Progress:{Colors.ENDC} [{Colors.YELLOW}{bar}{Colors.ENDC}] "
        f"{completed}/{total} "
        f"({Colors.GREEN}✓{completed}{Colors.ENDC} "
        f"{Colors.YELLOW}⟳{running}{Colors.ENDC} "
        f"{Colors.RED}✗{failed}{Colors.ENDC}) "
        f"[{elapsed:.1f}s]"
    )
    sys.stdout.flush()


def generate_report(results: List[Dict], metrics_history: List[Dict], output_dir: Path, wall_time: float) -> str:
    """Generate benchmark report."""
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    # Sort by AUC-ROC
    completed.sort(key=lambda x: x.get("test_metrics", {}).get("auc_roc") or 0, reverse=True)

    lines = [
        "# Standalone Benchmark Results (NO CLOVE)",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Mode:** Pure Python Multiprocessing (ProcessPoolExecutor)",
        f"**Total Experiments:** {len(results)}",
        f"**Completed:** {len(completed)}",
        f"**Failed:** {len(failed)}",
        f"**Wall-clock Time:** {wall_time:.1f}s",
        "",
        "---",
        "",
        "## Leaderboard (by Test AUC-ROC)",
        "",
        "| Rank | Dataset | Features | Model | AUC-ROC | Accuracy | F1 | Duration |",
        "|------|---------|----------|-------|---------|----------|-----|----------|",
    ]

    for rank, exp in enumerate(completed[:20], 1):
        metrics = exp.get("test_metrics", {})
        auc = metrics.get("auc_roc", "N/A")
        acc = metrics.get("accuracy", "N/A")
        f1 = metrics.get("f1", "N/A")
        duration = exp.get("duration_s", 0)

        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

        lines.append(f"| {rank} | {exp['dataset']} | {exp['features']} | {exp['model']} | {auc_str} | {acc_str} | {f1_str} | {duration:.1f}s |")

    # Process metrics summary
    if completed:
        avg_memory = sum(r.get("process_metrics", {}).get("peak_memory_mb", 0) for r in completed) / len(completed)
        max_memory = max(r.get("process_metrics", {}).get("peak_memory_mb", 0) for r in completed)
        total_cpu_user = sum(r.get("process_metrics", {}).get("user_cpu_time", 0) for r in completed)
        total_cpu_sys = sum(r.get("process_metrics", {}).get("system_cpu_time", 0) for r in completed)
        total_duration = sum(r.get("duration_s", 0) for r in completed)

        lines.extend([
            "",
            "---",
            "",
            "## Process Metrics Summary",
            "",
            f"- **Average Peak Memory per Process:** {avg_memory:.1f} MB",
            f"- **Maximum Peak Memory:** {max_memory:.1f} MB",
            f"- **Total User CPU Time:** {total_cpu_user:.1f}s",
            f"- **Total System CPU Time:** {total_cpu_sys:.1f}s",
            f"- **Total Compute Time:** {total_duration:.1f}s",
            f"- **Parallelization Efficiency:** {total_duration / wall_time:.2f}x",
        ])

    # System metrics summary
    if metrics_history:
        avg_cpu = sum(m["cpu_percent"] for m in metrics_history) / len(metrics_history)
        max_cpu = max(m["cpu_percent"] for m in metrics_history)
        avg_mem = sum(m["memory_percent"] for m in metrics_history) / len(metrics_history)
        max_mem = max(m["memory_percent"] for m in metrics_history)
        max_procs = max(m["num_processes"] for m in metrics_history)

        lines.extend([
            "",
            "---",
            "",
            "## System Metrics During Benchmark",
            "",
            f"- **Average CPU Usage:** {avg_cpu:.1f}%",
            f"- **Peak CPU Usage:** {max_cpu:.1f}%",
            f"- **Average Memory Usage:** {avg_mem:.1f}%",
            f"- **Peak Memory Usage:** {max_mem:.1f}%",
            f"- **Peak Process Count:** {max_procs}",
            f"- **Metrics Samples:** {len(metrics_history)}",
        ])

    # Failed experiments
    if failed:
        lines.extend([
            "",
            "---",
            "",
            "## Failed Experiments",
            "",
            "| Experiment | Dataset | Features | Model | Error |",
            "|------------|---------|----------|-------|-------|",
        ])
        for exp in failed:
            error = exp.get("error", "Unknown")[:50]
            lines.append(f"| {exp['experiment_id']} | {exp['dataset']} | {exp['features']} | {exp['model']} | {error} |")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by Standalone Benchmark (No Clove)*",
    ])

    report = "\n".join(lines)
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text(report)

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Benchmark (No Clove)")
    parser.add_argument("--datasets", nargs="+", default=["hERG", "AMES", "BBBP", "ClinTox"],
                        help="Datasets to benchmark")
    parser.add_argument("--features", nargs="+", default=["morgan", "maccs", "descriptors"],
                        help="Feature methods to use")
    parser.add_argument("--models", nargs="+", default=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                        help="Models to train")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Maximum parallel processes")
    parser.add_argument("--output-dir", default="benchmark_results_standalone",
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout per experiment in seconds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.quick:
        args.datasets = ["hERG", "AMES"]
        args.features = ["morgan", "maccs"]
        args.models = ["random_forest", "logistic_regression"]

    print_banner()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiments
    experiments = []
    exp_id = 0
    for dataset in args.datasets:
        for features in args.features:
            for model in args.models:
                exp_id += 1
                experiments.append((
                    f"exp_{exp_id:03d}",
                    dataset,
                    features,
                    model,
                    data_dir,
                    output_dir,
                    args.timeout,
                ))

    total = len(experiments)
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Datasets:    {', '.join(args.datasets)} ({len(args.datasets)})")
    print(f"  Features:    {', '.join(args.features)} ({len(args.features)})")
    print(f"  Models:      {', '.join(args.models)} ({len(args.models)})")
    print(f"  {Colors.YELLOW}Total experiments: {total}{Colors.ENDC}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Output: {output_dir}")
    print()
    print(f"{Colors.BOLD}Starting benchmark...{Colors.ENDC}")
    print()

    # Collect initial metrics
    psutil.cpu_percent(interval=None)  # Initialize
    time.sleep(0.1)

    start_time = time.time()
    results = []
    metrics_history = []
    completed = 0
    failed = 0

    try:
        with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
            running = len(futures)

            while futures:
                # Collect system metrics periodically
                metrics_history.append(metrics_to_dict(collect_system_metrics()))

                # Check for completed futures
                done_futures = []
                for future in futures:
                    if future.done():
                        done_futures.append(future)

                for future in done_futures:
                    exp = futures.pop(future)
                    running -= 1
                    try:
                        result = future.result(timeout=1)
                        results.append(result)
                        if result["status"] == "completed":
                            completed += 1
                            auc = result.get("test_metrics", {}).get("auc_roc", "N/A")
                            auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                            print(f"\n{Colors.GREEN}  ✓ {exp[1][:4]}/{exp[2][:4]}/{exp[3][:4]}: AUC={auc_str} ({result['duration_s']:.1f}s){Colors.ENDC}")
                        else:
                            failed += 1
                            print(f"\n{Colors.RED}  ✗ {exp[1][:4]}/{exp[2][:4]}/{exp[3][:4]}: {result.get('error', 'Unknown')[:50]}{Colors.ENDC}")
                    except Exception as e:
                        failed += 1
                        results.append({
                            "experiment_id": exp[0],
                            "dataset": exp[1],
                            "features": exp[2],
                            "model": exp[3],
                            "status": "failed",
                            "error": str(e),
                        })
                        print(f"\n{Colors.RED}  ✗ {exp[1][:4]}/{exp[2][:4]}/{exp[3][:4]}: {str(e)[:50]}{Colors.ENDC}")

                print_progress(completed, failed, len(futures), total, time.time() - start_time)
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")

    end_time = time.time()
    wall_time = end_time - start_time

    # Final summary
    print("\n\n")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}STANDALONE BENCHMARK COMPLETE{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print()
    print(f"  Total experiments:  {total}")
    print(f"  {Colors.GREEN}Completed:{Colors.ENDC}          {completed}")
    print(f"  {Colors.RED}Failed:{Colors.ENDC}             {failed}")
    print(f"  Wall-clock time:    {wall_time:.1f}s")
    print()

    # Generate report
    if completed > 0:
        print(f"{Colors.BOLD}Generating report...{Colors.ENDC}")
        generate_report(results, metrics_history, output_dir, wall_time)

        # Print top 5
        print()
        print(f"{Colors.BOLD}Top 5 Results (by AUC-ROC):{Colors.ENDC}")
        print()

        completed_results = sorted(
            [r for r in results if r["status"] == "completed"],
            key=lambda x: x.get("test_metrics", {}).get("auc_roc") or 0,
            reverse=True
        )

        print(f"  {'Rank':<6} {'Dataset':<10} {'Features':<12} {'Model':<18} {'AUC-ROC':<10} {'Accuracy':<10}")
        print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*18} {'-'*10} {'-'*10}")

        for rank, exp in enumerate(completed_results[:5], 1):
            metrics = exp.get("test_metrics", {})
            auc = metrics.get("auc_roc", "N/A")
            acc = metrics.get("accuracy", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            print(f"  {rank:<6} {exp['dataset']:<10} {exp['features']:<12} {exp['model']:<18} {auc_str:<10} {acc_str:<10}")

        print()
        print(f"{Colors.YELLOW}Full report saved to: {output_dir}/benchmark_report.md{Colors.ENDC}")

    # Save all results and metrics
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "system_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
