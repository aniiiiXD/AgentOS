#!/usr/bin/env python3
"""Benchmark orchestrator - spawns parallel experiments across datasets, features, and models.

This demonstrates Clove's ability to manage many concurrent processes with resource isolation.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ensure_sdk_on_path, load_config, normalize_limits, write_json

ensure_sdk_on_path()
from clove_sdk import CloveClient


# ANSI color codes for terminal output
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
class Experiment:
    id: str
    dataset: str
    features: str
    model: str
    status: str = "pending"
    result: Optional[Dict] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    config_sent: bool = False
    worker_name: Optional[str] = None  # worker_{pid}


def print_banner():
    """Print the benchmark banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║           CLOVE DRUG DISCOVERY BENCHMARK SUITE                    ║
║                                                                   ║
║   Parallel ML Pipeline Demonstration                              ║
║   - Multiple datasets from Therapeutics Data Commons              ║
║   - Multiple featurization methods (Morgan, MACCS, Descriptors)   ║
║   - Multiple ML models (RF, XGBoost, SVM, LogReg, GradBoost)     ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)


def print_progress(experiments: List[Experiment], start_time: float):
    """Print progress update."""
    completed = sum(1 for e in experiments if e.status == "completed")
    failed = sum(1 for e in experiments if e.status == "failed")
    running = sum(1 for e in experiments if e.status == "running")
    pending = sum(1 for e in experiments if e.status == "pending")
    total = len(experiments)

    elapsed = time.time() - start_time
    bar_width = 40
    progress = completed / total if total > 0 else 0
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Clear line and print progress
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.write(
        f"{Colors.BOLD}Progress:{Colors.ENDC} [{Colors.GREEN}{bar}{Colors.ENDC}] "
        f"{completed}/{total} "
        f"({Colors.GREEN}✓{completed}{Colors.ENDC} "
        f"{Colors.YELLOW}⟳{running}{Colors.ENDC} "
        f"{Colors.RED}✗{failed}{Colors.ENDC} "
        f"{Colors.DIM}○{pending}{Colors.ENDC}) "
        f"[{elapsed:.1f}s]"
    )
    sys.stdout.flush()


def print_experiment_status(exp: Experiment, action: str):
    """Print experiment status update."""
    status_colors = {
        "started": Colors.BLUE,
        "completed": Colors.GREEN,
        "failed": Colors.RED,
    }
    color = status_colors.get(action, Colors.ENDC)

    # Abbreviate for cleaner output
    exp_str = f"{exp.dataset[:4]}/{exp.features[:4]}/{exp.model[:4]}"

    if action == "completed" and exp.result:
        metrics = exp.result.get("test_metrics", {})
        auc = metrics.get("auc_roc", "N/A")
        acc = metrics.get("accuracy", "N/A")
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
        acc_str = f"{acc:.3f}" if isinstance(acc, float) else str(acc)
        duration = exp.result.get("duration_s", 0)
        print(f"\n{color}  ✓ {exp_str}: AUC={auc_str} Acc={acc_str} ({duration:.1f}s){Colors.ENDC}")
    elif action == "failed":
        error = exp.result.get("error", "Unknown") if exp.result else "Unknown"
        print(f"\n{color}  ✗ {exp_str}: {error[:50]}{Colors.ENDC}")


def generate_leaderboard(experiments: List[Experiment], output_dir: Path) -> str:
    """Generate leaderboard from completed experiments."""
    completed = [e for e in experiments if e.status == "completed" and e.result]

    # Sort by test AUC-ROC
    def get_auc(exp):
        metrics = exp.result.get("test_metrics", {})
        auc = metrics.get("auc_roc")
        return auc if auc is not None else 0

    completed.sort(key=get_auc, reverse=True)

    lines = [
        "# Drug Discovery Benchmark Results",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Experiments:** {len(experiments)}",
        f"**Completed:** {len(completed)}",
        f"**Failed:** {sum(1 for e in experiments if e.status == 'failed')}",
        "",
        "---",
        "",
        "## Leaderboard (by Test AUC-ROC)",
        "",
        "| Rank | Dataset | Features | Model | AUC-ROC | Accuracy | F1 | Duration |",
        "|------|---------|----------|-------|---------|----------|-----|----------|",
    ]

    for rank, exp in enumerate(completed[:20], 1):  # Top 20
        metrics = exp.result.get("test_metrics", {})
        auc = metrics.get("auc_roc", "N/A")
        acc = metrics.get("accuracy", "N/A")
        f1 = metrics.get("f1", "N/A")
        duration = exp.result.get("duration_s", 0)

        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

        lines.append(
            f"| {rank} | {exp.dataset} | {exp.features} | {exp.model} | "
            f"{auc_str} | {acc_str} | {f1_str} | {duration:.1f}s |"
        )

    # Best per dataset
    lines.extend([
        "",
        "---",
        "",
        "## Best Model per Dataset",
        "",
        "| Dataset | Features | Model | AUC-ROC | Accuracy |",
        "|---------|----------|-------|---------|----------|",
    ])

    datasets = set(e.dataset for e in completed)
    for dataset in sorted(datasets):
        dataset_exps = [e for e in completed if e.dataset == dataset]
        if dataset_exps:
            best = max(dataset_exps, key=get_auc)
            metrics = best.result.get("test_metrics", {})
            auc = metrics.get("auc_roc", "N/A")
            acc = metrics.get("accuracy", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            lines.append(f"| {dataset} | {best.features} | {best.model} | {auc_str} | {acc_str} |")

    # Best per model type
    lines.extend([
        "",
        "---",
        "",
        "## Average Performance by Model Type",
        "",
        "| Model | Avg AUC-ROC | Avg Accuracy | Experiments |",
        "|-------|-------------|--------------|-------------|",
    ])

    models = set(e.model for e in completed)
    model_stats = []
    for model in models:
        model_exps = [e for e in completed if e.model == model]
        aucs = [e.result.get("test_metrics", {}).get("auc_roc", 0) for e in model_exps if e.result.get("test_metrics", {}).get("auc_roc")]
        accs = [e.result.get("test_metrics", {}).get("accuracy", 0) for e in model_exps]
        if aucs:
            avg_auc = sum(aucs) / len(aucs)
            avg_acc = sum(accs) / len(accs)
            model_stats.append((model, avg_auc, avg_acc, len(model_exps)))

    model_stats.sort(key=lambda x: x[1], reverse=True)
    for model, avg_auc, avg_acc, count in model_stats:
        lines.append(f"| {model} | {avg_auc:.4f} | {avg_acc:.4f} | {count} |")

    # Feature method comparison
    lines.extend([
        "",
        "---",
        "",
        "## Average Performance by Feature Method",
        "",
        "| Features | Avg AUC-ROC | Avg Accuracy | Experiments |",
        "|----------|-------------|--------------|-------------|",
    ])

    features = set(e.features for e in completed)
    feature_stats = []
    for feat in features:
        feat_exps = [e for e in completed if e.features == feat]
        aucs = [e.result.get("test_metrics", {}).get("auc_roc", 0) for e in feat_exps if e.result.get("test_metrics", {}).get("auc_roc")]
        accs = [e.result.get("test_metrics", {}).get("accuracy", 0) for e in feat_exps]
        if aucs:
            avg_auc = sum(aucs) / len(aucs)
            avg_acc = sum(accs) / len(accs)
            feature_stats.append((feat, avg_auc, avg_acc, len(feat_exps)))

    feature_stats.sort(key=lambda x: x[1], reverse=True)
    for feat, avg_auc, avg_acc, count in feature_stats:
        lines.append(f"| {feat} | {avg_auc:.4f} | {avg_acc:.4f} | {count} |")

    # Summary statistics
    all_durations = [e.result.get("duration_s", 0) for e in completed if e.result]
    total_duration = sum(all_durations)
    avg_duration = total_duration / len(all_durations) if all_durations else 0

    lines.extend([
        "",
        "---",
        "",
        "## Execution Statistics",
        "",
        f"- **Total experiments:** {len(experiments)}",
        f"- **Completed:** {len(completed)}",
        f"- **Failed:** {sum(1 for e in experiments if e.status == 'failed')}",
        f"- **Total compute time:** {total_duration:.1f}s",
        f"- **Average per experiment:** {avg_duration:.1f}s",
        f"- **Parallelization speedup:** Processes ran concurrently via Clove",
        "",
        "---",
        "",
        "*Generated by Clove Drug Discovery Benchmark Suite*",
    ])

    report = "\n".join(lines)

    # Save report
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text(report)

    return report


def wait_for_worker_registration(client: CloveClient, worker_name: str, timeout: float = 10) -> bool:
    """Wait for a worker to register its name."""
    start = time.time()
    while time.time() - start < timeout:
        # Try to ping the worker
        result = client.send_message({"type": "ping"}, to_name=worker_name)
        if result and result.get("success"):
            return True
        time.sleep(0.1)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Clove Drug Discovery Benchmark Suite")
    parser.add_argument("--datasets", nargs="+", default=["hERG", "AMES", "BBBP", "ClinTox"],
                        help="Datasets to benchmark")
    parser.add_argument("--features", nargs="+", default=["morgan", "maccs", "descriptors"],
                        help="Feature methods to use")
    parser.add_argument("--models", nargs="+", default=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                        help="Models to train")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Maximum parallel processes")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Output directory")
    parser.add_argument("--socket-path", default="/tmp/clove.sock",
                        help="Clove socket path")
    parser.add_argument("--sandboxed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer experiments for testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Quick mode for testing
    if args.quick:
        args.datasets = ["hERG", "AMES"]
        args.features = ["morgan", "maccs"]
        args.models = ["random_forest", "logistic_regression"]

    print_banner()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_script = base_dir / "benchmark_worker.py"

    # Load limits config
    configs_dir = base_dir / "configs"
    limits_config = load_config(configs_dir / "clove_limits.yaml")
    default_limits = normalize_limits(limits_config.get("limits", {}).get("benchmark_worker", {}))
    if not default_limits:
        default_limits = normalize_limits(limits_config.get("limits", {}).get("train", {}))

    # Generate all experiments
    experiments: List[Experiment] = []
    exp_id = 0
    for dataset in args.datasets:
        for features in args.features:
            for model in args.models:
                exp_id += 1
                experiments.append(Experiment(
                    id=f"exp_{exp_id:03d}",
                    dataset=dataset,
                    features=features,
                    model=model,
                ))

    total_experiments = len(experiments)
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Datasets:    {', '.join(args.datasets)} ({len(args.datasets)})")
    print(f"  Features:    {', '.join(args.features)} ({len(args.features)})")
    print(f"  Models:      {', '.join(args.models)} ({len(args.models)})")
    print(f"  {Colors.YELLOW}Total experiments: {total_experiments}{Colors.ENDC}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Output: {output_dir}")
    print()

    # Connect to Clove
    client = CloveClient(socket_path=args.socket_path)
    if not client.connect():
        print(f"{Colors.RED}ERROR: Failed to connect to Clove kernel{Colors.ENDC}")
        print("Make sure the kernel is running: ./build/clove_kernel")
        return 1

    client.register_name("benchmark_orchestrator")

    print(f"{Colors.GREEN}Connected to Clove kernel{Colors.ENDC}")
    print()
    print(f"{Colors.BOLD}Starting benchmark...{Colors.ENDC}")
    print()

    start_time = time.time()
    running_experiments: Dict[str, Experiment] = {}

    try:
        while True:
            # Check for completed experiments
            result = client.recv_messages()
            for msg in result.get("messages", []):
                payload = msg.get("message", {})
                if payload.get("type") == "experiment_complete":
                    exp_id = payload.get("experiment_id")
                    if exp_id in running_experiments:
                        exp = running_experiments.pop(exp_id)
                        exp.result = payload.get("result", {})
                        exp.status = exp.result.get("status", "failed")
                        print_experiment_status(exp, exp.status)

            # Send config to workers that are running but haven't received config yet
            for exp_id, exp in list(running_experiments.items()):
                if not exp.config_sent and exp.worker_name:
                    # Try to send config to worker (using PID-based name)
                    send_result = client.send_message({
                        "type": "run_experiment",
                        "experiment_id": exp.id,
                        "dataset": exp.dataset,
                        "features": exp.features,
                        "model": exp.model,
                        "output_dir": str(output_dir),
                    }, to_name=exp.worker_name)

                    if send_result and send_result.get("success"):
                        exp.config_sent = True

            # Start new experiments if we have capacity
            pending = [e for e in experiments if e.status == "pending"]
            while len(running_experiments) < args.max_parallel and pending:
                exp = pending.pop(0)
                exp.status = "running"
                exp.start_time = time.time()

                # Spawn worker (no args - config sent via IPC)
                spawn_result = client.spawn(
                    name=exp.id,
                    script=str(worker_script),
                    sandboxed=args.sandboxed,
                    limits=default_limits,
                    restart_policy="never",
                )

                if spawn_result and spawn_result.get("status") == "running":
                    exp.pid = spawn_result.get("pid")
                    exp.worker_name = f"worker_{exp.pid}"  # Worker registers with this name
                    running_experiments[exp.id] = exp
                else:
                    exp.status = "failed"
                    exp.result = {"error": f"Failed to spawn worker: {spawn_result}", "status": "failed"}
                    print_experiment_status(exp, "failed")

            # Print progress
            print_progress(experiments, start_time)

            # Check if done
            completed = sum(1 for e in experiments if e.status in ("completed", "failed"))
            if completed == total_experiments:
                break

            # Handle stuck experiments (timeout after 180s)
            current_time = time.time()
            for exp_id, exp in list(running_experiments.items()):
                if exp.start_time and (current_time - exp.start_time) > 180:
                    exp.status = "failed"
                    exp.result = {"error": "Timeout after 180s", "status": "failed"}
                    running_experiments.pop(exp_id)
                    print_experiment_status(exp, "failed")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
    finally:
        client.disconnect()

    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    completed_count = sum(1 for e in experiments if e.status == "completed")
    failed_count = sum(1 for e in experiments if e.status == "failed")

    print("\n")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}BENCHMARK COMPLETE{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print()
    print(f"  Total experiments:  {total_experiments}")
    print(f"  {Colors.GREEN}Completed:{Colors.ENDC}          {completed_count}")
    print(f"  {Colors.RED}Failed:{Colors.ENDC}             {failed_count}")
    print(f"  Wall-clock time:    {total_duration:.1f}s")
    print()

    # Generate leaderboard
    if completed_count > 0:
        print(f"{Colors.BOLD}Generating leaderboard...{Colors.ENDC}")
        report = generate_leaderboard(experiments, output_dir)

        # Print top 5
        print()
        print(f"{Colors.BOLD}Top 5 Results (by AUC-ROC):{Colors.ENDC}")
        print()

        completed_exps = sorted(
            [e for e in experiments if e.status == "completed" and e.result],
            key=lambda e: e.result.get("test_metrics", {}).get("auc_roc", 0) or 0,
            reverse=True
        )

        print(f"  {'Rank':<6} {'Dataset':<10} {'Features':<12} {'Model':<18} {'AUC-ROC':<10} {'Accuracy':<10}")
        print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*18} {'-'*10} {'-'*10}")

        for rank, exp in enumerate(completed_exps[:5], 1):
            metrics = exp.result.get("test_metrics", {})
            auc = metrics.get("auc_roc", "N/A")
            acc = metrics.get("accuracy", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            print(f"  {rank:<6} {exp.dataset:<10} {exp.features:<12} {exp.model:<18} {auc_str:<10} {acc_str:<10}")

        print()
        print(f"{Colors.GREEN}Full report saved to: {output_dir}/benchmark_report.md{Colors.ENDC}")

    # Save all results
    all_results = []
    for exp in experiments:
        all_results.append({
            "experiment_id": exp.id,
            "dataset": exp.dataset,
            "features": exp.features,
            "model": exp.model,
            "status": exp.status,
            "result": exp.result,
        })
    write_json(output_dir / "all_results.json", all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
