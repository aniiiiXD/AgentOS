#!/usr/bin/env python3
"""Benchmark Comparison - Runs both Clove and Standalone benchmarks and compares them.

This script:
1. Runs the standalone benchmark (no Clove) with system metrics
2. Runs the Clove-orchestrated benchmark with system metrics
3. Generates a deep comparison report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class MetricsCollector:
    """Collects system metrics in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.metrics: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._initial_disk = None
        self._initial_net = None

    def start(self):
        self.running = True
        self.metrics = []
        psutil.cpu_percent(interval=None)  # Initialize
        self._initial_disk = psutil.disk_io_counters()
        self._initial_net = psutil.net_io_counters()
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.metrics

    def _collect_loop(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(percpu=True)
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                disk = psutil.disk_io_counters()
                net = psutil.net_io_counters()

                # Count benchmark-related processes
                benchmark_procs = 0
                benchmark_memory = 0
                for proc in psutil.process_iter(['name', 'cmdline', 'memory_info']):
                    try:
                        cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                        if 'benchmark' in cmdline.lower() or 'clove' in cmdline.lower():
                            benchmark_procs += 1
                            if proc.info.get('memory_info'):
                                benchmark_memory += proc.info['memory_info'].rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                self.metrics.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu,
                    "cpu_per_core": cpu_per_core,
                    "memory_percent": mem.percent,
                    "memory_used_mb": mem.used / (1024 * 1024),
                    "memory_available_mb": mem.available / (1024 * 1024),
                    "swap_percent": swap.percent,
                    "swap_used_mb": swap.used / (1024 * 1024),
                    "num_processes": len(psutil.pids()),
                    "benchmark_processes": benchmark_procs,
                    "benchmark_memory_mb": benchmark_memory / (1024 * 1024),
                    "disk_read_mb": (disk.read_bytes - self._initial_disk.read_bytes) / (1024 * 1024) if disk else 0,
                    "disk_write_mb": (disk.write_bytes - self._initial_disk.write_bytes) / (1024 * 1024) if disk else 0,
                    "net_sent_mb": (net.bytes_sent - self._initial_net.bytes_sent) / (1024 * 1024) if net else 0,
                    "net_recv_mb": (net.bytes_recv - self._initial_net.bytes_recv) / (1024 * 1024) if net else 0,
                    "load_avg": os.getloadavg(),
                })
            except Exception as e:
                pass

            time.sleep(self.interval)


def print_banner():
    banner = f"""
{Colors.MAGENTA}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                    CLOVE vs STANDALONE DEEP COMPARISON                    ║
║                                                                           ║
║   Comparing:                                                              ║
║   • Clove Orchestrator (process isolation, cgroups, IPC)                  ║
║   • Pure Python Multiprocessing (no isolation, direct execution)          ║
║                                                                           ║
║   Metrics: Wall time, CPU usage, Memory, Process overhead, Throughput     ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)


def run_benchmark(script: str, args: List[str], name: str, output_dir: Path) -> Dict:
    """Run a benchmark script and collect metrics."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Running: {name}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    collector = MetricsCollector(interval=0.5)
    collector.start()

    start_time = time.time()

    try:
        # Run the benchmark
        result = subprocess.run(
            ["python3", script] + args,
            cwd=Path(script).parent,
            capture_output=False,
            text=True,
        )
        exit_code = result.returncode
    except Exception as e:
        print(f"{Colors.RED}Error running benchmark: {e}{Colors.ENDC}")
        exit_code = 1

    end_time = time.time()
    metrics = collector.stop()

    wall_time = end_time - start_time

    return {
        "name": name,
        "wall_time": wall_time,
        "exit_code": exit_code,
        "metrics": metrics,
        "start_time": start_time,
        "end_time": end_time,
    }


def analyze_metrics(metrics: List[Dict]) -> Dict:
    """Analyze collected metrics."""
    if not metrics:
        return {}

    cpu_values = [m["cpu_percent"] for m in metrics]
    mem_values = [m["memory_percent"] for m in metrics]
    mem_used = [m["memory_used_mb"] for m in metrics]
    proc_counts = [m["num_processes"] for m in metrics]
    benchmark_procs = [m.get("benchmark_processes", 0) for m in metrics]
    benchmark_mem = [m.get("benchmark_memory_mb", 0) for m in metrics]
    load_avgs = [m.get("load_avg", (0, 0, 0)) for m in metrics]

    return {
        "samples": len(metrics),
        "duration": metrics[-1]["timestamp"] - metrics[0]["timestamp"] if len(metrics) > 1 else 0,
        "cpu": {
            "avg": sum(cpu_values) / len(cpu_values),
            "max": max(cpu_values),
            "min": min(cpu_values),
            "std": (sum((x - sum(cpu_values)/len(cpu_values))**2 for x in cpu_values) / len(cpu_values)) ** 0.5,
        },
        "memory": {
            "avg_percent": sum(mem_values) / len(mem_values),
            "max_percent": max(mem_values),
            "avg_used_mb": sum(mem_used) / len(mem_used),
            "max_used_mb": max(mem_used),
        },
        "processes": {
            "avg": sum(proc_counts) / len(proc_counts),
            "max": max(proc_counts),
            "min": min(proc_counts),
        },
        "benchmark_processes": {
            "avg": sum(benchmark_procs) / len(benchmark_procs) if benchmark_procs else 0,
            "max": max(benchmark_procs) if benchmark_procs else 0,
        },
        "benchmark_memory_mb": {
            "avg": sum(benchmark_mem) / len(benchmark_mem) if benchmark_mem else 0,
            "max": max(benchmark_mem) if benchmark_mem else 0,
        },
        "load_avg": {
            "1min": sum(l[0] for l in load_avgs) / len(load_avgs),
            "5min": sum(l[1] for l in load_avgs) / len(load_avgs),
            "15min": sum(l[2] for l in load_avgs) / len(load_avgs),
        },
        "disk_io_mb": {
            "read": metrics[-1].get("disk_read_mb", 0),
            "write": metrics[-1].get("disk_write_mb", 0),
        },
        "network_mb": {
            "sent": metrics[-1].get("net_sent_mb", 0),
            "recv": metrics[-1].get("net_recv_mb", 0),
        },
    }


def load_benchmark_results(results_dir: Path) -> Dict:
    """Load results from a benchmark run."""
    all_results_path = results_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            return json.load(f)
    return []


def find_latest_run(base_dir: Path) -> Optional[Path]:
    """Find the most recent run directory."""
    if not base_dir.exists():
        return None
    runs = sorted(base_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def generate_comparison_report(
    clove_run: Dict,
    standalone_run: Dict,
    clove_results: List[Dict],
    standalone_results: List[Dict],
    output_dir: Path,
) -> str:
    """Generate a deep comparison report."""

    clove_analysis = analyze_metrics(clove_run["metrics"])
    standalone_analysis = analyze_metrics(standalone_run["metrics"])

    clove_completed = [r for r in clove_results if r.get("status") == "completed"]
    standalone_completed = [r for r in standalone_results if r.get("status") == "completed"]

    # Calculate throughput
    clove_throughput = len(clove_completed) / clove_run["wall_time"] if clove_run["wall_time"] > 0 else 0
    standalone_throughput = len(standalone_completed) / standalone_run["wall_time"] if standalone_run["wall_time"] > 0 else 0

    # Calculate total compute time
    clove_compute = sum(r.get("duration_s", 0) for r in clove_completed)
    standalone_compute = sum(r.get("duration_s", 0) for r in standalone_completed)

    # Parallelization efficiency
    clove_efficiency = clove_compute / clove_run["wall_time"] if clove_run["wall_time"] > 0 else 0
    standalone_efficiency = standalone_compute / standalone_run["wall_time"] if standalone_run["wall_time"] > 0 else 0

    # Compare AUC-ROC results
    clove_aucs = {f"{r['dataset']}_{r['features']}_{r['model']}": r.get("test_metrics", {}).get("auc_roc")
                  for r in clove_completed}
    standalone_aucs = {f"{r['dataset']}_{r['features']}_{r['model']}": r.get("test_metrics", {}).get("auc_roc")
                       for r in standalone_completed}

    common_exps = set(clove_aucs.keys()) & set(standalone_aucs.keys())
    auc_diffs = []
    for exp in common_exps:
        if clove_aucs[exp] is not None and standalone_aucs[exp] is not None:
            auc_diffs.append(abs(clove_aucs[exp] - standalone_aucs[exp]))

    lines = [
        "# Deep Benchmark Comparison: Clove vs Standalone",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | Clove | Standalone | Difference | Winner |",
        "|--------|-------|------------|------------|--------|",
    ]

    # Wall time comparison
    time_diff = standalone_run["wall_time"] - clove_run["wall_time"]
    time_pct = (time_diff / standalone_run["wall_time"] * 100) if standalone_run["wall_time"] > 0 else 0
    time_winner = "Clove" if clove_run["wall_time"] < standalone_run["wall_time"] else "Standalone"
    lines.append(f"| Wall Clock Time | {clove_run['wall_time']:.1f}s | {standalone_run['wall_time']:.1f}s | {abs(time_diff):.1f}s ({abs(time_pct):.1f}%) | **{time_winner}** |")

    # Throughput comparison
    tp_diff = clove_throughput - standalone_throughput
    tp_winner = "Clove" if clove_throughput > standalone_throughput else "Standalone"
    lines.append(f"| Throughput (exp/s) | {clove_throughput:.3f} | {standalone_throughput:.3f} | {abs(tp_diff):.3f} | **{tp_winner}** |")

    # Parallelization efficiency
    eff_winner = "Clove" if clove_efficiency > standalone_efficiency else "Standalone"
    lines.append(f"| Parallelization Efficiency | {clove_efficiency:.2f}x | {standalone_efficiency:.2f}x | {abs(clove_efficiency - standalone_efficiency):.2f}x | **{eff_winner}** |")

    # Completion rate
    clove_rate = len(clove_completed) / len(clove_results) * 100 if clove_results else 0
    standalone_rate = len(standalone_completed) / len(standalone_results) * 100 if standalone_results else 0
    rate_winner = "Clove" if clove_rate >= standalone_rate else "Standalone"
    lines.append(f"| Completion Rate | {clove_rate:.1f}% | {standalone_rate:.1f}% | {abs(clove_rate - standalone_rate):.1f}% | **{rate_winner}** |")

    # CPU usage
    if clove_analysis and standalone_analysis:
        cpu_winner = "Standalone" if clove_analysis["cpu"]["avg"] > standalone_analysis["cpu"]["avg"] else "Clove"
        lines.append(f"| Avg CPU Usage | {clove_analysis['cpu']['avg']:.1f}% | {standalone_analysis['cpu']['avg']:.1f}% | {abs(clove_analysis['cpu']['avg'] - standalone_analysis['cpu']['avg']):.1f}% | **{cpu_winner}** |")

        mem_winner = "Standalone" if clove_analysis["memory"]["max_used_mb"] > standalone_analysis["memory"]["max_used_mb"] else "Clove"
        lines.append(f"| Peak Memory | {clove_analysis['memory']['max_used_mb']:.0f}MB | {standalone_analysis['memory']['max_used_mb']:.0f}MB | {abs(clove_analysis['memory']['max_used_mb'] - standalone_analysis['memory']['max_used_mb']):.0f}MB | **{mem_winner}** |")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Time Analysis",
        "",
        "### Clove Orchestrator",
        f"- **Wall Clock Time:** {clove_run['wall_time']:.2f}s",
        f"- **Total Compute Time:** {clove_compute:.2f}s",
        f"- **Parallelization Efficiency:** {clove_efficiency:.2f}x",
        f"- **Experiments Completed:** {len(clove_completed)}/{len(clove_results)}",
        f"- **Average Time per Experiment:** {clove_compute / len(clove_completed):.2f}s" if clove_completed else "",
        "",
        "### Standalone (Python Multiprocessing)",
        f"- **Wall Clock Time:** {standalone_run['wall_time']:.2f}s",
        f"- **Total Compute Time:** {standalone_compute:.2f}s",
        f"- **Parallelization Efficiency:** {standalone_efficiency:.2f}x",
        f"- **Experiments Completed:** {len(standalone_completed)}/{len(standalone_results)}",
        f"- **Average Time per Experiment:** {standalone_compute / len(standalone_completed):.2f}s" if standalone_completed else "",
        "",
        "### Overhead Analysis",
        "",
    ])

    # Calculate Clove overhead
    if standalone_run["wall_time"] > 0:
        overhead = ((clove_run["wall_time"] - standalone_run["wall_time"]) / standalone_run["wall_time"]) * 100
        if overhead > 0:
            lines.append(f"**Clove Overhead:** {overhead:.1f}% slower than standalone")
            lines.append("")
            lines.append("This overhead comes from:")
            lines.append("- Process isolation (Linux namespaces)")
            lines.append("- Resource limit enforcement (cgroups v2)")
            lines.append("- IPC message passing")
            lines.append("- Agent lifecycle management")
        else:
            lines.append(f"**Clove Advantage:** {abs(overhead):.1f}% faster than standalone")
            lines.append("")
            lines.append("Clove may be faster due to:")
            lines.append("- Better process scheduling")
            lines.append("- Optimized IPC")
            lines.append("- Kernel-level resource management")

    # Resource usage comparison
    if clove_analysis and standalone_analysis:
        lines.extend([
            "",
            "---",
            "",
            "## Resource Usage Comparison",
            "",
            "### CPU Usage",
            "",
            "| Metric | Clove | Standalone |",
            "|--------|-------|------------|",
            f"| Average | {clove_analysis['cpu']['avg']:.1f}% | {standalone_analysis['cpu']['avg']:.1f}% |",
            f"| Peak | {clove_analysis['cpu']['max']:.1f}% | {standalone_analysis['cpu']['max']:.1f}% |",
            f"| Minimum | {clove_analysis['cpu']['min']:.1f}% | {standalone_analysis['cpu']['min']:.1f}% |",
            f"| Std Dev | {clove_analysis['cpu']['std']:.1f}% | {standalone_analysis['cpu']['std']:.1f}% |",
            "",
            "### Memory Usage",
            "",
            "| Metric | Clove | Standalone |",
            "|--------|-------|------------|",
            f"| Average % | {clove_analysis['memory']['avg_percent']:.1f}% | {standalone_analysis['memory']['avg_percent']:.1f}% |",
            f"| Peak % | {clove_analysis['memory']['max_percent']:.1f}% | {standalone_analysis['memory']['max_percent']:.1f}% |",
            f"| Average Used | {clove_analysis['memory']['avg_used_mb']:.0f}MB | {standalone_analysis['memory']['avg_used_mb']:.0f}MB |",
            f"| Peak Used | {clove_analysis['memory']['max_used_mb']:.0f}MB | {standalone_analysis['memory']['max_used_mb']:.0f}MB |",
            "",
            "### Process Count",
            "",
            "| Metric | Clove | Standalone |",
            "|--------|-------|------------|",
            f"| Average | {clove_analysis['processes']['avg']:.0f} | {standalone_analysis['processes']['avg']:.0f} |",
            f"| Peak | {clove_analysis['processes']['max']} | {standalone_analysis['processes']['max']} |",
            "",
            "### System Load",
            "",
            "| Metric | Clove | Standalone |",
            "|--------|-------|------------|",
            f"| 1-min Load Avg | {clove_analysis['load_avg']['1min']:.2f} | {standalone_analysis['load_avg']['1min']:.2f} |",
            f"| 5-min Load Avg | {clove_analysis['load_avg']['5min']:.2f} | {standalone_analysis['load_avg']['5min']:.2f} |",
            "",
            "### I/O",
            "",
            "| Metric | Clove | Standalone |",
            "|--------|-------|------------|",
            f"| Disk Read | {clove_analysis['disk_io_mb']['read']:.1f}MB | {standalone_analysis['disk_io_mb']['read']:.1f}MB |",
            f"| Disk Write | {clove_analysis['disk_io_mb']['write']:.1f}MB | {standalone_analysis['disk_io_mb']['write']:.1f}MB |",
            f"| Network Sent | {clove_analysis['network_mb']['sent']:.1f}MB | {standalone_analysis['network_mb']['sent']:.1f}MB |",
            f"| Network Recv | {clove_analysis['network_mb']['recv']:.1f}MB | {standalone_analysis['network_mb']['recv']:.1f}MB |",
        ])

    # ML Results comparison
    if auc_diffs:
        avg_diff = sum(auc_diffs) / len(auc_diffs)
        max_diff = max(auc_diffs)

        lines.extend([
            "",
            "---",
            "",
            "## ML Results Consistency",
            "",
            f"Compared **{len(common_exps)}** matching experiments.",
            "",
            f"- **Average AUC-ROC Difference:** {avg_diff:.6f}",
            f"- **Maximum AUC-ROC Difference:** {max_diff:.6f}",
            "",
        ])

        if max_diff < 0.001:
            lines.append("✅ **Results are highly consistent** - differences are within numerical precision.")
        elif max_diff < 0.01:
            lines.append("✅ **Results are consistent** - minor variations likely from random state differences.")
        else:
            lines.append("⚠️ **Results show some variation** - may be due to different random seeds or execution order.")

    # Top results comparison
    clove_top = sorted(clove_completed, key=lambda x: x.get("test_metrics", {}).get("auc_roc") or 0, reverse=True)[:5]
    standalone_top = sorted(standalone_completed, key=lambda x: x.get("test_metrics", {}).get("auc_roc") or 0, reverse=True)[:5]

    lines.extend([
        "",
        "---",
        "",
        "## Top 5 Results Comparison",
        "",
        "### Clove Top 5",
        "",
        "| Rank | Dataset | Features | Model | AUC-ROC | Time |",
        "|------|---------|----------|-------|---------|------|",
    ])

    for i, r in enumerate(clove_top, 1):
        auc = r.get("test_metrics", {}).get("auc_roc", 0)
        lines.append(f"| {i} | {r['dataset']} | {r['features']} | {r['model']} | {auc:.4f} | {r.get('duration_s', 0):.1f}s |")

    lines.extend([
        "",
        "### Standalone Top 5",
        "",
        "| Rank | Dataset | Features | Model | AUC-ROC | Time |",
        "|------|---------|----------|-------|---------|------|",
    ])

    for i, r in enumerate(standalone_top, 1):
        auc = r.get("test_metrics", {}).get("auc_roc", 0)
        lines.append(f"| {i} | {r['dataset']} | {r['features']} | {r['model']} | {auc:.4f} | {r.get('duration_s', 0):.1f}s |")

    # Clove benefits section
    lines.extend([
        "",
        "---",
        "",
        "## Clove Benefits Analysis",
        "",
        "### What Clove Provides (Not Measured in Raw Performance)",
        "",
        "| Feature | Benefit | Standalone Equivalent |",
        "|---------|---------|----------------------|",
        "| **Process Isolation** | Prevents runaway processes from affecting others | None - shared process space |",
        "| **Memory Limits (cgroups)** | OOM kills only offending process | OOM can kill entire Python interpreter |",
        "| **CPU Quotas** | Fair scheduling, prevents CPU hogging | No limits - processes compete freely |",
        "| **Namespace Isolation** | Processes can't see each other's filesystem | Full visibility |",
        "| **IPC Message Passing** | Structured communication with audit trail | Shared memory, no audit |",
        "| **Hot Reload** | Automatic restart of failed processes | Manual intervention required |",
        "| **Metrics Collection** | Per-process resource tracking | Requires custom instrumentation |",
        "",
        "### When to Use Clove vs Standalone",
        "",
        "**Use Clove when:**",
        "- Running untrusted or experimental code",
        "- Need guaranteed resource isolation",
        "- Building multi-tenant systems",
        "- Require audit trails and compliance",
        "- Processes may crash or misbehave",
        "",
        "**Use Standalone when:**",
        "- Maximum raw performance is critical",
        "- All code is trusted and well-tested",
        "- Simple batch processing",
        "- Minimal infrastructure overhead needed",
    ])

    # Conclusion
    lines.extend([
        "",
        "---",
        "",
        "## Conclusion",
        "",
    ])

    if clove_run["wall_time"] <= standalone_run["wall_time"] * 1.1:  # Within 10%
        lines.extend([
            "**Clove performs comparably to standalone** while providing:",
            "- Full process isolation",
            "- Resource limit enforcement",
            "- Structured IPC",
            "- Built-in fault tolerance",
            "",
            "The ~10% overhead (if any) is a reasonable trade-off for production safety.",
        ])
    else:
        overhead_pct = ((clove_run["wall_time"] / standalone_run["wall_time"]) - 1) * 100
        lines.extend([
            f"**Clove has ~{overhead_pct:.0f}% overhead** compared to standalone.",
            "",
            "This overhead is the cost of:",
            "- Process spawning through the kernel",
            "- IPC message serialization",
            "- cgroup resource tracking",
            "",
            "For production workloads requiring isolation, this is acceptable.",
            "For maximum throughput on trusted code, standalone may be preferred.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "*Generated by Clove Benchmark Comparison Tool*",
    ])

    report = "\n".join(lines)

    # Save report
    report_path = output_dir / "comparison_report.md"
    report_path.write_text(report)

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Clove vs Standalone Benchmark Comparison")
    parser.add_argument("--datasets", nargs="+", default=["hERG", "AMES", "BBBP", "ClinTox"],
                        help="Datasets to benchmark")
    parser.add_argument("--features", nargs="+", default=["morgan", "maccs", "descriptors"],
                        help="Feature methods")
    parser.add_argument("--models", nargs="+", default=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                        help="Models to train")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Maximum parallel processes")
    parser.add_argument("--output-dir", default="benchmark_comparison",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer experiments")
    parser.add_argument("--skip-standalone", action="store_true",
                        help="Skip standalone benchmark (use existing results)")
    parser.add_argument("--skip-clove", action="store_true",
                        help="Skip Clove benchmark (use existing results)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.quick:
        args.datasets = ["hERG", "AMES"]
        args.features = ["morgan", "maccs"]
        args.models = ["random_forest", "logistic_regression"]

    print_banner()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir / time.strftime("comparison_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build common args
    common_args = [
        "--datasets"] + args.datasets + [
        "--features"] + args.features + [
        "--models"] + args.models + [
        "--max-parallel", str(args.max_parallel),
    ]

    total_experiments = len(args.datasets) * len(args.features) * len(args.models)
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Datasets:    {', '.join(args.datasets)} ({len(args.datasets)})")
    print(f"  Features:    {', '.join(args.features)} ({len(args.features)})")
    print(f"  Models:      {', '.join(args.models)} ({len(args.models)})")
    print(f"  {Colors.CYAN}Total experiments per benchmark: {total_experiments}{Colors.ENDC}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Output: {output_dir}")

    # Run standalone benchmark
    standalone_run = None
    standalone_results = []

    if not args.skip_standalone:
        standalone_script = base_dir / "benchmark_standalone.py"
        standalone_run = run_benchmark(
            str(standalone_script),
            common_args + ["--output-dir", str(output_dir / "standalone")],
            "Standalone (Python Multiprocessing)",
            output_dir,
        )

        # Load results
        standalone_dir = find_latest_run(output_dir / "standalone")
        if standalone_dir:
            standalone_results = load_benchmark_results(standalone_dir)

        # Save metrics
        with open(output_dir / "standalone_metrics.json", "w") as f:
            json.dump(standalone_run, f, indent=2, default=str)
    else:
        print(f"\n{Colors.YELLOW}Skipping standalone benchmark (--skip-standalone){Colors.ENDC}")
        # Try to load existing results
        standalone_dir = find_latest_run(base_dir / "benchmark_results_standalone")
        if standalone_dir:
            standalone_results = load_benchmark_results(standalone_dir)
            standalone_run = {"wall_time": 0, "metrics": [], "name": "Standalone (cached)"}

    # Run Clove benchmark
    clove_run = None
    clove_results = []

    if not args.skip_clove:
        # Check if Clove kernel is running
        kernel_running = False
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                if 'clove_kernel' in cmdline:
                    kernel_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if not kernel_running:
            print(f"\n{Colors.RED}ERROR: Clove kernel is not running!{Colors.ENDC}")
            print("Start it with: ./build/clove_kernel")
            print("Skipping Clove benchmark...")
        else:
            clove_script = base_dir / "benchmark_orchestrator.py"
            clove_run = run_benchmark(
                str(clove_script),
                common_args + ["--output-dir", str(output_dir / "clove")],
                "Clove Orchestrator",
                output_dir,
            )

            # Load results
            clove_dir = find_latest_run(output_dir / "clove")
            if clove_dir:
                clove_results = load_benchmark_results(clove_dir)

            # Save metrics
            with open(output_dir / "clove_metrics.json", "w") as f:
                json.dump(clove_run, f, indent=2, default=str)
    else:
        print(f"\n{Colors.YELLOW}Skipping Clove benchmark (--skip-clove){Colors.ENDC}")
        # Try to load existing results
        clove_dir = find_latest_run(base_dir / "benchmark_results")
        if clove_dir:
            clove_results = load_benchmark_results(clove_dir)
            clove_run = {"wall_time": 0, "metrics": [], "name": "Clove (cached)"}

    # Generate comparison report
    if clove_run and standalone_run:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Generating Comparison Report{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

        report = generate_comparison_report(
            clove_run,
            standalone_run,
            clove_results,
            standalone_results,
            output_dir,
        )

        # Print summary
        print(f"{Colors.BOLD}COMPARISON SUMMARY{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print()
        print(f"  {Colors.CYAN}Clove:{Colors.ENDC}")
        print(f"    Wall time:    {clove_run['wall_time']:.1f}s")
        print(f"    Completed:    {len([r for r in clove_results if r.get('status') == 'completed'])}/{len(clove_results)}")
        print()
        print(f"  {Colors.YELLOW}Standalone:{Colors.ENDC}")
        print(f"    Wall time:    {standalone_run['wall_time']:.1f}s")
        print(f"    Completed:    {len([r for r in standalone_results if r.get('status') == 'completed'])}/{len(standalone_results)}")
        print()

        if clove_run['wall_time'] > 0 and standalone_run['wall_time'] > 0:
            diff = clove_run['wall_time'] - standalone_run['wall_time']
            pct = (diff / standalone_run['wall_time']) * 100
            if diff > 0:
                print(f"  {Colors.RED}Clove overhead: +{diff:.1f}s ({pct:.1f}%){Colors.ENDC}")
            else:
                print(f"  {Colors.GREEN}Clove faster by: {abs(diff):.1f}s ({abs(pct):.1f}%){Colors.ENDC}")

        print()
        print(f"{Colors.GREEN}Full comparison report saved to: {output_dir}/comparison_report.md{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}Could not generate comparison - missing benchmark results{Colors.ENDC}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
