#!/usr/bin/env python3
"""
Simple benchmark harness for Phase 4.
Defines a small default experiment matrix, invokes the C++ benchmark executable,
and writes results to experiments/results/benchmark_results.csv
"""
import subprocess
import csv
import os
import sys
import itertools
import shutil

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BENCHMARK_BIN = os.path.join(BUILD_DIR, "benchmark")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "experiments", "results", "benchmark_results.csv")

# Fallback: also check ./build/benchmark
if not os.path.exists(BENCHMARK_BIN):
    BENCHMARK_BIN = os.path.join(PROJECT_ROOT, "build", "benchmark")

def run_single(config, writer):
    """
    Run benchmark binary for one workload config and write rows to CSV writer.
    Returns list of rows (dicts).
    """
    cmd = [
        BENCHMARK_BIN,
        "--array-size", str(config["array_size"]),
        "--task-count", str(config["task_count"]),
        "--pattern", config["pattern"],
        "--stride", str(config["stride"]),
        "--workers", str(config["workers"]),
        "--repeat-count", str(config["repeat_count"]),
        "--seed", str(config.get("seed", 42)),
        "--runs", str(config.get("runs", 5)),
        "--warmup", str(config.get("warmup", 1)),
    ]
    # Run and capture stdout (CSV)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] benchmark failed for {config}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Parse CSV output
    lines = result.stdout.strip().splitlines()
    if not lines:
        print(f"[error] no output for {config}", file=sys.stderr)
        sys.exit(1)
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        vals = [v.strip() for v in line.split(",")]
        row = dict(zip(header, vals))
        # checksum validation
        if row.get("checksum_valid") == "0":
            print(f"[error] checksum mismatch in {row}", file=sys.stderr)
            sys.exit(1)
        rows.append(row)
        writer.writerow(row)
    return rows

def main():
    # Use small default matrix suitable for development (as per spec example)
    # To expand later, modify these lists.
    default_matrix = {
        "array_size": [1000000],          # can be [100000, 500000, 1000000]
        "task_count": [100],
        "pattern": ["sequential", "strided", "random"],
        "stride": [8],
        "workers": [1, 2, 4],
        "repeat_count": [5],
        "seed": [42],
        "runs": [5],
        "warmup": [1],
    }

    # Optionally allow larger matrix via --full flag
    if "--full" in sys.argv:
        default_matrix["array_size"] = [100000, 500000, 1000000]
        default_matrix["task_count"] = [10, 100, 1000]
        default_matrix["workers"] = [1, 2, 4, 8]

    # Check benchmark binary
    if not os.path.exists(BENCHMARK_BIN):
        print(f"[error] benchmark binary not found at {BENCHMARK_BIN}", file=sys.stderr)
        print("Run: cmake -S . -B build && cmake --build build -j", file=sys.stderr)
        sys.exit(1)

    # Auto-cap workers to available hardware threads
    try:
        import multiprocessing
        hw = multiprocessing.cpu_count()
    except:
        hw = 4
    print(f"Detected {hw} hardware threads")

    # Filter workers that exceed hw (skip)
    orig_workers = default_matrix["workers"][:]
    default_matrix["workers"] = [w for w in orig_workers if w <= hw]
    skipped = set(orig_workers) - set(default_matrix["workers"])
    if skipped:
        print(f"Skipping worker counts {sorted(skipped)} > hw threads ({hw})")

    # Expand matrix
    keys = ["array_size", "task_count", "pattern", "workers", "repeat_count"]
    # Also need stride, but stride is fixed; could vary per pattern but keep simple
    combos = list(itertools.product(
        default_matrix["array_size"],
        default_matrix["task_count"],
        default_matrix["pattern"],
        default_matrix["workers"],
        default_matrix["repeat_count"],
    ))

    # Filter task_count <= array_size
    combos = [c for c in combos if c[1] <= c[0]]
    total_experiments = len(combos)
    print(f"Running {total_experiments} workload configs × 5 schedulers (Sequential+4)")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Write CSV with header from first run
    # We need to know header upfront; run one config to get it or use known header
    # Instead, open file, write header manually by invoking benchmark once
    # Simpler: open file, run experiments, handle header on first write

    # We'll collect header from metrics lib via first run
    first = True
    total_rows = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = None
        header_written = False

        for idx, (array_size, task_count, pattern, workers, repeat_count) in enumerate(combos, 1):
            stride = default_matrix["stride"][0] if pattern != "sequential" else 1
            # For random/sequential stride is still stored but not used meaningfully
            # Use configured stride for strided, else 1
            if pattern == "strided":
                stride = 8
            elif pattern == "random":
                stride = 1
            else:
                stride = 1

            config = {
                "array_size": array_size,
                "task_count": task_count,
                "pattern": pattern,
                "stride": stride,
                "workers": workers,
                "repeat_count": repeat_count,
                "seed": 42,
                "runs": 5,
                "warmup": 1,
            }
            print(f"[{idx}/{total_experiments}] array={array_size} tasks={task_count} pattern={pattern} workers={workers} repeat={repeat_count} ...", end=" ", flush=True)

            # Prepare writer on first iteration: need header
            if not header_written:
                # Run once to get header, but we can just create DictWriter after first run
                # Instead run and let run_single handle header
                # Create a temporary to get header
                cmd = [
                    BENCHMARK_BIN,
                    "--array-size", str(array_size),
                    "--task-count", str(task_count),
                    "--pattern", pattern,
                    "--stride", str(stride),
                    "--workers", str(workers),
                    "--repeat-count", str(repeat_count),
                    "--seed", "42",
                    "--runs", "5",
                    "--warmup", "1",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print("FAILED")
                    print(result.stderr)
                    sys.exit(1)
                lines = result.stdout.strip().splitlines()
                header = [h.strip() for h in lines[0].split(",")]
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                header_written = True
                # Write rows from this run
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    vals = [v.strip() for v in line.split(",")]
                    row = dict(zip(header, vals))
                    if row.get("checksum_valid") == "0":
                        print("checksum mismatch")
                        sys.exit(1)
                    writer.writerow(row)
                    total_rows += 1
                print("done")
            else:
                rows = run_single(config, writer)
                total_rows += len(rows)
                print("done")

    print(f"\nResults written to {OUTPUT_CSV} ({total_rows} rows)")
    # Print sample
    print("\nSample CSV rows:")
    with open(OUTPUT_CSV) as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i >= 6:
                break

if __name__ == "__main__":
    main()
