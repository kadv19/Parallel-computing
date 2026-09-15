#!/usr/bin/env python3
"""
Generate isolated per-task burst time dataset.
Invokes the C++ task_benchmark executable for each workload configuration
and aggregates results into experiments/results/task_burst_dataset.csv

Each row = one task execution (median burst_time_ms over 5 runs, warmup 1)
Features are known before execution; target is burst_time_ms.
No scheduler contention (1 worker, isolated process_task timing).
"""
import subprocess
import os
import sys
import csv
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BIN = os.path.join(BUILD_DIR, "task_benchmark")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "experiments", "results", "task_burst_dataset.csv")

if not os.path.exists(BIN):
    BIN = os.path.join(PROJECT_ROOT, "build", "task_benchmark")

def run_config(array_size, task_count, pattern, stride, repeat_count, seed=42, warmup=1, runs=5):
    cmd = [
        BIN,
        "--array-size", str(array_size),
        "--task-count", str(task_count),
        "--pattern", pattern,
        "--stride", str(stride),
        "--repeat-count", str(repeat_count),
        "--seed", str(seed),
        "--warmup", str(warmup),
        "--runs", str(runs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] task_benchmark failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    lines = result.stdout.strip().splitlines()
    if not lines:
        print("[error] no output from task_benchmark", file=sys.stderr)
        sys.exit(1)
    header = lines[0].strip().split(",")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        vals = line.strip().split(",")
        rows.append(dict(zip(header, vals)))
    return header, rows

def main():
    if not os.path.exists(BIN):
        print(f"[error] task_benchmark not found at {BIN}", file=sys.stderr)
        print("Run: cmake -S . -B build && cmake --build build -j", file=sys.stderr)
        sys.exit(1)

    # Manageable workload configs for academic dataset
    # Vary array_size, task granularity, pattern, repeat for measurable burst times
    configs = []
    for array_size, task_count, pattern, repeat_count in itertools.product(
        [100000, 500000, 1000000],   # array sizes
        [10, 100, 500],              # task counts (will filter > array_size)
        ["sequential", "strided", "random"],
        [1, 5],
    ):
        if task_count > array_size:
            continue
        # Stride meaningful only for strided; keep 8 for strided, 1 otherwise
        stride = 8 if pattern == "strided" else 1
        # For larger repeat counts we get more measurable burst times; keep both 1 and 5
        # Skip overly tiny working sets that are below timer resolution? We'll keep but note
        # working_set = element_count *8
        # e.g. 1000000/500=2000 elements -> 16KB, measurable; 100000/10=10000 ->80KB
        configs.append((array_size, task_count, pattern, stride, repeat_count))

    # Also add some intermediate granularities for diversity
    extra = [
        (200000, 20, "sequential", 1, 5),
        (200000, 20, "strided", 8, 5),
        (200000, 20, "random", 1, 5),
    ]
    configs.extend(extra)

    print(f"Generating task burst dataset from {len(configs)} configs")
    all_rows = []
    header = None
    for idx, (array_size, task_count, pattern, stride, repeat_count) in enumerate(configs, 1):
        element_count = array_size // task_count
        print(f"[{idx}/{len(configs)}] array={array_size} tasks={task_count} pattern={pattern} repeat={repeat_count} (elem_per_task~{element_count}) ...", end=" ", flush=True)
        h, rows = run_config(array_size, task_count, pattern, stride, repeat_count)
        if header is None:
            header = h
        all_rows.extend(rows)
        print(f"done ({len(rows)} tasks)")

    # Filter very small burst times that are below reliable resolution? Keep all but warn
    small = sum(1 for r in all_rows if float(r["burst_time_ms"]) < 0.005)
    if small:
        print(f"[note] {small}/{len(all_rows)} tasks have burst_time <0.005ms (near timer resolution); increase repeat_count for more measurable times")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDataset written to {OUTPUT_CSV} ({len(all_rows)} rows)")
    # Quick stats
    patterns = {}
    for r in all_rows:
        patterns[r["access_pattern"]] = patterns.get(r["access_pattern"], 0) + 1
    print(f"Patterns: {patterns}")
    print(f"Distinct configs: {len(configs)}")
    print("Sample rows:")
    with open(OUTPUT_CSV) as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i >= 5:
                break
    # Validate
    import csv as csv2
    with open(OUTPUT_CSV) as f:
        reader = csv2.DictReader(f)
        rows = list(reader)
        assert all(r["burst_time_ms"] for r in rows), "missing burst_time"
        assert len(rows) >= 100, "dataset too small"
        assert len(set(r["access_pattern"] for r in rows)) >= 3, "need all patterns"
    print("[ok] dataset validation passed")

if __name__ == "__main__":
    main()
