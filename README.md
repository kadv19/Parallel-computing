# AI-Assisted Cache-Aware Parallel Data Processing with Burst-Time Prediction

> **Phase 7 — Visualization + Comparison + Conclusions (Final).** Analysis of benchmark and ML results with plots, summary tables, and lessons learned.
> *Also referred to as: AI-Driven Cache-Aware Parallel Image Processing Scheduler (legacy project title)*

## Problem & Objectives

This project studies how **memory-access patterns** affect numerical processing performance when work is executed in parallel. The workload is a large in-memory `std::vector<double>` divided into independent tasks. By varying traversal order (Sequential, Strided, Random) and computational intensity (`repeat_count`), later phases will measure how cache behavior and scheduling policies (FCFS, SJF, Round Robin, Dynamic, AI) impact speedup and efficiency.

## Repository Structure

```
.
├── CMakeLists.txt
├── config/default.yaml
├── src/
│   ├── main.cpp
│   ├── config/  config_loader.hpp/.cpp
│   ├── data/  array_generator.hpp/.cpp
│   ├── core/  task.hpp, task_generator, access_pattern
│   ├── kernel/  numeric_kernel.hpp/.cpp
│   ├── engine/  thread_pool.hpp/.cpp
│   ├── scheduler/  ischeduler.hpp + FCFS/SJF/RoundRobin/Dynamic/AI
│   ├── ml/  model.hpp/.cpp
│   ├── benchmark/  metrics.hpp/.cpp, benchmark.cpp, task_benchmark.cpp
├── experiments/
│   ├── run_experiments.py
│   ├── generate_task_dataset.py
│   └── results/
│       ├── benchmark_results.csv        # 54 rows (9 configs ×6 schedulers)
│       └── task_burst_dataset.csv       # 11040 rows
├── analysis/
│   ├── visualize.py
│   ├── plots/
│   │   ├── execution_time_by_pattern.png
│   │   ├── scheduler_comparison.png
│   │   ├── speedup_vs_workers.png
│   │   ├── efficiency_vs_workers.png
│   │   ├── throughput_vs_workers.png
│   │   ├── ai_prediction_quality.png
│   │   ├── ai_vs_traditional.png
│   │   └── burst_time_by_pattern.png
│   ├── results_summary.txt
│   ├── final_comparison.csv
│   └── model_summary.txt
├── ml/
│   ├── ml_pipeline.py
│   ├── model.json
│   ├── model_metrics.json
│   ├── test_predictions.csv
│   └── model.pkl
├── tests/cpp/
│   ├── test_config_loader.cpp
│   ├── test_task_generator.cpp
│   ├── test_access_pattern.cpp
│   ├── test_numeric_kernel.cpp
│   ├── test_schedulers.cpp
│   ├── test_benchmark.cpp
│   └── test_ai_scheduler.cpp
```

## Architecture

The system is layered: **Data layer** (`array_generator` + `task_generator` + `access_pattern`) → **Kernel layer** (`numeric_kernel` traversing via Sequential/Strided/Random) → **Engine layer** (`thread_pool` with `std::thread` + `mutex` + `condition_variable`) → **Scheduler layer** (`IScheduler` interface: FCFS, SJF, RoundRobin, Dynamic, AI) → **Benchmark/Metrics layer** (`metrics`, `benchmark.cpp`, CSV output) → **ML layer** (`ml_pipeline.py` + `model.json` loader) → **Analysis layer** (`visualize.py` plots & summaries). Tasks are independent disjoint `[start,end)` ranges (`src/core/task.hpp`), enabling embarrassingly parallel execution.

## Prerequisites

- CMake >= 3.14, C++17 compiler, Threads (pthreads)
- Python 3 with pandas, numpy, scikit-learn, joblib, matplotlib
- Internet on first build (Catch2 via FetchContent)

## How to Run the Project

> Beginner-friendly quick-start for a fresh clone. All commands below have been verified against the current repository (68 tests passing, zero warnings).

### 1. Clone the Repository

```bash
git clone https://github.com/kadv19/Parallel-computing.git
cd Parallel-computing
```

### 2. Build the Project

```bash
cmake -S . -B build
cmake --build build -j
```

This builds the C++ executables (`build/main`, `build/benchmark`, `build/task_benchmark`) and test suite (`build/tests`) with `-Wall -Wextra -Wpedantic` zero warnings.

### 3. Run All Tests

```bash
ctest --test-dir build --output-on-failure
```

Expected result:

```text
100% tests passed, 0 tests failed out of 68

Total Test time (real) =   0.15 sec
```

The current project contains **68 C++ tests** (Catch2, verified via `ctest --test-dir build --output-on-failure`).

### 4. Run a Small Benchmark

```bash
./build/benchmark --array-size 100000 --task-count 10 --workers 4
```

This runs the sequential baseline and all parallel scheduling strategies (FCFS, SJF, RoundRobin, Dynamic, AI) on a small workload and prints CSV to stdout. Verified help and defaults (`src/benchmark/benchmark.cpp:36`):

- `--array-size N` (default 1000000), `--task-count N` (default 100), `--pattern sequential|strided|random` (default sequential), `--stride N` (default 8), `--workers N` (default 4), `--repeat-count N` (default 5), `--seed`, `--runs`, `--warmup`, `--output`.

Example with explicit pattern (also verified):

```bash
./build/benchmark --array-size 100000 --task-count 10 --pattern sequential --workers 4 --repeat-count 5
```

### 5. Run the Experiment Suite

```bash
python3 experiments/run_experiments.py
```

This invokes `build/benchmark` for the configured matrix and generates/updates:

```text
experiments/results/benchmark_results.csv
```

The default matrix (`experiments/run_experiments.py:72`) evaluates combinations of:

- `array_size` = 1000000
- `task_count` = 100
- `access pattern` = sequential / strided / random
- `worker_count` = 1, 2, 4 (auto-capped to hardware threads)
- `scheduling strategy` = Sequential + FCFS + SJF + RoundRobin + Dynamic + AI

Default output is **54 data rows** (9 configs × 6 schedulers; 55 lines with header, verified `wc -l experiments/results/benchmark_results.csv` = 55). Use `--full` for a larger matrix.

### 6. Train / Evaluate the ML Model

```bash
python3 ml/ml_pipeline.py
```

Inspected `ml/ml_pipeline.py:1` — the pipeline loads `experiments/results/task_burst_dataset.csv` (11040 rows), does a config-based 75/25 split (14 configs held out, 2560 test rows), trains Linear Regression (primary) and Decision Tree comparison, validates export, and produces:

```text
ml/model.json
ml/model_metrics.json
ml/model.pkl
ml/test_predictions.csv
```

Current metrics (from `ml/model_metrics.json`): Linear Regression `MAE~0.040 RMSE~0.076 R²~0.793`, Decision Tree `R²~0.879`. Linear Regression is the deployed model for C++ inference because it is lightweight, transparent, and easy to implement directly in C++ (`src/ml/model.cpp:89`), even though Decision Tree had better prediction accuracy.

### 7. Generate Visualizations

```bash
python3 analysis/visualize.py
```

Verified `analysis/visualize.py:1` uses `matplotlib` Agg backend (headless). It loads `experiments/results/benchmark_results.csv`, `ml/model_metrics.json`, and `ml/test_predictions.csv` and generates outputs under:

```text
analysis/plots/
```

including:

```text
execution_time_by_pattern.png
scheduler_comparison.png
speedup_vs_workers.png
efficiency_vs_workers.png
throughput_vs_workers.png
ai_prediction_quality.png
ai_vs_traditional.png
burst_time_by_pattern.png
```

Also:

```text
analysis/results_summary.txt
analysis/final_comparison.csv
analysis/model_summary.txt
```

### 8. Typical Complete Workflow

```bash
git clone https://github.com/kadv19/Parallel-computing.git
cd Parallel-computing

cmake -S . -B build
cmake --build build -j

ctest --test-dir build --output-on-failure

./build/benchmark --array-size 100000 --task-count 10 --workers 4

python3 ml/ml_pipeline.py
python3 experiments/run_experiments.py
python3 analysis/visualize.py
```

Each command above is valid in the current repository. The safest order is to **build and test first**, then run a small benchmark to sanity-check, then regenerate ML model (requires existing `task_burst_dataset.csv`), then regenerate full benchmark results, then regenerate visualizations (which depend on both). Running `experiments/run_experiments.py` or `ml/ml_pipeline.py` will overwrite `benchmark_results.csv` or model artifacts, which is intentional for reproducibility — committed artifacts are the reference results.

### Requirements

Only dependencies actually required (inspected from `CMakeLists.txt` and Python imports):

- **C++:** Compiler with C++17 support (e.g., g++ >= 9, clang++ >= 10), CMake >= 3.14, `Threads` (pthreads)
- **C++ FetchContent:** Internet on first build for Catch2 (`https://github.com/catchorg/Catch2.git` tag `v3.7.1`)
- **Python:** Python 3 with:
  - `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib` (required by `ml/ml_pipeline.py:16` and `analysis/visualize.py:11`)
  - No TensorFlow, PyTorch, CUDA, Streamlit, or Plotly required
- **Verification:** `python3 -c "import pandas, sklearn, matplotlib, numpy; print('ok')"` → `ok`

## Build Instructions

```bash
cmake -S . -B build
cmake --build build -j
```

Produces `build/main`, `build/benchmark`, `build/task_benchmark`, `build/tests` with `-Wall -Wextra -Wpedantic` zero warnings.

## Running Tests

```bash
ctest --test-dir build --output-on-failure
# Expected: 68 tests, 100% passing, zero warnings
```

## Running

```bash
./build/main
./build/benchmark --array-size 100000 --task-count 10 --pattern sequential --workers 4 --repeat-count 5
./build/benchmark --array-size 1000000 --task-count 100 --pattern sequential --workers 4 --repeat-count 5
python3 experiments/run_experiments.py          # Full benchmark matrix -> experiments/results/benchmark_results.csv (54 rows)
python3 experiments/generate_task_dataset.py    # Per-task burst dataset -> experiments/results/task_burst_dataset.csv (11040 rows)
python3 ml/ml_pipeline.py                       # Train & export Linear Regression -> ml/model.json (+ test_predictions.csv)
python3 analysis/visualize.py                   # Generate 8 plots + summaries -> analysis/plots/*.png, final_comparison.csv
```

## Benchmarking

### Controlled Workload Matrix
`array_size=1000000 task_count=100 patterns=Seq/Strided/Random workers=1,2,4 repeat=5` → 54 rows.

### Repeated Measurements
`warmup=1` + `measurement=5`, median for speedup; timing only scheduler execution.

### Metrics
- `speedup = T_seq/T_parallel`
- `parallel_efficiency = speedup/workers`
- `throughput = elements/seconds`
- `locality_score`: Sequential `1.0`, Strided `1/(1+stride/8)`, Random `0.1`

> Parallel efficiency is calculated as speedup divided by worker count. It is not the same as hardware CPU utilization measured using performance counters.

### CSV Output
`experiments/results/benchmark_results.csv`

## Access Patterns

Three traversal orders vary spatial locality and cache behavior (`src/core/access_pattern.cpp`):

- **Sequential** (`locality_score=1.0`): contiguous `start … end-1`; best prefetch, lowest latency.
- **Strided** (`locality_score=1/(1+stride/8)`, stride=8 → 0.5): `(start + i*stride) % element_count` modulo wrap; intermediate locality, less cache-friendly than sequential.
- **Random** (`locality_score=0.1`): per-task shuffled permutation (deterministic via seed); scattered accesses, highest stalls.

> The locality score is a derived heuristic based on access pattern and stride. It is NOT a direct hardware cache-hit measurement.

## Burst-Time Prediction Model

### What is burst time?
Isolated `process_task` median over 5 runs.

### Features (known BEFORE execution)
`element_count`, `working_set_bytes`, `access_pattern_encoded` (0/1/2), `stride`, `repeat_count`, `operation_count`, `locality_score`

### Target
`burst_time_ms`

### Model
Linear Regression (`MAE~0.040 RMSE~0.076 R²~0.79`) + DecisionTree (`MAE~0.027 R²~0.88`), config-based 75/25 split.

### Export
`ml/model.json` validated manual vs sklearn within `1e-6`.

> The model predicts task cost; it does not directly measure cache hits.

## AI Scheduler

### Step 1 — Feature extraction
Construct features exactly as training (`src/ml/model.cpp:66`).

### Step 2 — Burst prediction
Load `ml/model.json` once, `pred = intercept + sum(coef*feature)` clamped `1e-9` (`src/ml/model.cpp:89`).

> The ML model predicts task cost; the scheduler uses that prediction to make worker-assignment decisions.

### Step 3 — Predicted load balancing
Sort descending predicted burst, greedy least-loaded assignment (`src/scheduler/ai_scheduler.cpp:33`).

### Step 4 — Execution
Static assignment before execution via `thread_pool` + `process_task()`.

## Scheduling Strategies (Parallel Scheduling)

> The scheduler determines which worker receives each task; worker threads execute the numeric kernel.

### FCFS
Static contiguous blocks.

### SJF
Sorts by `element_count*repeat_count`.

### Round Robin
Cyclic `i%W`, non-preemptive.

### Dynamic
Atomic `fetch_add` shared queue.

### AI
Predicted burst-time load balancing.

## Visualization & How to Generate Visualizations

```bash
python3 analysis/visualize.py
```

Generates `analysis/plots/*.png` (8 plots, headless Agg backend), `analysis/final_comparison.csv`, `analysis/results_summary.txt`, `analysis/model_summary.txt` from actual measured CSVs (`experiments/results/benchmark_results.csv`, `ml/model_metrics.json`, `ml/test_predictions.csv`).

## How to Run Benchmark

```bash
# Single workload (existing build):
./build/benchmark --array-size 100000 --task-count 10 --pattern sequential --workers 4 --repeat-count 5

# Full experiment matrix (9 configs × 6 schedulers = 54 rows):
python3 experiments/run_experiments.py
# Output: experiments/results/benchmark_results.csv

# Full with larger matrix (optional):
python3 experiments/run_experiments.py --full
```

## Results and Conclusions

### Access Patterns
- **Sequential** generally lowest median time (e.g., 12.1 ms avg at 1M/100/workers 4) due to best spatial locality (locality_score 1.0), contiguous accesses benefit from cache prefetch.
- **Random** highest (22.3 ms) with poor locality (0.1), scattered accesses increase memory stalls.
- **Strided** intermediate (16.1 ms, score 0.5 for stride 8) — modulo wrap still less cache-friendly than sequential. Ordering is measurable but not forced; actual results vary with scheduler and may overlap.

### Scheduling
- For uniform tasks (100 tasks ×10k elements, 1M array, sequential, 4 workers): **FCFS 4.78 ms** fastest, followed by SJF 8.46 ms, RoundRobin 13.16 ms, Dynamic 16.85 ms, AI 17.51 ms (`analysis/results_summary.txt`). Static contiguous assignment wins when task costs are equal (no heterogeneity to balance).
- For heterogeneous or random patterns, Dynamic and AI can close gap or win (e.g., random pattern: AI 20.45 ms vs FCFS 23.86 ms in `final_comparison.csv`).
- No single scheduler dominates all workloads.

### AI Burst Prediction
- Predicted `burst_time_ms` from 7 pre-execution features.
- Linear Regression `MAE 0.040 RMSE 0.076 R² 0.793` (test: 2560 rows, 14 configs held out). Scatter `ai_prediction_quality.png` shows correlation with ideal `y=x` line, but small-task noise near timer resolution causes spread.
- Prediction used solely for worker assignment; no future info leakage.

### Scalability
- **Speedup** (`speedup_vs_workers.png`) rises with workers but is sublinear: e.g., FCFS sequential 1M/100: workers 1→1.00, 2→1.72, 4→3.78 (ideal 4). Dynamic/AI show similar sublinear trends.
- **Efficiency** (`efficiency_vs_workers.png`) drops with more workers: FCFS 100%→85%→94% (4 workers), Dynamic 99%→93%→26% on same workload — overhead and bandwidth limit scaling.
- **Throughput** (`throughput_vs_workers.png`) plateaus: adding workers beyond 2–4 yields diminishing returns; memory bandwidth saturation noted.

### Limitations
- Timing noise for tiny tasks (microseconds) near timer resolution; repeat_count increased where possible.
- Memory bandwidth caps throughput; not CPU-bound.
- Hardware/environment dependent; results not universal.
- Derived locality score `1/(1+stride/8)` is **not** hardware cache-hit rate.
- Linear Regression has error (MAE 0.04ms); AI not guaranteed to beat every policy on uniform workloads.
- AI assumes isolated cost predicts parallel load; contention not modeled.

### Conclusion
The experiment connects access pattern → locality → task cost → ML prediction → scheduling. Sequential access benefits cache; strided/random degrade locality. Among schedulers, static FCFS excels for uniform costs, while predicted-load AI provides a simple, explainable learned alternative that achieves correctness (checksums match) and competitive performance on heterogeneous workloads without universally outperforming traditional policies. This validates the case-study premise: cache-aware, AI-assisted scheduling is feasible and educational, but **optimization requires matching scheduler choice to workload heterogeneity** — e.g., use predicted burst-time balancing when task sizes vary, prefer simpler static or dynamic for uniform tasks, and avoid excessive workers when bandwidth saturates.

## Case-Study Requirement Mapping

| Requirement | Implementation |
|-------------|----------------|
| 1. Define Problem & Objectives | `README.md#Problem & Objectives` + `config/default.yaml` workload definition |
| 2. Identify Parallelism & Dependencies | `src/core/task.hpp` disjoint `[start,end)` ranges; `README.md#Task Independence` |
| 3. Develop Sequential Baseline | `src/kernel/numeric_kernel.cpp:process_tasks_sequential` + `src/main.cpp` sequential timing + checksum |
| 4. Implement Parallel Scheduler | `src/engine/thread_pool.*` + `src/scheduler/ischeduler.hpp` interface |
| 5. Apply Multiple Scheduling Strategies | `src/scheduler/fcfs|sjf|round_robin|dynamic|ai_scheduler.*` (5 strategies) |
| 6. Test Different Workloads & Cores | `experiments/run_experiments.py` matrix 1M/100/patterns workers 1,2,4 + `src/benchmark/benchmark.cpp` CLI |
| 7. Measure Time, Speedup & Utilization/Efficiency | `src/benchmark/metrics.*` speedup/efficiency/throughput + `benchmark_results.csv` |
| 8. Build & Train AI Model for Burst-Time Prediction | `experiments/generate_task_dataset.py` 11040 rows + `ml/ml_pipeline.py` Linear Regression + `ml/model.json` |
| 9. Integrate AI with Parallel Scheduler | `src/ml/model.*` loader + `src/scheduler/ai_scheduler.*` predicted load balancing + benchmark/main integration |
| 10. Compare, Optimize, Visualize & Conclude | `analysis/visualize.py` 8 plots + `analysis/final_comparison.csv` + `README.md#Results and Conclusions` |

## Design Decisions

- **ThreadPool:** `std::thread` + `mutex` + `condition_variable`.
- **Array:** `mt19937`+`uniform_real_distribution`.
- **Locality:** derived score, not hardware counter.

## License

TBD.
