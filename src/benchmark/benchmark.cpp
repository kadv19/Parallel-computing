#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "benchmark/metrics.hpp"
#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"
#include "scheduler/ai_scheduler.hpp"
#include "scheduler/dynamic_scheduler.hpp"
#include "scheduler/fcfs_scheduler.hpp"
#include "scheduler/round_robin_scheduler.hpp"
#include "scheduler/sjf_scheduler.hpp"

struct BenchConfig {
    size_t array_size = 1000000;
    size_t task_count = 100;
    AccessPattern pattern = AccessPattern::Sequential;
    size_t stride = 8;
    size_t worker_count = 4;
    int repeat_count = 5;
    unsigned seed = 42;
    double kernel_a = 1.0001;
    double kernel_b = 0.5;
    double kernel_c = 1.00001;
    int warmup_runs = 1;
    int measurement_runs = 5;
    std::string output_path;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --array-size N\n"
              << "  --task-count N\n"
              << "  --pattern sequential|strided|random\n"
              << "  --stride N\n"
              << "  --workers N\n"
              << "  --repeat-count N\n"
              << "  --seed N\n"
              << "  --runs N (measurement runs, default 5)\n"
              << "  --warmup N (default 1)\n"
              << "  --output path (CSV output, default stdout)\n";
}

static bool parse_args(int argc, char* argv[], BenchConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](std::string& out) -> bool {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return false; }
            out = argv[++i];
            return true;
        };
        std::string val;
        if (arg == "--array-size") {
            if (!need_val(val)) return false;
            cfg.array_size = static_cast<size_t>(std::stoull(val));
        } else if (arg == "--task-count") {
            if (!need_val(val)) return false;
            cfg.task_count = static_cast<size_t>(std::stoull(val));
        } else if (arg == "--pattern") {
            if (!need_val(val)) return false;
            cfg.pattern = parse_access_pattern(val);
        } else if (arg == "--stride") {
            if (!need_val(val)) return false;
            cfg.stride = static_cast<size_t>(std::stoull(val));
        } else if (arg == "--workers") {
            if (!need_val(val)) return false;
            cfg.worker_count = static_cast<size_t>(std::stoull(val));
        } else if (arg == "--repeat-count") {
            if (!need_val(val)) return false;
            cfg.repeat_count = std::stoi(val);
        } else if (arg == "--seed") {
            if (!need_val(val)) return false;
            cfg.seed = static_cast<unsigned>(std::stoul(val));
        } else if (arg == "--runs") {
            if (!need_val(val)) return false;
            cfg.measurement_runs = std::stoi(val);
        } else if (arg == "--warmup") {
            if (!need_val(val)) return false;
            cfg.warmup_runs = std::stoi(val);
        } else if (arg == "--output") {
            if (!need_val(val)) return false;
            cfg.output_path = val;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// Measure function repeatedly; returns vector of times in ms, and last checksum
static std::vector<double> measure_sequential(const std::vector<double>& original,
                                              const std::vector<Task>& tasks,
                                              BenchConfig& cfg,
                                              double& out_checksum) {
    std::vector<double> times;
    times.reserve(static_cast<size_t>(cfg.measurement_runs));
    // warmup
    for (int i = 0; i < cfg.warmup_runs; ++i) {
        auto copy = original;
        process_tasks_sequential(copy, tasks, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
    }
    out_checksum = 0;
    for (int i = 0; i < cfg.measurement_runs; ++i) {
        auto copy = original;
        auto start = std::chrono::high_resolution_clock::now();
        process_tasks_sequential(copy, tasks, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        double cs = std::accumulate(copy.begin(), copy.end(), 0.0);
        if (i == 0) out_checksum = cs;
        else {
            if (std::abs(cs - out_checksum) > 1e-6) {
                std::cerr << "Warning: sequential checksum mismatch across runs\n";
            }
        }
        volatile double sink = cs;
        (void)sink;
    }
    return times;
}

static std::vector<double> measure_scheduler(IScheduler& sched,
                                             const std::vector<double>& original,
                                             const std::vector<Task>& tasks,
                                             BenchConfig& cfg,
                                             double& out_checksum) {
    std::vector<double> times;
    times.reserve(static_cast<size_t>(cfg.measurement_runs));
    for (int i = 0; i < cfg.warmup_runs; ++i) {
        auto copy = original;
        sched.execute(copy, tasks, cfg.worker_count, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
    }
    out_checksum = 0;
    for (int i = 0; i < cfg.measurement_runs; ++i) {
        auto copy = original;
        auto start = std::chrono::high_resolution_clock::now();
        sched.execute(copy, tasks, cfg.worker_count, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        double cs = std::accumulate(copy.begin(), copy.end(), 0.0);
        if (i == 0) out_checksum = cs;
        volatile double sink = cs;
        (void)sink;
    }
    return times;
}

int main(int argc, char* argv[]) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    if (cfg.task_count > cfg.array_size) {
        std::cerr << "task_count must be <= array_size\n";
        return 1;
    }
    if (cfg.kernel_c == 0.0) {
        std::cerr << "kernel_c must not be zero\n";
        return 1;
    }

    // Preparation outside timing (not measured)
    std::vector<double> original = generate_array(cfg.array_size, cfg.seed);
    std::vector<Task> tasks = generate_tasks(cfg.array_size, cfg.task_count, cfg.pattern, cfg.stride, cfg.seed, cfg.repeat_count);
    size_t total_elements = cfg.array_size;
    double locality = locality_score(cfg.pattern, cfg.stride);
    std::string pattern_str = to_string(cfg.pattern);

    // Sequential baseline
    double seq_checksum = 0;
    auto seq_times = measure_sequential(original, tasks, cfg, seq_checksum);
    double seq_median = calc_median(seq_times);
    double seq_mean = calc_mean(seq_times);
    double seq_std = calc_stddev(seq_times, seq_mean);
    double seq_min = calc_min(seq_times);

    std::vector<BenchmarkResult> results;

    // Sequential result entry (worker_count=1 for efficiency base)
    {
        BenchmarkResult r;
        r.scheduler = "Sequential";
        r.array_size = cfg.array_size;
        r.task_count = cfg.task_count;
        r.access_pattern = pattern_str;
        r.stride = cfg.stride;
        r.repeat_count = cfg.repeat_count;
        r.worker_count = 1;
        r.median_time_ms = seq_median;
        r.mean_time_ms = seq_mean;
        r.stddev_time_ms = seq_std;
        r.min_time_ms = seq_min;
        r.speedup = 1.0;
        r.parallel_efficiency = 1.0;
        r.parallel_efficiency_percent = 100.0;
        r.throughput = calc_throughput(total_elements, seq_median);
        r.locality = locality;
        r.total_elements_processed = total_elements;
        r.checksum = seq_checksum;
        r.checksum_valid = true;
        results.push_back(r);
    }

    // Parallel schedulers (including AI)
    std::vector<std::unique_ptr<IScheduler>> schedulers;
    schedulers.emplace_back(std::make_unique<FcfsScheduler>());
    schedulers.emplace_back(std::make_unique<SjfScheduler>());
    schedulers.emplace_back(std::make_unique<RoundRobinScheduler>());
    schedulers.emplace_back(std::make_unique<DynamicScheduler>());
    try {
        schedulers.emplace_back(std::make_unique<AiScheduler>("ml/model.json"));
    } catch (const std::exception& e) {
        std::cerr << "Warning: AI scheduler unavailable: " << e.what() << "\n";
    }

    for (auto& sched : schedulers) {
        double cs = 0;
        auto times = measure_scheduler(*sched, original, tasks, cfg, cs);
        double median = calc_median(times);
        double mean = calc_mean(times);
        double stddev = calc_stddev(times, mean);
        double minv = calc_min(times);
        double speedup = calc_speedup(seq_median, median);
        double eff = calc_efficiency(speedup, cfg.worker_count);
        double thr = calc_throughput(total_elements, median);
        bool valid = std::abs(cs - seq_checksum) < 1e-4; // tolerance

        if (!valid) {
            std::cerr << "Checksum mismatch for " << sched->name()
                      << " expected " << seq_checksum << " got " << cs << "\n";
        }

        BenchmarkResult r;
        r.scheduler = sched->name();
        r.array_size = cfg.array_size;
        r.task_count = cfg.task_count;
        r.access_pattern = pattern_str;
        r.stride = cfg.stride;
        r.repeat_count = cfg.repeat_count;
        r.worker_count = cfg.worker_count;
        r.median_time_ms = median;
        r.mean_time_ms = mean;
        r.stddev_time_ms = stddev;
        r.min_time_ms = minv;
        r.speedup = speedup;
        r.parallel_efficiency = eff;
        r.parallel_efficiency_percent = eff * 100.0;
        r.throughput = thr;
        r.locality = locality;
        r.total_elements_processed = total_elements;
        r.checksum = cs;
        r.checksum_valid = valid;
        results.push_back(r);
    }

    // Output CSV
    auto output_csv = [&](std::ostream& os) {
        os << csv_header() << "\n";
        for (auto& r : results) {
            os << to_csv_row(r) << "\n";
        }
    };

    if (!cfg.output_path.empty()) {
        std::ofstream out(cfg.output_path);
        if (!out) {
            std::cerr << "Failed to open output file: " << cfg.output_path << "\n";
            return 1;
        }
        output_csv(out);
        std::cout << "Results written to " << cfg.output_path << " (" << results.size() << " rows)\n";
    } else {
        output_csv(std::cout);
    }

    // Check for invalid checksums and exit non-zero if any
    for (auto& r : results) {
        if (!r.checksum_valid) {
            std::cerr << "Benchmark failed checksum validation\n";
            return 1;
        }
    }

    return 0;
}
