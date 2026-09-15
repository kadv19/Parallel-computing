#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "benchmark/metrics.hpp"
#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"

// Isolated per-task burst time measurement.
// For each config, generates array and tasks, then measures each task
// individually with 1 worker (no scheduler contention), warmup+median.

struct TaskBenchConfig {
    size_t array_size = 100000;
    size_t task_count = 10;
    AccessPattern pattern = AccessPattern::Sequential;
    size_t stride = 1;
    int repeat_count = 5;
    unsigned seed = 42;
    double kernel_a = 1.0001;
    double kernel_b = 0.5;
    double kernel_c = 1.00001;
    int warmup = 1;
    int runs = 5;
    std::string output;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --array-size N\n"
              << "  --task-count N\n"
              << "  --pattern sequential|strided|random\n"
              << "  --stride N\n"
              << "  --repeat-count N\n"
              << "  --seed N\n"
              << "  --warmup N\n"
              << "  --runs N\n"
              << "  --output path (CSV)\n";
}

static bool parse_args(int argc, char* argv[], TaskBenchConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](std::string& out) -> bool {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return false; }
            out = argv[++i];
            return true;
        };
        std::string val;
        if (arg == "--array-size") { if (!need(val)) return false; cfg.array_size = std::stoull(val); }
        else if (arg == "--task-count") { if (!need(val)) return false; cfg.task_count = std::stoull(val); }
        else if (arg == "--pattern") { if (!need(val)) return false; cfg.pattern = parse_access_pattern(val); }
        else if (arg == "--stride") { if (!need(val)) return false; cfg.stride = std::stoull(val); }
        else if (arg == "--repeat-count") { if (!need(val)) return false; cfg.repeat_count = std::stoi(val); }
        else if (arg == "--seed") { if (!need(val)) return false; cfg.seed = static_cast<unsigned>(std::stoul(val)); }
        else if (arg == "--warmup") { if (!need(val)) return false; cfg.warmup = std::stoi(val); }
        else if (arg == "--runs") { if (!need(val)) return false; cfg.runs = std::stoi(val); }
        else if (arg == "--output") { if (!need(val)) return false; cfg.output = val; }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown arg: " << arg << "\n"; print_usage(argv[0]); return false; }
    }
    return true;
}

int main(int argc, char* argv[]) {
    TaskBenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;
    if (cfg.task_count > cfg.array_size) { std::cerr << "task_count <= array_size required\n"; return 1; }
    if (cfg.kernel_c == 0.0) { std::cerr << "kernel_c !=0\n"; return 1; }

    // Generate original array and tasks outside timing
    std::vector<double> original = generate_array(cfg.array_size, cfg.seed);
    std::vector<Task> tasks = generate_tasks(cfg.array_size, cfg.task_count, cfg.pattern, cfg.stride, cfg.seed, cfg.repeat_count);

    std::string pat_str = to_string(cfg.pattern);

    // Prepare output
    std::ostream* out = &std::cout;
    std::ofstream fout;
    if (!cfg.output.empty()) {
        fout.open(cfg.output);
        if (!fout) { std::cerr << "Failed to open " << cfg.output << "\n"; return 1; }
        out = &fout;
    }

    // Header
    *out << "array_size,task_count,task_id,element_count,working_set_bytes,access_pattern,access_pattern_encoded,stride,repeat_count,operation_count,locality_score,burst_time_ms\n";

    for (const auto& task : tasks) {
        // Measure this task in isolation, 1 worker, fresh copy each run
        // Warmup
        for (int w = 0; w < cfg.warmup; ++w) {
            auto copy = original;
            process_task(copy, task, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
        }
        std::vector<double> times;
        times.reserve(static_cast<size_t>(cfg.runs));
        for (int r = 0; r < cfg.runs; ++r) {
            auto copy = original;
            auto start = std::chrono::high_resolution_clock::now();
            process_task(copy, task, cfg.kernel_a, cfg.kernel_b, cfg.kernel_c, cfg.repeat_count);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms);
            // Prevent optimization: use copy
            volatile double sink = copy[task.start_index];
            (void)sink;
        }
        double median = calc_median(times);
        // If median is 0 (below timer resolution), use mean of larger batch? Documented: we keep as is.
        // For very small tasks, we may under-measure; caller should use larger repeat_count.
        size_t working_set = task.element_count * sizeof(double);
        int ap_enc = (cfg.pattern == AccessPattern::Sequential ? 0 : (cfg.pattern == AccessPattern::Strided ? 1 : 2));
        double locality = locality_score(cfg.pattern, cfg.stride);

        *out << cfg.array_size << ","
             << cfg.task_count << ","
             << task.task_id << ","
             << task.element_count << ","
             << working_set << ","
             << pat_str << ","
             << ap_enc << ","
             << cfg.stride << ","
             << cfg.repeat_count << ","
             << task.operation_count << ","
             << locality << ","
             << median << "\n";
    }

    if (fout.is_open()) {
        std::cout << "Task burst dataset written to " << cfg.output << " (" << tasks.size() << " tasks)\n";
    }
    return 0;
}
