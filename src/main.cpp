#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "config/config_loader.hpp"
#include "core/access_pattern.hpp"
#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"
#include "ml/model.hpp"
#include "scheduler/ai_scheduler.hpp"
#include "scheduler/dynamic_scheduler.hpp"
#include "scheduler/fcfs_scheduler.hpp"
#include "scheduler/round_robin_scheduler.hpp"
#include "scheduler/sjf_scheduler.hpp"

int main() {
    try {
        AppConfig config = ConfigLoader::load("config/default.yaml");

        std::cout << "Config loaded successfully:\n";
        std::cout << "  dataset_path: " << config.dataset_path << "\n";
        std::cout << "  task_count: " << config.task_count << "\n";
        std::cout << "  default_core_count: " << config.default_core_count << "\n";
        std::cout << "  default_access_pattern: " << config.default_access_pattern << "\n";
        std::cout << "  default_stride: " << config.default_stride << "\n";
        std::cout << "  log_output_path: " << config.log_output_path << "\n";
        std::cout << "  array_size: " << config.array_size << "\n";
        std::cout << "  array_seed: " << config.array_seed << "\n";
        std::cout << "  kernel_a: " << config.kernel_a << "\n";
        std::cout << "  kernel_b: " << config.kernel_b << "\n";
        std::cout << "  kernel_c: " << config.kernel_c << "\n";
        std::cout << "  repeat_count: " << config.repeat_count << "\n";
        std::cout << "  worker_count: " << config.worker_count << "\n";
        std::cout << "  ai_model_path: " << config.ai_model_path << "\n";

        AccessPattern pattern = parse_access_pattern(config.default_access_pattern);

        // Generation outside timed regions
        std::vector<double> original_array = generate_array(config.array_size, config.array_seed);
        std::vector<Task> tasks = generate_tasks(
            config.array_size,
            static_cast<size_t>(config.task_count),
            pattern,
            static_cast<size_t>(config.default_stride),
            config.array_seed,
            config.repeat_count
        );

        std::cout << "\nSequential baseline:\n";
        std::cout << "  access_pattern: " << to_string(pattern) << "\n";
        std::cout << "  array_size: " << config.array_size << "\n";
        std::cout << "  task_count: " << config.task_count << "\n";
        std::cout << "  repeat_count: " << config.repeat_count << "\n";

        size_t total_elements = 0;
        for (const auto& t : tasks) total_elements += t.element_count;

        // Sequential
        std::vector<double> seq_array = original_array;
        auto start = std::chrono::high_resolution_clock::now();
        process_tasks_sequential(seq_array, tasks, config.kernel_a, config.kernel_b, config.kernel_c, config.repeat_count);
        auto end = std::chrono::high_resolution_clock::now();
        auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto seq_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double seq_checksum = std::accumulate(seq_array.begin(), seq_array.end(), 0.0);
        std::cout << "  sequential_time_ms: " << seq_ms << "\n";
        std::cout << "  sequential_time_us: " << seq_us << "\n";
        std::cout << "  total_elements_processed: " << total_elements << "\n";
        std::cout << "  checksum: " << seq_checksum << "\n";
        volatile double sink = seq_checksum;
        (void)sink;

        // Parallel schedulers — each gets fresh copy of original array
        std::vector<std::unique_ptr<IScheduler>> schedulers;
        schedulers.emplace_back(std::make_unique<FcfsScheduler>());
        schedulers.emplace_back(std::make_unique<SjfScheduler>());
        schedulers.emplace_back(std::make_unique<RoundRobinScheduler>());
        schedulers.emplace_back(std::make_unique<DynamicScheduler>());
        try {
            schedulers.emplace_back(std::make_unique<AiScheduler>(config.ai_model_path));
        } catch (const std::exception& e) {
            std::cerr << "Warning: AI scheduler not available: " << e.what() << "\n";
        }

        for (auto& sched : schedulers) {
            std::vector<double> arr_copy = original_array;
            auto s = std::chrono::high_resolution_clock::now();
            sched->execute(arr_copy, tasks, static_cast<size_t>(config.worker_count),
                           config.kernel_a, config.kernel_b, config.kernel_c, config.repeat_count);
            auto e = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
            double checksum = std::accumulate(arr_copy.begin(), arr_copy.end(), 0.0);
            std::cout << "\nScheduler: " << sched->name() << "\n";
            std::cout << "  Workers: " << config.worker_count << "\n";
            std::cout << "  Time: " << ms << " ms (" << us << " us)\n";
            std::cout << "  Checksum: " << checksum << "\n";
            std::cout << "  Checksum match: " << (std::abs(checksum - seq_checksum) < 1e-6 ? "YES" : "NO") << "\n";
            if (sched->name() == "AI") {
                auto* ai = dynamic_cast<AiScheduler*>(sched.get());
                if (ai) {
                    std::cout << "  AI prediction_time_ms: " << ai->last_prediction_time_ms() << "\n";
                    std::cout << "  AI assignment_time_ms: " << ai->last_assignment_time_ms() << "\n";
                }
                // Also show predicted burst stats if available
                // (prediction stats are inside scheduler, but we can recompute for display)
                BurstModel m = BurstModel::load(config.ai_model_path);
                double min_pred = 1e100, max_pred = -1e100, sum_pred = 0;
                for (auto& t : tasks) {
                    double p = m.predict(t);
                    if (p < min_pred) min_pred = p;
                    if (p > max_pred) max_pred = p;
                    sum_pred += p;
                }
                double avg_pred = sum_pred / static_cast<double>(tasks.size());
                std::cout << "  AI predictions: min=" << min_pred << " max=" << max_pred << " avg=" << avg_pred << " ms\n";
            }
            volatile double sink2 = checksum;
            (void)sink2;
        }

    } catch (const std::exception& e) {
        std::cerr << "Failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
