#include "scheduler/ai_scheduler.hpp"

#include "core/task.hpp"
#include "engine/thread_pool.hpp"
#include "kernel/numeric_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

AiScheduler::AiScheduler(const std::string& model_path) : model_(BurstModel::load(model_path)) {}

void AiScheduler::execute(std::vector<double>& array,
                          const std::vector<Task>& tasks,
                          size_t worker_count,
                          double a, double b, double c,
                          int repeat_count) {
    if (tasks.empty()) return;
    if (worker_count == 0) throw std::runtime_error("worker_count must be > 0");

    size_t n = tasks.size();
    size_t w = std::min(worker_count, n);

    // Prediction phase (outside worker execution, but measured separately)
    auto pred_start = std::chrono::high_resolution_clock::now();

    struct PredictedTask {
        Task task;
        double predicted;
    };
    std::vector<PredictedTask> predicted;
    predicted.reserve(n);
    double min_pred = 1e100, max_pred = -1e100, sum_pred = 0;
    for (const auto& t : tasks) {
        double p = model_.predict(t);
        predicted.push_back({t, p});
        if (p < min_pred) min_pred = p;
        if (p > max_pred) max_pred = p;
        sum_pred += p;
    }

    auto pred_end = std::chrono::high_resolution_clock::now();
    last_prediction_ms_ = std::chrono::duration<double, std::milli>(pred_end - pred_start).count();

    // Sort descending by predicted burst time
    auto assign_start = std::chrono::high_resolution_clock::now();
    std::sort(predicted.begin(), predicted.end(), [](const PredictedTask& a, const PredictedTask& b) {
        return a.predicted > b.predicted;
    });

    // Greedy least-loaded assignment
    std::vector<double> worker_load(w, 0.0);
    std::vector<std::vector<Task>> per_worker(w);

    for (auto& pt : predicted) {
        // Find worker with smallest load
        size_t best = 0;
        double best_load = worker_load[0];
        for (size_t i = 1; i < w; ++i) {
            if (worker_load[i] < best_load) {
                best_load = worker_load[i];
                best = i;
            }
        }
        per_worker[best].push_back(pt.task);
        worker_load[best] += pt.predicted;
    }

    auto assign_end = std::chrono::high_resolution_clock::now();
    last_assignment_ms_ = std::chrono::duration<double, std::milli>(assign_end - assign_start).count();

    // Optional diagnostics (not per-task, just aggregate) — could be printed by caller if needed
    // Keep prediction/assignment outside main workload timer per spec: they are preparation.
    // However we store timings for optional reporting.

    // Execute via thread pool (this is the measured region in benchmark)
    ThreadPool pool(w);
    for (size_t worker = 0; worker < w; ++worker) {
        pool.enqueue([&, worker] {
            for (const auto& t : per_worker[worker]) {
                process_task(array, t, a, b, c, repeat_count);
            }
        });
    }
    pool.wait();
}
