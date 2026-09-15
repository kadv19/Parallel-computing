#pragma once

#include "ml/model.hpp"
#include "scheduler/ischeduler.hpp"
#include <string>

/// AI Scheduler — predicted-burst-time load balancing.
/// Steps:
///  1. For each task, predict burst time via Linear Regression model (pre-execution features only).
///  2. Sort tasks by descending predicted burst time.
///  3. Greedily assign each task to currently least-loaded worker (worker_load += prediction).
///  4. Execute assigned tasks via existing thread pool + numeric kernel.
/// Static assignment: predict/sort/assign happens BEFORE execution.
/// Model is loaded once at construction, not per task.
class AiScheduler : public IScheduler {
public:
    explicit AiScheduler(const std::string& model_path = "ml/model.json");
    std::string name() const override { return "AI"; }

    void execute(std::vector<double>& array,
                 const std::vector<Task>& tasks,
                 size_t worker_count,
                 double a, double b, double c,
                 int repeat_count) override;

    // For benchmarking diagnostics: time spent in prediction/assignment (ms)
    double last_prediction_time_ms() const { return last_prediction_ms_; }
    double last_assignment_time_ms() const { return last_assignment_ms_; }

private:
    BurstModel model_;
    double last_prediction_ms_{0};
    double last_assignment_ms_{0};
};
