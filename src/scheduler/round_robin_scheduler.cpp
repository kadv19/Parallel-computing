#include "scheduler/round_robin_scheduler.hpp"
#include "core/task.hpp"
#include "engine/thread_pool.hpp"
#include "kernel/numeric_kernel.hpp"

#include <stdexcept>

void RoundRobinScheduler::execute(std::vector<double>& array,
                                  const std::vector<Task>& tasks,
                                  size_t worker_count,
                                  double a, double b, double c,
                                  int repeat_count) {
    if (tasks.empty()) return;
    if (worker_count == 0) throw std::runtime_error("worker_count must be > 0");

    size_t n = tasks.size();
    size_t w = std::min(worker_count, n);

    // Build per-worker task lists via cyclic assignment
    std::vector<std::vector<Task>> per_worker(w);
    for (size_t i = 0; i < n; ++i) {
        per_worker[i % w].push_back(tasks[i]);
    }

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
