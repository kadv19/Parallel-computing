#include "scheduler/dynamic_scheduler.hpp"
#include "core/task.hpp"
#include "engine/thread_pool.hpp"
#include "kernel/numeric_kernel.hpp"

#include <atomic>
#include <mutex>
#include <stdexcept>

void DynamicScheduler::execute(std::vector<double>& array,
                               const std::vector<Task>& tasks,
                               size_t worker_count,
                               double a, double b, double c,
                               int repeat_count) {
    if (tasks.empty()) return;
    if (worker_count == 0) throw std::runtime_error("worker_count must be > 0");

    size_t n = tasks.size();
    size_t w = std::min(worker_count, n);

    // Shared index protected by atomic + mutex alternative; use atomic for simple pull
    std::atomic<size_t> next_index{0};

    ThreadPool pool(w);
    for (size_t worker = 0; worker < w; ++worker) {
        pool.enqueue([&, worker] {
            while (true) {
                size_t idx = next_index.fetch_add(1);
                if (idx >= n) break;
                process_task(array, tasks[idx], a, b, c, repeat_count);
            }
        });
    }
    pool.wait();
}
