#include "scheduler/sjf_scheduler.hpp"
#include "core/task.hpp"
#include "engine/thread_pool.hpp"
#include "kernel/numeric_kernel.hpp"

#include <algorithm>
#include <stdexcept>

void SjfScheduler::execute(std::vector<double>& array,
                           const std::vector<Task>& tasks,
                           size_t worker_count,
                           double a, double b, double c,
                           int repeat_count) {
    if (tasks.empty()) return;
    if (worker_count == 0) throw std::runtime_error("worker_count must be > 0");

    // Copy and sort by estimated cost = element_count * repeat_count
    // repeat_count is per-task operation_count, but we use the scheduler arg for estimate
    // to keep it deterministic and simple. Alternatively use task.element_count * task.operation_count.
    std::vector<Task> sorted = tasks;
    std::sort(sorted.begin(), sorted.end(), [](const Task& x, const Task& y) {
        size_t cx = x.element_count * static_cast<size_t>(x.operation_count);
        size_t cy = y.element_count * static_cast<size_t>(y.operation_count);
        if (cx != cy) return cx < cy;
        return x.task_id < y.task_id; // stable tie-break
    });

    size_t n = sorted.size();
    size_t w = std::min(worker_count, n);
    size_t base = n / w;
    size_t rem = n % w;

    ThreadPool pool(w);
    size_t offset = 0;
    for (size_t worker = 0; worker < w; ++worker) {
        size_t count = base + (worker < rem ? 1 : 0);
        size_t start = offset;
        size_t end = start + count;
        offset = end;
        pool.enqueue([&, start, end] {
            for (size_t i = start; i < end; ++i) {
                process_task(array, sorted[i], a, b, c, repeat_count);
            }
        });
    }
    pool.wait();
}
