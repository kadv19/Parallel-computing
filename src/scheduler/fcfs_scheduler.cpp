#include "scheduler/fcfs_scheduler.hpp"
#include "core/task.hpp"
#include "engine/thread_pool.hpp"
#include "kernel/numeric_kernel.hpp"

#include <stdexcept>

void FcfsScheduler::execute(std::vector<double>& array,
                            const std::vector<Task>& tasks,
                            size_t worker_count,
                            double a, double b, double c,
                            int repeat_count) {
    if (tasks.empty()) return;
    if (worker_count == 0) throw std::runtime_error("worker_count must be > 0");

    size_t n = tasks.size();
    size_t w = std::min(worker_count, n);

    // Static partitioning: contiguous blocks, remainder to first workers
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
                process_task(array, tasks[i], a, b, c, repeat_count);
            }
        });
    }
    pool.wait();
}
