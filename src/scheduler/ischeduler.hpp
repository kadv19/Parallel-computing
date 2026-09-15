#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Task;

/// Common scheduler interface.
/// The scheduler decides which worker executes each task;
/// workers execute the existing numeric kernel (process_task).
class IScheduler {
public:
    virtual ~IScheduler() = default;
    virtual std::string name() const = 0;

    /// Execute tasks in parallel using worker_count threads.
    /// @param array        Shared array (disjoint ranges, no per-element lock needed)
    /// @param tasks        Vector of tasks to execute
    /// @param worker_count Number of worker threads (>0)
    /// @param a,b,c        Kernel coefficients
    /// @param repeat_count Kernel repeat count
    virtual void execute(std::vector<double>& array,
                         const std::vector<Task>& tasks,
                         size_t worker_count,
                         double a, double b, double c,
                         int repeat_count) = 0;
};
