#include "kernel/numeric_kernel.hpp"

#include "core/access_pattern.hpp"

#include <stdexcept>

void process_task(std::vector<double>& array,
                  const Task& task,
                  double a,
                  double b,
                  double c,
                  int repeat_count) {
    if (c == 0.0) {
        throw std::runtime_error("kernel_c must not be zero");
    }
    if (repeat_count <= 0) {
        throw std::runtime_error("repeat_count must be > 0");
    }
    // Iterate via access pattern without allocating global vector.
    for_each_index(task, [&](size_t idx) {
        double val = array[idx];
        for (int r = 0; r < repeat_count; ++r) {
            val = (val * a + b) / c;
        }
        array[idx] = val;
    });
}

void process_tasks_sequential(std::vector<double>& array,
                              const std::vector<Task>& tasks,
                              double a,
                              double b,
                              double c,
                              int repeat_count) {
    for (const auto& task : tasks) {
        process_task(array, task, a, b, c, repeat_count);
    }
}
