#pragma once

#include "core/task.hpp"
#include <cstddef>
#include <vector>

/// Deterministic per-element numerical kernel.
///
/// For each selected index i:
///   repeat repeat_count times:
///     array[i] = (array[i] * a + b) / c
///
/// a, b, c come from configuration; c must not be zero.
/// The operation depends only on array[i], so tasks remain independent.

/// Apply kernel to a single element with given repeat count.
inline double apply_kernel(double value, double a, double b, double c, int repeat_count) {
    for (int r = 0; r < repeat_count; ++r) {
        value = (value * a + b) / c;
    }
    return value;
}

/// Process a single task's range via its access pattern.
/// Iterates indices via for_each_index (no global traversal vector).
/// @param array  Global array to modify in-place.
/// @param task   Task describing range and pattern.
/// @param a,b,c  Kernel coefficients (c != 0).
/// @param repeat_count Number of repetitions per element.
void process_task(std::vector<double>& array,
                  const Task& task,
                  double a,
                  double b,
                  double c,
                  int repeat_count);

/// Process all tasks sequentially (baseline).
void process_tasks_sequential(std::vector<double>& array,
                              const std::vector<Task>& tasks,
                              double a,
                              double b,
                              double c,
                              int repeat_count);
