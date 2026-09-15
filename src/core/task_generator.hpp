#pragma once

#include "task.hpp"
#include <cstddef>
#include <vector>

/// Generate tasks that partition [0, array_size) into contiguous non-overlapping ranges.
///
/// @param array_size     Total number of elements in the array.
/// @param task_count     Number of tasks to generate (must be >0 and <= array_size).
/// @param pattern        Access pattern to assign to every task.
/// @param stride         Stride value for strided pattern (ignored for sequential/random but stored).
/// @param base_seed      Base seed; each task gets random_seed = base_seed + task_id (for deterministic random pattern).
/// @param operation_count repeat_count / operation count per element (stored in Task).
/// @return Vector of task_count Tasks covering the array without gaps/overlaps.
/// @throws std::runtime_error on invalid task_count.
///
/// Strategy: first (array_size % task_count) tasks receive one extra element.
/// This yields balanced partitioning and ensures sum element_count == array_size.
std::vector<Task> generate_tasks(size_t array_size,
                                 size_t task_count,
                                 AccessPattern pattern,
                                 size_t stride,
                                 unsigned base_seed,
                                 int operation_count);
