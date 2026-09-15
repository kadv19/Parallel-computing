#pragma once

#include "core/task.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

/// Access pattern utilities.
///
/// The large-array execution path avoids allocating a giant traversal vector.
/// Instead it iterates indices on-the-fly via for_each_index.
/// For Sequential: zero allocation, direct loop.
/// For Strided: O(n) visited bitmap (1 byte per element) plus modulo stepping,
///              with visited handling for non-coprime stride/count.
/// For Random: per-task permutation (size == element_count), avoids global array-size allocation.
/// For unit tests, generate_traversal_indices produces a small verifiable vector.

/// Generate traversal order as a vector (for tests / small ranges).
std::vector<size_t> generate_traversal_indices(const Task& task);

/// Non-template overload (defined in .cpp)
void for_each_index(const Task& task, const std::function<void(size_t)>& func);

/// Templated inline iteration without std::function overhead.
/// Calls func(global_index) exactly element_count times.
template <typename Func>
void for_each_index(const Task& task, Func&& func) {
    size_t n = task.element_count;
    if (n == 0) return;

    switch (task.access_pattern) {
        case AccessPattern::Sequential: {
            for (size_t i = task.start_index; i < task.end_index; ++i) {
                func(i);
            }
            break;
        }
        case AccessPattern::Strided: {
            size_t stride = task.stride == 0 ? 1 : task.stride;
            std::vector<char> visited(n, 0);
            size_t current_local = 0;
            for (size_t k = 0; k < n; ++k) {
                while (visited[current_local]) {
                    current_local = (current_local + 1) % n;
                }
                visited[current_local] = 1;
                func(task.start_index + current_local);
                current_local = (current_local + stride) % n;
            }
            break;
        }
        case AccessPattern::Random: {
            std::vector<size_t> local(n);
            std::iota(local.begin(), local.end(), 0);
            std::mt19937 rng(task.random_seed);
            std::shuffle(local.begin(), local.end(), rng);
            for (size_t v : local) {
                func(task.start_index + v);
            }
            break;
        }
    }
}
