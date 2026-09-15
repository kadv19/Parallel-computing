#include "core/access_pattern.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

std::vector<size_t> generate_traversal_indices(const Task& task) {
    size_t n = task.element_count;
    std::vector<size_t> result;
    if (n == 0) return result;
    result.reserve(n);

    switch (task.access_pattern) {
        case AccessPattern::Sequential: {
            for (size_t i = task.start_index; i < task.end_index; ++i) {
                result.push_back(i);
            }
            break;
        }
        case AccessPattern::Strided: {
            // Strided with modulo wrap-around.
            // Example n=10 stride3 -> 0,3,6,9,2,5,8,1,4,7  == (i*stride)%n
            // For non-coprime, we use visited-set to ensure each element visited exactly once.
            size_t stride = task.stride == 0 ? 1 : task.stride;
            std::vector<char> visited(n, 0);
            size_t current_local = 0;
            for (size_t k = 0; k < n; ++k) {
                // Find next unvisited if current already visited (handles non-coprime)
                while (visited[current_local]) {
                    current_local = (current_local + 1) % n;
                }
                visited[current_local] = 1;
                result.push_back(task.start_index + current_local);
                current_local = (current_local + stride) % n;
            }
            break;
        }
        case AccessPattern::Random: {
            // Deterministic shuffled permutation
            std::vector<size_t> local(n);
            std::iota(local.begin(), local.end(), 0);
            std::mt19937 rng(task.random_seed);
            std::shuffle(local.begin(), local.end(), rng);
            for (size_t v : local) {
                result.push_back(task.start_index + v);
            }
            break;
        }
    }
    return result;
}

void for_each_index(const Task& task, const std::function<void(size_t)>& func) {
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
            // Per-task permutation (size n). This is the only allocation for Random.
            // It is per-task, not per-array, so avoids a giant array-size vector.
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

// Templated inline version: defined in header via generic. To satisfy the
// templated declaration, we provide an inline implementation there implicitly
// via the non-template overload above for std::function. For generic lambdas,
// we rely on the header's template being instantiated there.
// To make templated for_each_index work without separate definition, we define
// it as a header-inline helper: the declaration in the header is sufficient
// because we implement logic directly in the header? Instead we explicitly
// instantiate via inclusion. The simplest is to keep the template in the header
// as inline. Since we can't define template in .cpp for all types, we provide
// a header inline implementation below by re-including.

// To allow generic Func without std::function overhead, provide template in header.
// But we already declared template in header; we need to define it there.
// We'll define it here as an explicit template pattern via header inclusion trick:
// Actually we can define the template body in the header itself (inline).
// For now, ensure the header template is defined inline (we will patch header).

