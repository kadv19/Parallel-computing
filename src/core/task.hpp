#pragma once

#include <cstddef>
#include <string>

/// Memory access pattern for a task.
enum class AccessPattern {
    Sequential,
    Strided,
    Random
};

/// Convert string to AccessPattern (case-insensitive).
/// Throws std::runtime_error on invalid input.
AccessPattern parse_access_pattern(const std::string& s);

/// Convert AccessPattern to canonical lower-case string.
std::string to_string(AccessPattern p);

/// Task represents a contiguous range of work on the global array.
/// Each task owns [start_index, end_index) with no overlap between tasks.
struct Task {
    size_t task_id{0};
    size_t start_index{0};
    size_t end_index{0};
    size_t element_count{0};
    AccessPattern access_pattern{AccessPattern::Sequential};
    size_t stride{1};
    int operation_count{0};  // repeat_count
    unsigned random_seed{0};
};
