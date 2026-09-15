#include "core/task_generator.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

AccessPattern parse_access_pattern(const std::string& s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "sequential") return AccessPattern::Sequential;
    if (lower == "strided") return AccessPattern::Strided;
    if (lower == "random") return AccessPattern::Random;
    throw std::runtime_error("Invalid access pattern: " + s);
}

std::string to_string(AccessPattern p) {
    switch (p) {
        case AccessPattern::Sequential: return "sequential";
        case AccessPattern::Strided: return "strided";
        case AccessPattern::Random: return "random";
    }
    return "unknown";
}

std::vector<Task> generate_tasks(size_t array_size,
                                 size_t task_count,
                                 AccessPattern pattern,
                                 size_t stride,
                                 unsigned base_seed,
                                 int operation_count) {
    if (task_count == 0) {
        throw std::runtime_error("task_count must be > 0");
    }
    if (array_size == 0) {
        throw std::runtime_error("array_size must be > 0");
    }
    if (task_count > array_size) {
        throw std::runtime_error("task_count must be <= array_size");
    }

    std::vector<Task> tasks;
    tasks.reserve(task_count);

    size_t base = array_size / task_count;
    size_t remainder = array_size % task_count;

    size_t current = 0;
    for (size_t i = 0; i < task_count; ++i) {
        size_t count = base + (i < remainder ? 1 : 0);
        size_t start = current;
        size_t end = start + count;
        Task t;
        t.task_id = i;
        t.start_index = start;
        t.end_index = end;
        t.element_count = count;
        t.access_pattern = pattern;
        t.stride = stride;
        t.operation_count = operation_count;
        t.random_seed = base_seed + static_cast<unsigned>(i);
        tasks.push_back(t);
        current = end;
    }

    return tasks;
}
