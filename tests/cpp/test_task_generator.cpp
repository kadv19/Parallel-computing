#include <catch2/catch_test_macros.hpp>

#include "core/task_generator.hpp"
#include <stdexcept>

TEST_CASE("Task generation evenly divisible", "[task]") {
    size_t array_size = 100;
    size_t task_count = 10;
    auto tasks = generate_tasks(array_size, task_count, AccessPattern::Sequential, 1, 42, 5);
    REQUIRE(tasks.size() == 10);
    size_t total = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        CHECK(tasks[i].task_id == i);
        CHECK(tasks[i].element_count == 10);
        CHECK(tasks[i].start_index + tasks[i].element_count == tasks[i].end_index);
        total += tasks[i].element_count;
        if (i > 0) {
            CHECK(tasks[i].start_index == tasks[i-1].end_index);
        }
    }
    CHECK(tasks.front().start_index == 0);
    CHECK(tasks.back().end_index == array_size);
    CHECK(total == array_size);
}

TEST_CASE("Task generation unevenly divisible distributes remainder", "[task]") {
    // Example: 103 elements / 10 tasks -> first 3 tasks get 11, rest 10
    size_t array_size = 103;
    size_t task_count = 10;
    auto tasks = generate_tasks(array_size, task_count, AccessPattern::Sequential, 1, 0, 1);
    REQUIRE(tasks.size() == 10);
    size_t total = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i < 3) {
            CHECK(tasks[i].element_count == 11);
        } else {
            CHECK(tasks[i].element_count == 10);
        }
        total += tasks[i].element_count;
    }
    CHECK(total == 103);
    CHECK(tasks.front().start_index == 0);
    CHECK(tasks.back().end_index == 103);
    // No gaps, no overlaps
    for (size_t i = 1; i < tasks.size(); ++i) {
        CHECK(tasks[i].start_index == tasks[i-1].end_index);
        CHECK(tasks[i].start_index < tasks[i].end_index);
    }
}

TEST_CASE("Task generation very small arrays", "[task]") {
    // array 5, tasks 5 -> each 1
    {
        auto tasks = generate_tasks(5, 5, AccessPattern::Sequential, 1, 0, 1);
        REQUIRE(tasks.size() == 5);
        for (size_t i = 0; i < 5; ++i) {
            CHECK(tasks[i].element_count == 1);
            CHECK(tasks[i].start_index == i);
            CHECK(tasks[i].end_index == i+1);
        }
    }
    // array 1, tasks 1 -> single element
    {
        auto tasks = generate_tasks(1, 1, AccessPattern::Sequential, 1, 99, 1);
        REQUIRE(tasks.size() == 1);
        CHECK(tasks[0].element_count == 1);
        CHECK(tasks[0].start_index == 0);
        CHECK(tasks[0].end_index == 1);
        CHECK(tasks[0].random_seed == 99);
    }
    // array 3, tasks 2 -> 2 and 1
    {
        auto tasks = generate_tasks(3, 2, AccessPattern::Sequential, 1, 10, 1);
        REQUIRE(tasks.size() == 2);
        CHECK(tasks[0].element_count == 2);
        CHECK(tasks[1].element_count == 1);
        CHECK(tasks[0].start_index == 0);
        CHECK(tasks[0].end_index == 2);
        CHECK(tasks[1].start_index == 2);
        CHECK(tasks[1].end_index == 3);
    }
}

TEST_CASE("Task generation task count smaller than array size", "[task]") {
    // Large array small task count
    auto tasks = generate_tasks(1000000, 4, AccessPattern::Sequential, 8, 42, 5);
    REQUIRE(tasks.size() == 4);
    CHECK(tasks[0].element_count == 250000);
    size_t total = 0;
    for (auto& t : tasks) total += t.element_count;
    CHECK(total == 1000000);
}

TEST_CASE("Task generation invalid task counts throw", "[task]") {
    REQUIRE_THROWS_AS(generate_tasks(100, 0, AccessPattern::Sequential, 1, 0, 1), std::runtime_error);
    REQUIRE_THROWS_AS(generate_tasks(10, 11, AccessPattern::Sequential, 1, 0, 1), std::runtime_error);
    REQUIRE_THROWS_AS(generate_tasks(0, 1, AccessPattern::Sequential, 1, 0, 1), std::runtime_error);
}

TEST_CASE("Task generation assigns access pattern and stride correctly", "[task]") {
    auto tasks = generate_tasks(20, 2, AccessPattern::Strided, 8, 42, 7);
    for (auto& t : tasks) {
        CHECK(t.access_pattern == AccessPattern::Strided);
        CHECK(t.stride == 8);
        CHECK(t.operation_count == 7);
    }
    CHECK(tasks[0].random_seed == 42);
    CHECK(tasks[1].random_seed == 43);
}

TEST_CASE("Task generation covers array without gaps or overlaps exhaustive", "[task]") {
    // Test many combinations
    for (size_t array_size : {10, 17, 100, 101, 1000}) {
        for (size_t task_count : {1, 2, 3, 7, 10}) {
            if (task_count > array_size) continue;
            auto tasks = generate_tasks(array_size, task_count, AccessPattern::Random, 4, 123, 1);
            size_t total = 0;
            for (size_t i = 0; i < tasks.size(); ++i) {
                total += tasks[i].element_count;
                if (i > 0) {
                    REQUIRE(tasks[i].start_index == tasks[i-1].end_index);
                }
            }
            CHECK(total == array_size);
            CHECK(tasks.front().start_index == 0);
            CHECK(tasks.back().end_index == array_size);
        }
    }
}

TEST_CASE("Task generation element_count equals end minus start", "[task]") {
    auto tasks = generate_tasks(50, 7, AccessPattern::Sequential, 1, 0, 1);
    for (auto& t : tasks) {
        CHECK(t.element_count == t.end_index - t.start_index);
    }
}
