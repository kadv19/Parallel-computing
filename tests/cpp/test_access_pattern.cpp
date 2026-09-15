#include <catch2/catch_test_macros.hpp>

#include "core/access_pattern.hpp"
#include "core/task_generator.hpp"

#include <algorithm>
#include <unordered_set>

// Helper to create a task with given range and pattern
Task make_task(size_t start, size_t end, AccessPattern pat, size_t stride, unsigned seed) {
    Task t;
    t.task_id = 0;
    t.start_index = start;
    t.end_index = end;
    t.element_count = end - start;
    t.access_pattern = pat;
    t.stride = stride;
    t.operation_count = 1;
    t.random_seed = seed;
    return t;
}

TEST_CASE("Sequential access visits in order", "[access]") {
    Task t = make_task(10, 20, AccessPattern::Sequential, 1, 0);
    auto indices = generate_traversal_indices(t);
    REQUIRE(indices.size() == 10);
    for (size_t i = 0; i < 10; ++i) {
        CHECK(indices[i] == 10 + i);
    }
    // Also check for_each_index
    std::vector<size_t> via_for_each;
    for_each_index(t, [&](size_t idx){ via_for_each.push_back(idx); });
    CHECK(via_for_each == indices);
}

TEST_CASE("Strided access with stride 3 for n=10 matches spec example", "[access]") {
    // Task [0,10) stride 3 -> expected 0,3,6,9,2,5,8,1,4,7 (local) => global same since start 0
    Task t = make_task(0, 10, AccessPattern::Strided, 3, 0);
    auto indices = generate_traversal_indices(t);
    std::vector<size_t> expected = {0,3,6,9,2,5,8,1,4,7};
    REQUIRE(indices == expected);
}

TEST_CASE("Strided access with offset start", "[access]") {
    // Task [10,20) stride 3 -> local pattern same but offset by 10
    Task t = make_task(10, 20, AccessPattern::Strided, 3, 0);
    auto indices = generate_traversal_indices(t);
    std::vector<size_t> expected_local = {0,3,6,9,2,5,8,1,4,7};
    REQUIRE(indices.size() == 10);
    for (size_t i = 0; i < 10; ++i) {
        CHECK(indices[i] == 10 + expected_local[i]);
    }
}

TEST_CASE("Strided access visits every element exactly once", "[access]") {
    // Test various sizes and strides, including non-coprime
    for (size_t n : {6, 8, 10, 12, 7, 9}) {
        for (size_t stride : {2, 3, 4, 5}) {
            Task t = make_task(0, n, AccessPattern::Strided, stride, 0);
            auto indices = generate_traversal_indices(t);
            REQUIRE(indices.size() == n);
            std::vector<char> seen(n, 0);
            for (size_t idx : indices) {
                REQUIRE(idx < n);
                size_t local = idx; // start 0
                REQUIRE(seen[local] == 0);
                seen[local] = 1;
            }
            for (size_t i = 0; i < n; ++i) CHECK(seen[i] == 1);
            // Ensure exactly n accesses
            size_t count = 0;
            for_each_index(t, [&](size_t){ ++count; });
            CHECK(count == n);
        }
    }
}

TEST_CASE("Strided access with stride 1 equals sequential", "[access]") {
    Task t_seq = make_task(5, 15, AccessPattern::Sequential, 1, 0);
    Task t_str = make_task(5, 15, AccessPattern::Strided, 1, 0);
    CHECK(generate_traversal_indices(t_seq) == generate_traversal_indices(t_str));
}

TEST_CASE("Random access is deterministic with same seed", "[access]") {
    Task t1 = make_task(0, 10, AccessPattern::Random, 1, 42);
    Task t2 = make_task(0, 10, AccessPattern::Random, 1, 42);
    CHECK(generate_traversal_indices(t1) == generate_traversal_indices(t2));
    // for_each also deterministic
    std::vector<size_t> v1, v2;
    for_each_index(t1, [&](size_t i){ v1.push_back(i); });
    for_each_index(t2, [&](size_t i){ v2.push_back(i); });
    CHECK(v1 == v2);
}

TEST_CASE("Random access different seeds produce different orderings", "[access]") {
    Task t1 = make_task(0, 20, AccessPattern::Random, 1, 42);
    Task t2 = make_task(0, 20, AccessPattern::Random, 1, 43);
    auto a = generate_traversal_indices(t1);
    auto b = generate_traversal_indices(t2);
    CHECK(a != b);
}

TEST_CASE("Random access covers all indices exactly once and valid", "[access]") {
    Task t = make_task(10, 20, AccessPattern::Random, 1, 123);
    auto indices = generate_traversal_indices(t);
    REQUIRE(indices.size() == 10);
    std::vector<char> seen(10, 0);
    for (size_t idx : indices) {
        REQUIRE(idx >= 10);
        REQUIRE(idx < 20);
        size_t local = idx - 10;
        REQUIRE(seen[local] == 0);
        seen[local] = 1;
    }
    for (size_t i = 0; i < 10; ++i) CHECK(seen[i] == 1);
}

TEST_CASE("Random access is not sequential", "[access]") {
    Task t = make_task(0, 20, AccessPattern::Random, 1, 42);
    auto indices = generate_traversal_indices(t);
    std::vector<size_t> sequential;
    for (size_t i = 0; i < 20; ++i) sequential.push_back(i);
    CHECK(indices != sequential);
}

TEST_CASE("Random access for different tasks with derived seeds", "[access]") {
    // generate_tasks uses base_seed + task_id
    auto tasks = generate_tasks(20, 2, AccessPattern::Random, 1, 100, 1);
    // tasks[0] seed 100, tasks[1] seed 101, ranges [0,10) and [10,20)
    auto idx0 = generate_traversal_indices(tasks[0]);
    auto idx1 = generate_traversal_indices(tasks[1]);
    // Ensure each task's indices stay within its range
    for (auto i : idx0) { CHECK(i < 10); }
    for (auto i : idx1) { CHECK(i >= 10); CHECK(i < 20); }
    // Same task regenerated with same seed should match
    Task same0 = make_task(0, 10, AccessPattern::Random, 1, 100);
    CHECK(generate_traversal_indices(same0) == idx0);
}

TEST_CASE("Access pattern every task performs exactly element_count accesses", "[access]") {
    for (auto pat : {AccessPattern::Sequential, AccessPattern::Strided, AccessPattern::Random}) {
        Task t = make_task(0, 15, pat, 3, 77);
        size_t count = 0;
        for_each_index(t, [&](size_t){ ++count; });
        CHECK(count == 15);
        CHECK(generate_traversal_indices(t).size() == 15);
    }
}

TEST_CASE("parse_access_pattern case-insensitive", "[access]") {
    CHECK(parse_access_pattern("sequential") == AccessPattern::Sequential);
    CHECK(parse_access_pattern("Sequential") == AccessPattern::Sequential);
    CHECK(parse_access_pattern("SEQUENTIAL") == AccessPattern::Sequential);
    CHECK(parse_access_pattern("strided") == AccessPattern::Strided);
    CHECK(parse_access_pattern("random") == AccessPattern::Random);
    REQUIRE_THROWS(parse_access_pattern("invalid"));
}

TEST_CASE("Strided with large stride wraps correctly", "[access]") {
    // n=5 stride=8 -> stride % n =3 effectively
    Task t = make_task(0, 5, AccessPattern::Strided, 8, 0);
    auto indices = generate_traversal_indices(t);
    REQUIRE(indices.size() == 5);
    std::unordered_set<size_t> set(indices.begin(), indices.end());
    CHECK(set.size() == 5);
}
