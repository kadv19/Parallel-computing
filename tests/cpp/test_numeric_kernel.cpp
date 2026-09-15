#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Numeric kernel repeat_count 1 hand-calculated", "[kernel]") {
    // array[i] = (array[i] * a + b) / c
    // a=2, b=1, c=1, val=3 => (3*2+1)/1=7
    double val = 3.0;
    double res = apply_kernel(val, 2.0, 1.0, 1.0, 1);
    CHECK(res == 7.0);
}

TEST_CASE("Numeric kernel repeat_count >1 hand-calculated", "[kernel]") {
    // val=1, a=2, b=0, c=1
    // iter1: (1*2)/1=2
    // iter2: (2*2)/1=4
    // iter3: (4*2)/1=8
    CHECK(apply_kernel(1.0, 2.0, 0.0, 1.0, 3) == 8.0);
}

TEST_CASE("Numeric kernel with division", "[kernel]") {
    // val=10, a=1, b=0, c=2, repeat2
    // iter1: 10/2=5, iter2:5/2=2.5
    double res = apply_kernel(10.0, 1.0, 0.0, 2.0, 2);
    CHECK(res == 2.5);
}

TEST_CASE("Numeric kernel c==0 throws", "[kernel]") {
    std::vector<double> arr = {1.0, 2.0};
    Task t;
    t.task_id = 0;
    t.start_index = 0;
    t.end_index = 2;
    t.element_count = 2;
    t.access_pattern = AccessPattern::Sequential;
    t.stride = 1;
    t.operation_count = 1;
    t.random_seed = 0;
    REQUIRE_THROWS_AS(process_task(arr, t, 1.0, 0.0, 0.0, 1), std::runtime_error);
    REQUIRE_THROWS_AS(process_tasks_sequential(arr, {t}, 1.0, 0.0, 0.0, 1), std::runtime_error);
}

TEST_CASE("process_task sequential applies to correct range", "[kernel]") {
    std::vector<double> arr = {0.0, 1.0, 2.0, 3.0, 4.0};
    Task t;
    t.task_id = 0;
    t.start_index = 1;
    t.end_index = 4;
    t.element_count = 3;
    t.access_pattern = AccessPattern::Sequential;
    t.stride = 1;
    t.operation_count = 1;
    t.random_seed = 0;
    // a=1,b=1,c=1 => val+1
    process_task(arr, t, 1.0, 1.0, 1.0, 1);
    CHECK(arr[0] == 0.0); // untouched
    CHECK(arr[1] == 2.0);
    CHECK(arr[2] == 3.0);
    CHECK(arr[3] == 4.0);
    CHECK(arr[4] == 4.0); // untouched
}

TEST_CASE("Different traversal orders produce same final values", "[kernel]") {
    // This verifies operation independence per element.
    // Same array, same tasks covering same range but different patterns should yield identical array.
    size_t n = 100;
    auto base_array = generate_array(n, 42);
    auto array_seq = base_array;
    auto array_str = base_array;
    auto array_rand = base_array;

    // Single task covering whole array
    Task t_seq{0, 0, n, n, AccessPattern::Sequential, 1, 5, 123};
    Task t_str{0, 0, n, n, AccessPattern::Strided, 3, 5, 123};
    Task t_rand{0, 0, n, n, AccessPattern::Random, 1, 5, 123};

    double a = 1.0001, b = 0.5, c = 1.00001;
    int repeat = 5;

    process_task(array_seq, t_seq, a, b, c, repeat);
    process_task(array_str, t_str, a, b, c, repeat);
    process_task(array_rand, t_rand, a, b, c, repeat);

    // All should be identical (within floating point tolerance)
    for (size_t i = 0; i < n; ++i) {
        CHECK_THAT(array_seq[i], WithinRel(array_str[i], 1e-12) || WithinAbs(array_str[i], 1e-12));
        CHECK_THAT(array_seq[i], WithinRel(array_rand[i], 1e-12) || WithinAbs(array_rand[i], 1e-12));
    }
}

TEST_CASE("process_tasks_sequential covers all elements", "[kernel]") {
    size_t n = 50;
    auto arr = generate_array(n, 99);
    auto arr_copy = arr;
    auto tasks = generate_tasks(n, 5, AccessPattern::Sequential, 1, 0, 3);
    process_tasks_sequential(arr, tasks, 1.0, 1.0, 1.0, 3); // val = val+? Let's compute

    // Manually compute expected: each element transformed 3 times with a=1,b=1,c=1 => +1 each iter => +3
    for (size_t i = 0; i < n; ++i) {
        double expected = apply_kernel(arr_copy[i], 1.0, 1.0, 1.0, 3);
        CHECK(arr[i] == expected);
    }
}

TEST_CASE("Kernel is deterministic", "[kernel]") {
    auto arr1 = generate_array(100, 42);
    auto arr2 = arr1;
    auto tasks = generate_tasks(100, 4, AccessPattern::Random, 2, 42, 5);
    process_tasks_sequential(arr1, tasks, 1.0001, 0.5, 1.00001, 5);
    process_tasks_sequential(arr2, tasks, 1.0001, 0.5, 1.00001, 5);
    CHECK(arr1 == arr2);
}

TEST_CASE("Kernel repeat_count from task operation_count", "[kernel]") {
    // Ensure task.operation_count is used (via generate_tasks)
    auto tasks = generate_tasks(10, 2, AccessPattern::Sequential, 1, 0, 7);
    CHECK(tasks[0].operation_count == 7);
    std::vector<double> arr(10, 1.0);
    process_tasks_sequential(arr, tasks, 2.0, 0.0, 1.0, tasks[0].operation_count);
    // 1.0 *2^7 =128
    for (double v : arr) CHECK(v == 128.0);
}

TEST_CASE("Large array execution does not allocate giant traversal vector (conceptual)", "[kernel]") {
    // This test documents the requirement: process_task uses for_each_index, not a giant vector.
    // We verify that processing a large array via tasks still yields correct checksum.
    size_t n = 10000;
    auto arr = generate_array(n, 1);
    auto tasks = generate_tasks(n, 10, AccessPattern::Random, 1, 1, 2);
    double before_sum = 0;
    for (double v : arr) before_sum += v;
    process_tasks_sequential(arr, tasks, 1.0, 1.0, 1.0, 2);
    double after_sum = 0;
    for (double v : arr) after_sum += v;
    // Each element +2 (since a=1,b=1,c=1, repeat2); use tolerance for floating point
    double expected = before_sum + 2.0 * static_cast<double>(n);
    CHECK_THAT(after_sum, WithinRel(expected, 1e-9) || WithinAbs(expected, 1e-9));
}
