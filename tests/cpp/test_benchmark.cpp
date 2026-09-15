#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "benchmark/metrics.hpp"
#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"
#include "scheduler/dynamic_scheduler.hpp"
#include "scheduler/fcfs_scheduler.hpp"
#include "scheduler/round_robin_scheduler.hpp"
#include "scheduler/sjf_scheduler.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// Metric calculation tests
TEST_CASE("Metric speedup calculation", "[benchmark]") {
    CHECK(calc_speedup(10.0, 5.0) == 2.0);
    CHECK(calc_speedup(10.0, 10.0) == 1.0);
    CHECK(calc_speedup(0.0, 5.0) == 0.0);
    CHECK(calc_speedup(10.0, 0.0) == 0.0);
}

TEST_CASE("Metric efficiency calculation", "[benchmark]") {
    CHECK(calc_efficiency(2.0, 2) == 1.0);
    CHECK(calc_efficiency(4.0, 4) == 1.0);
    CHECK(calc_efficiency(2.0, 4) == 0.5);
    CHECK(calc_efficiency(1.0, 0) == 0.0);
}

TEST_CASE("Metric throughput calculation", "[benchmark]") {
    // 1000 elements in 10 ms = 1000 / 0.01 = 100000
    CHECK(calc_throughput(1000, 10.0) == 100000.0);
    CHECK(calc_throughput(500, 0.0) == 0.0);
}

TEST_CASE("Metric mean/median/stddev/min", "[benchmark]") {
    std::vector<double> v = {1, 2, 3, 4, 5};
    CHECK(calc_mean(v) == 3.0);
    CHECK(calc_median(v) == 3.0);
    CHECK(calc_min(v) == 1.0);
    // stddev of 1..5: mean 3, var 2, stddev sqrt(2) ~1.414
    CHECK_THAT(calc_stddev(v, 3.0), WithinRel(std::sqrt(2.0), 1e-9));

    std::vector<double> even = {1, 2, 3, 4};
    CHECK(calc_median(even) == 2.5);
}

TEST_CASE("Locality score sequential highest, random lowest", "[benchmark]") {
    double seq = locality_score(AccessPattern::Sequential, 1);
    double strided = locality_score(AccessPattern::Strided, 8);
    double rnd = locality_score(AccessPattern::Random, 1);
    CHECK(seq == 1.0);
    CHECK(rnd == 0.1);
    CHECK(seq > strided);
    CHECK(strided > rnd);
    CHECK(strided == locality_score(AccessPattern::Strided, 8)); // deterministic
}

TEST_CASE("Locality score strided depends on stride", "[benchmark]") {
    double s2 = locality_score(AccessPattern::Strided, 2);
    double s8 = locality_score(AccessPattern::Strided, 8);
    double s16 = locality_score(AccessPattern::Strided, 16);
    CHECK(s2 != s8);
    CHECK(s8 != s16);
    CHECK(s2 > s8);
    CHECK(s8 >= s16); // clamped to 0.2
}

TEST_CASE("CSV header contains required fields", "[benchmark]") {
    std::string h = csv_header();
    CHECK(h.find("scheduler") != std::string::npos);
    CHECK(h.find("array_size") != std::string::npos);
    CHECK(h.find("task_count") != std::string::npos);
    CHECK(h.find("access_pattern") != std::string::npos);
    CHECK(h.find("worker_count") != std::string::npos);
    CHECK(h.find("median_time_ms") != std::string::npos);
    CHECK(h.find("speedup") != std::string::npos);
    CHECK(h.find("parallel_efficiency") != std::string::npos);
    CHECK(h.find("throughput") != std::string::npos);
    CHECK(h.find("locality_score") != std::string::npos);
    CHECK(h.find("checksum") != std::string::npos);
}

TEST_CASE("CSV row format", "[benchmark]") {
    BenchmarkResult r;
    r.scheduler = "FCFS";
    r.array_size = 1000;
    r.task_count = 10;
    r.access_pattern = "sequential";
    r.stride = 1;
    r.repeat_count = 5;
    r.worker_count = 4;
    r.median_time_ms = 5.0;
    r.mean_time_ms = 5.1;
    r.stddev_time_ms = 0.2;
    r.min_time_ms = 4.9;
    r.speedup = 2.0;
    r.parallel_efficiency = 0.5;
    r.parallel_efficiency_percent = 50.0;
    r.throughput = 200000;
    r.locality = 1.0;
    r.total_elements_processed = 1000;
    r.checksum = 1234.5;
    r.checksum_valid = true;
    std::string row = to_csv_row(r);
    CHECK(row.find("FCFS") != std::string::npos);
    CHECK(row.find("1000") != std::string::npos);
}

TEST_CASE("Benchmark correctness all schedulers match sequential", "[benchmark]") {
    size_t n = 5000;
    unsigned seed = 42;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 10, AccessPattern::Sequential, 1, seed, 3);
    double a = 1.0001, b = 0.5, c = 1.00001;
    int repeat = 3;

    // Sequential
    auto seq = original;
    process_tasks_sequential(seq, tasks, a, b, c, repeat);
    double expected = std::accumulate(seq.begin(), seq.end(), 0.0);

    // Each scheduler must match
    FcfsScheduler fcfs; SjfScheduler sjf; RoundRobinScheduler rr; DynamicScheduler dyn;
    std::vector<double> p;
    for (auto* sched : {static_cast<IScheduler*>(&fcfs), static_cast<IScheduler*>(&sjf), static_cast<IScheduler*>(&rr), static_cast<IScheduler*>(&dyn)}) {
        p = original;
        sched->execute(p, tasks, 4, a, b, c, repeat);
        double cs = std::accumulate(p.begin(), p.end(), 0.0);
        CHECK_THAT(cs, WithinRel(expected, 1e-9));
    }
}

TEST_CASE("Benchmark metrics use median for speedup", "[benchmark]") {
    // Simulate timings
    std::vector<double> seq_times = {10, 10, 10, 10, 10};
    std::vector<double> par_times = {5, 5, 5, 5, 5};
    double seq_median = calc_median(seq_times);
    double par_median = calc_median(par_times);
    double speedup = calc_speedup(seq_median, par_median);
    CHECK(speedup == 2.0);
    double eff = calc_efficiency(speedup, 4);
    CHECK(eff == 0.5);
}
