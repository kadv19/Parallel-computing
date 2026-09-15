#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"
#include "scheduler/dynamic_scheduler.hpp"
#include "scheduler/fcfs_scheduler.hpp"
#include "scheduler/round_robin_scheduler.hpp"
#include "scheduler/sjf_scheduler.hpp"

#include <memory>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinRel;

static double checksum(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

// Helper to run sequential and a scheduler and compare checksums
static void verify_scheduler_matches_sequential(IScheduler& sched, size_t worker_count = 4) {
    size_t n = 10000;
    unsigned seed = 42;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 20, AccessPattern::Sequential, 1, seed, 5);
    double a = 1.0001, b = 0.5, c = 1.00001;
    int repeat = 5;

    auto seq = original;
    process_tasks_sequential(seq, tasks, a, b, c, repeat);
    double seq_sum = checksum(seq);

    auto par = original;
    sched.execute(par, tasks, worker_count, a, b, c, repeat);
    double par_sum = checksum(par);

    // Use tolerance for floating point
    CHECK_THAT(par_sum, WithinRel(seq_sum, 1e-9));
    // Also element-wise check
    for (size_t i = 0; i < n; ++i) {
        CHECK_THAT(par[i], WithinRel(seq[i], 1e-9));
    }
}

TEST_CASE("FCFS scheduler correctness matches sequential", "[schedulers]") {
    FcfsScheduler s;
    verify_scheduler_matches_sequential(s, 4);
}

TEST_CASE("SJF scheduler correctness matches sequential", "[schedulers]") {
    SjfScheduler s;
    verify_scheduler_matches_sequential(s, 4);
}

TEST_CASE("RoundRobin scheduler correctness matches sequential", "[schedulers]") {
    RoundRobinScheduler s;
    verify_scheduler_matches_sequential(s, 4);
}

TEST_CASE("Dynamic scheduler correctness matches sequential", "[schedulers]") {
    DynamicScheduler s;
    verify_scheduler_matches_sequential(s, 4);
}

TEST_CASE("Schedulers handle worker-count edge cases", "[schedulers]") {
    // 1 worker and more workers than tasks
    std::vector<std::unique_ptr<IScheduler>> schedulers;
    schedulers.emplace_back(std::make_unique<FcfsScheduler>());
    schedulers.emplace_back(std::make_unique<SjfScheduler>());
    schedulers.emplace_back(std::make_unique<RoundRobinScheduler>());
    schedulers.emplace_back(std::make_unique<DynamicScheduler>());

    for (auto& sched : schedulers) {
        // 1 worker
        verify_scheduler_matches_sequential(*sched, 1);
        // more workers than tasks (10 tasks, 20 workers)
        size_t n = 100;
        unsigned seed = 7;
        auto original = generate_array(n, seed);
        auto tasks = generate_tasks(n, 10, AccessPattern::Random, 2, seed, 3);
        double a = 1.0001, b = 0.5, c = 1.00001;
        int repeat = 3;
        auto seq = original;
        process_tasks_sequential(seq, tasks, a, b, c, repeat);
        auto par = original;
        sched->execute(par, tasks, 20, a, b, c, repeat);
        CHECK_THAT(checksum(par), WithinRel(checksum(seq), 1e-9));
    }
}

TEST_CASE("Scheduler names are correct", "[schedulers]") {
    FcfsScheduler fcfs;
    SjfScheduler sjf;
    RoundRobinScheduler rr;
    DynamicScheduler dyn;
    CHECK(fcfs.name() == "FCFS");
    CHECK(sjf.name() == "SJF");
    CHECK(rr.name() == "RoundRobin");
    CHECK(dyn.name() == "Dynamic");
}

TEST_CASE("All schedulers produce same checksum for same workload", "[schedulers]") {
    size_t n = 5000;
    unsigned seed = 123;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 10, AccessPattern::Strided, 3, seed, 4);
    double a = 1.0001, b = 0.5, c = 1.00001;
    int repeat = 4;

    auto seq = original;
    process_tasks_sequential(seq, tasks, a, b, c, repeat);
    double seq_sum = checksum(seq);

    FcfsScheduler fcfs; SjfScheduler sjf; RoundRobinScheduler rr; DynamicScheduler dyn;
    std::vector<double> p1 = original, p2 = original, p3 = original, p4 = original;
    fcfs.execute(p1, tasks, 4, a, b, c, repeat);
    sjf.execute(p2, tasks, 4, a, b, c, repeat);
    rr.execute(p3, tasks, 4, a, b, c, repeat);
    dyn.execute(p4, tasks, 4, a, b, c, repeat);

    CHECK_THAT(checksum(p1), WithinRel(seq_sum, 1e-9));
    CHECK_THAT(checksum(p2), WithinRel(seq_sum, 1e-9));
    CHECK_THAT(checksum(p3), WithinRel(seq_sum, 1e-9));
    CHECK_THAT(checksum(p4), WithinRel(seq_sum, 1e-9));
}
