#pragma once

#include "core/task.hpp"
#include <cstddef>
#include <string>
#include <vector>

/// Metrics and statistics for benchmarking.

// Basic statistics
double calc_mean(const std::vector<double>& values);
double calc_median(std::vector<double> values); // copy for sorting
double calc_stddev(const std::vector<double>& values, double mean);
double calc_min(const std::vector<double>& values);

// Parallel metrics
double calc_speedup(double seq_time_ms, double parallel_time_ms);
double calc_efficiency(double speedup, size_t worker_count); // speedup / workers
double calc_throughput(size_t total_elements, double time_ms); // elements per sec

// Derived locality score (not hardware cache hit rate)
double locality_score(AccessPattern pattern, size_t stride);

// CSV header
std::string csv_header();

struct BenchmarkResult {
    std::string scheduler;
    size_t array_size{0};
    size_t task_count{0};
    std::string access_pattern;
    size_t stride{0};
    int repeat_count{0};
    size_t worker_count{0};
    // per-run times are stored externally; this struct holds aggregated
    double median_time_ms{0};
    double mean_time_ms{0};
    double stddev_time_ms{0};
    double min_time_ms{0};
    double speedup{0};
    double parallel_efficiency{0};
    double parallel_efficiency_percent{0};
    double throughput{0};
    double locality{0};
    size_t total_elements_processed{0};
    double checksum{0};
    bool checksum_valid{false};
};

std::string to_csv_row(const BenchmarkResult& r);
