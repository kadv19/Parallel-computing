#include "benchmark/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

double calc_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double calc_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) return values[n/2];
    return (values[n/2 -1] + values[n/2]) / 2.0;
}

double calc_stddev(const std::vector<double>& values, double mean) {
    if (values.size() <= 1) return 0.0;
    double acc = 0.0;
    for (double v : values) {
        double d = v - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(values.size()));
}

double calc_min(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

double calc_speedup(double seq_time_ms, double parallel_time_ms) {
    if (parallel_time_ms <= 0.0) return 0.0;
    return seq_time_ms / parallel_time_ms;
}

double calc_efficiency(double speedup, size_t worker_count) {
    if (worker_count == 0) return 0.0;
    return speedup / static_cast<double>(worker_count);
}

double calc_throughput(size_t total_elements, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double sec = time_ms / 1000.0;
    return static_cast<double>(total_elements) / sec;
}

double locality_score(AccessPattern pattern, size_t stride) {
    // Derived locality score — NOT a hardware cache hit rate.
    // Sequential: 1.0 (best spatial locality)
    // Strided: depends on stride — larger stride = worse locality
    // Random: 0.1 (poor locality)
    // Formula for strided: 1.0 / (1.0 + stride/8.0) — monotonic decreasing,
    // e.g. stride 2 -> 0.8, stride 8 -> 0.5, stride 16 -> 0.33.
    // Documented for ML feature use, not as measured cache metric.
    if (pattern == AccessPattern::Sequential) return 1.0;
    if (pattern == AccessPattern::Random) return 0.1;
    // Strided
    if (stride == 0) return 0.5;
    return 1.0 / (1.0 + static_cast<double>(stride) / 8.0);
}

std::string csv_header() {
    return "scheduler,array_size,task_count,access_pattern,stride,repeat_count,worker_count,median_time_ms,mean_time_ms,stddev_time_ms,min_time_ms,speedup,parallel_efficiency,parallel_efficiency_percent,throughput_elements_per_sec,locality_score,total_elements_processed,checksum,checksum_valid";
}

std::string to_csv_row(const BenchmarkResult& r) {
    std::ostringstream oss;
    oss << r.scheduler << ","
        << r.array_size << ","
        << r.task_count << ","
        << r.access_pattern << ","
        << r.stride << ","
        << r.repeat_count << ","
        << r.worker_count << ","
        << r.median_time_ms << ","
        << r.mean_time_ms << ","
        << r.stddev_time_ms << ","
        << r.min_time_ms << ","
        << r.speedup << ","
        << r.parallel_efficiency << ","
        << r.parallel_efficiency_percent << ","
        << r.throughput << ","
        << r.locality << ","
        << r.total_elements_processed << ","
        << r.checksum << ","
        << (r.checksum_valid ? "1" : "0");
    return oss.str();
}
