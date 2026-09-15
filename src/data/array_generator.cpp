#include "data/array_generator.hpp"

#include <random>

std::vector<double> generate_array(size_t n, unsigned seed) {
    std::vector<double> arr;
    arr.reserve(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < n; ++i) {
        arr.push_back(dist(rng));
    }
    return arr;
}
