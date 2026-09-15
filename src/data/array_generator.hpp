#pragma once

#include <cstddef>
#include <vector>

/// Generate a deterministic numerical array of size n using a seeded PRNG.
///
/// @param n    Number of elements to generate.
/// @param seed Seed for the deterministic generator; same n+seed yields identical array.
/// @return Vector of n doubles, uniformly distributed in [0, 1).
///
/// Uses std::mt19937 with std::uniform_real_distribution<double>.
/// No file I/O, no external dependencies.
std::vector<double> generate_array(size_t n, unsigned seed);
