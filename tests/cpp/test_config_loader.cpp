#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "config_loader.hpp"

// Helper to create a temporary YAML file for testing.
namespace {

std::string write_temp_file(const std::string& filename, const std::string& content) {
    std::string path = "/tmp/" + filename;
    std::ofstream out(path);
    out << content;
    return path;
}

// Helper to append Phase 2 fields to a minimal config for Phase 1 legacy tests.
std::string phase2_defaults() {
    return
        "array_size: 1000\n"
        "array_seed: 42\n"
        "kernel_a: 1.0001\n"
        "kernel_b: 0.5\n"
        "kernel_c: 1.00001\n"
        "repeat_count: 5\n"
        "worker_count: 4\n";
}

} // namespace

// ---------------------------------------------------------------------------
// Valid config loading
// ---------------------------------------------------------------------------

TEST_CASE("ConfigLoader loads default.yaml without error", "[config]") {
    // Try multiple candidate paths so the test passes regardless of CWD
    // (ctest runs from build/ or with WORKING_DIRECTORY, ./build/tests runs from root).
    AppConfig cfg;
    bool loaded = false;
    std::string last_err;
    for (const auto& candidate : {
             std::string("config/default.yaml"),
             std::string("../config/default.yaml"),
             std::string("../../config/default.yaml"),
         }) {
        try {
            cfg = ConfigLoader::load(candidate);
            loaded = true;
            break;
        } catch (const std::exception& e) {
            last_err = e.what();
        }
    }
    if (!loaded) {
        FAIL("Failed to load default.yaml from any candidate path: " << last_err);
    }

    CHECK(cfg.dataset_path == "data/sample");
    CHECK(cfg.task_count == 10);
    CHECK(cfg.default_core_count == 4);
    CHECK(cfg.default_access_pattern == "sequential");
    CHECK(cfg.default_stride == 8);
    CHECK(cfg.log_output_path == "experiments/results/run.log");
    CHECK(cfg.array_size == 1000000);
    CHECK(cfg.array_seed == 42);
    CHECK(cfg.kernel_a == 1.0001);
    CHECK(cfg.kernel_b == 0.5);
    CHECK(cfg.kernel_c == 1.00001);
    CHECK(cfg.repeat_count == 5);
    CHECK(cfg.worker_count == 4);
}

TEST_CASE("ConfigLoader correctly parses quoted strings and integers", "[config]") {
    std::string path = write_temp_file("test_quoted.yaml",
        "dataset_path: \"my/data/path\"\n"
        "task_count: 42\n"
        "default_core_count: 8\n"
        "default_access_pattern: 'random'\n"
        "default_stride: 16\n"
        "log_output_path: \"logs/out.log\"\n"
        + phase2_defaults());

    AppConfig cfg = ConfigLoader::load(path);

    CHECK(cfg.dataset_path == "my/data/path");
    CHECK(cfg.task_count == 42);
    CHECK(cfg.default_core_count == 8);
    CHECK(cfg.default_access_pattern == "random");
    CHECK(cfg.default_stride == 16);
    CHECK(cfg.log_output_path == "logs/out.log");

    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader handles comments and blank lines", "[config]") {
    std::string path = write_temp_file("test_comments.yaml",
        "# This is a comment\n"
        "\n"
        "dataset_path: data/sample  # inline comment\n"
        "task_count: 10\n"
        "# another comment\n"
        "default_core_count: 2\n"
        "default_access_pattern: sequential\n"
        "default_stride: 4\n"
        "log_output_path: experiments/results/run.log\n"
        + phase2_defaults());

    AppConfig cfg = ConfigLoader::load(path);

    CHECK(cfg.dataset_path == "data/sample");
    CHECK(cfg.task_count == 10);
    CHECK(cfg.default_core_count == 2);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Error handling — malformed / missing config
// ---------------------------------------------------------------------------

TEST_CASE("ConfigLoader throws on missing file", "[config]") {
    REQUIRE_THROWS_AS(ConfigLoader::load("nonexistent/path/missing.yaml"),
                      std::runtime_error);
}

TEST_CASE("ConfigLoader throws on empty file (missing required fields)", "[config]") {
    std::string path = write_temp_file("test_empty.yaml", "");
    REQUIRE_THROWS_AS(ConfigLoader::load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader throws on missing required field", "[config]") {
    std::string path = write_temp_file("test_missing_field.yaml",
        "dataset_path: data/sample\n"
        "task_count: 10\n"
        // missing default_core_count, default_access_pattern, etc.
        "default_stride: 1\n"
        "log_output_path: experiments/results/run.log\n");

    REQUIRE_THROWS_AS(ConfigLoader::load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader throws on invalid integer value", "[config]") {
    std::string path = write_temp_file("test_bad_int.yaml",
        "dataset_path: data/sample\n"
        "task_count: not_a_number\n"
        "default_core_count: 4\n"
        "default_access_pattern: sequential\n"
        "default_stride: 1\n"
        "log_output_path: experiments/results/run.log\n"
        + phase2_defaults());

    REQUIRE_THROWS_AS(ConfigLoader::load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader throws on malformed line (no colon)", "[config]") {
    std::string path = write_temp_file("test_no_colon.yaml",
        "dataset_path data/sample\n"
        "task_count: 10\n"
        "default_core_count: 4\n"
        "default_access_pattern: sequential\n"
        "default_stride: 1\n"
        "log_output_path: experiments/results/run.log\n"
        + phase2_defaults());

    REQUIRE_THROWS_AS(ConfigLoader::load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader does not crash on malformed input", "[config]") {
    // This test ensures no undefined behavior — exception is expected, not a crash.
    std::string path = write_temp_file("test_garbage.yaml",
        "::::\n"
        "garbage garbage garbage\n");

    REQUIRE_THROWS(ConfigLoader::load(path));
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader rejects kernel_c == 0", "[config]") {
    std::string path = write_temp_file("test_zero_c.yaml",
        "dataset_path: data/sample\n"
        "task_count: 10\n"
        "default_core_count: 4\n"
        "default_access_pattern: sequential\n"
        "default_stride: 1\n"
        "log_output_path: experiments/results/run.log\n"
        "array_size: 1000\n"
        "array_seed: 42\n"
        "kernel_a: 1.0\n"
        "kernel_b: 0.5\n"
        "kernel_c: 0.0\n"
        "repeat_count: 5\n"
        "worker_count: 4\n");
    REQUIRE_THROWS_AS(ConfigLoader::load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ConfigLoader correctly parses Phase 2 double and size_t fields", "[config]") {
    std::string path = write_temp_file("test_phase2.yaml",
        "dataset_path: data/sample\n"
        "task_count: 10\n"
        "default_core_count: 4\n"
        "default_access_pattern: strided\n"
        "default_stride: 8\n"
        "log_output_path: experiments/results/run.log\n"
        "array_size: 500000\n"
        "array_seed: 123\n"
        "kernel_a: 2.5\n"
        "kernel_b: 1.5\n"
        "kernel_c: 0.5\n"
        "repeat_count: 10\n"
        "worker_count: 4\n");
    AppConfig cfg = ConfigLoader::load(path);
    CHECK(cfg.array_size == 500000);
    CHECK(cfg.array_seed == 123);
    CHECK(cfg.kernel_a == 2.5);
    CHECK(cfg.kernel_b == 1.5);
    CHECK(cfg.kernel_c == 0.5);
    CHECK(cfg.repeat_count == 10);
    CHECK(cfg.default_access_pattern == "strided");
    CHECK(cfg.default_stride == 8);
    CHECK(cfg.worker_count == 4);
    std::remove(path.c_str());
}
