#pragma once

#include <string>

/// AppConfig holds all typed configuration fields parsed from config/default.yaml.
/// This interface is depended on by Phase 2 and must remain stable.
struct AppConfig {
    // Phase 1 fields (kept)
    std::string dataset_path;
    int task_count{0};
    int default_core_count{0};
    std::string default_access_pattern;
    int default_stride{0};
    std::string log_output_path;

    // Phase 2 fields
    size_t array_size{0};
    unsigned array_seed{0};
    double kernel_a{0.0};
    double kernel_b{0.0};
    double kernel_c{0.0};
    int repeat_count{0};

    // Phase 3 fields
    int worker_count{0};

    // Phase 6 fields
    std::string ai_model_path;
};

/// ConfigLoader provides a single static entry point to parse YAML configuration.
///
/// Usage:
///   AppConfig cfg = ConfigLoader::load("config/default.yaml");
///
/// Throws std::runtime_error on I/O failure or malformed / missing required fields.
class ConfigLoader {
public:
    /// Load and parse the YAML file at @p path.
    /// @param path Filesystem path to the YAML config file.
    /// @return Populated AppConfig.
    /// @throws std::runtime_error if the file cannot be opened, read, or required
    ///         fields are missing / have invalid types.
    static AppConfig load(const std::string& path);
};
