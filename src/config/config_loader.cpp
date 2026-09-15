#include "config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Minimal YAML parser — flat key: value subset only.
// Supports:
//   - Line comments starting with '#'
//   - Scalar string (optionally quoted with " or ') and integer / double values
//   - Whitespace trimming around keys and values
// ---------------------------------------------------------------------------

namespace {

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

std::string strip_inline_comment(const std::string& value) {
    bool in_single = false;
    bool in_double = false;
    for (size_t i = 0; i < value.size(); ++i) {
        char c = value[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (c == '#' && !in_single && !in_double) {
            return trim(value.substr(0, i));
        }
    }
    return value;
}

int parse_int_field(const std::unordered_map<std::string, std::string>& kv,
                    const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::runtime_error("Missing required config field: " + key);
    }
    const std::string& raw = it->second;
    try {
        size_t pos = 0;
        int val = std::stoi(raw, &pos);
        std::string remainder = trim(raw.substr(pos));
        if (!remainder.empty()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid integer value for field '" + key + "': " + raw);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Integer out of range for field '" + key + "': " + raw);
    }
}

size_t parse_size_t_field(const std::unordered_map<std::string, std::string>& kv,
                          const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::runtime_error("Missing required config field: " + key);
    }
    const std::string& raw = it->second;
    try {
        size_t pos = 0;
        size_t val = static_cast<size_t>(std::stoull(raw, &pos));
        std::string remainder = trim(raw.substr(pos));
        if (!remainder.empty()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid integer value for field '" + key + "': " + raw);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Integer out of range for field '" + key + "': " + raw);
    }
}

unsigned parse_unsigned_field(const std::unordered_map<std::string, std::string>& kv,
                              const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::runtime_error("Missing required config field: " + key);
    }
    const std::string& raw = it->second;
    try {
        size_t pos = 0;
        unsigned long val = std::stoul(raw, &pos);
        std::string remainder = trim(raw.substr(pos));
        if (!remainder.empty()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<unsigned>(val);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid integer value for field '" + key + "': " + raw);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Integer out of range for field '" + key + "': " + raw);
    }
}

double parse_double_field(const std::unordered_map<std::string, std::string>& kv,
                          const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::runtime_error("Missing required config field: " + key);
    }
    const std::string& raw = it->second;
    try {
        size_t pos = 0;
        double val = std::stod(raw, &pos);
        std::string remainder = trim(raw.substr(pos));
        if (!remainder.empty()) {
            throw std::invalid_argument("trailing characters");
        }
        return val;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid double value for field '" + key + "': " + raw);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Double out of range for field '" + key + "': " + raw);
    }
}

std::string parse_string_field(const std::unordered_map<std::string, std::string>& kv,
                               const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::runtime_error("Missing required config field: " + key);
    }
    return strip_quotes(it->second);
}

} // namespace

AppConfig ConfigLoader::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        ++line_num;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (trimmed == "---" || trimmed == "...") {
            continue;
        }
        auto colon = trimmed.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("Malformed line " + std::to_string(line_num) +
                                     " (missing ':'): " + line);
        }
        std::string key = trim(trimmed.substr(0, colon));
        std::string value = trim(trimmed.substr(colon + 1));
        if (key.empty()) {
            throw std::runtime_error("Malformed line " + std::to_string(line_num) +
                                     " (empty key): " + line);
        }
        value = strip_inline_comment(value);
        value = trim(value);
        kv[key] = value;
    }

    AppConfig config;
    // Phase 1 fields
    config.dataset_path = parse_string_field(kv, "dataset_path");
    config.task_count = parse_int_field(kv, "task_count");
    config.default_core_count = parse_int_field(kv, "default_core_count");
    config.default_access_pattern = parse_string_field(kv, "default_access_pattern");
    config.default_stride = parse_int_field(kv, "default_stride");
    config.log_output_path = parse_string_field(kv, "log_output_path");

    // Phase 2 fields
    config.array_size = parse_size_t_field(kv, "array_size");
    config.array_seed = parse_unsigned_field(kv, "array_seed");
    config.kernel_a = parse_double_field(kv, "kernel_a");
    config.kernel_b = parse_double_field(kv, "kernel_b");
    config.kernel_c = parse_double_field(kv, "kernel_c");
    config.repeat_count = parse_int_field(kv, "repeat_count");

    // Phase 3 fields
    config.worker_count = parse_int_field(kv, "worker_count");

    // Phase 6 fields (optional, default ml/model.json)
    auto it = kv.find("ai_model_path");
    if (it != kv.end()) {
        config.ai_model_path = strip_quotes(it->second);
        if (config.ai_model_path.empty()) config.ai_model_path = "ml/model.json";
    } else {
        config.ai_model_path = "ml/model.json";
    }

    // Validation
    if (config.dataset_path.empty()) {
        throw std::runtime_error("Config field 'dataset_path' must not be empty");
    }
    if (config.default_access_pattern.empty()) {
        throw std::runtime_error("Config field 'default_access_pattern' must not be empty");
    }
    if (config.log_output_path.empty()) {
        throw std::runtime_error("Config field 'log_output_path' must not be empty");
    }
    if (config.array_size == 0) {
        throw std::runtime_error("Config field 'array_size' must be > 0");
    }
    if (config.task_count <= 0) {
        throw std::runtime_error("Config field 'task_count' must be > 0");
    }
    if (static_cast<size_t>(config.task_count) > config.array_size) {
        throw std::runtime_error("task_count must be <= array_size");
    }
    if (config.kernel_c == 0.0) {
        throw std::runtime_error("Config field 'kernel_c' must not be zero");
    }
    if (config.repeat_count <= 0) {
        throw std::runtime_error("Config field 'repeat_count' must be > 0");
    }
    if (config.default_stride <= 0) {
        throw std::runtime_error("Config field 'default_stride' must be > 0");
    }
    if (config.worker_count <= 0) {
        throw std::runtime_error("Config field 'worker_count' must be > 0");
    }

    std::string pat = config.default_access_pattern;
    std::transform(pat.begin(), pat.end(), pat.begin(), ::tolower);
    if (pat != "sequential" && pat != "strided" && pat != "random") {
        throw std::runtime_error("Invalid default_access_pattern: " + config.default_access_pattern +
                                 " (must be sequential, strided, or random)");
    }

    return config;
}
