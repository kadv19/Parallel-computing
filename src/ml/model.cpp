#include "ml/model.hpp"

#include "benchmark/metrics.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open model file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
    return s.substr(a, b - a);
}

// Extract string value for key: "key" : "value"
std::string extract_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = json.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed key: " + key);
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') throw std::runtime_error("Expected string for key: " + key);
    ++pos;
    size_t end = pos;
    while (end < json.size() && json[end] != '"') {
        if (json[end] == '\\' && end + 1 < json.size()) end += 2;
        else ++end;
    }
    if (end >= json.size()) throw std::runtime_error("Unterminated string for key: " + key);
    return json.substr(pos, end - pos);
}

// Extract double for key: "key" : number
double extract_double(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = json.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed key: " + key);
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    size_t end = pos;
    while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end]=='.' || json[end]=='-' || json[end]=='+' || json[end]=='e' || json[end]=='E')) ++end;
    if (end == pos) throw std::runtime_error("Expected number for key: " + key);
    return std::stod(json.substr(pos, end - pos));
}

// Extract array of strings: "key": ["a","b"]
std::vector<std::string> extract_string_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = json.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed key: " + key);
    pos = json.find('[', pos);
    if (pos == std::string::npos) throw std::runtime_error("Expected array for key: " + key);
    size_t end = json.find(']', pos);
    if (end == std::string::npos) throw std::runtime_error("Unterminated array for key: " + key);
    std::string inner = json.substr(pos + 1, end - pos - 1);
    std::vector<std::string> result;
    size_t cur = 0;
    while (cur < inner.size()) {
        while (cur < inner.size() && (std::isspace(static_cast<unsigned char>(inner[cur])) || inner[cur]==',')) ++cur;
        if (cur >= inner.size()) break;
        if (inner[cur] != '"') {
            // skip non-string? should not happen
            ++cur;
            continue;
        }
        ++cur;
        size_t e = cur;
        while (e < inner.size() && inner[e] != '"') {
            if (inner[e] == '\\' && e + 1 < inner.size()) e += 2;
            else ++e;
        }
        if (e >= inner.size()) throw std::runtime_error("Unterminated string in array");
        result.push_back(inner.substr(cur, e - cur));
        cur = e + 1;
    }
    return result;
}

// Extract array of doubles: "key": [1.0, 2.0]
std::vector<double> extract_double_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = json.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed key: " + key);
    pos = json.find('[', pos);
    if (pos == std::string::npos) throw std::runtime_error("Expected array for key: " + key);
    size_t end = json.find(']', pos);
    if (end == std::string::npos) throw std::runtime_error("Unterminated array for key: " + key);
    std::string inner = json.substr(pos + 1, end - pos - 1);
    std::vector<double> result;
    std::istringstream iss(inner);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::string t = trim(token);
        if (t.empty()) continue;
        result.push_back(std::stod(t));
    }
    return result;
}

} // namespace

BurstModel BurstModel::load(const std::string& path) {
    std::string json = read_file(path);
    BurstModel m;
    m.model_type_ = extract_string(json, "model_type");
    if (m.model_type_ != "linear_regression") {
        throw std::runtime_error("Unsupported model_type: " + m.model_type_ + " (expected linear_regression)");
    }
    m.features_ = extract_string_array(json, "features");
    // Prefer coefficients, fallback to checking alternative key
    m.coefficients_ = extract_double_array(json, "coefficients");
    m.intercept_ = extract_double(json, "intercept");

    if (m.features_.empty()) throw std::runtime_error("Model has no features");
    if (m.coefficients_.size() != m.features_.size()) {
        throw std::runtime_error("Coefficient count mismatch: features " + std::to_string(m.features_.size()) +
                                 " vs coefficients " + std::to_string(m.coefficients_.size()));
    }
    return m;
}

double BurstModel::predict_from_features(const std::vector<double>& features) const {
    if (features.size() != coefficients_.size()) {
        throw std::runtime_error("Feature count mismatch in predict");
    }
    double result = intercept_;
    for (size_t i = 0; i < features.size(); ++i) {
        result += coefficients_[i] * features[i];
    }
    // Clamp negative predictions to small positive (execution time cannot be negative)
    if (result < 1e-9) result = 1e-9;
    return result;
}

double BurstModel::predict(const Task& task) const {
    // Construct features in exact order as training
    // Feature order from Phase 5: element_count, working_set_bytes, access_pattern_encoded, stride, repeat_count, operation_count, locality_score
    // Must match ml/ml_pipeline.py FEATURES
    std::vector<double> feats;
    feats.reserve(features_.size());
    for (const auto& fname : features_) {
        if (fname == "element_count") {
            feats.push_back(static_cast<double>(task.element_count));
        } else if (fname == "working_set_bytes") {
            feats.push_back(static_cast<double>(task.element_count * sizeof(double)));
        } else if (fname == "access_pattern_encoded") {
            int enc = 0;
            if (task.access_pattern == AccessPattern::Sequential) enc = 0;
            else if (task.access_pattern == AccessPattern::Strided) enc = 1;
            else if (task.access_pattern == AccessPattern::Random) enc = 2;
            feats.push_back(static_cast<double>(enc));
        } else if (fname == "stride") {
            feats.push_back(static_cast<double>(task.stride));
        } else if (fname == "repeat_count") {
            feats.push_back(static_cast<double>(task.operation_count));
        } else if (fname == "operation_count") {
            feats.push_back(static_cast<double>(task.operation_count));
        } else if (fname == "locality_score") {
            double ls = locality_score(task.access_pattern, task.stride);
            feats.push_back(ls);
        } else if (fname == "array_size") {
            // Not in default but handle if present (fallback to element_count related)
            feats.push_back(static_cast<double>(task.element_count));
        } else {
            throw std::runtime_error("Unknown feature: " + fname);
        }
    }
    return predict_from_features(feats);
}
