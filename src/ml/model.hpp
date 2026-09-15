#pragma once

#include "core/task.hpp"
#include <string>
#include <vector>

/// Simple Linear Regression model loader for burst-time prediction.
/// Loads ml/model.json exported by Phase 5, validates, and provides predict().
class BurstModel {
public:
    /// Load model from JSON file. Throws on malformed/invalid model.
    static BurstModel load(const std::string& path);

    /// Predict burst time for a task using pre-execution features.
    /// Features are constructed identically to training: see ml/ml_pipeline.py
    double predict(const Task& task) const;

    /// Predict from explicit feature values (in feature_order).
    double predict_from_features(const std::vector<double>& features) const;

    const std::vector<std::string>& features() const { return features_; }
    const std::vector<double>& coefficients() const { return coefficients_; }
    double intercept() const { return intercept_; }
    const std::string& model_type() const { return model_type_; }

private:
    std::string model_type_;
    std::vector<std::string> features_;
    std::vector<double> coefficients_;
    double intercept_{0.0};
};
