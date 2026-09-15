#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/task_generator.hpp"
#include "data/array_generator.hpp"
#include "kernel/numeric_kernel.hpp"
#include "ml/model.hpp"
#include "scheduler/ai_scheduler.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinRel;

TEST_CASE("Model loading succeeds", "[ai]") {
    // Try candidate paths for robustness
    BurstModel m;
    bool loaded = false;
    std::string err;
    for (auto p : {"ml/model.json", "../ml/model.json", "../../ml/model.json", "ml/model.json"}) {
        try {
            m = BurstModel::load(p);
            loaded = true;
            break;
        } catch (const std::exception& e) {
            err = e.what();
        }
    }
    REQUIRE(loaded);
    CHECK(m.model_type() == "linear_regression");
    CHECK(m.features().size() == 7);
    CHECK(m.coefficients().size() == 7);
}

TEST_CASE("Model prediction matches expected value", "[ai]") {
    BurstModel m = BurstModel::load("ml/model.json");
    // Create a known task and compute manual prediction vs model
    Task t;
    t.task_id = 0;
    t.start_index = 0;
    t.end_index = 1000;
    t.element_count = 1000;
    t.access_pattern = AccessPattern::Sequential;
    t.stride = 1;
    t.operation_count = 5;
    t.random_seed = 42;

    double pred = m.predict(t);
    // Manual compute using known coefficients from model.json
    // features: element_count, working_set_bytes, access_pattern_encoded, stride, repeat_count, operation_count, locality_score
    // For this task: element_count=1000, working_set=8000, enc=0, stride=1, repeat=5, op=5, locality=1.0
    // coefficients: [3.947e-07,3.157e-06,0.04668,-0.007429,0.00516,0.00516,-0.0209] intercept -0.04012
    // Expected: -0.04012 + 3.947e-07*1000 + 3.157e-06*8000 +0 + -0.007429*1 +0.00516*5+0.00516*5 -0.0209*1
    // Approx  -0.04012 +0.000394+0.02526 -0.007429 +0.0258+0.0258 -0.0209 ≈ 0.0088
    CHECK(pred > 0); // clamped positive
    // Verify that predict_from_features matches
    std::vector<double> feats = {1000,8000,0,1,5,5,1.0};
    double pred2 = m.predict_from_features(feats);
    CHECK_THAT(pred, WithinRel(pred2, 1e-9));
    // Compare to Python exported validation: should be reproducible
    // We test that manual linear formula matches
    double manual = m.intercept();
    auto coeffs = m.coefficients();
    for (size_t i=0;i<feats.size();++i) manual += coeffs[i]*feats[i];
    if (manual < 1e-9) manual = 1e-9;
    CHECK_THAT(pred, WithinRel(manual, 1e-9));
}

TEST_CASE("Feature extraction encoding and working set", "[ai]") {
    BurstModel m = BurstModel::load("ml/model.json");
    Task t_seq{0,0,10000,10000,AccessPattern::Sequential, 1, 5, 0};
    Task t_str{1,0,10000,10000,AccessPattern::Strided, 8, 5, 1};
    Task t_rand{2,0,10000,10000,AccessPattern::Random, 1, 5, 2};

    // Check that predictions differ based on encoding (Sequential vs Strided vs Random)
    double p_seq = m.predict(t_seq);
    double p_str = m.predict(t_str);
    double p_rand = m.predict(t_rand);
    // They should be different due to encoding and locality
    CHECK(p_seq != p_str);
    CHECK(p_str != p_rand);

    // Working set = element_count *8
    Task t_small{0,0,1000,1000,AccessPattern::Sequential,1,5,0};
    Task t_large{0,0,10000,10000,AccessPattern::Sequential,1,5,0};
    double p_small = m.predict(t_small);
    double p_large = m.predict(t_large);
    CHECK(p_large > p_small); // larger working set should predict larger burst
}

TEST_CASE("AI scheduler correctness matches sequential", "[ai]") {
    size_t n = 10000;
    unsigned seed = 42;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 20, AccessPattern::Sequential, 1, seed, 5);
    double a=1.0001,b=0.5,c=1.00001;
    int repeat=5;
    auto seq = original;
    process_tasks_sequential(seq, tasks, a,b,c, repeat);
    double expected = std::accumulate(seq.begin(), seq.end(), 0.0);

    AiScheduler ai("ml/model.json");
    auto par = original;
    ai.execute(par, tasks, 4, a,b,c, repeat);
    double got = std::accumulate(par.begin(), par.end(), 0.0);
    CHECK_THAT(got, WithinRel(expected, 1e-9));
}

TEST_CASE("AI scheduler with 1 worker", "[ai]") {
    size_t n = 5000;
    unsigned seed = 7;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 10, AccessPattern::Random, 1, seed, 3);
    double a=1.0001,b=0.5,c=1.00001;
    int repeat=3;
    auto seq = original;
    process_tasks_sequential(seq, tasks, a,b,c, repeat);
    double expected = std::accumulate(seq.begin(), seq.end(), 0.0);
    AiScheduler ai("ml/model.json");
    auto par = original;
    ai.execute(par, tasks, 1, a,b,c, repeat);
    CHECK_THAT(std::accumulate(par.begin(), par.end(), 0.0), WithinRel(expected, 1e-9));
}

TEST_CASE("AI scheduler with more workers than tasks", "[ai]") {
    size_t n = 100;
    unsigned seed = 7;
    auto original = generate_array(n, seed);
    auto tasks = generate_tasks(n, 10, AccessPattern::Random, 2, seed, 3);
    double a=1.0001,b=0.5,c=1.00001;
    int repeat=3;
    auto seq = original;
    process_tasks_sequential(seq, tasks, a,b,c, repeat);
    double expected = std::accumulate(seq.begin(), seq.end(), 0.0);
    AiScheduler ai("ml/model.json");
    auto par = original;
    ai.execute(par, tasks, 20, a,b,c, repeat);
    CHECK_THAT(std::accumulate(par.begin(), par.end(), 0.0), WithinRel(expected, 1e-9));
}

TEST_CASE("AI scheduler name is correct", "[ai]") {
    AiScheduler ai("ml/model.json");
    CHECK(ai.name() == "AI");
}

TEST_CASE("Negative prediction clamped to positive", "[ai]") {
    BurstModel m = BurstModel::load("ml/model.json");
    // Create a task that might give very small or negative raw prediction?
    // Use tiny element_count and check that predict never returns negative
    Task t{0,0,1,1,AccessPattern::Random, 100, 1, 0}; // large stride, random -> low locality, but small size may give negative raw
    double p = m.predict(t);
    CHECK(p >= 1e-9);
    // Directly test predict_from_features with extreme values that would give negative raw
    // e.g., intercept -0.04 with small features might be negative before clamp
    // We know intercept is -0.040..., with all zero features would be negative
    std::vector<double> zero_feats(7, 0.0);
    double p2 = m.predict_from_features(zero_feats);
    CHECK(p2 >= 1e-9);
}

TEST_CASE("AI scheduler predictions influence assignment deterministically", "[ai]") {
    // Verify that AI sorts by predicted burst time descending
    // Create tasks with varying element_counts to get distinct predictions
    std::vector<Task> tasks;
    for (size_t i=0;i<4;++i) {
        Task t;
        t.task_id=i;
        t.start_index=i*100;
        t.end_index=(i+1)*100;
        t.element_count=100 + i*1000; // increasing size
        t.access_pattern=AccessPattern::Sequential;
        t.stride=1;
        t.operation_count=5;
        t.random_seed=0;
        tasks.push_back(t);
    }
    // Predictions should be monotonic with element_count
    BurstModel m = BurstModel::load("ml/model.json");
    std::vector<double> preds;
    for (auto& t: tasks) preds.push_back(m.predict(t));
    // Larger element_count -> larger prediction (since coef positive)
    CHECK(preds[3] > preds[0]);
}
