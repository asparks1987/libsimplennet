#include "simplennet/simple_predictor.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void test_int_prediction() {
    simplennet::TrainingOptions options;
    options.epochs = 30;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
    predictor.fit({{0}, {1}, {2}, {3}}, {0, 2, 4, 6});
    const auto result = predictor.predict_ints({{4}});
    assert(result.size() == 1);
}

void test_float_prediction() {
    simplennet::TrainingOptions options;
    options.epochs = 30;
    options.learning_rate = 0.000001;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Float, options);
    predictor.fit({{600}, {900}, {1200}}, {1200, 1800, 2400});
    const auto result = predictor.predict_numbers({{1000}});
    assert(result.size() == 1);
}

void test_category_prediction() {
    simplennet::TrainingOptions options;
    options.epochs = 20;
    options.learning_rate = 0.05;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Category, options);
    predictor.fit_labels({{0.1, 0.9}, {0.8, 0.2}, {0.2, 0.7}, {0.9, 0.1}}, {"cold", "hot", "cold", "hot"});
    const auto result = predictor.predict_labels({{0.85, 0.15}});
    assert(result.size() == 1);
    assert(result[0] == "cold" || result[0] == "hot");
}

void test_save_load_round_trip() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "libsimplennet-test-model.snet";
    std::filesystem::remove_all(path);

    simplennet::TrainingOptions options;
    options.epochs = 10;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
    predictor.fit({{0}, {1}, {2}, {3}}, {0, 2, 4, 6});
    predictor.save(path);

    const auto loaded = simplennet::SimplePredictor::load(path);
    const auto result = loaded.predict_ints({{4}});
    assert(result.size() == 1);
    assert(std::filesystem::exists(path / "model.json"));
    assert(std::filesystem::exists(path / "weights.bin"));

    std::filesystem::remove_all(path);
}

void test_text_int_prediction_and_save_load() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "libsimplennet-text-model.snet";
    std::filesystem::remove_all(path);

    simplennet::TrainingOptions options;
    options.epochs = 10;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};
    options.seed = 123;

    simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
    predictor.fit_text(
        {
            R"({"kind":"cart","items":1})",
            R"({"kind":"cart","items":2})",
            R"({"kind":"cart","items":3})",
        },
        {2, 4, 6},
        32);

    const auto result = predictor.predict_text_ints({R"({"kind":"cart","items":4})"});
    assert(result.size() == 1);

    predictor.save(path);
    const auto loaded = simplennet::SimplePredictor::load(path);
    const auto reloaded = loaded.predict_text_ints({R"({"kind":"cart","items":4})"});
    assert(reloaded.size() == 1);

    std::filesystem::remove_all(path);
}

void test_text_category_prediction() {
    simplennet::TrainingOptions options;
    options.epochs = 20;
    options.learning_rate = 0.05;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Category, options);
    predictor.fit_text_labels(
        {
            R"({"user":"new","plan":"free"})",
            R"({"user":"team","plan":"enterprise"})",
            R"({"user":"new","plan":"trial"})",
            R"({"user":"team","plan":"business"})",
        },
        {"self_serve", "sales", "self_serve", "sales"},
        32);
    const auto result = predictor.predict_text_labels({R"({"user":"team","plan":"enterprise"})"});
    assert(result.size() == 1);
    assert(result[0] == "self_serve" || result[0] == "sales");
}

void test_deterministic_training() {
    simplennet::TrainingOptions options;
    options.epochs = 10;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};
    options.seed = 123;

    simplennet::SimplePredictor first(simplennet::OutputType::Float, options);
    simplennet::SimplePredictor second(simplennet::OutputType::Float, options);
    first.fit({{0}, {1}, {2}}, {0, 1, 2});
    second.fit({{0}, {1}, {2}}, {0, 1, 2});

    assert(first.predict_numbers({{3}})[0] == second.predict_numbers({{3}})[0]);
}

void test_invalid_dimensions() {
    bool failed = false;
    simplennet::SimplePredictor predictor(simplennet::OutputType::Float);
    try {
        predictor.fit({{1}, {1, 2}}, {1, 2});
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
}

void test_missing_model_errors() {
    bool failed = false;
    try {
        (void)simplennet::SimplePredictor::load(std::filesystem::temp_directory_path() / "libsimplennet-missing-model.snet");
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
}

}  // namespace

int main() {
    test_int_prediction();
    test_float_prediction();
    test_category_prediction();
    test_save_load_round_trip();
    test_text_int_prediction_and_save_load();
    test_text_category_prediction();
    test_deterministic_training();
    test_invalid_dimensions();
    test_missing_model_errors();
    std::cout << "simplennet C++ core tests passed\n";
    return 0;
}
