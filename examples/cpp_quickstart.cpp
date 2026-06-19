#include "simplennet/simple_predictor.hpp"

#include <iostream>

int main() {
    simplennet::TrainingOptions options;
    options.epochs = 30;
    options.learning_rate = 0.001;
    options.hidden_layers = {4};

    simplennet::SimplePredictor predictor(simplennet::OutputType::Int, options);
    predictor.fit({{0}, {1}, {2}, {3}}, {0, 2, 4, 6});

    const auto result = predictor.predict_ints({{4}});
    std::cout << result.front() << "\n";
    return 0;
}
