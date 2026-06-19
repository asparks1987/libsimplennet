#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace simplennet {

enum class OutputType {
    Int,
    Float,
    Category,
};

struct TrainingOptions {
    std::vector<std::size_t> hidden_layers{32, 16};
    std::size_t epochs{50};
    double learning_rate{0.01};
    std::uint32_t seed{42};
};

using Matrix = std::vector<std::vector<double>>;

class SimplePredictor {
public:
    explicit SimplePredictor(OutputType output_type = OutputType::Float, TrainingOptions options = {});

    void fit(const Matrix& inputs, const std::vector<double>& targets);
    void fit_labels(const Matrix& inputs, const std::vector<std::string>& targets);
    void fit_text(const std::vector<std::string>& inputs, const std::vector<double>& targets, std::size_t width = 64);
    void fit_text_labels(const std::vector<std::string>& inputs, const std::vector<std::string>& targets, std::size_t width = 64);

    std::vector<double> predict_numbers(const Matrix& inputs) const;
    std::vector<int> predict_ints(const Matrix& inputs) const;
    std::vector<std::string> predict_labels(const Matrix& inputs) const;
    std::vector<double> predict_text_numbers(const std::vector<std::string>& inputs) const;
    std::vector<int> predict_text_ints(const std::vector<std::string>& inputs) const;
    std::vector<std::string> predict_text_labels(const std::vector<std::string>& inputs) const;

    void save(const std::filesystem::path& directory) const;
    static SimplePredictor load(const std::filesystem::path& directory);

    OutputType output_type() const;
    std::string output_type_name() const;
    bool is_fit() const;

    void set_training_options(const TrainingOptions& options);
    const TrainingOptions& training_options() const;

    void set_feature_names(std::vector<std::string> feature_names);
    const std::vector<std::string>& feature_names() const;
    const std::vector<std::string>& class_labels() const;

private:
    struct Layer {
        std::size_t input_size{};
        std::size_t output_size{};
        std::vector<double> weights;
        std::vector<double> biases;
    };

    OutputType output_type_;
    TrainingOptions options_;
    std::size_t input_dim_{0};
    std::string input_encoding_{"numeric"};
    std::size_t text_width_{64};
    std::vector<Layer> layers_;
    std::vector<std::string> feature_names_;
    std::vector<std::string> class_labels_;

    void validate_inputs(const Matrix& inputs) const;
    void initialize(std::size_t input_dim, std::size_t output_dim);
    Matrix forward(const std::vector<double>& input, std::vector<std::vector<double>>* activations = nullptr) const;
    std::vector<double> forward_values(const std::vector<double>& input) const;
    void train_numeric(const Matrix& inputs, const std::vector<double>& targets);
    void train_category(const Matrix& inputs, const std::vector<std::size_t>& targets);
    Matrix encode_text_inputs(const std::vector<std::string>& inputs, std::size_t width) const;
};

OutputType output_type_from_string(const std::string& value);
std::string output_type_to_string(OutputType value);

}  // namespace simplennet
