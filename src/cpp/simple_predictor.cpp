#include "simplennet/simple_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace simplennet {
namespace {

constexpr int kFormatVersion = 1;

double relu(double value) {
    return value > 0.0 ? value : 0.0;
}

double relu_derivative(double value) {
    return value > 0.0 ? 1.0 : 0.0;
}

std::vector<double> softmax(const std::vector<double>& values) {
    const double max_value = *std::max_element(values.begin(), values.end());
    std::vector<double> result(values.size(), 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = std::exp(values[i] - max_value);
        total += result[i];
    }
    if (total == 0.0) {
        return result;
    }
    for (double& value : result) {
        value /= total;
    }
    return result;
}

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned char ch : value) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

void add_text_feature(std::vector<double>& row, const std::string& token, double amount = 1.0) {
    const std::uint64_t hash = fnv1a(token);
    const std::size_t index = static_cast<std::size_t>(hash % row.size());
    const double sign = ((hash >> 8) & 1ull) == 0ull ? 1.0 : -1.0;
    row[index] += sign * amount;
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            out << '\\';
        }
        out << ch;
    }
    return out.str();
}

std::string string_array_json(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << "\"" << escape_json(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

std::string size_array_json(const std::vector<std::size_t>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string json_string_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Missing string field in model.json: " + key);
    }
    return match[1].str();
}

std::size_t json_size_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Missing numeric field in model.json: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

double json_double_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Missing numeric field in model.json: " + key);
    }
    return std::stod(match[1].str());
}

std::vector<std::string> json_string_array(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[(.*?)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return {};
    }

    std::vector<std::string> values;
    const std::string body = match[1].str();
    const std::regex item_pattern("\"([^\"]*)\"");
    for (std::sregex_iterator it(body.begin(), body.end(), item_pattern), end; it != end; ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

std::vector<std::size_t> json_size_array(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[(.*?)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return {};
    }

    std::vector<std::size_t> values;
    std::stringstream stream(match[1].str());
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(static_cast<std::size_t>(std::stoull(item)));
        }
    }
    return values;
}

template <typename T>
void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of weights.bin");
    }
    return value;
}

}  // namespace

SimplePredictor::SimplePredictor(OutputType output_type, TrainingOptions options)
    : output_type_(output_type), options_(std::move(options)) {}

void SimplePredictor::fit(const Matrix& inputs, const std::vector<double>& targets) {
    if (output_type_ == OutputType::Category) {
        throw std::runtime_error("Use fit_labels for categorical predictors");
    }
    if (inputs.size() != targets.size()) {
        throw std::runtime_error("inputs and targets must contain the same number of rows");
    }
    if (inputs.empty()) {
        throw std::runtime_error("inputs must not be empty");
    }
    validate_inputs(inputs);
    input_dim_ = inputs[0].size();
    input_encoding_ = "numeric";
    initialize(input_dim_, 1);
    train_numeric(inputs, targets);
}

void SimplePredictor::fit_labels(const Matrix& inputs, const std::vector<std::string>& targets) {
    if (output_type_ != OutputType::Category) {
        throw std::runtime_error("fit_labels requires a categorical predictor");
    }
    if (inputs.size() != targets.size()) {
        throw std::runtime_error("inputs and targets must contain the same number of rows");
    }
    if (inputs.empty()) {
        throw std::runtime_error("inputs must not be empty");
    }
    validate_inputs(inputs);

    class_labels_.clear();
    std::vector<std::size_t> encoded;
    for (const std::string& target : targets) {
        auto it = std::find(class_labels_.begin(), class_labels_.end(), target);
        if (it == class_labels_.end()) {
            class_labels_.push_back(target);
            encoded.push_back(class_labels_.size() - 1);
        } else {
            encoded.push_back(static_cast<std::size_t>(std::distance(class_labels_.begin(), it)));
        }
    }

    input_dim_ = inputs[0].size();
    input_encoding_ = "numeric";
    initialize(input_dim_, class_labels_.size());
    train_category(inputs, encoded);
}

void SimplePredictor::fit_text(const std::vector<std::string>& inputs, const std::vector<double>& targets, std::size_t width) {
    Matrix encoded = encode_text_inputs(inputs, width);
    text_width_ = width;
    input_encoding_ = "text";
    fit(encoded, targets);
    input_encoding_ = "text";
    text_width_ = width;
}

void SimplePredictor::fit_text_labels(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& targets,
    std::size_t width) {
    Matrix encoded = encode_text_inputs(inputs, width);
    text_width_ = width;
    input_encoding_ = "text";
    fit_labels(encoded, targets);
    input_encoding_ = "text";
    text_width_ = width;
}

std::vector<double> SimplePredictor::predict_numbers(const Matrix& inputs) const {
    if (!is_fit()) {
        throw std::runtime_error("predictor has not been fit");
    }
    validate_inputs(inputs);

    std::vector<double> results;
    results.reserve(inputs.size());
    for (const auto& row : inputs) {
        const auto output = forward_values(row);
        results.push_back(output.front());
    }
    return results;
}

std::vector<int> SimplePredictor::predict_ints(const Matrix& inputs) const {
    std::vector<double> numbers = predict_numbers(inputs);
    std::vector<int> results;
    results.reserve(numbers.size());
    for (double value : numbers) {
        results.push_back(static_cast<int>(std::llround(value)));
    }
    return results;
}

std::vector<std::string> SimplePredictor::predict_labels(const Matrix& inputs) const {
    if (output_type_ != OutputType::Category) {
        throw std::runtime_error("predict_labels requires a categorical predictor");
    }
    if (!is_fit()) {
        throw std::runtime_error("predictor has not been fit");
    }
    validate_inputs(inputs);

    std::vector<std::string> results;
    results.reserve(inputs.size());
    for (const auto& row : inputs) {
        const auto output = softmax(forward_values(row));
        const auto index = static_cast<std::size_t>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
        results.push_back(class_labels_.at(index));
    }
    return results;
}

std::vector<double> SimplePredictor::predict_text_numbers(const std::vector<std::string>& inputs) const {
    return predict_numbers(encode_text_inputs(inputs, text_width_));
}

std::vector<int> SimplePredictor::predict_text_ints(const std::vector<std::string>& inputs) const {
    return predict_ints(encode_text_inputs(inputs, text_width_));
}

std::vector<std::string> SimplePredictor::predict_text_labels(const std::vector<std::string>& inputs) const {
    return predict_labels(encode_text_inputs(inputs, text_width_));
}

void SimplePredictor::save(const std::filesystem::path& directory) const {
    if (!is_fit()) {
        throw std::runtime_error("predictor has not been fit");
    }
    std::filesystem::create_directories(directory);

    std::vector<std::size_t> layer_sizes;
    layer_sizes.push_back(input_dim_);
    for (const auto& layer : layers_) {
        layer_sizes.push_back(layer.output_size);
    }

    std::ofstream metadata(directory / "model.json");
    if (!metadata) {
        throw std::runtime_error("Could not write model.json");
    }
    metadata << "{\n";
    metadata << "  \"snet_format_version\": " << kFormatVersion << ",\n";
    metadata << "  \"output_type\": \"" << output_type_name() << "\",\n";
    metadata << "  \"input_dim\": " << input_dim_ << ",\n";
    metadata << "  \"input_encoding\": \"" << input_encoding_ << "\",\n";
    metadata << "  \"text_width\": " << text_width_ << ",\n";
    metadata << "  \"layer_sizes\": " << size_array_json(layer_sizes) << ",\n";
    metadata << "  \"hidden_layers\": " << size_array_json(options_.hidden_layers) << ",\n";
    metadata << "  \"epochs\": " << options_.epochs << ",\n";
    metadata << "  \"learning_rate\": " << options_.learning_rate << ",\n";
    metadata << "  \"seed\": " << options_.seed << ",\n";
    metadata << "  \"feature_names\": " << string_array_json(feature_names_) << ",\n";
    metadata << "  \"class_labels\": " << string_array_json(class_labels_) << "\n";
    metadata << "}\n";

    std::ofstream weights(directory / "weights.bin", std::ios::binary);
    if (!weights) {
        throw std::runtime_error("Could not write weights.bin");
    }
    write_value<std::uint64_t>(weights, static_cast<std::uint64_t>(layers_.size()));
    for (const Layer& layer : layers_) {
        write_value<std::uint64_t>(weights, static_cast<std::uint64_t>(layer.input_size));
        write_value<std::uint64_t>(weights, static_cast<std::uint64_t>(layer.output_size));
        weights.write(reinterpret_cast<const char*>(layer.weights.data()), static_cast<std::streamsize>(layer.weights.size() * sizeof(double)));
        weights.write(reinterpret_cast<const char*>(layer.biases.data()), static_cast<std::streamsize>(layer.biases.size() * sizeof(double)));
    }
}

SimplePredictor SimplePredictor::load(const std::filesystem::path& directory) {
    const std::string metadata = read_text(directory / "model.json");
    const std::size_t version = json_size_value(metadata, "snet_format_version");
    if (version != kFormatVersion) {
        throw std::runtime_error("Unsupported .snet format version");
    }

    TrainingOptions options;
    options.hidden_layers = json_size_array(metadata, "hidden_layers");
    options.epochs = json_size_value(metadata, "epochs");
    options.learning_rate = json_double_value(metadata, "learning_rate");
    options.seed = static_cast<std::uint32_t>(json_size_value(metadata, "seed"));

    SimplePredictor predictor(output_type_from_string(json_string_value(metadata, "output_type")), options);
    predictor.input_dim_ = json_size_value(metadata, "input_dim");
    try {
        predictor.input_encoding_ = json_string_value(metadata, "input_encoding");
        predictor.text_width_ = json_size_value(metadata, "text_width");
    } catch (const std::runtime_error&) {
        predictor.input_encoding_ = "numeric";
        predictor.text_width_ = 64;
    }
    predictor.feature_names_ = json_string_array(metadata, "feature_names");
    predictor.class_labels_ = json_string_array(metadata, "class_labels");

    std::ifstream weights(directory / "weights.bin", std::ios::binary);
    if (!weights) {
        throw std::runtime_error("Could not open weights.bin");
    }
    const std::uint64_t layer_count = read_value<std::uint64_t>(weights);
    predictor.layers_.clear();
    predictor.layers_.reserve(static_cast<std::size_t>(layer_count));
    for (std::uint64_t i = 0; i < layer_count; ++i) {
        Layer layer;
        layer.input_size = static_cast<std::size_t>(read_value<std::uint64_t>(weights));
        layer.output_size = static_cast<std::size_t>(read_value<std::uint64_t>(weights));
        layer.weights.resize(layer.input_size * layer.output_size);
        layer.biases.resize(layer.output_size);
        weights.read(reinterpret_cast<char*>(layer.weights.data()), static_cast<std::streamsize>(layer.weights.size() * sizeof(double)));
        weights.read(reinterpret_cast<char*>(layer.biases.data()), static_cast<std::streamsize>(layer.biases.size() * sizeof(double)));
        if (!weights) {
            throw std::runtime_error("Unexpected end of weights.bin");
        }
        predictor.layers_.push_back(std::move(layer));
    }
    return predictor;
}

OutputType SimplePredictor::output_type() const {
    return output_type_;
}

std::string SimplePredictor::output_type_name() const {
    return output_type_to_string(output_type_);
}

bool SimplePredictor::is_fit() const {
    return !layers_.empty();
}

void SimplePredictor::set_training_options(const TrainingOptions& options) {
    options_ = options;
}

const TrainingOptions& SimplePredictor::training_options() const {
    return options_;
}

void SimplePredictor::set_feature_names(std::vector<std::string> feature_names) {
    feature_names_ = std::move(feature_names);
}

const std::vector<std::string>& SimplePredictor::feature_names() const {
    return feature_names_;
}

const std::vector<std::string>& SimplePredictor::class_labels() const {
    return class_labels_;
}

void SimplePredictor::validate_inputs(const Matrix& inputs) const {
    if (inputs.empty()) {
        throw std::runtime_error("inputs must not be empty");
    }
    const std::size_t cols = inputs.front().size();
    if (cols == 0) {
        throw std::runtime_error("inputs must contain at least one feature");
    }
    for (const auto& row : inputs) {
        if (row.size() != cols) {
            throw std::runtime_error("all input rows must have the same number of features");
        }
    }
    if (is_fit() && cols != input_dim_) {
        throw std::runtime_error("input feature count does not match the fitted predictor");
    }
}

void SimplePredictor::initialize(std::size_t input_dim, std::size_t output_dim) {
    layers_.clear();
    std::vector<std::size_t> sizes;
    sizes.push_back(input_dim);
    for (std::size_t hidden : options_.hidden_layers) {
        if (hidden > 0) {
            sizes.push_back(hidden);
        }
    }
    sizes.push_back(output_dim);

    std::mt19937 rng(options_.seed);
    for (std::size_t i = 1; i < sizes.size(); ++i) {
        Layer layer;
        layer.input_size = sizes[i - 1];
        layer.output_size = sizes[i];
        layer.weights.resize(layer.input_size * layer.output_size);
        layer.biases.assign(layer.output_size, 0.0);

        const double limit = std::sqrt(6.0 / static_cast<double>(layer.input_size + layer.output_size));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (double& weight : layer.weights) {
            weight = dist(rng);
        }
        layers_.push_back(std::move(layer));
    }
}

Matrix SimplePredictor::forward(const std::vector<double>& input, std::vector<std::vector<double>>* activations) const {
    std::vector<double> current = input;
    if (activations) {
        activations->clear();
        activations->push_back(current);
    }

    Matrix pre_activations;
    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        const Layer& layer = layers_[layer_index];
        std::vector<double> z(layer.output_size, 0.0);
        for (std::size_t out = 0; out < layer.output_size; ++out) {
            double value = layer.biases[out];
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                value += current[in] * layer.weights[in * layer.output_size + out];
            }
            z[out] = value;
        }
        pre_activations.push_back(z);
        if (layer_index + 1 < layers_.size()) {
            for (double& value : z) {
                value = relu(value);
            }
        }
        current = z;
        if (activations) {
            activations->push_back(current);
        }
    }
    return pre_activations;
}

std::vector<double> SimplePredictor::forward_values(const std::vector<double>& input) const {
    std::vector<std::vector<double>> activations;
    forward(input, &activations);
    return activations.back();
}

void SimplePredictor::train_numeric(const Matrix& inputs, const std::vector<double>& targets) {
    for (std::size_t epoch = 0; epoch < options_.epochs; ++epoch) {
        std::vector<std::vector<double>> grad_weights;
        std::vector<std::vector<double>> grad_biases;
        for (const Layer& layer : layers_) {
            grad_weights.emplace_back(layer.weights.size(), 0.0);
            grad_biases.emplace_back(layer.biases.size(), 0.0);
        }

        for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
            std::vector<std::vector<double>> activations;
            const Matrix pre_activations = forward(inputs[sample], &activations);
            std::vector<double> delta{(activations.back()[0] - targets[sample]) / static_cast<double>(inputs.size())};

            for (std::size_t layer_rev = layers_.size(); layer_rev-- > 0;) {
                const Layer& layer = layers_[layer_rev];
                const auto& prev_activation = activations[layer_rev];
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    for (std::size_t out = 0; out < layer.output_size; ++out) {
                        grad_weights[layer_rev][in * layer.output_size + out] += prev_activation[in] * delta[out];
                    }
                }
                for (std::size_t out = 0; out < layer.output_size; ++out) {
                    grad_biases[layer_rev][out] += delta[out];
                }

                if (layer_rev > 0) {
                    std::vector<double> next_delta(layer.input_size, 0.0);
                    for (std::size_t in = 0; in < layer.input_size; ++in) {
                        for (std::size_t out = 0; out < layer.output_size; ++out) {
                            next_delta[in] += layer.weights[in * layer.output_size + out] * delta[out];
                        }
                        next_delta[in] *= relu_derivative(pre_activations[layer_rev - 1][in]);
                    }
                    delta = std::move(next_delta);
                }
            }
        }

        for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
            for (std::size_t i = 0; i < layers_[layer_index].weights.size(); ++i) {
                layers_[layer_index].weights[i] -= options_.learning_rate * grad_weights[layer_index][i];
            }
            for (std::size_t i = 0; i < layers_[layer_index].biases.size(); ++i) {
                layers_[layer_index].biases[i] -= options_.learning_rate * grad_biases[layer_index][i];
            }
        }
    }
}

void SimplePredictor::train_category(const Matrix& inputs, const std::vector<std::size_t>& targets) {
    for (std::size_t epoch = 0; epoch < options_.epochs; ++epoch) {
        std::vector<std::vector<double>> grad_weights;
        std::vector<std::vector<double>> grad_biases;
        for (const Layer& layer : layers_) {
            grad_weights.emplace_back(layer.weights.size(), 0.0);
            grad_biases.emplace_back(layer.biases.size(), 0.0);
        }

        for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
            std::vector<std::vector<double>> activations;
            const Matrix pre_activations = forward(inputs[sample], &activations);
            std::vector<double> delta = softmax(activations.back());
            delta[targets[sample]] -= 1.0;
            for (double& value : delta) {
                value /= static_cast<double>(inputs.size());
            }

            for (std::size_t layer_rev = layers_.size(); layer_rev-- > 0;) {
                const Layer& layer = layers_[layer_rev];
                const auto& prev_activation = activations[layer_rev];
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    for (std::size_t out = 0; out < layer.output_size; ++out) {
                        grad_weights[layer_rev][in * layer.output_size + out] += prev_activation[in] * delta[out];
                    }
                }
                for (std::size_t out = 0; out < layer.output_size; ++out) {
                    grad_biases[layer_rev][out] += delta[out];
                }

                if (layer_rev > 0) {
                    std::vector<double> next_delta(layer.input_size, 0.0);
                    for (std::size_t in = 0; in < layer.input_size; ++in) {
                        for (std::size_t out = 0; out < layer.output_size; ++out) {
                            next_delta[in] += layer.weights[in * layer.output_size + out] * delta[out];
                        }
                        next_delta[in] *= relu_derivative(pre_activations[layer_rev - 1][in]);
                    }
                    delta = std::move(next_delta);
                }
            }
        }

        for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
            for (std::size_t i = 0; i < layers_[layer_index].weights.size(); ++i) {
                layers_[layer_index].weights[i] -= options_.learning_rate * grad_weights[layer_index][i];
            }
            for (std::size_t i = 0; i < layers_[layer_index].biases.size(); ++i) {
                layers_[layer_index].biases[i] -= options_.learning_rate * grad_biases[layer_index][i];
            }
        }
    }
}

Matrix SimplePredictor::encode_text_inputs(const std::vector<std::string>& inputs, std::size_t width) const {
    if (inputs.empty()) {
        throw std::runtime_error("text inputs must not be empty");
    }
    if (width == 0) {
        throw std::runtime_error("text encoder width must be greater than zero");
    }

    Matrix matrix;
    matrix.reserve(inputs.size());
    for (const std::string& input : inputs) {
        std::vector<double> row(width, 0.0);
        add_text_feature(row, "text:" + input);
        add_text_feature(row, "length", static_cast<double>(input.size()) / 128.0);
        for (std::size_t i = 0; i < input.size(); ++i) {
            add_text_feature(row, "char:" + input.substr(i, 1), 0.25);
            if (i + 3 <= input.size()) {
                add_text_feature(row, "tri:" + input.substr(i, 3), 0.5);
            }
        }
        matrix.push_back(std::move(row));
    }
    return matrix;
}

OutputType output_type_from_string(const std::string& value) {
    if (value == "int" || value == "integer") {
        return OutputType::Int;
    }
    if (value == "float" || value == "double") {
        return OutputType::Float;
    }
    if (value == "category" || value == "string" || value == "label") {
        return OutputType::Category;
    }
    throw std::runtime_error("Unknown output type: " + value);
}

std::string output_type_to_string(OutputType value) {
    switch (value) {
        case OutputType::Int:
            return "int";
        case OutputType::Float:
            return "float";
        case OutputType::Category:
            return "category";
    }
    throw std::runtime_error("Unknown output type");
}

}  // namespace simplennet
