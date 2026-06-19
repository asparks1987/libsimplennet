#include "simplennet/c_api.h"

#include "simplennet/simple_predictor.hpp"

#include <exception>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

struct snet_predictor {
    simplennet::SimplePredictor model;
};

namespace {

thread_local std::string last_error;

int fail(const std::exception& exc) {
    last_error = exc.what();
    return -1;
}

simplennet::OutputType convert_type(snet_output_type_t output_type) {
    switch (output_type) {
        case SNET_OUTPUT_INT:
            return simplennet::OutputType::Int;
        case SNET_OUTPUT_FLOAT:
            return simplennet::OutputType::Float;
        case SNET_OUTPUT_CATEGORY:
            return simplennet::OutputType::Category;
    }
    throw std::runtime_error("Unknown output type");
}

simplennet::Matrix matrix_from_row_major(const double* data, size_t rows, size_t cols) {
    if (data == nullptr) {
        throw std::runtime_error("input data pointer must not be null");
    }
    simplennet::Matrix matrix(rows, std::vector<double>(cols, 0.0));
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            matrix[row][col] = data[row * cols + col];
        }
    }
    return matrix;
}

std::vector<std::string> strings_from_c_array(const char* const* values, size_t count, const char* name) {
    if (values == nullptr) {
        throw std::runtime_error(std::string(name) + " pointer must not be null");
    }
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(values[i] == nullptr ? "" : values[i]);
    }
    return result;
}

}  // namespace

int snet_create(snet_output_type_t output_type, snet_predictor_t** out) {
    try {
        if (out == nullptr) {
            throw std::runtime_error("out pointer must not be null");
        }
        *out = new snet_predictor{simplennet::SimplePredictor(convert_type(output_type))};
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

void snet_destroy(snet_predictor_t* predictor) {
    delete predictor;
}

snet_output_type_t snet_get_output_type(const snet_predictor_t* predictor) {
    if (predictor == nullptr) {
        return SNET_OUTPUT_FLOAT;
    }
    switch (predictor->model.output_type()) {
        case simplennet::OutputType::Int:
            return SNET_OUTPUT_INT;
        case simplennet::OutputType::Float:
            return SNET_OUTPUT_FLOAT;
        case simplennet::OutputType::Category:
            return SNET_OUTPUT_CATEGORY;
    }
    return SNET_OUTPUT_FLOAT;
}

const char* snet_output_type_name(const snet_predictor_t* predictor) {
    if (predictor == nullptr) {
        last_error = "predictor must not be null";
        return nullptr;
    }
    static thread_local std::string name;
    name = predictor->model.output_type_name();
    return name.c_str();
}

int snet_fit_numeric(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const double* targets,
    size_t target_count,
    size_t epochs,
    double learning_rate) {
    try {
        if (predictor == nullptr || targets == nullptr) {
            throw std::runtime_error("predictor and targets must not be null");
        }
        if (rows != target_count) {
            throw std::runtime_error("target count must match row count");
        }
        auto options = predictor->model.training_options();
        options.epochs = epochs;
        options.learning_rate = learning_rate;
        predictor->model.set_training_options(options);
        predictor->model.fit(matrix_from_row_major(row_major_inputs, rows, cols), std::vector<double>(targets, targets + target_count));
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_fit_numeric_bits(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const double* targets,
    size_t target_count,
    size_t epochs,
    unsigned long long learning_rate_bits) {
    double learning_rate = 0.0;
    std::memcpy(&learning_rate, &learning_rate_bits, sizeof(double));
    return snet_fit_numeric(predictor, row_major_inputs, rows, cols, targets, target_count, epochs, learning_rate);
}

int snet_fit_labels(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const char* const* labels,
    size_t label_count,
    size_t epochs,
    double learning_rate) {
    try {
        if (predictor == nullptr || labels == nullptr) {
            throw std::runtime_error("predictor and labels must not be null");
        }
        if (rows != label_count) {
            throw std::runtime_error("label count must match row count");
        }
        std::vector<std::string> targets;
        targets.reserve(label_count);
        for (size_t i = 0; i < label_count; ++i) {
            targets.emplace_back(labels[i] == nullptr ? "" : labels[i]);
        }
        auto options = predictor->model.training_options();
        options.epochs = epochs;
        options.learning_rate = learning_rate;
        predictor->model.set_training_options(options);
        predictor->model.fit_labels(matrix_from_row_major(row_major_inputs, rows, cols), targets);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_fit_labels_bits(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const char* const* labels,
    size_t label_count,
    size_t epochs,
    unsigned long long learning_rate_bits) {
    double learning_rate = 0.0;
    std::memcpy(&learning_rate, &learning_rate_bits, sizeof(double));
    return snet_fit_labels(predictor, row_major_inputs, rows, cols, labels, label_count, epochs, learning_rate);
}

int snet_fit_text_numeric_bits(
    snet_predictor_t* predictor,
    const char* const* inputs,
    size_t input_count,
    const double* targets,
    size_t target_count,
    size_t width,
    size_t epochs,
    unsigned long long learning_rate_bits) {
    try {
        if (predictor == nullptr || targets == nullptr) {
            throw std::runtime_error("predictor and targets must not be null");
        }
        if (input_count != target_count) {
            throw std::runtime_error("target count must match input count");
        }
        double learning_rate = 0.0;
        std::memcpy(&learning_rate, &learning_rate_bits, sizeof(double));
        auto options = predictor->model.training_options();
        options.epochs = epochs;
        options.learning_rate = learning_rate;
        predictor->model.set_training_options(options);
        predictor->model.fit_text(strings_from_c_array(inputs, input_count, "inputs"), std::vector<double>(targets, targets + target_count), width);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_fit_text_labels_bits(
    snet_predictor_t* predictor,
    const char* const* inputs,
    size_t input_count,
    const char* const* labels,
    size_t label_count,
    size_t width,
    size_t epochs,
    unsigned long long learning_rate_bits) {
    try {
        if (predictor == nullptr) {
            throw std::runtime_error("predictor must not be null");
        }
        if (input_count != label_count) {
            throw std::runtime_error("label count must match input count");
        }
        double learning_rate = 0.0;
        std::memcpy(&learning_rate, &learning_rate_bits, sizeof(double));
        auto options = predictor->model.training_options();
        options.epochs = epochs;
        options.learning_rate = learning_rate;
        predictor->model.set_training_options(options);
        predictor->model.fit_text_labels(
            strings_from_c_array(inputs, input_count, "inputs"),
            strings_from_c_array(labels, label_count, "labels"),
            width);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_numeric(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    double* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < rows) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto predictions = predictor->model.predict_numbers(matrix_from_row_major(row_major_inputs, rows, cols));
        for (size_t i = 0; i < predictions.size(); ++i) {
            outputs[i] = predictions[i];
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_ints(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    long long* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < rows) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto predictions = predictor->model.predict_ints(matrix_from_row_major(row_major_inputs, rows, cols));
        for (size_t i = 0; i < predictions.size(); ++i) {
            outputs[i] = static_cast<long long>(predictions[i]);
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_class_indices(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    size_t* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < rows) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto labels = predictor->model.predict_labels(matrix_from_row_major(row_major_inputs, rows, cols));
        for (size_t i = 0; i < labels.size(); ++i) {
            const auto& class_labels = predictor->model.class_labels();
            const auto it = std::find(class_labels.begin(), class_labels.end(), labels[i]);
            if (it == class_labels.end()) {
                throw std::runtime_error("predicted label not found in class metadata");
            }
            outputs[i] = static_cast<size_t>(std::distance(class_labels.begin(), it));
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_text_numeric(
    const snet_predictor_t* predictor,
    const char* const* inputs,
    size_t input_count,
    double* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < input_count) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto predictions = predictor->model.predict_text_numbers(strings_from_c_array(inputs, input_count, "inputs"));
        for (size_t i = 0; i < predictions.size(); ++i) {
            outputs[i] = predictions[i];
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_text_ints(
    const snet_predictor_t* predictor,
    const char* const* inputs,
    size_t input_count,
    long long* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < input_count) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto predictions = predictor->model.predict_text_ints(strings_from_c_array(inputs, input_count, "inputs"));
        for (size_t i = 0; i < predictions.size(); ++i) {
            outputs[i] = static_cast<long long>(predictions[i]);
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_predict_text_class_indices(
    const snet_predictor_t* predictor,
    const char* const* inputs,
    size_t input_count,
    size_t* outputs,
    size_t output_count) {
    try {
        if (predictor == nullptr || outputs == nullptr) {
            throw std::runtime_error("predictor and outputs must not be null");
        }
        if (output_count < input_count) {
            throw std::runtime_error("output buffer is too small");
        }
        const auto labels = predictor->model.predict_text_labels(strings_from_c_array(inputs, input_count, "inputs"));
        for (size_t i = 0; i < labels.size(); ++i) {
            const auto& class_labels = predictor->model.class_labels();
            const auto it = std::find(class_labels.begin(), class_labels.end(), labels[i]);
            if (it == class_labels.end()) {
                throw std::runtime_error("predicted label not found in class metadata");
            }
            outputs[i] = static_cast<size_t>(std::distance(class_labels.begin(), it));
        }
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

size_t snet_class_label_count(const snet_predictor_t* predictor) {
    return predictor == nullptr ? 0 : predictor->model.class_labels().size();
}

int snet_class_label_copy(const snet_predictor_t* predictor, size_t index, char* buffer, size_t buffer_size) {
    try {
        if (predictor == nullptr || buffer == nullptr) {
            throw std::runtime_error("predictor and buffer must not be null");
        }
        const auto& labels = predictor->model.class_labels();
        if (index >= labels.size()) {
            throw std::runtime_error("class label index out of range");
        }
        const std::string& label = labels[index];
        if (buffer_size == 0 || label.size() + 1 > buffer_size) {
            throw std::runtime_error("buffer is too small");
        }
        std::memcpy(buffer, label.c_str(), label.size() + 1);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_save(const snet_predictor_t* predictor, const char* directory) {
    try {
        if (predictor == nullptr || directory == nullptr) {
            throw std::runtime_error("predictor and directory must not be null");
        }
        predictor->model.save(directory);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

int snet_load(const char* directory, snet_predictor_t** out) {
    try {
        if (directory == nullptr || out == nullptr) {
            throw std::runtime_error("directory and out pointer must not be null");
        }
        *out = new snet_predictor{simplennet::SimplePredictor::load(directory)};
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc);
    }
}

const char* snet_last_error(void) {
    return last_error.c_str();
}
