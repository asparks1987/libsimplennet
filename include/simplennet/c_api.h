#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SNET_API __declspec(dllexport)
#else
#define SNET_API
#endif

typedef enum snet_output_type {
    SNET_OUTPUT_INT = 0,
    SNET_OUTPUT_FLOAT = 1,
    SNET_OUTPUT_CATEGORY = 2
} snet_output_type_t;

typedef struct snet_predictor snet_predictor_t;

SNET_API int snet_create(snet_output_type_t output_type, snet_predictor_t** out);
SNET_API void snet_destroy(snet_predictor_t* predictor);
SNET_API snet_output_type_t snet_get_output_type(const snet_predictor_t* predictor);
SNET_API const char* snet_output_type_name(const snet_predictor_t* predictor);

SNET_API int snet_fit_numeric(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const double* targets,
    size_t target_count,
    size_t epochs,
    double learning_rate);

SNET_API int snet_fit_numeric_bits(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const double* targets,
    size_t target_count,
    size_t epochs,
    unsigned long long learning_rate_bits);

SNET_API int snet_fit_labels(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const char* const* labels,
    size_t label_count,
    size_t epochs,
    double learning_rate);

SNET_API int snet_fit_labels_bits(
    snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    const char* const* labels,
    size_t label_count,
    size_t epochs,
    unsigned long long learning_rate_bits);

SNET_API int snet_predict_numeric(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    double* outputs,
    size_t output_count);

SNET_API int snet_predict_ints(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    long long* outputs,
    size_t output_count);

SNET_API int snet_predict_class_indices(
    const snet_predictor_t* predictor,
    const double* row_major_inputs,
    size_t rows,
    size_t cols,
    size_t* outputs,
    size_t output_count);

SNET_API size_t snet_class_label_count(const snet_predictor_t* predictor);
SNET_API int snet_class_label_copy(const snet_predictor_t* predictor, size_t index, char* buffer, size_t buffer_size);

SNET_API int snet_save(const snet_predictor_t* predictor, const char* directory);
SNET_API int snet_load(const char* directory, snet_predictor_t** out);

SNET_API const char* snet_last_error(void);

#ifdef __cplusplus
}
#endif
