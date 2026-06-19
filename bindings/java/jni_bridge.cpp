#include "simplennet/simple_predictor.hpp"

#include <jni.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Predictor = simplennet::SimplePredictor;

void throw_java(JNIEnv* env, const std::string& message) {
    jclass exception_class = env->FindClass("java/lang/RuntimeException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

std::vector<double> copy_row(JNIEnv* env, jdoubleArray row_array) {
    jsize cols = env->GetArrayLength(row_array);
    std::vector<double> row(static_cast<std::size_t>(cols));
    env->GetDoubleArrayRegion(row_array, 0, cols, row.data());
    return row;
}

simplennet::Matrix to_matrix(JNIEnv* env, jobjectArray inputs) {
    jsize rows = env->GetArrayLength(inputs);
    simplennet::Matrix matrix;
    matrix.reserve(static_cast<std::size_t>(rows));
    std::size_t expected_cols = 0;
    for (jsize i = 0; i < rows; ++i) {
        auto* row_array = static_cast<jdoubleArray>(env->GetObjectArrayElement(inputs, i));
        if (row_array == nullptr) {
            throw std::runtime_error("input row must not be null");
        }
        std::vector<double> row = copy_row(env, row_array);
        env->DeleteLocalRef(row_array);
        if (i == 0) {
            expected_cols = row.size();
            if (expected_cols == 0) {
                throw std::runtime_error("input rows must contain at least one feature");
            }
        } else if (row.size() != expected_cols) {
            throw std::runtime_error("all input rows must have the same number of features");
        }
        matrix.push_back(std::move(row));
    }
    return matrix;
}

std::vector<double> to_doubles(JNIEnv* env, jdoubleArray array) {
    jsize count = env->GetArrayLength(array);
    std::vector<double> values(static_cast<std::size_t>(count));
    env->GetDoubleArrayRegion(array, 0, count, values.data());
    return values;
}

std::vector<std::string> to_strings(JNIEnv* env, jobjectArray array) {
    jsize count = env->GetArrayLength(array);
    std::vector<std::string> values;
    values.reserve(static_cast<std::size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        jstring value = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        if (value == nullptr) {
            throw std::runtime_error("string target must not be null");
        }
        const char* utf = env->GetStringUTFChars(value, nullptr);
        values.emplace_back(utf);
        env->ReleaseStringUTFChars(value, utf);
        env->DeleteLocalRef(value);
    }
    return values;
}

jlong to_handle(Predictor* predictor) {
    return reinterpret_cast<jlong>(predictor);
}

Predictor* from_handle(jlong handle) {
    return reinterpret_cast<Predictor*>(handle);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_simplennet_SimplePredictor_nativeCreate(JNIEnv* env, jclass, jint output_type) {
    try {
        auto type = simplennet::OutputType::Float;
        switch (output_type) {
            case 0:
                type = simplennet::OutputType::Int;
                break;
            case 1:
                type = simplennet::OutputType::Float;
                break;
            case 2:
                type = simplennet::OutputType::Category;
                break;
            default:
                throw std::runtime_error("unknown output type");
        }
        auto* predictor = new Predictor(type);
        return to_handle(predictor);
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_ai_simplennet_SimplePredictor_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete from_handle(handle);
}

JNIEXPORT void JNICALL Java_ai_simplennet_SimplePredictor_nativeFitNumeric(
    JNIEnv* env,
    jclass,
    jlong handle,
    jobjectArray inputs,
    jdoubleArray targets,
    jint epochs,
    jdouble learning_rate) {
    try {
        auto* predictor = from_handle(handle);
        simplennet::TrainingOptions options = predictor->training_options();
        options.epochs = static_cast<std::size_t>(epochs);
        options.learning_rate = static_cast<double>(learning_rate);
        predictor->set_training_options(options);
        predictor->fit(to_matrix(env, inputs), to_doubles(env, targets));
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
    }
}

JNIEXPORT void JNICALL Java_ai_simplennet_SimplePredictor_nativeFitLabels(
    JNIEnv* env,
    jclass,
    jlong handle,
    jobjectArray inputs,
    jobjectArray targets,
    jint epochs,
    jdouble learning_rate) {
    try {
        auto* predictor = from_handle(handle);
        simplennet::TrainingOptions options = predictor->training_options();
        options.epochs = static_cast<std::size_t>(epochs);
        options.learning_rate = static_cast<double>(learning_rate);
        predictor->set_training_options(options);
        predictor->fit_labels(to_matrix(env, inputs), to_strings(env, targets));
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
    }
}

JNIEXPORT jdoubleArray JNICALL Java_ai_simplennet_SimplePredictor_nativePredictNumbers(JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
    try {
        auto values = from_handle(handle)->predict_numbers(to_matrix(env, inputs));
        jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(values.size()));
        env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
        return result;
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return nullptr;
    }
}

JNIEXPORT jintArray JNICALL Java_ai_simplennet_SimplePredictor_nativePredictInts(JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
    try {
        auto values = from_handle(handle)->predict_ints(to_matrix(env, inputs));
        jintArray result = env->NewIntArray(static_cast<jsize>(values.size()));
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
        return result;
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return nullptr;
    }
}

JNIEXPORT jobjectArray JNICALL Java_ai_simplennet_SimplePredictor_nativePredictLabels(JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
    try {
        auto values = from_handle(handle)->predict_labels(to_matrix(env, inputs));
        jclass string_class = env->FindClass("java/lang/String");
        jobjectArray result = env->NewObjectArray(static_cast<jsize>(values.size()), string_class, nullptr);
        for (jsize i = 0; i < static_cast<jsize>(values.size()); ++i) {
            env->SetObjectArrayElement(result, i, env->NewStringUTF(values[static_cast<std::size_t>(i)].c_str()));
        }
        return result;
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_ai_simplennet_SimplePredictor_nativeSave(JNIEnv* env, jclass, jlong handle, jstring path) {
    try {
        const char* utf = env->GetStringUTFChars(path, nullptr);
        from_handle(handle)->save(std::filesystem::path(utf));
        env->ReleaseStringUTFChars(path, utf);
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
    }
}

JNIEXPORT jlong JNICALL Java_ai_simplennet_SimplePredictor_nativeLoad(JNIEnv* env, jclass, jstring path) {
    try {
        const char* utf = env->GetStringUTFChars(path, nullptr);
        Predictor predictor = Predictor::load(std::filesystem::path(utf));
        env->ReleaseStringUTFChars(path, utf);
        return to_handle(new Predictor(std::move(predictor)));
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_ai_simplennet_SimplePredictor_nativeOutputType(JNIEnv*, jclass, jlong handle) {
    switch (from_handle(handle)->output_type()) {
        case simplennet::OutputType::Int:
            return 0;
        case simplennet::OutputType::Float:
            return 1;
        case simplennet::OutputType::Category:
            return 2;
    }
    return 1;
}

JNIEXPORT jobjectArray JNICALL Java_ai_simplennet_SimplePredictor_nativeClassLabels(JNIEnv* env, jclass, jlong handle) {
    try {
        const auto& values = from_handle(handle)->class_labels();
        jclass string_class = env->FindClass("java/lang/String");
        jobjectArray result = env->NewObjectArray(static_cast<jsize>(values.size()), string_class, nullptr);
        for (jsize i = 0; i < static_cast<jsize>(values.size()); ++i) {
            env->SetObjectArrayElement(result, i, env->NewStringUTF(values[static_cast<std::size_t>(i)].c_str()));
        }
        return result;
    } catch (const std::exception& exc) {
        throw_java(env, exc.what());
        return nullptr;
    }
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK || env == nullptr) {
        return JNI_ERR;
    }

    jclass predictor_class = env->FindClass("ai/simplennet/SimplePredictor");
    if (predictor_class == nullptr) {
        return JNI_ERR;
    }

    JNINativeMethod methods[] = {
        {"nativeCreate", "(I)J", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeCreate)},
        {"nativeDestroy", "(J)V", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeDestroy)},
        {"nativeFitNumeric", "(J[[D[DID)V", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeFitNumeric)},
        {"nativeFitLabels", "(J[[D[Ljava/lang/String;ID)V", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeFitLabels)},
        {"nativePredictNumbers", "(J[[D)[D", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativePredictNumbers)},
        {"nativePredictInts", "(J[[D)[I", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativePredictInts)},
        {"nativePredictLabels", "(J[[D)[Ljava/lang/String;", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativePredictLabels)},
        {"nativeSave", "(JLjava/lang/String;)V", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeSave)},
        {"nativeLoad", "(Ljava/lang/String;)J", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeLoad)},
        {"nativeOutputType", "(J)I", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeOutputType)},
        {"nativeClassLabels", "(J)[Ljava/lang/String;", reinterpret_cast<void*>(Java_ai_simplennet_SimplePredictor_nativeClassLabels)},
    };

    if (env->RegisterNatives(predictor_class, methods, static_cast<jint>(sizeof(methods) / sizeof(methods[0]))) != 0) {
        env->DeleteLocalRef(predictor_class);
        return JNI_ERR;
    }
    env->DeleteLocalRef(predictor_class);
    return JNI_VERSION_1_8;
}

}  // extern "C"
