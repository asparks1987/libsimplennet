#include "simplennet/simple_predictor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

simplennet::TrainingOptions make_options(
    std::size_t epochs,
    double learning_rate,
    const std::vector<std::size_t>& hidden_units,
    std::uint32_t seed) {
    simplennet::TrainingOptions options;
    options.epochs = epochs;
    options.learning_rate = learning_rate;
    options.hidden_layers = hidden_units;
    options.seed = seed;
    return options;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    py::enum_<simplennet::OutputType>(m, "OutputType")
        .value("Int", simplennet::OutputType::Int)
        .value("Float", simplennet::OutputType::Float)
        .value("Category", simplennet::OutputType::Category);

    py::class_<simplennet::SimplePredictor>(m, "NativePredictor")
        .def(
            py::init([](
                         const std::string& output_type,
                         std::size_t epochs,
                         double learning_rate,
                         const std::vector<std::size_t>& hidden_units,
                         std::uint32_t seed) {
                return simplennet::SimplePredictor(
                    simplennet::output_type_from_string(output_type),
                    make_options(epochs, learning_rate, hidden_units, seed));
            }),
            py::arg("output_type"),
            py::arg("epochs") = 50,
            py::arg("learning_rate") = 0.01,
            py::arg("hidden_units") = std::vector<std::size_t>{32, 16},
            py::arg("seed") = 42)
        .def("fit_numeric", &simplennet::SimplePredictor::fit)
        .def("fit_labels", &simplennet::SimplePredictor::fit_labels)
        .def("predict_numbers", &simplennet::SimplePredictor::predict_numbers)
        .def("predict_ints", &simplennet::SimplePredictor::predict_ints)
        .def("predict_labels", &simplennet::SimplePredictor::predict_labels)
        .def("save", &simplennet::SimplePredictor::save)
        .def_static("load", &simplennet::SimplePredictor::load)
        .def_property_readonly("output_type", &simplennet::SimplePredictor::output_type_name)
        .def_property_readonly("class_labels", &simplennet::SimplePredictor::class_labels)
        .def_property("feature_names", &simplennet::SimplePredictor::feature_names, &simplennet::SimplePredictor::set_feature_names)
        .def_property_readonly("is_fit", &simplennet::SimplePredictor::is_fit);
}
