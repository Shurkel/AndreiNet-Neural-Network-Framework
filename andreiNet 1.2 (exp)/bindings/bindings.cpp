#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::pair
#include <pybind11/functional.h> // For std::function
#include <pybind11/numpy.h> // For potential numpy integration (optional here, stl is fine)

// It's crucial that these includes point to your actual header files.
// If your headers are in a subdirectory like "cpp_lib", adjust the paths.
#include "../utils/node.h"    // Relative path to your node.h
#include "../utils/layer.h"   // Relative path to your layer.h
#include "../utils/net.h"     // Relative path to your net.h
// util.h, timer.h, includes.h are included by the above typically

namespace py = pybind11;

// Helper to convert TrainingSet (std::vector<std::pair<std::vector<double>, std::vector<double>>>)
// from Python list of tuples of lists.
TrainingSet cast_training_set(const py::list& python_training_set) {
    TrainingSet cpp_training_set;
    cpp_training_set.reserve(python_training_set.size());
    for (const auto& item : python_training_set) {
        py::tuple pair_tuple = item.cast<py::tuple>();
        if (pair_tuple.size() != 2) {
            throw py::value_error("Training pair must be a tuple of two lists (input, target)");
        }
        InputData input_data = pair_tuple[0].cast<InputData>();
        TargetData target_data = pair_tuple[1].cast<TargetData>();
        cpp_training_set.emplace_back(input_data, target_data);
    }
    return cpp_training_set;
}


PYBIND11_MODULE(andreinet_py, m) {
    m.doc() = "Python bindings for AndreiNet C++ Neural Network Library";

    // Bind Node::Connection (if direct access is needed, otherwise skip)
    py::class_<Node::Connection>(m, "NodeConnection")
        .def_readonly("node_ptr_placeholder", &Node::Connection::node) // Node* is tricky, expose as placeholder or ID
        .def_readonly("weight", &Node::Connection::weight);

    py::class_<Node>(m, "Node")
        .def(py::init<double, double>(), py::arg("val") = 0.0, py::arg("b") = 0.0)
        .def_property_readonly("value", &Node::getValue)
        .def_property_readonly("unactivated_value", &Node::getUnactivatedValue)
        .def_property_readonly("bias", &Node::getBias)
        .def_property_readonly("delta", &Node::getDelta)
        .def_property_readonly("id", &Node::getId)
        .def_property_readonly("layer_id", &Node::getLayerId)
        .def_property_readonly("activation_type", &Node::getActivationType)
        .def("set_activation_type", &Node::setActivationType, py::arg("function_id"))
        .def_property_readonly("connections", &Node::getConnections) // Returns std::vector<Connection>
        .def("get_weight_to_idx", static_cast<double (Node::*)(int) const>(&Node::getWeightTo), py::arg("next_node_index_in_layer"));


    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>(), py::arg("num_nodes"), py::arg("id") = 0)
        .def_readwrite("nodes", &Layer::nodes) // Expose nodes vector
        .def_property_readonly("layer_id", [](const Layer&l){ return l.layerId;})
        .def("get_values", &Layer::getValues)
        .def("set_values_from_vector", &Layer::setValuesFromVector, py::arg("values"))
        .def("set_activation_function_all", &Layer::setActivationFunctionAll, py::arg("function_id"))
        .def("no_activate_all", &Layer::noActivateAll)
        .def("set_bias_all", &Layer::setBiasAll, py::arg("bias"))
        // Add more Layer methods if needed by GUI, like connectTo (though Net handles this)
        .def_property_readonly("node_count", [](const Layer& l){ return l.nodes.size(); });


    py::class_<Net>(m, "Net")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>(), py::arg("layer_sizes"))
        .def("set_layer_activation", &Net::setLayerActivation, py::arg("layer_id"), py::arg("function_id"))
        .def("randomize_all_weights", &Net::randomizeAllWeights, py::arg("min_val") = -0.1, py::arg("max_val") = 0.1)
        .def("set_all_biases", &Net::setAllBiases, py::arg("bias_value"))
        .def("set_all_weights", &Net::setAllWeights, py::arg("weight_value"))
        .def("predict", &Net::predict, py::arg("input_values"))
        .def("get_output", &Net::getOutput)
        .def("calculate_ssr", &Net::calculateSSR, py::arg("expected_values"))
        .def("calculate_cross_entropy", &Net::calculateCrossEntropy, py::arg("expected_values"))
        .def("train", [](Net &net, const py::list& python_training_set, int epochs, double learning_rate, 
                         int batch_size, bool shuffle, const Net::ProgressCallback& progress_cb) {
            TrainingSet cpp_ts = cast_training_set(python_training_set);
            net.train(cpp_ts, epochs, learning_rate, batch_size, shuffle, progress_cb);
        }, py::arg("training_data"), py::arg("epochs"), py::arg("learning_rate"), 
           py::arg("batch_size") = 1, py::arg("shuffle") = true, py::arg("progress_callback") = nullptr,
           py::call_guard<py::gil_scoped_release>()) // Release GIL for long C++ computation
        .def("get_network_structure_string", &Net::getNetworkStructureString)
        .def("get_layers", &Net::getLayers, py::return_value_policy::reference_internal) // Important for exposing vector of C++ objects
        .def_property_readonly("average_ssr", [](const Net& n){ return n.averageSSR; })
        .def_property_readonly("average_cross_entropy", [](const Net& n){ return n.averageCrossEntropy; });

    // Activation function constants (matching C++ Node::activationFunction)
    m.attr("ACT_LINEAR") = -1;
    m.attr("ACT_RELU") = 0;
    m.attr("ACT_SIGMOID") = 1;
    m.attr("ACT_SOFTPLUS") = 2;
}