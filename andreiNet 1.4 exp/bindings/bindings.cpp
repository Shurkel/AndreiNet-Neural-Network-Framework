// bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>         // Needed for std::vector, std::pair, etc.
#include <pybind11/eigen.h>       // Needed for Eigen types
#include <pybind11/numpy.h>       // Needed for py::array_t
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include <sstream>
#include <vector> // Ensure vector is included for TrainingSetEigen conversion
#include <stdexcept> // For exceptions

// Include your main library header (adjust path if needed)
#include "../cpp_src/andreinet.h" // Should include everything else

// Helper function to capture C++ cout (remains the same)
std::string capture_cout(const std::function<void()>& func) {
    std::stringstream buffer;
    std::streambuf* old_cout_buf = std::cout.rdbuf(buffer.rdbuf());
    func();
    std::cout.rdbuf(old_cout_buf);
    return buffer.str();
}

namespace py = pybind11;

PYBIND11_MODULE(andreinet_bindings, m) {
    m.doc() = "Python bindings for the andreiNET C++ library";

    // === Bind Enums ===
    // Use py::arithmetic() if you want to allow bitwise operations (not usually needed for these)
    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("ADAM", OptimizerType::ADAM)
        .export_values(); // Make enum values accessible directly

    py::enum_<Net::LossFunction>(m, "LossFunction")
        .value("MSE", Net::LossFunction::MSE)
        .value("CROSS_ENTROPY", Net::LossFunction::CROSS_ENTROPY)
        .export_values();

    // === Bind Net Class ===
    py::class_<Net>(m, "Net")
        .def(py::init<const std::vector<int>&>(), py::arg("layer_sizes"))
        // Config methods remain the same...
        .def("set_loss_function", &Net::setLossFunction, py::arg("loss_type"), "Set the loss function (MSE or CROSS_ENTROPY)")
        .def("set_optimizer", &Net::setOptimizer,
             py::arg("optimizer_type"), py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8,
             "Set the optimizer (SGD or ADAM) with optional Adam parameters")
        .def("set_L2_regularization", &Net::setL2Regularization, py::arg("lambda_val"), "Set the L2 regularization strength (lambda)")
        .def("set_learning_rate_decay", &Net::setLearningRateDecay, py::arg("decay"), "Set the learning rate decay factor per epoch")
        .def("set_layer_activation", [](Net &net, int layer_index, int activation_type) {
            if (layer_index >= 0 && layer_index < net.layers.size()) {
                net.layers[layer_index].setActivationFunction(activation_type);
            } else {
                throw std::out_of_range("Layer index out of range");
            }
        }, py::arg("layer_index"), py::arg("activation_type"), "Set activation function for a specific layer (-1:Lin, 0:ReLU, 1:Sig, 2:Soft+, 3:Tanh)")

        // Core Methods
        .def("predict", &Net::predict, py::arg("input_data"), py::return_value_policy::reference_internal, // Important policy for Eigen!
             "Perform a forward pass and return the output layer activations")

        // --- MODIFIED TRAIN BINDING ---
        .def("train", [](Net &net,
                         // Accept a list of tuples containing NumPy arrays
                         const py::list& training_data_py,
                         int epochs, double initial_learning_rate, int batch_size, bool shuffle) {

            // --- Convert Python list[tuple[numpy, numpy]] to C++ TrainingSetEigen ---
            TrainingSetEigen training_data_cpp;
            training_data_cpp.reserve(training_data_py.size()); // Optimize allocation

            int sample_idx = 0; // For better error messages
            for (const py::handle& item_handle : training_data_py) {
                // Each item should be a tuple
                if (!py::isinstance<py::tuple>(item_handle)) {
                    throw py::type_error("Training data must be a list of tuples (input, target).");
                }
                py::tuple item_tuple = item_handle.cast<py::tuple>();
                if (item_tuple.size() != 2) {
                    throw py::value_error("Each tuple in training data must contain exactly two elements (input, target).");
                }

                // Ensure tuple elements are numpy arrays of correct type (double)
                 if (!py::isinstance<py::array_t<double>>(item_tuple[0]) ||
                     !py::isinstance<py::array_t<double>>(item_tuple[1])) {
                     // Provide more context in the error message
                     std::string msg = "Elements within training data tuples must be NumPy arrays of float64 (double). Sample index: " + std::to_string(sample_idx);
                     throw py::type_error(msg.c_str());
                 }

                try {
                    // Convert NumPy arrays to Eigen::VectorXd using pybind11's automatic conversion
                    Eigen::VectorXd input_eigen = item_tuple[0].cast<Eigen::VectorXd>();
                    Eigen::VectorXd target_eigen = item_tuple[1].cast<Eigen::VectorXd>();

                    // Optional: Add dimension checks here if necessary, e.g.
                    // if (input_eigen.size() != net.layers[0].numNodes) { ... throw error ... }

                    training_data_cpp.emplace_back(input_eigen, target_eigen);
                } catch (const py::cast_error& e) {
                    // Catch potential casting errors for more specific feedback
                    std::string msg = "Error casting NumPy array to Eigen::VectorXd for sample index " + std::to_string(sample_idx) + ". Check array data type and shape. Details: " + e.what();
                    throw py::type_error(msg.c_str());
                } catch (const std::exception& e) {
                    // Catch other potential standard exceptions during conversion
                     std::string msg = "Error processing training data sample index " + std::to_string(sample_idx) + ". Details: " + e.what();
                     throw std::runtime_error(msg.c_str());
                }
                sample_idx++; // Increment sample index
            }
            // --- End Conversion ---


            // Redirect cout/cerr and release GIL (remains the same)
            // Uncomment the cout redirect if you want to see C++ prints in python console during training
            // py::scoped_ostream_redirect stream_redir_cout(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream_redir_err(std::cerr, py::module::import("sys").attr("stderr"));
            py::gil_scoped_release release; // Release Python GIL for C++ computation

            // Call the original C++ train function with the converted data
            net.train(training_data_cpp, epochs, initial_learning_rate, batch_size, shuffle);

         }, py::arg("training_data"), // Keep arg name the same for Python side
            py::arg("epochs"),
            py::arg("initial_learning_rate"),
            py::arg("batch_size") = 1,
            py::arg("shuffle") = true,
            "Train the network. Accepts list[tuple[np.ndarray, np.ndarray]]. Loss history saved to 'training_loss_eigen.txt'")
        // --- END MODIFIED TRAIN BINDING ---


        .def("save", &Net::save, py::arg("filename"), "Save the network weights and biases to a binary file")
        .def("load", &Net::load, py::arg("filename"), "Load network weights and biases from a file (Network topology must match!)")

        // Inspection Methods (remain the same)
        .def("get_layer_count", [](const Net& net){ return net.layers.size(); })
        .def("get_layer_nodes", [](const Net& net, int layer_index){
             if (layer_index >= 0 && layer_index < net.layers.size()) {
                return net.layers[layer_index].numNodes;
            } else {
                throw std::out_of_range("Layer index out of range");
            }
        }, py::arg("layer_index"))
        .def("get_layer_activation", [](const Net& net, int layer_index){
             if (layer_index >= 0 && layer_index < net.layers.size()) {
                return net.layers[layer_index].activationFunction;
            } else {
                throw std::out_of_range("Layer index out of range");
            }
        }, py::arg("layer_index"))
        .def("get_network_structure_str", [](const Net& net, bool show_matrices) {
             // Capture the output of printNetworkStructure
             return capture_cout([&]() {
                 net.printNetworkStructure(show_matrices);
             });
        }, py::arg("show_matrices") = false, "Get a string representation of the network structure");

} // End PYBIND11_MODULE