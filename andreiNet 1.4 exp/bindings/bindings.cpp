// bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>         // Needed for std::vector, std::pair, etc.
#include "../eigen/Eigen/Core"       // Needed for Eigen::VectorXd, MatrixXd
#include <pybind11/functional.h>  // If you were to use callbacks
#include <pybind11/iostream.h>   // For redirecting C++ streams

#include <sstream>              // To capture printNetworkStructure output

// Include your main library header (adjust path if needed)
#include "../cpp_src/andreinet.h" // Should include everything else

// Helper function to capture C++ cout
std::string capture_cout(const std::function<void()>& func) {
    std::stringstream buffer;
    // Redirect std::cout to buffer
    std::streambuf* old_cout_buf = std::cout.rdbuf(buffer.rdbuf());
    func(); // Execute the function that prints to cout
    std::cout.rdbuf(old_cout_buf); // Restore original cout buffer
    return buffer.str();
}


namespace py = pybind11;

PYBIND11_MODULE(andreinet_bindings, m) {
    m.doc() = "Python bindings for the andreiNET C++ library"; // Optional module docstring

    // === Bind Enums ===
    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("ADAM", OptimizerType::ADAM)
        .export_values(); // Make enum values accessible directly

    py::enum_<Net::LossFunction>(m, "LossFunction")
        .value("MSE", Net::LossFunction::MSE)
        .value("CROSS_ENTROPY", Net::LossFunction::CROSS_ENTROPY)
        .export_values();

    // === Bind Layer Class (Optional - only if needed directly) ===
    // Usually interacting via Net is sufficient. We might need setActivationFunction.
    // py::class_<Layer>(m, "Layer")
    //    .def("setActivationFunction", &Layer::setActivationFunction, py::arg("funcType"));
       // Expose other Layer methods/attributes if necessary

    // === Bind Net Class ===
    py::class_<Net>(m, "Net")
        // Constructor: Takes a list of integers for topology
        .def(py::init<const std::vector<int>&>(), py::arg("layer_sizes"))

        // Configuration Methods
        .def("set_loss_function", &Net::setLossFunction, py::arg("loss_type"), "Set the loss function (MSE or CROSS_ENTROPY)")
        .def("set_optimizer", &Net::setOptimizer,
             py::arg("optimizer_type"), py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8,
             "Set the optimizer (SGD or ADAM) with optional Adam parameters")
        .def("set_L2_regularization", &Net::setL2Regularization, py::arg("lambda_val"), "Set the L2 regularization strength (lambda)")
        .def("set_learning_rate_decay", &Net::setLearningRateDecay, py::arg("decay"), "Set the learning rate decay factor per epoch")

        // Layer-specific configuration (Need layer index)
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
        .def("train", [](Net &net, const TrainingSetEigen& training_data, int epochs, double initial_learning_rate, int batch_size, bool shuffle) {
             // Redirect cout during training to avoid cluttering Python output unless desired
             // py::scoped_ostream_redirect stream_redir(std::cout, py::module::import("sys").attr("stdout")); // Uncomment to see C++ prints
             py::scoped_ostream_redirect stream_redir_err(std::cerr, py::module::import("sys").attr("stderr")); // Capture errors
             py::gil_scoped_release release; // Release GIL for potentially long C++ computation
             net.train(training_data, epochs, initial_learning_rate, batch_size, shuffle);
             // Note: Loss history is written to 'training_loss_eigen.txt' by the C++ code
         }, py::arg("training_data"), py::arg("epochs"), py::arg("initial_learning_rate"), py::arg("batch_size") = 1, py::arg("shuffle") = true,
            "Train the network. Loss history is saved to 'training_loss_eigen.txt'")

        .def("save", &Net::save, py::arg("filename"), "Save the network weights and biases to a binary file")
        .def("load", &Net::load, py::arg("filename"), "Load network weights and biases from a file (Network topology must match!)")

        // Inspection Methods
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

        // Method to get structure as a string
        .def("get_network_structure_str", [](const Net& net, bool show_matrices) {
             // Capture the output of printNetworkStructure
             return capture_cout([&]() {
                 net.printNetworkStructure(show_matrices);
             });
        }, py::arg("show_matrices") = false, "Get a string representation of the network structure");


    // Bind the training data type alias if needed, pybind11/stl.h and eigen.h handle it well
    // py::bind_vector<TrainingSetEigen>(m, "TrainingSetEigen");
}