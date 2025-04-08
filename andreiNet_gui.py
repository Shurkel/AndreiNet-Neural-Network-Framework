import sys
import os
import numpy as np
import subprocess
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                            QSlider, QProgressBar, QListWidget, QTextEdit,
                            QGroupBox, QGridLayout, QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt"""
    def __init__(self):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class NetworkTrainerThread(QThread):
    """Thread for running neural network training without blocking the UI"""
    update_progress = pyqtSignal(int, float)
    training_complete = pyqtSignal()
    
    def __init__(self, network_config, dataset_config, training_config):
        super().__init__()
        self.network_config = network_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        
    def run(self):
        # Here we'll generate a C++ command to run the andreiNet training
        # And communicate with it through files or subprocess I/O
        
        # For now, let's simulate training with a loop
        epochs = self.training_config.get('epochs', 100)
        
        for epoch in range(epochs):
            # Simulate training one epoch
            loss = 1.0 / (epoch + 1) + 0.1 * np.random.random()
            
            # Report progress back to main thread
            progress = int((epoch + 1) / epochs * 100)
            self.update_progress.emit(progress, loss)
            
            # Simulate work
            self.msleep(50)
            
        # Signal that training is complete
        self.training_complete.emit()
        
        # In the real implementation, we would:
        # 1. Create a temporary C++ file with the configured network
        # 2. Compile or use a pre-built binary with these configs
        # 3. Run the training process
        # 4. Parse the output or read results from a file

class NetworkVisualizer(QWidget):
    """Widget for visualizing neural network architecture"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = MplCanvas()
        self.layout.addWidget(self.canvas)
        self.setMinimumHeight(300)
        
    def visualize_network(self, layer_sizes):
        self.canvas.axes.clear()
        
        n_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        # Create positions for neurons
        neuron_positions = []
        for i, size in enumerate(layer_sizes):
            layer_positions = []
            for j in range(size):
                # Calculate position to center neurons vertically
                y_pos = 0.5 + (j - (size - 1) / 2) / max_neurons * 0.8
                layer_positions.append((i / (n_layers - 1), y_pos))
            neuron_positions.append(layer_positions)
        
        # Draw connections between layers
        for i in range(n_layers - 1):
            for j, pos1 in enumerate(neuron_positions[i]):
                for k, pos2 in enumerate(neuron_positions[i + 1]):
                    # Draw a light gray line for each connection
                    self.canvas.axes.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                         color='lightgray', linewidth=0.5)
        
        # Draw neurons
        for i, layer in enumerate(neuron_positions):
            x = [pos[0] for pos in layer]
            y = [pos[1] for pos in layer]
            
            # Input layer is green, output layer is red, hidden layers are blue
            if i == 0:
                color = 'green'
                label = f'Input ({layer_sizes[i]})'
            elif i == n_layers - 1:
                color = 'red'
                label = f'Output ({layer_sizes[i]})'
            else:
                color = 'blue'
                label = f'Hidden {i} ({layer_sizes[i]})'
                
            self.canvas.axes.scatter(x, y, s=100, color=color, label=label)
        
        # Add labels and finalize
        self.canvas.axes.set_xlim(-0.1, 1.1)
        self.canvas.axes.set_ylim(0, 1)
        self.canvas.axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        self.canvas.axes.axis('off')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

class TrainingVisualizer(QWidget):
    """Widget for visualizing training progress"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = MplCanvas()
        self.layout.addWidget(self.canvas)
        
        self.epochs = []
        self.losses = []
        
    def update_plot(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        self.canvas.axes.clear()
        self.canvas.axes.plot(self.epochs, self.losses, 'b-')
        self.canvas.axes.set_xlabel('Epoch')
        self.canvas.axes.set_ylabel('Loss')
        self.canvas.axes.set_title('Training Progress')
        
        if len(self.epochs) > 1:
            # Only adjust ylim if we have enough data points
            self.canvas.axes.set_ylim(0, max(self.losses) * 1.1)
            
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def reset(self):
        self.epochs = []
        self.losses = []
        self.canvas.axes.clear()
        self.canvas.draw()

class DatasetGenerator(QWidget):
    """Widget for generating and visualizing datasets"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Dataset selection
        self.dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["XOR Problem", "Sine Wave", "Polynomial Function", "Concentric Circles"])
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(self.dataset_combo)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(10, 10000)
        self.sample_size_spin.setValue(1000)
        self.sample_size_spin.setSingleStep(100)
        self.sample_size_spin.setPrefix("Samples: ")
        dataset_layout.addWidget(self.sample_size_spin)
        
        self.noise_slider = QDoubleSpinBox()
        self.noise_slider.setRange(0, 1)
        self.noise_slider.setValue(0.1)
        self.noise_slider.setSingleStep(0.05)
        self.noise_slider.setPrefix("Noise: ")
        dataset_layout.addWidget(self.noise_slider)
        
        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self.generate_dataset)
        dataset_layout.addWidget(self.generate_btn)
        
        self.dataset_group.setLayout(dataset_layout)
        self.layout.addWidget(self.dataset_group)
        
        # Dataset visualization
        self.canvas = MplCanvas()
        self.layout.addWidget(self.canvas)
        
        # Store dataset properties
        self.current_dataset = None
        self.input_dim = 0
        self.output_dim = 0
        
    def on_dataset_changed(self, index):
        # Update UI based on dataset selection
        if index == 0:  # XOR
            self.noise_slider.setEnabled(True)
        elif index == 1:  # Sine
            self.noise_slider.setEnabled(True)
        elif index == 2:  # Polynomial
            self.noise_slider.setEnabled(True)
        elif index == 3:  # Circles
            self.noise_slider.setEnabled(True)
    
    def generate_dataset(self):
        dataset_type = self.dataset_combo.currentIndex()
        num_samples = self.sample_size_spin.value()
        noise_level = self.noise_slider.value()
        
        self.canvas.axes.clear()
        
        # Generate dataset based on selection
        if dataset_type == 0:  # XOR Problem
            X = np.random.rand(num_samples, 2)
            X = (X > 0.5).astype(float)  # Convert to binary
            noise = np.random.normal(0, noise_level, X.shape)
            X += noise
            y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(float).reshape(-1, 1)
            
            # Visualize
            self.canvas.axes.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm', alpha=0.7)
            self.canvas.axes.set_xlabel('X1')
            self.canvas.axes.set_ylabel('X2')
            self.canvas.axes.set_title('XOR Dataset')
            
            self.input_dim = 2
            self.output_dim = 1
            
        elif dataset_type == 1:  # Sine Wave
            X = np.random.uniform(0, 2*np.pi, (num_samples, 1))
            noise = np.random.normal(0, noise_level, (num_samples, 1))
            y = np.sin(X) + noise
            
            # Visualize
            sort_idx = np.argsort(X[:, 0])
            self.canvas.axes.plot(X[sort_idx, 0], y[sort_idx, 0], 'b.')
            self.canvas.axes.set_xlabel('X')
            self.canvas.axes.set_ylabel('sin(X)')
            self.canvas.axes.set_title('Sine Wave Dataset')
            
            self.input_dim = 1
            self.output_dim = 1
            
        elif dataset_type == 2:  # Polynomial Function
            X = np.random.uniform(-5, 5, (num_samples, 1))
            noise = np.random.normal(0, noise_level * 2, (num_samples, 1))
            y = X**2 + 3*X - 2 + noise
            
            # Visualize
            sort_idx = np.argsort(X[:, 0])
            self.canvas.axes.plot(X[sort_idx, 0], y[sort_idx, 0], 'b.')
            self.canvas.axes.set_xlabel('X')
            self.canvas.axes.set_ylabel('y = x² + 3x - 2')
            self.canvas.axes.set_title('Polynomial Function Dataset')
            
            self.input_dim = 1
            self.output_dim = 1
            
        elif dataset_type == 3:  # Concentric Circles
            samples_per_class = num_samples // 2
            
            # Inner circle (class 0)
            angles = np.random.uniform(0, 2*np.pi, samples_per_class)
            radius = 1.0 + np.random.normal(0, noise_level, samples_per_class)
            X1 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            y1 = np.zeros((samples_per_class, 1))
            
            # Outer circle (class 1)
            angles = np.random.uniform(0, 2*np.pi, samples_per_class)
            radius = 3.0 + np.random.normal(0, noise_level, samples_per_class)
            X2 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            y2 = np.ones((samples_per_class, 1))
            
            # Combine
            X = np.vstack([X1, X2])
            y = np.vstack([y1, y2])
            
            # Visualize
            self.canvas.axes.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm', alpha=0.7)
            self.canvas.axes.set_xlabel('X1')
            self.canvas.axes.set_ylabel('X2')
            self.canvas.axes.axis('equal')
            self.canvas.axes.set_title('Concentric Circles Dataset')
            
            self.input_dim = 2
            self.output_dim = 1
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
        # Store dataset
        self.current_dataset = {
            'type': dataset_type,
            'X': X,
            'y': y,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
        
        return self.current_dataset

class NetworkDesigner(QWidget):
    """Widget for designing neural network architecture"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Layout for controls
        controls_layout = QHBoxLayout()
        
        # Network configuration
        self.network_group = QGroupBox("Network Architecture")
        network_layout = QVBoxLayout()
        
        # Input layer (will be set by dataset)
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Layer:"))
        self.input_size_label = QLabel("2")
        input_layout.addWidget(self.input_size_label)
        network_layout.addLayout(input_layout)
        
        # Hidden layers controls
        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setRange(0, 5)
        self.hidden_layers_spin.setValue(1)
        self.hidden_layers_spin.setPrefix("Hidden Layers: ")
        self.hidden_layers_spin.valueChanged.connect(self.update_hidden_layer_controls)
        network_layout.addWidget(self.hidden_layers_spin)
        
        # Container for hidden layer size controls
        self.hidden_layer_container = QWidget()
        self.hidden_layer_layout = QVBoxLayout(self.hidden_layer_container)
        network_layout.addWidget(self.hidden_layer_container)
        
        # Output layer (will be set by dataset)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Layer:"))
        self.output_size_label = QLabel("1")
        output_layout.addWidget(self.output_size_label)
        network_layout.addLayout(output_layout)
        
        # Activation functions
        self.activation_group = QGroupBox("Activation Functions")
        activation_layout = QVBoxLayout()
        
        # Hidden layers activation
        hidden_act_layout = QHBoxLayout()
        hidden_act_layout.addWidget(QLabel("Hidden Layers:"))
        self.hidden_activation_combo = QComboBox()
        self.hidden_activation_combo.addItems(["Linear", "Step", "Sigmoid", "Tanh", "ReLU", "Leaky ReLU"])
        self.hidden_activation_combo.setCurrentIndex(2)  # Sigmoid default
        hidden_act_layout.addWidget(self.hidden_activation_combo)
        activation_layout.addLayout(hidden_act_layout)
        
        # Output layer activation
        output_act_layout = QHBoxLayout()
        output_act_layout.addWidget(QLabel("Output Layer:"))
        self.output_activation_combo = QComboBox()
        self.output_activation_combo.addItems(["Linear", "Step", "Sigmoid", "Tanh", "ReLU", "Leaky ReLU"])
        self.output_activation_combo.setCurrentIndex(2)  # Sigmoid default
        output_act_layout.addWidget(self.output_activation_combo)
        activation_layout.addLayout(output_act_layout)
        
        self.activation_group.setLayout(activation_layout)
        
        self.network_group.setLayout(network_layout)
        controls_layout.addWidget(self.network_group)
        controls_layout.addWidget(self.activation_group)
        
        self.layout.addLayout(controls_layout)
        
        # Initialize hidden layer spinboxes
        self.hidden_layer_spinboxes = []
        self.update_hidden_layer_controls()
        
        # Network visualization
        self.network_viz = NetworkVisualizer(self)
        self.layout.addWidget(self.network_viz)
        
        # Update button
        self.update_btn = QPushButton("Update Network Visualization")
        self.update_btn.clicked.connect(self.update_network_visualization)
        self.layout.addWidget(self.update_btn)
        
        # Initial visualization
        self.update_network_visualization()
        
    def update_hidden_layer_controls(self):
        # Clear existing spinboxes
        for i in reversed(range(self.hidden_layer_layout.count())): 
            item = self.hidden_layer_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # If it's a layout, we need to clear it recursively
                layout = item.layout()
                for j in reversed(range(layout.count())):
                    widget = layout.itemAt(j).widget()
                    if widget:
                        widget.deleteLater()
                # Remove the layout
                self.hidden_layer_layout.removeItem(item)
        
        self.hidden_layer_spinboxes = []
        
        # Add spinbox for each hidden layer
        for i in range(self.hidden_layers_spin.value()):
            layer_layout = QHBoxLayout()
            layer_layout.addWidget(QLabel(f"Layer {i+1}:"))
            
            spinbox = QSpinBox()
            spinbox.setRange(1, 100)
            spinbox.setValue(5)  # Default value
            layer_layout.addWidget(spinbox)
            
            self.hidden_layer_spinboxes.append(spinbox)
            self.hidden_layer_layout.addLayout(layer_layout)
            
        # Update visualization after changing architecture
        QTimer.singleShot(100, self.update_network_visualization)
    
    def update_network_visualization(self):
        # Get current architecture
        layer_sizes = []
        
        # Input layer
        layer_sizes.append(int(self.input_size_label.text()))
        
        # Hidden layers
        for spinbox in self.hidden_layer_spinboxes:
            layer_sizes.append(spinbox.value())
            
        # Output layer
        layer_sizes.append(int(self.output_size_label.text()))
        
        # Update visualization
        self.network_viz.visualize_network(layer_sizes)
        
    def set_io_size(self, input_dim, output_dim):
        """Set input and output layer sizes based on dataset"""
        self.input_size_label.setText(str(input_dim))
        self.output_size_label.setText(str(output_dim))
        self.update_network_visualization()
        
    def get_network_config(self):
        """Get current network configuration"""
        layer_sizes = [int(self.input_size_label.text())]
        
        for spinbox in self.hidden_layer_spinboxes:
            layer_sizes.append(spinbox.value())
            
        layer_sizes.append(int(self.output_size_label.text()))
        
        # Map activation functions to andreiNet indices
        activation_map = {
            0: -1,  # Linear
            1: 0,   # Step
            2: 1,   # Sigmoid
            3: 2,   # Tanh
            4: 3,   # ReLU
            5: 4    # Leaky ReLU
        }
        
        hidden_activation = activation_map[self.hidden_activation_combo.currentIndex()]
        output_activation = activation_map[self.output_activation_combo.currentIndex()]
        
        return {
            'layer_sizes': layer_sizes,
            'hidden_activation': hidden_activation,
            'output_activation': output_activation
        }

class TrainingConfigurator(QWidget):
    """Widget for configuring training parameters"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Training parameters group
        self.training_group = QGroupBox("Training Parameters")
        training_layout = QGridLayout()
        
        # Epochs
        training_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSingleStep(10)
        training_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Learning rate
        training_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        training_layout.addWidget(self.lr_spin, 1, 1)
        
        # Batch size
        training_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setValue(10)
        training_layout.addWidget(self.batch_spin, 2, 1)
        
        # Loss function
        training_layout.addWidget(QLabel("Loss Function:"), 3, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["Mean Squared Error", "Cross Entropy"])
        training_layout.addWidget(self.loss_combo, 3, 1)
        
        self.training_group.setLayout(training_layout)
        self.layout.addWidget(self.training_group)
        
        # Progress visualization
        progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Loss: N/A")
        progress_layout.addWidget(self.progress_label)
        
        self.layout.addLayout(progress_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setEnabled(False)  # Disabled until dataset and network are ready
        buttons_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset")
        buttons_layout.addWidget(self.reset_btn)
        
        self.layout.addLayout(buttons_layout)
        
        # Training visualization
        self.train_viz = TrainingVisualizer(self)
        self.layout.addWidget(self.train_viz)
    
    def update_progress(self, progress, loss):
        """Update training progress UI"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Loss: {loss:.6f}")
        
        # Update visualization
        self.train_viz.update_plot(len(self.train_viz.epochs) + 1, loss)
    
    def reset_progress(self):
        """Reset training progress UI"""
        self.progress_bar.setValue(0)
        self.progress_label.setText("Loss: N/A")
        self.train_viz.reset()
    
    def get_training_config(self):
        """Get current training configuration"""
        return {
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': self.batch_spin.value(),
            'loss_function': self.loss_combo.currentIndex()  # 0: MSE, 1: CrossEntropy
        }

class TestingWidget(QWidget):
    """Widget for testing trained models with input ranges and visualization"""
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create a splitter for flexible layout
        self.splitter = QSplitter(Qt.Vertical)
        
        # === Top section - Input parameters ===
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Title
        input_title = QLabel("Model Testing")
        input_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        input_layout.addWidget(input_title)
        
        # Add tabs for different testing modes
        self.test_tabs = QTabWidget()
        
        # === SINGLE TEST TAB ===
        single_test_tab = QWidget()
        single_test_layout = QVBoxLayout(single_test_tab)
        
        # Description
        single_test_desc = QLabel("Enter specific values to test the trained model:")
        single_test_desc.setWordWrap(True)
        single_test_layout.addWidget(single_test_desc)
        
        # Container for input fields
        self.input_container = QWidget()
        self.input_layout = QVBoxLayout(self.input_container)
        self.input_layout.setSpacing(10)
        single_test_layout.addWidget(self.input_container)
        
        # Test button
        self.test_btn = QPushButton("Run Single Prediction")
        self.test_btn.setMinimumHeight(30)
        self.test_btn.setEnabled(False)  # Disabled until model is trained
        single_test_layout.addWidget(self.test_btn)
        
        self.test_tabs.addTab(single_test_tab, "Single Test")
        
        # === RANGE TEST TAB ===
        range_test_tab = QWidget()
        range_test_layout = QVBoxLayout(range_test_tab)
        
        range_test_desc = QLabel("Test model across a range of input values:")
        range_test_desc.setWordWrap(True)
        range_test_layout.addWidget(range_test_desc)
        
        # Input dimension selection (only relevant for multi-dimensional inputs)
        self.dim_layout = QHBoxLayout()
        self.dim_layout.addWidget(QLabel("Vary input dimension:"))
        self.dim_selector = QComboBox()
        self.dim_selector.addItem("Input 1")  # Default, will be updated with dataset
        self.dim_layout.addWidget(self.dim_selector)
        range_test_layout.addLayout(self.dim_layout)
        
        # Range configuration
        range_grid = QGridLayout()
        range_grid.addWidget(QLabel("Start:"), 0, 0)
        self.range_start = QDoubleSpinBox()
        self.range_start.setRange(-100, 100)
        self.range_start.setValue(-5)
        self.range_start.setDecimals(2)
        range_grid.addWidget(self.range_start, 0, 1)
        
        range_grid.addWidget(QLabel("End:"), 1, 0)
        self.range_end = QDoubleSpinBox()
        self.range_end.setRange(-100, 100)
        self.range_end.setValue(5)
        self.range_end.setDecimals(2)
        range_grid.addWidget(self.range_end, 1, 1)
        
        range_grid.addWidget(QLabel("Steps:"), 2, 0)
        self.range_steps = QSpinBox()
        self.range_steps.setRange(2, 1000)
        self.range_steps.setValue(100)
        range_grid.addWidget(self.range_steps, 2, 1)
        
        range_test_layout.addLayout(range_grid)
        
        # Fixed values for other inputs (when input dimension > 1)
        self.fixed_values_container = QWidget()
        self.fixed_values_layout = QVBoxLayout(self.fixed_values_container)
        self.fixed_values_layout.addWidget(QLabel("Fixed values for other inputs:"))
        
        # Will be populated when we know the input dimensions
        self.fixed_value_fields = []
        
        range_test_layout.addWidget(self.fixed_values_container)
        
        # Test range button
        self.test_range_btn = QPushButton("Run Range Prediction")
        self.test_range_btn.setMinimumHeight(30) 
        self.test_range_btn.setEnabled(False)  # Disabled until model is trained
        self.test_range_btn.clicked.connect(self.run_range_prediction)
        range_test_layout.addWidget(self.test_range_btn)
        
        self.test_tabs.addTab(range_test_tab, "Range Test")
        
        input_layout.addWidget(self.test_tabs)
        
        # === Bottom section - Output display and visualization ===
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        
        # Create results tabs
        self.results_tabs = QTabWidget()
        
        # Single test result tab
        single_result_tab = QWidget()
        single_result_layout = QVBoxLayout(single_result_tab)
        
        # Output display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(100)
        single_result_layout.addWidget(self.output_text)
        
        self.results_tabs.addTab(single_result_tab, "Text Output")
        
        # Range test visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        self.viz_canvas = MplCanvas()
        viz_layout.addWidget(self.viz_canvas)
        
        self.results_tabs.addTab(viz_tab, "Visualization")
        
        output_layout.addWidget(self.results_tabs)
        
        # Add widgets to splitter
        self.splitter.addWidget(input_widget)
        self.splitter.addWidget(output_widget)
        
        main_layout.addWidget(self.splitter)
        
        # Initialize with default input fields
        self.update_input_fields(2)  # Default to 2 input fields
        
        # Connect signals
        self.test_btn.clicked.connect(self.run_single_prediction)
        self.dim_selector.currentIndexChanged.connect(self.update_fixed_value_fields)
    
    def update_input_fields(self, input_count):
        """Update input fields based on input dimension"""
        # Clear existing fields for single test
        for i in reversed(range(self.input_layout.count())):
            item = self.input_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # If it's a layout, we need to clear it recursively
                layout = item.layout()
                for j in reversed(range(layout.count())):
                    widget = layout.itemAt(j).widget()
                    if widget:
                        widget.deleteLater()
                # Remove the layout
                self.input_layout.removeItem(item)
        
        self.input_fields = []
        
        # Add spinbox for each input dimension for single tests
        for i in range(input_count):
            field_layout = QHBoxLayout()
            
            label = QLabel(f"Input {i+1}:")
            field_layout.addWidget(label)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-100, 100)
            spinbox.setValue(0)
            spinbox.setDecimals(4)
            spinbox.setMinimumWidth(150)
            field_layout.addWidget(spinbox)
            
            self.input_fields.append(spinbox)
            self.input_layout.addLayout(field_layout)
        
        # Update dimension selector for range test
        self.dim_selector.clear()
        for i in range(input_count):
            self.dim_selector.addItem(f"Input {i+1}")
        
        # Update fixed value fields
        self.update_fixed_value_fields()
    
    def update_fixed_value_fields(self):
        """Update fixed value fields for inputs not being varied in range test"""
        # Clear existing fields
        for i in reversed(range(self.fixed_values_layout.count())):
            item = self.fixed_values_layout.itemAt(i)
            if i == 0:  # Skip the label
                continue
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # If it's a layout, we need to clear it recursively
                layout = item.layout()
                for j in reversed(range(layout.count())):
                    widget = layout.itemAt(j).widget()
                    if widget:
                        widget.deleteLater()
                # Remove the layout
                self.fixed_values_layout.removeItem(item)
        
        self.fixed_value_fields = []
        
        # If there's only one input dimension, hide the container
        if len(self.input_fields) <= 1:
            self.fixed_values_container.setVisible(False)
            return
            
        # Show the container for multi-dimensional inputs
        self.fixed_values_container.setVisible(True)
        
        # Get the selected dimension (0-based)
        selected_dim = self.dim_selector.currentIndex()
        
        # Add fixed value fields for other dimensions
        for i in range(len(self.input_fields)):
            if i == selected_dim:
                continue  # Skip the dimension being varied
                
            field_layout = QHBoxLayout()
            label = QLabel(f"Fixed Input {i+1}:")
            field_layout.addWidget(label)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-100, 100)
            spinbox.setValue(0)
            spinbox.setDecimals(4)
            spinbox.setMinimumWidth(150)
            field_layout.addWidget(spinbox)
            
            self.fixed_value_fields.append((i, spinbox))  # Store the input index and spinbox
            self.fixed_values_layout.addLayout(field_layout)
    
    def enable_testing(self, enabled):
        """Enable or disable testing functionality"""
        self.test_btn.setEnabled(enabled)
        self.test_range_btn.setEnabled(enabled)
    
    def run_single_prediction(self):
        """Handle single prediction execution"""
        # Collect inputs from the fields
        inputs = [field.value() for field in self.input_fields]
        
        # Get parent window to access dataset and model info
        main_window = self.window()
        
        # Call parent's prediction method and display results
        result = main_window.run_prediction_internal(inputs)
        self.display_output(result)
        
        # Switch to output tab
        self.results_tabs.setCurrentIndex(0)
    
    def run_range_prediction(self):
        """Handle range prediction execution and visualization"""
        # Get parent window to access dataset and model info
        main_window = self.window()
        
        # Get selected dimension to vary
        selected_dim = self.dim_selector.currentIndex()
        
        # Get range parameters
        start = self.range_start.value()
        end = self.range_end.value()
        steps = self.range_steps.value()
        
        # Generate range of input values
        input_range = np.linspace(start, end, steps)
        
        # Generate predictions for each value
        results = []
        
        for val in input_range:
            # Set up the input array with fixed values for all other dimensions
            test_input = [0.0] * len(self.input_fields)
            
            # Set the varied dimension
            test_input[selected_dim] = val
            
            # Set fixed values for other dimensions
            for idx, field in self.fixed_value_fields:
                test_input[idx] = field.value()
            
            # Get prediction result
            result = main_window.run_prediction_internal(test_input)
            
            # For now, assume result is a list with one value (the prediction)
            results.append(result[0])
        
        # Visualize results
        self.visualize_range_results(input_range, results, selected_dim)
        
        # Switch to visualization tab
        self.results_tabs.setCurrentIndex(1)
    
    def visualize_range_results(self, input_range, results, selected_dim):
        """Create a visualization of predictions over an input range"""
        self.viz_canvas.axes.clear()
        
        # Plot the results
        self.viz_canvas.axes.plot(input_range, results, 'b-', linewidth=2)
        self.viz_canvas.axes.set_xlabel(f"Input {selected_dim + 1} Value")
        self.viz_canvas.axes.set_ylabel("Prediction")
        self.viz_canvas.axes.set_title("Model Prediction Across Input Range")
        self.viz_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        
        # Add a reference line at y=0.5 for classification tasks
        if all(0 <= r <= 1 for r in results):
            self.viz_canvas.axes.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Decision Boundary")
            self.viz_canvas.axes.legend()
        
        # Format the plot
        self.viz_canvas.fig.tight_layout()
        self.viz_canvas.draw()
    
    def display_output(self, output_data):
        """Display prediction output for single test"""
        self.output_text.clear()
        
        # Create formatted output
        html_output = "<div style='font-size: 14px;'>"
        
        if isinstance(output_data, list) or isinstance(output_data, np.ndarray):
            for i, val in enumerate(output_data):
                html_output += f"<p><b>Output {i+1}:</b> {val:.6f}</p>"
        else:
            html_output += f"<p><b>Output:</b> {output_data:.6f}</p>"
            
        # Add interpretation for classification
        if len(output_data) == 1 and (output_data[0] <= 1.0 and output_data[0] >= 0.0):
            confidence = max(output_data[0], 1.0 - output_data[0])
            predicted_class = 1 if output_data[0] >= 0.5 else 0
            
            # Add styled classification result
            html_output += "<div style='margin-top: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>"
            html_output += "<h3 style='margin-top: 0;'>Classification Result</h3>"
            html_output += f"<p><b>Predicted Class:</b> {predicted_class}</p>"
            html_output += f"<p><b>Confidence:</b> {confidence:.2%}</p>"
            html_output += "</div>"
        
        html_output += "</div>"
        self.output_text.setHtml(html_output)

class InfoWidget(QWidget):
    """Widget for displaying educational information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.text_browser = QTextEdit()
        self.text_browser.setReadOnly(True)
        
        self.layout.addWidget(self.text_browser)
        
        # Set initial content
        self.set_neural_network_info()
    
    def set_neural_network_info(self):
        info_text = """
<h1 style="text-align:center;">Neural Network Basics</h1>

<h2>What is a Neural Network?</h2>
<p>A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes or 'neurons' that process information.</p>

<h2>Key Components:</h2>
<ul>
  <li><b>Input Layer:</b> Receives the initial data</li>
  <li><b>Hidden Layers:</b> Process the data through weighted connections</li>
  <li><b>Output Layer:</b> Produces the final result</li>
  <li><b>Weights:</b> Connection strengths between neurons</li>
  <li><b>Biases:</b> Offset values for each neuron</li>
  <li><b>Activation Functions:</b> Introduce non-linearity</li>
</ul>

<h2>Learning Process:</h2>
<ol>
  <li><b>Forward Pass:</b> Data flows through the network</li>
  <li><b>Error Calculation:</b> Compare output to expected result</li>
  <li><b>Backpropagation:</b> Distribute error through the network</li>
  <li><b>Weight Updates:</b> Adjust connections to reduce error</li>
</ol>

<h2>Common Applications:</h2>
<ul>
  <li>Classification (e.g., XOR problem, image recognition)</li>
  <li>Regression (e.g., function approximation)</li>
  <li>Pattern Recognition</li>
  <li>Time Series Prediction</li>
</ul>

<h2>About activation functions:</h2>
<ul>
  <li><b>Linear:</b> f(x) = x. No transformation, useful for regression problems.</li>
  <li><b>Step:</b> Binary output (0 or 1). Very simple but not differentiable.</li>
  <li><b>Sigmoid:</b> f(x) = 1/(1+e^-x). Outputs between 0 and 1, good for binary classification.</li>
  <li><b>Tanh:</b> Similar to sigmoid but outputs between -1 and 1.</li>
  <li><b>ReLU:</b> f(x) = max(0, x). Fast to compute, helps with vanishing gradient problem.</li>
  <li><b>Leaky ReLU:</b> Like ReLU but allows small negative values.</li>
</ul>

<p>This program uses andreiNet, a C++ neural network library developed by Roman Andrei Dan (2023-2024).</p>
"""
        self.text_browser.setHtml(info_text)

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("andreiNet Educational GUI")
        self.setMinimumSize(900, 700)
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create widgets for each tab
        self.dataset_tab = DatasetGenerator()
        self.network_tab = NetworkDesigner()
        self.training_tab = TrainingConfigurator()
        self.testing_tab = TestingWidget()
        self.info_tab = InfoWidget()
        
        # Add tabs
        self.tabs.addTab(self.dataset_tab, "1. Dataset")
        self.tabs.addTab(self.network_tab, "2. Network Design")
        self.tabs.addTab(self.training_tab, "3. Training")
        self.tabs.addTab(self.testing_tab, "4. Testing")
        self.tabs.addTab(self.info_tab, "Information")
        
        self.layout.addWidget(self.tabs)
        
        # Status bar message
        self.statusBar().showMessage("Welcome to andreiNet Educational GUI")
        
        # Connect signals
        self.connect_signals()
        
        # Store model state
        self.model_trained = False
        self.current_model = None
    
    def connect_signals(self):
        # Dataset tab signals
        self.dataset_tab.generate_btn.clicked.connect(self.on_dataset_generated)
        
        # Training tab signals
        self.training_tab.start_btn.clicked.connect(self.start_training)
        self.training_tab.stop_btn.clicked.connect(self.stop_training)
        self.training_tab.reset_btn.clicked.connect(self.reset_training)
        
        # Testing tab signals
        self.testing_tab.test_btn.clicked.connect(self.run_prediction)
    
    def on_dataset_generated(self):
        dataset = self.dataset_tab.current_dataset
        if dataset:
            # Update network design tab with input/output dimensions
            self.network_tab.set_io_size(dataset['input_dim'], dataset['output_dim'])
            
            # Update testing tab with input fields
            self.testing_tab.update_input_fields(dataset['input_dim'])
            
            # Enable training button
            self.training_tab.start_btn.setEnabled(True)
            
            # Set appropriate tab
            self.tabs.setCurrentIndex(1)  # Switch to network design tab
            
            # Update status
            self.statusBar().showMessage(f"Dataset generated successfully: {self.dataset_tab.dataset_combo.currentText()}")
    
    def start_training(self):
        # Get configurations
        network_config = self.network_tab.get_network_config()
        dataset_config = self.dataset_tab.current_dataset
        training_config = self.training_tab.get_training_config()
        
        # Create and start training thread
        self.training_thread = NetworkTrainerThread(
            network_config, dataset_config, training_config)
        self.training_thread.update_progress.connect(self.training_tab.update_progress)
        self.training_thread.training_complete.connect(self.on_training_complete)
        
        # Update UI
        self.training_tab.start_btn.setEnabled(False)
        self.training_tab.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Training in progress...")
        
        self.training_thread.start()
    
    def stop_training(self):
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            
            # Update UI
            self.training_tab.stop_btn.setEnabled(False)
            self.training_tab.start_btn.setEnabled(True)
            self.statusBar().showMessage("Training stopped by user")
    
    def reset_training(self):
        # Stop training if running
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        
        # Reset UI
        self.training_tab.reset_progress()
        self.training_tab.start_btn.setEnabled(True)
        self.training_tab.stop_btn.setEnabled(False)
        self.model_trained = False
        self.testing_tab.enable_testing(False)
        self.statusBar().showMessage("Training reset")
    
    def on_training_complete(self):
        # Update UI
        self.training_tab.stop_btn.setEnabled(False)
        self.training_tab.start_btn.setEnabled(True)
        self.model_trained = True
        self.testing_tab.enable_testing(True)
        
        # In a real implementation, we would save the trained model here
        # or keep it in memory for inference
        
        # Notify user
        self.statusBar().showMessage("Training completed successfully!")
        
        # Switch to testing tab
        self.tabs.setCurrentIndex(3)  # Testing tab
    
    def run_prediction(self):
        if not self.model_trained:
            QMessageBox.warning(self, "Model Not Ready", "Please train a model first.")
            return
        
        # Get inputs from testing tab
        inputs = [field.value() for field in self.testing_tab.input_fields]
        
        # In a real implementation, we would:
        # 1. Pass the inputs to the C++ model
        # 2. Get the predictions
        # 3. Display the results
        
        # For now, simulate a prediction result
        if len(inputs) == 1:  # Regression (sine or polynomial)
            x = inputs[0]
            # Choose different simulation based on dataset
            dataset_type = self.dataset_tab.dataset_combo.currentIndex()
            if dataset_type == 1:  # Sine
                result = [np.sin(x)]
            elif dataset_type == 2:  # Polynomial
                result = [x*x + 3*x - 2]
            else:
                result = [np.random.random()]
                
        elif len(inputs) == 2:  # Classification (XOR or circles)
            x1, x2 = inputs
            dataset_type = self.dataset_tab.dataset_combo.currentIndex()
            if dataset_type == 0:  # XOR
                result = [(x1 > 0.5) != (x2 > 0.5)]  # XOR logic
            elif dataset_type == 3:  # Circles
                distance = np.sqrt(x1*x1 + x2*x2)
                result = [1.0 if distance > 2.0 else 0.0]  # Circle boundary at radius=2
            else:
                result = [np.random.random()]
        else:
            result = [np.random.random()]
        
        # Display prediction
        self.testing_tab.display_output(result)
        
        # Update status
        self.statusBar().showMessage("Prediction completed")

    def run_prediction_internal(self, inputs):
        if not self.model_trained:
            return [0.0]  # Dummy return value for testing
        
        # In a real implementation, we would:
        # 1. Pass the inputs to the C++ model
        # 2. Get the predictions
        # 3. Return the results
        
        # For now, simulate a prediction result
        if len(inputs) == 1:  # Regression (sine or polynomial)
            x = inputs[0]
            # Choose different simulation based on dataset
            dataset_type = self.dataset_tab.dataset_combo.currentIndex()
            if dataset_type == 1:  # Sine
                result = [np.sin(x)]
            elif dataset_type == 2:  # Polynomial
                result = [x*x + 3*x - 2]
            else:
                result = [np.random.random()]
                
        elif len(inputs) == 2:  # Classification (XOR or circles)
            x1, x2 = inputs
            dataset_type = self.dataset_tab.dataset_combo.currentIndex()
            if dataset_type == 0:  # XOR
                result = [(x1 > 0.5) != (x2 > 0.5)]  # XOR logic
            elif dataset_type == 3:  # Circles
                distance = np.sqrt(x1*x1 + x2*x2)
                result = [1.0 if distance > 2.0 else 0.0]  # Circle boundary at radius=2
            else:
                result = [np.random.random()]
        else:
            result = [np.random.random()]
        
        return result

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())