# python_src/utils.py
import numpy as np
import matplotlib.pyplot as plt
import os

# Define activation function mapping for user display
ACTIVATION_MAP = {
    -1: "Linear",
    0: "ReLU",
    1: "Sigmoid",
    2: "Softplus",
    3: "Tanh"
}
ACTIVATION_IDS = {v.lower(): k for k, v in ACTIVATION_MAP.items()}

def load_xor_data(file_path="datasets/xor.csv"):
    """Loads the XOR dataset from a CSV file."""
    print(f"Loading XOR data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Creating default XOR data.")
        # Create default file if it doesn't exist
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        default_data = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        np.savetxt(file_path, default_data, delimiter=',', fmt='%d')

    try:
        data = np.loadtxt(file_path, delimiter=',')
        if data.shape[1] < 2:
             raise ValueError("CSV file must have at least 2 columns (features + target).")
        # Separate features (all columns except last) and targets (last column)
        features = data[:, :-1].astype(np.float64)
        targets = data[:, -1:].astype(np.float64) # Keep it as a 2D array (N, 1)

        # Convert to the C++ expected format: List[Tuple[np.ndarray, np.ndarray]]
        training_data_py = []
        for i in range(features.shape[0]):
            # Ensure features[i] and targets[i] are 1D vectors for Eigen binding
            training_data_py.append( (features[i].flatten(), targets[i].flatten()) )

        print(f"Loaded {len(training_data_py)} samples.")
        print("Sample 0 Features:", training_data_py[0][0])
        print("Sample 0 Target:", training_data_py[0][1])
        return training_data_py
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def plot_loss_history(log_file="training_loss_eigen.txt"):
    """Plots the loss history from the log file created by C++ train."""
    try:
        # Load data: Epoch, Avg Cost, Learning Rate
        loss_data = np.loadtxt(log_file, skiprows=0) # Assumes no header
        if loss_data.ndim == 1: # Handle case with only one epoch logged
            loss_data = loss_data.reshape(1, -1)

        if loss_data.shape[1] < 2:
            print(f"Warning: Log file '{log_file}' has unexpected format.")
            return

        epochs = loss_data[:, 0]
        avg_cost = loss_data[:, 1]
        # lr = loss_data[:, 2] # Learning rate is also available

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, avg_cost, marker='.')
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Average Cost (Loss)")
        plt.grid(True)
        plt.tight_layout()
        # Make plot interactive and non-blocking if possible
        plt.ion()
        plt.show()
        plt.pause(0.1) # Allow plot to render

        print(f"\nLoss plot displayed (from {log_file}). Close plot window to continue.")
        # Keep plot window open until user closes it (simplest approach)
        # For non-blocking, plt.show(block=False) might work depending on backend
        # plt.show(block=True) # Ensure plot stays open

    except FileNotFoundError:
        print(f"Warning: Loss log file '{log_file}' not found. Cannot plot loss.")
    except Exception as e:
        print(f"Error plotting loss history: {e}")

def clear_loss_log(log_file="training_loss_eigen.txt"):
    """Deletes the loss log file."""
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print(f"Cleared previous loss log: {log_file}")
        except Exception as e:
            print(f"Warning: Could not clear loss log file: {e}")