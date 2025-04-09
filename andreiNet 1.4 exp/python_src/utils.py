# python_src/utils.py
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split # Use scikit-learn for splitting
from sklearn.preprocessing import MinMaxScaler # Use scikit-learn for scaling

# --- Installation Check for Scikit-learn ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("------------------------------------------------------")
    print("WARNING: scikit-learn not found.")
    print("         Data splitting and normalization features will be disabled.")
    print("         Please install it: pip install scikit-learn")
    print("------------------------------------------------------")
    # Define dummy functions or disable features in GUI if scikit-learn is missing
    MinMaxScaler = None # Flag that it's unavailable
    train_test_split = None
# --- End Installation Check ---


# Activation mapping (remains the same)
ACTIVATION_MAP = {
    -1: "Linear", 0: "ReLU", 1: "Sigmoid", 2: "Softplus", 3: "Tanh"
}
ACTIVATION_IDS = {v.lower(): k for k, v in ACTIVATION_MAP.items()}

# --- Data Loading ---

def load_csv_data(file_path, delimiter=',', has_header=False):
    """Loads features and targets from a generic CSV file."""
    print(f"Loading CSV data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # skiprows calculation remains the same
        skiprows = 1 if has_header else 0
        # CORRECTED: Ensure default comment handling is used (no 'comments' arg needed)
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows) # No 'comments' arg

        # ... (rest of the function remains the same) ...
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 2: raise ValueError("CSV file must have >= 2 columns.")
        features = data[:, :-1].astype(np.float64)
        targets = data[:, -1:].astype(np.float64)
        print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features.")
        return features, targets

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def format_data_for_cpp(features, targets):
    """Converts numpy features/targets into the list[tuple[np.ndarray, np.ndarray]] format for C++."""
    if features.shape[0] != targets.shape[0]:
        raise ValueError("Number of samples in features and targets must match.")

    cpp_data = []
    for i in range(features.shape[0]):
        input_vec = features[i].flatten().astype(np.float64)
        target_vec = targets[i].flatten().astype(np.float64)
        cpp_data.append( (input_vec, target_vec) )
    return cpp_data


# --- Data Preprocessing ---

def split_data(features, targets, validation_split=0.2, shuffle=True):
    """Splits features and targets into training and validation sets."""
    if not train_test_split: # Check if scikit-learn is available
        print("Warning: scikit-learn not available, cannot split data.")
        return features, targets, None, None # Return original data as train, None for validation

    if not (0 < validation_split < 1):
        # If split is 0 or 1, don't split
        return features, targets, None, None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=validation_split, shuffle=shuffle, random_state=42 # Use fixed state for reproducibility if shuffle=True
        )
        print(f"Data split: {len(X_train)} train samples, {len(X_val)} validation samples.")
        return X_train, y_train, X_val, y_val
    except Exception as e:
        print(f"Error splitting data: {e}")
        # Fallback: return original data as training set
        return features, targets, None, None


def normalize_data(X_train, X_val=None, feature_range=(0, 1)):
    """Normalizes features using MinMaxScaler."""
    if not MinMaxScaler:
        print("Warning: scikit-learn not available, cannot normalize data.")
        return X_train, X_val, None # Return original data, no scaler

    try:
        scaler = MinMaxScaler(feature_range=feature_range)
        # Fit scaler ONLY on training data
        X_train_scaled = scaler.fit_transform(X_train)
        print(f"Training data normalized to range {feature_range}.")

        # Transform validation data using the *same* scaler
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            print("Validation data normalized using training data scaler.")

        return X_train_scaled, X_val_scaled, scaler # Return scaler for potential inverse transform later

    except Exception as e:
        print(f"Error normalizing data: {e}")
        # Fallback: return original data
        return X_train, X_val, None


# --- Plotting (minor change for clarity) ---

def plot_loss_history(log_file="training_loss_eigen.txt"):
    """Plots the loss history from the log file."""
    print(f"Attempting to plot loss from: {log_file}")
    try:
        if not os.path.exists(log_file):
             print(f"Log file '{log_file}' not found.")
             return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.ion()

        loss_data = np.loadtxt(log_file, skiprows=0)
        if loss_data.ndim == 1: loss_data = loss_data.reshape(1, -1)
        if loss_data.shape[1] < 2:
            print(f"Warning: Log file '{log_file}' has unexpected format.")
            return None

        epochs = loss_data[:, 0]
        avg_cost = loss_data[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, avg_cost, marker='.')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Cost (Loss)")
        ax.grid(True)
        fig.tight_layout()
        fig.canvas.manager.set_window_title("Loss Plot") # Set window title
        fig.show()
        print("Loss plot displayed.")
        return fig

    except Exception as e:
        print(f"Error plotting loss history: {e}")
        return None

# --- Other Utils (Unchanged) ---
def clear_loss_log(log_file="training_loss_eigen.txt"):
    """Deletes the loss log file."""
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print(f"Cleared previous loss log: {log_file}")
            return True
        except Exception as e:
            print(f"Warning: Could not clear loss log file: {e}")
            return False
    return True