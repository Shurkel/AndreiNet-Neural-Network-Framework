# create_dataset.py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_moons

# --- Parameters ---
N_SAMPLES = 400      # Total number of data points
NOISE_LEVEL = 0.15   # Amount of random noise added (0.0 to 1.0)
FILENAME = "two_moons_data.csv"
DATASET_DIR = "datasets"
PLOT_DATA = True     # Set to True to visualize the generated data

# --- Generate Data ---
print(f"Generating '{FILENAME}'...")
print(f"  Samples: {N_SAMPLES}")
print(f"  Noise:   {NOISE_LEVEL}")

# make_moons generates two interleaving half circles
# X will be (N_SAMPLES, 2) array of features (coordinates)
# y will be (N_SAMPLES,) array of labels (0 or 1)
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE_LEVEL, random_state=42) # Use random_state for reproducibility

# --- Combine Features and Target for CSV ---
# Reshape y to be a column vector for stacking
y_col = y.reshape(-1, 1)
# Stack features and target horizontally
data_to_save = np.hstack((X, y_col))

# --- Save to CSV ---
output_path = os.path.join(DATASET_DIR, FILENAME)
try:
    os.makedirs(DATASET_DIR, exist_ok=True)
    header = "Feature_X,Feature_Y,Target_Class"
    # CORRECTED: Remove comments='' to let savetxt add the default '#'
    np.savetxt(output_path, data_to_save, delimiter=',', fmt='%.6f', header=header)
    print(f"Dataset saved successfully to: {output_path}")
    print(f"Format: #{header}") # Show that '#' is expected
except Exception as e:
    print(f"Error saving dataset: {e}")

# --- Optional: Plot the Data ---
if PLOT_DATA:
    print("Plotting generated data...")
    try:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, edgecolors='k', s=50)
        plt.title(f'Generated Two Moons Dataset (Noise={NOISE_LEVEL})')
        plt.xlabel('Feature X')
        plt.ylabel('Feature Y')
        plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.show()
    except Exception as e:
        print(f"Error plotting data: {e}")
        print("Ensure matplotlib and scikit-learn are installed (`pip install matplotlib scikit-learn`)")