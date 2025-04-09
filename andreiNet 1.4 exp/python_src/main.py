# python_src/main.py
import numpy as np
import sys
import os
import time

# Use relative imports for modules within the same package
from . import utils
from . import ui

# Import the C++ bindings module - ensure it's built and accessible
try:
    import andreinet_bindings as anet
except ImportError:
    # ui.py already prints a detailed error, just exit here
    sys.exit(1)


# --- Global State ---
current_net = None
training_data = None
model_filename = None # Keep track of loaded/saved model file

# --- Main Application Logic ---

def display_main_menu():
    """Prints the main menu options."""
    print("\n--- AndreiNET Interactive Demo ---")
    print(" Status:")
    print(f"  - Network Loaded: {'Yes' if current_net else 'No'}")
    print(f"  - Data Loaded: {'Yes' if training_data else 'No'}")
    print(f"  - Model File: {model_filename if model_filename else 'N/A'}")
    print("\n Main Menu:")
    print("  1. Create New Network")
    print("  2. Configure Network")
    print("  3. Load XOR Dataset")
    # print("  4. Load Data from CSV") # TODO: Implement CSV loading
    print("  5. Display Network Structure")
    print("  6. Train Network")
    print("  7. Evaluate Network (on loaded data)")
    print("  8. Predict Single Input")
    print("  9. Save Network")
    print(" 10. Load Network")
    print("  0. Exit")
    print("---------------------------------")

def handle_train():
    global current_net, training_data
    if current_net is None:
        print("Error: Create or load a network first.")
        return
    if training_data is None:
        print("Error: Load training data first.")
        return

    print("\n--- Train Network ---")
    epochs = ui.get_int_input("Enter number of epochs (e.g., 1000): ", 1)
    batch_size = ui.get_int_input(f"Enter batch size (1 to {len(training_data)}): ", 1, len(training_data))
    initial_lr = ui.get_float_input("Enter initial learning rate (e.g., 0.01): ", 1e-7, 10.0)
    shuffle = ui.get_yes_no_input("Shuffle data each epoch?")

    # Clear previous loss log before training
    utils.clear_loss_log()

    print("\nStarting training...")
    start_time = time.time()
    try:
        # Redirect C++ output streams (optional, uncomment in bindings.cpp if desired)
        # This prevents C++ prints from cluttering the Python UI during training
        # You might want to see them initially for debugging C++ side.
        print("(C++ training output is redirected, check 'training_loss_eigen.txt' for loss)")

        # --- Call the C++ train function via bindings ---
        current_net.train(training_data, epochs, initial_lr, batch_size, shuffle)
        # ------------------------------------------------

        end_time = time.time()
        print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

        # Plot loss history from the log file
        plot_loss = ui.get_yes_no_input("Plot loss history?")
        if plot_loss:
             utils.plot_loss_history()

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

def handle_evaluate():
    global current_net, training_data
    if current_net is None:
        print("Error: Create or load a network first.")
        return
    if training_data is None:
        print("Error: Load data first to evaluate.")
        return

    print("\n--- Evaluate Network ---")
    print("Predictions on loaded dataset:")
    correct_predictions = 0
    total_samples = len(training_data)

    for i, (features, target) in enumerate(training_data):
        try:
            prediction = current_net.predict(features) # Get Eigen::VectorXd back

            # Assuming binary classification for XOR with Sigmoid output
            target_val = target[0] # Get scalar value
            pred_val = prediction[0]
            predicted_class = 1 if pred_val > 0.5 else 0
            target_class = int(round(target_val))

            is_correct = (predicted_class == target_class)
            if is_correct:
                correct_predictions += 1

            print(f"Sample {i}: Input={features} Target={target_val:.1f} -> Prediction={pred_val:.4f} (Class: {predicted_class}) {'[Correct]' if is_correct else '[Incorrect]'}")

        except Exception as e:
            print(f"Error predicting sample {i}: {e}")

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")


def handle_predict_single():
    global current_net
    if current_net is None:
        print("Error: Create or load a network first.")
        return

    print("\n--- Predict Single Input ---")
    try:
        # Determine expected input size from network's first layer
        input_size = current_net.get_layer_nodes(0)
        print(f"Network expects input with {input_size} features.")
        input_str = input(f"Enter {input_size} comma-separated feature values (e.g., 1.0,0.0): ").strip()
        features_list = [float(x.strip()) for x in input_str.split(',')]

        if len(features_list) != input_size:
            print(f"Error: Expected {input_size} features, but got {len(features_list)}.")
            return

        # Convert to NumPy array (required by Eigen binding)
        features_np = np.array(features_list, dtype=np.float64)

        prediction = current_net.predict(features_np)
        output_size = current_net.get_layer_nodes(current_net.get_layer_count() - 1)

        print(f"\nInput: {features_np}")
        print(f"Raw Prediction (Output Layer Activation): {prediction}")
        # Add interpretation based on output layer size/activation if needed
        if output_size == 1:
             pred_val = prediction[0]
             # Check output activation (assuming it's available via binding if needed)
             # last_layer_act = current_net.get_layer_activation(current_net.get_layer_count() - 1)
             print(f"Predicted Value (Output Node 0): {pred_val:.5f}")
             if -1 <= pred_val <= 1: # Tanh range check is tricky, sigmoid is 0 to 1
                 print(f"Predicted Class (Threshold 0.5 for Sigmoid): {1 if pred_val > 0.5 else 0}")


    except ValueError:
        print("Invalid input format. Please enter comma-separated numbers.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def handle_save():
    global current_net, model_filename
    if current_net is None:
        print("Error: Create or load a network first.")
        return

    print("\n--- Save Network ---")
    default_name = model_filename if model_filename else "my_andreinet.bin"
    filename = input(f"Enter filename to save model [{default_name}]: ").strip()
    if not filename:
        filename = default_name

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", filename)

    try:
        current_net.save(save_path)
        print(f"Network saved successfully to: {save_path}")
        model_filename = filename # Update tracked filename
    except Exception as e:
        print(f"Error saving network: {e}")


def handle_load():
    global current_net, model_filename
    print("\n--- Load Network ---")
    print("IMPORTANT: You must create a NEW network with the EXACT SAME topology")
    print("           (layer sizes) as the saved network BEFORE loading.")

    default_name = model_filename if model_filename else "my_andreinet.bin"
    filename = input(f"Enter filename to load model [{default_name}]: ").strip()
    if not filename:
        filename = default_name

    load_path = os.path.join("models", filename)

    if not os.path.exists(load_path):
        print(f"Error: File not found: {load_path}")
        return

    if current_net is None:
         print("Please create a network with the correct topology first (Option 1).")
         # Optionally prompt to create one now?
         return

    try:
        print(f"Attempting to load weights into the current network from: {load_path}")
        current_net.load(load_path)
        print("Network loaded successfully!")
        print("Note: Runtime settings (optimizer type, L2, decay) are not saved in the file.")
        print("      You may need to reconfigure them if continuing training.")
        model_filename = filename # Update tracked filename
        # Reset Adam state is handled internally in C++ load now
    except Exception as e:
        print(f"Error loading network: {e}")
        print("Ensure the current network's topology matches the saved file.")


def main_loop():
    """Runs the main interactive loop."""
    global current_net, training_data # Allow modification

    while True:
        display_main_menu()
        choice = ui.get_int_input("Enter your choice: ", 0, 10)

        if choice == 1:
            # Create Network (replaces current one)
            new_net = ui.ui_create_network()
            if new_net:
                current_net = new_net
                model_filename = None # Reset model file tracking
        elif choice == 2:
            # Configure Network
            ui.ui_configure_network(current_net)
        elif choice == 3:
            # Load XOR Data
            training_data = utils.load_xor_data()
        elif choice == 4:
            # Load Data from CSV (Placeholder)
            print("Load from CSV - Not implemented yet.")
            # file_path = input("Enter CSV file path: ").strip()
            # training_data = utils.load_csv_data(file_path) # Need to implement load_csv_data
        elif choice == 5:
            # Display Structure
            ui.ui_display_structure(current_net)
        elif choice == 6:
            # Train Network
            handle_train()
        elif choice == 7:
            # Evaluate Network
            handle_evaluate()
        elif choice == 8:
            # Predict Single Input
            handle_predict_single()
        elif choice == 9:
            # Save Network
            handle_save()
        elif choice == 10:
            # Load Network
            handle_load()
        elif choice == 0:
            # Exit
            print("Exiting AndreiNET Demo. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

        # Pause briefly before showing menu again
        # input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    # Start the main application loop
    main_loop()