# python_src/ui.py
import sys
from .utils import ACTIVATION_MAP, ACTIVATION_IDS # Use relative import

# Import the C++ bindings module - ensure it's built and accessible
try:
    import andreinet_bindings as anet
except ImportError:
    print("FATAL ERROR: Could not import 'andreinet_bindings'.")
    print("Please ensure the C++ bindings are compiled correctly:")
    print("1. Navigate to the project root directory (andreiNET_python_demo/).")
    print("2. Run: python setup.py build_ext --inplace")
    sys.exit(1)


# --- Input Functions ---

def get_int_input(prompt, min_val=None, max_val=None):
    """Gets integer input from the user with validation."""
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float_input(prompt, min_val=None, max_val=None):
    """Gets float input from the user with validation."""
    while True:
        try:
            value = float(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
            elif max_val is not None and value > max_val:
                 print(f"Value must be at most {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 0.01).")

def get_choice_input(prompt, choices):
    """Gets a choice from a list."""
    print(prompt)
    choices_lower = [c.lower() for c in choices]
    for i, choice in enumerate(choices):
        print(f"  {i+1}. {choice}")
    while True:
        raw_input = input("Enter choice (number or name): ").strip().lower()
        try:
            # Try by number
            choice_index = int(raw_input) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index]
        except ValueError:
            # Try by name
            if raw_input in choices_lower:
                return choices[choices_lower.index(raw_input)] # Return original case
        print("Invalid choice. Please enter a valid number or name from the list.")

def get_yes_no_input(prompt):
    """Gets a yes/no answer."""
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

# --- UI Action Functions ---

def ui_create_network():
    """Guides user to create a network."""
    print("\n--- Create New Network ---")
    print("Define the network topology (number of nodes per layer).")
    print("Example: 2,4,1 for Input(2) -> Hidden(4) -> Output(1)")
    while True:
        topo_str = input("Enter layer sizes separated by commas (e.g., 2,4,1): ").strip()
        try:
            topology = [int(n.strip()) for n in topo_str.split(',')]
            if len(topology) < 2:
                print("Network must have at least an input and output layer.")
            elif any(n <= 0 for n in topology):
                 print("Number of nodes in each layer must be positive.")
            else:
                print(f"Creating network with topology: {topology}")
                # Use the C++ binding constructor
                net = anet.Net(topology)
                print("Network created successfully!")
                return net
        except ValueError:
            print("Invalid format. Please enter comma-separated integers (e.g., 2,4,1).")
        except Exception as e:
             print(f"Error creating network: {e}")
             return None # Indicate failure

def ui_configure_network(net):
    """Guides user to configure the network."""
    if net is None:
        print("No network created yet.")
        return

    print("\n--- Configure Network ---")
    num_layers = net.get_layer_count()

    # 1. Configure Activations
    print("\nConfigure Activation Functions:")
    print("Available: " + ", ".join(ACTIVATION_MAP.values()))
    print("IDs: Linear(-1), ReLU(0), Sigmoid(1), Softplus(2), Tanh(3)")
    print("Recommendation: ReLU/Tanh for hidden, Sigmoid/Linear for output.")
    for i in range(num_layers):
        layer_type = "Input" if i == 0 else "Output" if i == num_layers - 1 else f"Hidden {i}"
        current_act_id = net.get_layer_activation(i)
        current_act_name = ACTIVATION_MAP.get(current_act_id, "Unknown")
        prompt = f"  Layer {i} ({layer_type}, {net.get_layer_nodes(i)} nodes) [Current: {current_act_name}]: Enter activation ID (-1 to 3): "
        while True:
             try:
                 act_id = int(input(prompt).strip())
                 if act_id in ACTIVATION_MAP:
                     net.set_layer_activation(i, act_id)
                     print(f"  Layer {i} set to {ACTIVATION_MAP[act_id]}.")
                     break
                 else:
                     print("  Invalid ID. Choose from -1, 0, 1, 2, 3.")
             except ValueError:
                 print("  Invalid input. Please enter an integer ID.")

    # 2. Configure Loss Function
    print("\nConfigure Loss Function:")
    loss_choices = ["MSE", "CROSS_ENTROPY"]
    loss_choice = get_choice_input("Choose loss function:", loss_choices)
    if loss_choice == "MSE":
        net.set_loss_function(anet.LossFunction.MSE)
        print("Loss set to MSE. Often paired with Linear output.")
    else:
        net.set_loss_function(anet.LossFunction.CROSS_ENTROPY)
        print("Loss set to Cross-Entropy. Often paired with Sigmoid output.")

    # 3. Configure Optimizer
    print("\nConfigure Optimizer:")
    opt_choices = ["SGD", "ADAM"]
    opt_choice = get_choice_input("Choose optimizer:", opt_choices)
    if opt_choice == "ADAM":
        use_defaults = get_yes_no_input("Use default Adam parameters (beta1=0.9, beta2=0.999, eps=1e-8)?")
        if use_defaults:
            net.set_optimizer(anet.OptimizerType.ADAM)
        else:
            b1 = get_float_input("Enter Adam beta1 (e.g., 0.9): ", 0.0, 1.0)
            b2 = get_float_input("Enter Adam beta2 (e.g., 0.999): ", 0.0, 1.0)
            eps = get_float_input("Enter Adam epsilon (e.g., 1e-8): ", 1e-12, 1.0)
            net.set_optimizer(anet.OptimizerType.ADAM, b1, b2, eps)
        print("Optimizer set to ADAM.")
    else:
         net.set_optimizer(anet.OptimizerType.SGD)
         print("Optimizer set to SGD.")

    # 4. Configure L2 Regularization
    print("\nConfigure L2 Regularization:")
    l2_lambda = get_float_input("Enter L2 lambda (0 for none, e.g., 0.001): ", 0.0)
    net.set_L2_regularization(l2_lambda)
    print(f"L2 Lambda set to {l2_lambda}.")

    # 5. Configure Learning Rate Decay
    print("\nConfigure Learning Rate Decay:")
    lr_decay = get_float_input("Enter LR decay factor per epoch (0 for none, e.g., 0.0005): ", 0.0)
    net.set_learning_rate_decay(lr_decay)
    print(f"LR Decay set to {lr_decay}.")

    print("\nConfiguration complete.")

def ui_display_structure(net):
    if net is None:
        print("No network created yet.")
        return
    print("\n--- Network Structure ---")
    try:
        # Get structure string from C++ binding
        structure_str = net.get_network_structure_str(show_matrices=False)
        print(structure_str)
    except Exception as e:
        print(f"Error getting network structure: {e}")