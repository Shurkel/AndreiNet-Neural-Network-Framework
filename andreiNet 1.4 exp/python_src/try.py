
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


#small test
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
                print(net.get_network_structure_str())
                print("Network created successfully!")
                return net
        except ValueError:
            print("Invalid format. Please enter comma-separated integers (e.g., 2,4,1).")
        except Exception as e:
             print(f"Error creating network: {e}")
             return None # Indicate failure

# Test the C++ bindings
#main
if __name__ == "__main__":
    ui_create_network()
    