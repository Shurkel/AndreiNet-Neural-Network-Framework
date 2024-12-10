import matplotlib.pyplot as plt

# Function to read SSR progress from the file
def read_ssr_progress(filename):
    ssr_values = []

    # Open the file and read the SSR values
    with open(filename, 'r') as f:
        for line in f:
            # Assuming each line contains just the SSR value
            try:
                ssr_value = float(line.strip())
                ssr_values.append(ssr_value)
            except ValueError:
                continue  # Skip invalid lines (if any)
    
    # Generate epochs based on the number of SSR values
    epochs = list(range(1, len(ssr_values) + 1))

    return epochs, ssr_values

# Function to plot the SSR progress over epochs
def plot_ssr_progress(epochs, ssr_values):
    plt.figure(figsize=(8, 6))

    # Plotting SSR over epochs
    plt.plot(epochs, ssr_values, marker='o', linestyle='-', color='b', label='SSR')
    
    plt.xlabel('Epochs')
    plt.ylabel('SSR (Sum of Squared Residues)')
    plt.title('SSR Progress Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to read and plot SSR progress
def main():
    # The file containing SSR progress (no epoch numbers, only SSR values)
    filename = 'test.txt'

    # Read SSR data from file
    epochs, ssr_values = read_ssr_progress(filename)
    
    # Plot SSR evolution over epochs
    plot_ssr_progress(epochs, ssr_values)

# Run the main function
if __name__ == "__main__":
    main()
