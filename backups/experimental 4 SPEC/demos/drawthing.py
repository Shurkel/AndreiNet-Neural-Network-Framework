import matplotlib.pyplot as plt

# Function to read training data from the file
def read_training_data(filename):
    inputs = []
    labels = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            x1, x2, label = float(parts[0]), float(parts[1]), float(parts[2])
            inputs.append((x1, x2))
            labels.append(label)
    
    return inputs, labels

# Function to plot the data
def plot_training_data(inputs, labels):
    # Separate the inputs into x1 and x2
    x1 = [input[0] for input in inputs]
    x2 = [input[1] for input in inputs]
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    
    # Scatter plot for inputs
    plt.scatter(x1, x2, c=labels, cmap=plt.cm.RdYlBu, edgecolors='k', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')
    plt.colorbar(label='Expected Output')
    plt.show()

# Main function to read and plot the data
def main():
    # File where the data is stored
    filename = 'trainingData.txt'
    
    # Read the data from file
    inputs, labels = read_training_data(filename)
    
    # Plot the data
    plot_training_data(inputs, labels)

# Run the main function
if __name__ == "__main__":
    main()
