import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Set plot style
plt.style.use('seaborn-darkgrid')

def plot_decision_boundary():
    # Load test grid data
    data = np.loadtxt('test_grid.txt')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    # Reshape to grid
    grid_size = int(math.sqrt(len(x)))
    X = x.reshape(grid_size, grid_size)
    Y = y.reshape(grid_size, grid_size)
    Z = z.reshape(grid_size, grid_size)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Output value')
    
    # Load and plot sample training data
    try:
        train_data = np.loadtxt('training_data_sample.txt')
        # Plot points with different colors based on class
        class1 = train_data[train_data[:, 2] > 0.5]
        class2 = train_data[train_data[:, 2] <= 0.5]
        
        plt.scatter(class1[:, 0], class1[:, 1], c='white', edgecolor='black', label='Class 1 (Inside)', s=50)
        plt.scatter(class2[:, 0], class2[:, 1], c='red', edgecolor='black', label='Class 0 (Outside)', s=50)
    except Exception as e:
        print(f"Could not plot training data: {e}")
    
    # Draw the true circle (radius=5)
    circle = plt.Circle((0, 0), 5, fill=False, color='red', linestyle='--', linewidth=2, label='True boundary')
    plt.gca().add_patch(circle)
    
    plt.title('Neural Network Decision Boundary (Circular Dataset)', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
    
def plot_training_metrics():
    try:
        # Load training metrics
        df = pd.read_csv('training_metrics.txt')
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot loss
        color = 'tab:red'
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Cross-Entropy Loss', color=color, fontsize=14)
        ax1.plot(df['Epoch'], df['Loss'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for accuracy
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color, fontsize=14)
        ax2.plot(df['Epoch'], df['Accuracy'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.05])
        
        plt.title('Training Metrics - Cross-Entropy Loss and Accuracy', fontsize=16)
        plt.grid(True)
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"Could not plot training metrics: {e}")

if __name__ == "__main__":
    plot_decision_boundary()
    plot_training_metrics()
    plt.show()
