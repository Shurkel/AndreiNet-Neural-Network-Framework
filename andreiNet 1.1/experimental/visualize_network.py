import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
import math

class NetworkVisualizer:
    def __init__(self, network_dir="output/network"):
        self.network_dir = network_dir
        self.network_files = sorted(glob.glob(f"{network_dir}/ce_network_epoch_*.dat"),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if not self.network_files:
            raise FileNotFoundError(f"No network files found in {network_dir}")
        
        print(f"Found {len(self.network_files)} network state files")
        
        # Load the first file to get network architecture
        self.load_network(self.network_files[0])
        
    def load_network(self, file_path):
        """Load network architecture and weights from file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Parse number of layers
            num_layers = int(lines[0].strip())
            
            # Parse layer sizes
            layer_sizes = list(map(int, lines[1].strip().split()))
            
            # Parse weights
            weights_flat = list(map(float, lines[2].strip().split()))
            
            # Parse biases
            biases_flat = list(map(float, lines[3].strip().split()))
            
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        
        # Reshape weights into a list of matrices
        self.weights = []
        idx = 0
        for i in range(num_layers - 1):
            layer_weights = np.zeros((layer_sizes[i], layer_sizes[i+1]))
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i+1]):
                    layer_weights[j, k] = weights_flat[idx]
                    idx += 1
            self.weights.append(layer_weights)
        
        # Reshape biases into a list of vectors
        self.biases = []
        idx = 0
        for i in range(1, num_layers):  # Skip input layer (no biases)
            layer_biases = np.zeros(layer_sizes[i])
            for j in range(layer_sizes[i]):
                layer_biases[j] = biases_flat[idx]
                idx += 1
            self.biases.append(layer_biases)
            
        return self.layer_sizes, self.weights, self.biases
        
    def visualize_network(self, ax, weights, title):
        """Create a visualization of the network architecture with weights"""
        ax.clear()
        ax.axis('off')
        ax.set_title(title)
        
        # Spacing parameters
        vertical_spacing = 1.0
        horizontal_spacing = 2.5
        
        # Calculate the maximum layer size for vertical centering
        max_layer_size = max(self.layer_sizes)
        
        # Node positions
        node_positions = []
        
        # Draw nodes for each layer
        for l, layer_size in enumerate(self.layer_sizes):
            layer_positions = []
            
            # Calculate vertical positions for each node in this layer
            start_y = (max_layer_size - layer_size) * vertical_spacing / 2
            
            for n in range(layer_size):
                # Position the node
                x = l * horizontal_spacing
                y = start_y + n * vertical_spacing
                
                # Store position
                layer_positions.append((x, y))
                
                # Draw the node
                circle = plt.Circle((x, y), 0.2, fill=True, 
                                    color='lightblue' if l == 0 else 
                                          'lightgreen' if l == len(self.layer_sizes)-1 else 
                                          'lightgray')
                ax.add_artist(circle)
                
                # Add node label 
                if l == 0:
                    ax.text(x, y, f"Input {n+1}", ha='center', va='center', fontsize=8)
                elif l == len(self.layer_sizes)-1:
                    ax.text(x, y, f"Output", ha='center', va='center', fontsize=8)
                else:
                    ax.text(x, y, f"H{l}_{n+1}", ha='center', va='center', fontsize=8)
            
            node_positions.append(layer_positions)
        
        # Draw connections between layers
        for l in range(len(self.layer_sizes) - 1):
            layer_weights = weights[l]
            
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l+1]):
                    start_x, start_y = node_positions[l][i]
                    end_x, end_y = node_positions[l+1][j]
                    
                    # Calculate weight value and normalize for visualization
                    weight = layer_weights[i, j]
                    
                    # Determine line thickness based on absolute weight value
                    # Scale between 0.5 and 3.0
                    abs_weight = abs(weight)
                    max_weight = np.max(np.abs(layer_weights))
                    thickness = 0.5 + 2.5 * (abs_weight / max_weight) if max_weight > 0 else 1.0
                    
                    # Determine line color based on weight sign
                    color = 'red' if weight < 0 else 'blue'
                    
                    # Draw the connection
                    ax.plot([start_x, end_x], [start_y, end_y], 
                           color=color, linewidth=thickness, alpha=0.6)
        
        # Set axis limits with some padding
        ax.set_xlim(-1, horizontal_spacing * (len(self.layer_sizes) - 1) + 1)
        ax.set_ylim(-1, vertical_spacing * max_layer_size + 1)
        
        # Add legend
        ax.plot([], [], color='blue', linewidth=2, label='Positive Weight')
        ax.plot([], [], color='red', linewidth=2, label='Negative Weight')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        return ax

    def create_animation(self, interval=500):
        """Create an animation of the network weights changing during training"""
        # Set up the figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        def update(frame):
            file_path = self.network_files[frame]
            epoch = int(file_path.split('_')[-1].split('.')[0])
            _, weights, _ = self.load_network(file_path)
            self.visualize_network(ax, weights, f"Network Architecture and Weights (Epoch {epoch})")
            return ax,
        
        ani = FuncAnimation(fig, update, frames=len(self.network_files), 
                            interval=interval, blit=True)
        
        # Save the animation
        ani.save('output/visuals/network_evolution.gif', writer='pillow', fps=2)
        plt.close()
        print("Animation saved to output/visuals/network_evolution.gif")
        
        # Also show the final state
        fig, ax = plt.subplots(figsize=(12, 8))
        _, weights, _ = self.load_network(self.network_files[-1])
        epoch = int(self.network_files[-1].split('_')[-1].split('.')[0])
        self.visualize_network(ax, weights, f"Final Network Architecture and Weights (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig('output/visuals/final_network.png')
        plt.show()

    def visualize_weight_distributions(self):
        """Visualize how weight distributions change over training"""
        # Select a subset of network files to analyze
        num_files = len(self.network_files)
        files_to_analyze = [self.network_files[i] for i in 
                          np.linspace(0, num_files-1, min(5, num_files), dtype=int)]
        
        # Create figure
        fig, axes = plt.subplots(len(files_to_analyze), len(self.layer_sizes)-1, 
                                 figsize=(15, 3*len(files_to_analyze)))
        
        if len(files_to_analyze) == 1:
            axes = np.array([axes])
        
        for i, file_path in enumerate(files_to_analyze):
            epoch = int(file_path.split('_')[-1].split('.')[0])
            _, weights, _ = self.load_network(file_path)
            
            for j, weight_matrix in enumerate(weights):
                ax = axes[i, j]
                ax.hist(weight_matrix.flatten(), bins=30, alpha=0.7)
                ax.set_title(f"Layer {j+1}->{j+2}, Epoch {epoch}")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig('output/visuals/weight_distributions.png')
        plt.show()

if __name__ == "__main__":
    try:
        visualizer = NetworkVisualizer()
        
        print("Creating network animation...")
        visualizer.create_animation()
        
        print("Creating weight distribution visualizations...")
        visualizer.visualize_weight_distributions()
        
        print("Visualization complete. Check the output/visuals directory for results.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the training demo first to generate network state files.")
