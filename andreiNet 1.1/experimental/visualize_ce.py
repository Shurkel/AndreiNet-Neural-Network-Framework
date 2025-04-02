
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import os

# Load the dataset
data = np.loadtxt('output/data/ce_dataset.dat')
X = data[:, :2]  # features
y = data[:, 2]   # labels

# Load loss data
loss_data = np.loadtxt('output/data/ce_loss.dat')
epochs = loss_data[:, 0]
losses = loss_data[:, 1]

# Create figure with subplots
fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(121)  # For decision boundary
ax2 = plt.subplot(122)  # For loss plot

# Plot the loss curve
ax2.plot(epochs, losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

# Plot the dataset points
class0 = X[y == 0]
class1 = X[y == 1]
ax1.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0', alpha=0.7)
ax1.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1', alpha=0.7)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Decision Boundary Evolution')
ax1.grid(True)
ax1.legend()

# Find all prediction files
pred_files = sorted([f for f in os.listdir('output/predictions') if f.startswith('ce_pred_')])
epochs_saved = [int(f.split('_')[-1].split('.')[0]) for f in pred_files]

# Create colormap for decision boundary
contour = None

def update(frame):
    global contour
    
    # Clear previous contour
    if contour:
        for coll in contour.collections:
            coll.remove()
    
    # Load prediction data
    epoch = epochs_saved[frame]
    pred_data = np.loadtxt(f'output/predictions/ce_pred_{epoch}.dat')
    
    # Reshape data for contour plot
    x_unique = np.sort(np.unique(pred_data[:, 0]))
    y_unique = np.sort(np.unique(pred_data[:, 1]))
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
    Z_grid = np.zeros_like(X_grid)
    
    # Fill Z values from predictions
    for i, (x, y, z) in enumerate(pred_data):
        ix = np.where(x_unique == x)[0][0]
        iy = np.where(y_unique == y)[0][0]
        Z_grid[iy, ix] = z
    
    # Plot decision boundary
    contour = ax1.contourf(X_grid, Y_grid, Z_grid, 20, alpha=0.6, cmap=cm.coolwarm)
    ax1.set_title(f'Decision Boundary (Epoch {epoch})')
    
    return contour,

# Create animation
ani = FuncAnimation(fig, update, frames=len(pred_files), interval=300)

# Save animation
ani.save('output/visuals/ce_training_animation.gif', writer='pillow', fps=5)

plt.tight_layout()
plt.savefig('output/visuals/ce_final_plot.png')
plt.show()
