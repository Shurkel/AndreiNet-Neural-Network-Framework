import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Read the data
with open('test_ce.txt', 'r') as f:
    data = [float(line.strip()) for line in f.readlines()]

# Create x and y arrays
x = np.arange(len(data))
y = np.array(data)

# Calculate differences between consecutive points
dy = np.diff(y)

# Create points and segments for LineCollection
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a color array: red if y increases, green if y decreases
colors = ['red' if dy_i > 0 else 'green' for dy_i in dy]

# Create the line collection
lc = LineCollection(segments, colors=colors, linewidth=2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.add_collection(lc)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(min(y), max(y))
ax.set_title('Cross-Entropy Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.grid(True)

plt.savefig('cross_entropy_progress.png')
plt.show()