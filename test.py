import numpy as np
import matplotlib.pyplot as plt

# Create a grid with one extra point in each dimension
x = np.linspace(-3, 2, 13)  # 13 points for 12 cells
y = np.linspace(-3, 2, 13)
X, Y = np.meshgrid(x, y)

# Create the intensity values on the cell centers (one fewer in each dimension)
x_centers = (x[:-1] + x[1:]) / 2  # Cell centers
y_centers = (y[:-1] + y[1:]) / 2
Xc, Yc = np.meshgrid(x_centers, y_centers)

# Create the intensity values with multiple peaks
Z = (np.exp(-(Xc**2 + Yc**2)/0.5) +  # Central peak
     0.7 * np.exp(-((Xc-2)**2 + Yc**2)/0.8) +  # Right peak
     0.5 * np.exp(-((Xc+2)**2 + Yc**2)/0.8))   # Left peak

# Create the plot
plt.figure(figsize=(8, 8))
plt.pcolormesh(X, Y, Z, cmap='viridis', shading='flat')

# Set the axis limits
plt.xlim(-3, 2)
plt.ylim(-3, 2)

# Adjust aspect ratio
plt.gca().set_aspect('equal')

plt.show()

# save
plt.savefig('seaborn.png')
