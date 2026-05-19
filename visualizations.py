import matplotlib.pyplot as plt
import numpy as np


fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Visualizations Part 1', fontsize=14, fontweight='bold')

# Plot 1: Simple Bar Graph
categories = ('A', 'B', 'C', 'D', 'E')
values = [23, 45, 56, 78, 32]
axs1[0].bar(x=categories, height=values, label='abc', color='red', width=0.5)
axs1[0].set_xlabel('Categories')
axs1[0].set_ylabel('Values')
axs1[0].set_title('Simple Bar Graph')
axs1[0].legend()

# Plot 2: Line Graph with Markers
x = ["Jan", "Feb", "Mar", "Apr", "May"]
y = [10, 20, 25, 30, 15]
axs1[1].plot(x, y, marker='o')
axs1[1].set_xlabel('Month')
axs1[1].set_ylabel('Sale')
axs1[1].set_title('Line Graph with Markers')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Visualizations Part 2', fontsize=14, fontweight='bold')

# Plot 3: Scatter Plot with Regression Line
x_scat = [1, 2, 3, 4, 5]
y_scat = [10, 15, 20, 25, 30]
slope, intercept = np.polyfit(x_scat, y_scat, 1)
fit_line = [slope * i + intercept for i in x_scat]
axs2[0].scatter(x_scat, y_scat)
axs2[0].plot(x_scat, fit_line, color='red')
axs2[0].set_xlabel('X-axis')
axs2[0].set_ylabel('Y-axis')
axs2[0].set_title('Scatter Plot with Regression Line')

# Plot 4: Histogram
data_hist = [1, 2, 3, 4, 5, 6, 3, 8, 8, 9, 8, 9, 8, 9, 8, 9, 9, 10, 2]
axs2[1].hist(data_hist, bins=3, alpha=1, color='g', edgecolor='b')
axs2[1].set_title('Histogram')
axs2[1].set_xlabel('Value')
axs2[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()