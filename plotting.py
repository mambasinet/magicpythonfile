import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def blurring_matrix(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][(i-2)%n] = 0.125
        matrix[i][(i-1)%n] = 0.250
        matrix[i][i] = 0.250
        matrix[i][(i+1)%n] = 0.250
        matrix[i][(i+2)%n] = 0.125
    return matrix

# distribution = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
distribution = np.array([0.19959409, 0.22200935, 0.16217505, 0.09357177, 0.06299155,
       0.12200935, 0.19877759, 0.23017432, 0.16299155, 0.08540681,
       0.06217505, 0.13017432])


blurring_matrix = blurring_matrix(12)
blurred_distribution = np.dot(blurring_matrix, distribution)

def richardson_lucy(blurred_distribution, blurring_matrix, iterations):
    # f_0=(1/12)*np.array([1, 1, 1,1,1,1,1,1,1,1,1,1])
    estimate=(1/12)*np.array([1, 1, 1,1,1,1,1,1,1,1,1,1])
    # estimate = blurred_distribution.copy()
    for i in range(iterations):
        blurred_estimate = np.dot(blurring_matrix, estimate)
        error = np.divide(blurred_distribution, blurred_estimate, where=blurred_estimate != 0)
        correction_factor = np.dot(np.transpose(blurring_matrix), error)
        estimate *= correction_factor
    return estimate

rl = richardson_lucy(blurred_distribution, blurring_matrix, iterations=10000)

# Generate the table
table_data = np.column_stack((np.arange(12), distribution, blurred_distribution, rl))
table_headers = ['Index', 'Original', 'Blurred', 'Deblurred']
table = tabulate(table_data, headers=table_headers, floatfmt=".3f", tablefmt="latex")
print(table)
# Print the table
# print(table)

# Plot the data
plt.plot(np.arange(12), rl, 'k-', linewidth=2.5, label='Deblurred')
plt.plot(np.arange(12), blurred_distribution, '--', linewidth=2.5, label='Blurred')
plt.plot(np.arange(12), distribution, 'r:', linewidth=2.5, label='Original')

# Set the plot limits and labels
plt.xlim([0, 11])
# plt.ylim([0, 1])
plt.xlabel('Index')
plt.ylabel('Value')

# Add a legend
plt.legend(loc='upper right')

# Save the figure as a high-resolution image
plt.savefig('fig.png')

# Show the plot
plt.show()
rl
