import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Number of samples
N = 1000

# Generate N random two-dimensional data points
X = np.random.rand(N, 2)

# Define a line
line_points = np.array([[0, 0], [1, 1]])
line_vec = np.diff(line_points, axis=0)

# Compute the signed distance of each point to the line
signed_distances = np.cross(line_vec, np.subtract(X, line_points[0]))

# Classify points as -1 or 1
y = np.sign(signed_distances)

# Create a DataFrame
df = pd.DataFrame(np.column_stack([X, y]), columns=['x1', 'x2', 'label'])

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)
print(df)