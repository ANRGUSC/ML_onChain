import numpy as np
import pandas as pd

# Number of data points
num_samples = 1000

# Randomly generate data for 5 features
data = np.random.rand(num_samples, 5)

# For the sake of simplicity, let's generate labels based on some arbitrary condition.
# Here's an example where we classify based on the sum of the 5 features:
labels = np.where(data.sum(axis=1) > 2.5, 1, 0)

# Convert data and labels to a pandas DataFrame
df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
df['label'] = labels

# Save to csv
df.to_csv('synthetic_data.csv', index=False)
