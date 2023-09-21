import numpy as np

# Predicted labels from 2 committee members
y1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

# Calculate disagreement
disagreement = np.abs(y1 - y2)

# Find index of point with max disagreement
query_index = np.argmax(disagreement)

print(f"Data point {query_index+1} selected with maximum disagreement")

