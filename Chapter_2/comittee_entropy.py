import numpy as np

p1 = np.array([[0.6, 0.4], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3],
               [0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])

p2 = np.array([[0.3, 0.7], [0.1, 0.9], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6],
               [0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8]])

# Average probabilities per class
p_class0 = (p1[:, 0] + p2[:, 0]) / 2
p_class1 = (p1[:, 1] + p2[:, 1]) / 2

p_avg = np.concatenate((p_class0.reshape(-1, 1), p_class1.reshape(-1, 1)), axis=1)

# Calculate entropy
H = -np.sum(p_avg * np.log2(p_avg), axis=1)

query_index = np.argmax(H)
print(f"Data point {query_index+1} selected with maximum entropy of {H[query_index]}")
