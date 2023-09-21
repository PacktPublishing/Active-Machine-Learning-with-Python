import numpy as np

p1 = np.array([[0.6, 0.4], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3],
               [0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])

p2 = np.array([[0.3, 0.7], [0.1, 0.9], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6],
               [0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8]])

KL1 = np.sum(p1 * np.log(p1 / p2), axis=1)
KL2 = np.sum(p2 * np.log(p2 / p1), axis=1)
avg_KL = (KL1 + KL2) / 2

print("KL(M1||M2):", KL1)
print("KL(M2||M1):", KL2)
print("Average KL:", avg_KL)

query_index = np.argmax(avg_KL)
print("\nData point", query_index+1, "selected with max average KL of", avg_KL[query_index])
