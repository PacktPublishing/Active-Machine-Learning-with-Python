import numpy as np

# Model's probabilities of sample 1 and 2 for the 3 classes
probs_sample_1 = np.array([0.05, 0.85, 0.10])
probs_sample_2 = np.array([0.35, 0.15, 0.50])
def entropy_score(predicted_probs):
    return -np.multiply(predicted_probs, np.nan_to_num(np.log2(predicted_probs))).sum()

# For sample 1
entropy_score_sample_1 = entropy_score(probs_sample_1)
print(f'The margin score of sample 1 is: {entropy_score_sample_1}')

# For sample 2
entropy_score_sample_2 = entropy_score(probs_sample_2)
print(f'The margin score of sample 2 is: {entropy_score_sample_2}')

entropy_scores = np.array([entropy_score_sample_1, entropy_score_sample_2])

most_informative_sample = np.argmax(entropy_scores)
print(f'The most informative sample is sample {most_informative_sample+1}')





