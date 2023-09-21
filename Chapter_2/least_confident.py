import numpy as np

# Model's probabilities of sample 1 and 2 for the 3 classes
probs_sample_1 = np.array([0.05, 0.85, 0.10])
probs_sample_2 = np.array([0.35, 0.15, 0.50])

def least_confident_score(predicted_probs):
    return 1 - predicted_probs[np.argmax(predicted_probs)]

LC_samples_scores = np.array([least_confident_score(probs_sample_1), least_confident_score(probs_sample_2)])
print(f'Least confident score for sample 1 is: {LC_samples_scores[0]}')
print(f'Least confident score for sample 2 is: {LC_samples_scores[1]}')

most_informative_sample = np.argmax(LC_samples_scores)
print(f'The most informative sample is sample {most_informative_sample+1}')







