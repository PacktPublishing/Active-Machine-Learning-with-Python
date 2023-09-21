import numpy as np

# Model's probabilities of sample 1 and 2 for the 3 classes
probs_sample_1 = np.array([0.05, 0.85, 0.10])
probs_sample_2 = np.array([0.35, 0.15, 0.50])

def margin_score(predicted_probs):
    predicted_probs_max_1 = np.sort(predicted_probs)[-1]
    predicted_probs_max_2 = np.sort(predicted_probs)[-2]
    margin_score = predicted_probs_max_1 - predicted_probs_max_2
    return margin_score

# For sample 1
margin_score_sample_1 = margin_score(probs_sample_1)
print(f'The margin score of sample 1 is: {margin_score_sample_1}')

# For sample 2
margin_score_sample_2 = margin_score(probs_sample_2)
print(f'The margin score of sample 2 is: {margin_score_sample_2}')

margin_scores = np.array([margin_score_sample_1, margin_score_sample_2])

most_informative_sample = np.argmin(margin_scores)
print(f'The most informative sample is sample {most_informative_sample+1}')




