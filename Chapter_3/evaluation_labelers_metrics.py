from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

dummy_annotator_labels = ['positive', 'negative', 'positive', 'positive', 'positive']
dummy_known_labels = ['negative', 'negative', 'positive', 'positive', 'negative']
accuracy = accuracy_score(dummy_annotator_labels, dummy_known_labels)
print(f"Annotator accuracy: {accuracy*100:.2f}%")

kappa = cohen_kappa_score(dummy_annotator_labels, dummy_known_labels)
print(f"Cohen's Kappa: {kappa:.3f}")
