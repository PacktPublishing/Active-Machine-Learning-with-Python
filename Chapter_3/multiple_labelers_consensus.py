import pandas as pd

dummy_annotator_labels_1 = ['positive', 'negative', 'positive', 'positive', 'positive']
dummy_annotator_labels_2 = ['positive', 'negative', 'positive', 'negative', 'positive']
dummy_annotator_labels_3 = ['negative', 'negative', 'positive', 'positive', 'negative']

# DataFrame with multiple labels
df = pd.DataFrame({
    "Annotator1": dummy_annotator_labels_1,
    "Annotator2": dummy_annotator_labels_2,
    "Annotator3": dummy_annotator_labels_3
})

# Take majority vote
df["MajorityVote"] = df.mode(axis=1)[0]
print(df["MajorityVote"])
