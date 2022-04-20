from textfier.core import Dataset

# Defines the input sequences
sentences = ["hey", "lets go"]

# Defines the input labels
labels = [0, 0]

# Creates the dataset
dataset = Dataset(sentences=sentences, labels=labels)

print(dataset.sentences, dataset.labels)
