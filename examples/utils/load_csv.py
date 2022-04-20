from textfier.utils import loader

# Loads an input .csv file
samples, labels = loader.load_csv("data/csv/train.csv")

# Printing loaded text
print(samples, labels)
