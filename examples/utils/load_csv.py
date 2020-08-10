import textfier.utils.loader as l

# Loads an input .csv file
labels, samples = l.load_csv('data/csv/train.csv')

# Printing loaded text
print(labels, samples)
