import textfier.utils.loader as l

# Loads an input .csv file
samples, labels = l.load_csv('data/csv/train.csv')

# Printing loaded text
print(samples, labels)
