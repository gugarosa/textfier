from core.extractor import Extractor

# Defining file to be extracted
FILE_PATH = 'data/CNSP-344-2016.pdf'

# Creating the extractor itself
e = Extractor(FILE_PATH)

# Printing out the raw extracted text
print(e.raw)
