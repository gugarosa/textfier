from nltk.stem import RSLPStemmer


def stem_tokens(tokens):
    """
    """

    #
    stemmer = RSLPStemmer()

    #
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    return stemmed_tokens