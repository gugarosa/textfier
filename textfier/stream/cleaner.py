"""Cleaning-based utilities, such as stemmers and stopword removal.
"""

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

import textfier.stream.tokenizer as t


def clean_sentences(sentences, remove_stopwords=False, language='portuguese'):
    """Stems and removes stopwords from a set of sentence-level tokens using the RSLPStemmer.

    Args:
        sentences (list): Sentences to be stemmed.
        remove_stopwords (bool): Whether stopwords should be removed or not.

    Returns:
        List of stemmed tokens.

    """

    # Creates a list of stemmed sentences
    stemmed_sentences = []

    # For every possible sentence
    for s in sentences:
        # Tokenizes the sentence into words
        words = t.tokenize_to_words(s)

        # Stems the words
        stemmed_words = clean_words(words, remove_stopwords, language)

        # Re-builds the sentence and appends to list
        stemmed_sentences.append(' '.join(stemmed_words))

    return stemmed_sentences


def clean_words(words, remove_stopwords=False, language='portuguese'):
    """Stems and removes stopwords from a set of word-level tokens using the RSLPStemmer.

    Args:
        words (list): Tokens to be stemmed.
        remove_stopwords (bool): Whether stopwords should be removed or not.
        language (str): Identifier of stopwords' language.

    Returns:
        List of stemmed tokens.

    """

    # Creates the RSLP stemmer
    stemmer = RSLPStemmer()

    # Checks if stopwords are supposed to be removed
    if remove_stopwords:
        # Gathers the stopwords
        stop_words = stopwords.words(language)

        # Stems and removes the stopwords
        stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]

    # If stopwords are not supposed to be removed
    else:
        # Just stems the words
        stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words
