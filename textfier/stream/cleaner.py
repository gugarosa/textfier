"""Cleaning-based utilities, such as stemmers and stopword removal.
"""

from typing import List, Optional

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

import textfier.stream.tokenizer as t


def clean_sentences(
    sentences: List[str],
    remove_stopwords: Optional[bool] = False,
    language: Optional[str] = "portuguese",
) -> List[str]:
    """Stems and removes stopwords from a set of sentence-level tokens using the RSLPStemmer.

    Args:
        sentences: Sentences to be stemmed.
        remove_stopwords: Whether stopwords should be removed or not.

    Returns:
        (List[str]): Stemmed tokens.

    """

    # Creates a list of stemmed sentences
    stemmed_sentences = []

    for s in sentences:
        # Tokenizes the sentence into words
        words = t.tokenize_to_words(s)

        # Stems the words
        stemmed_words = clean_words(words, remove_stopwords, language)

        # Re-builds the sentence and appends to list
        stemmed_sentences.append(" ".join(stemmed_words))

    return stemmed_sentences


def clean_words(
    words: List[str],
    remove_stopwords: Optional[bool] = False,
    language: Optional[str] = "portuguese",
) -> List[str]:
    """Stems and removes stopwords from a set of word-level tokens using the RSLPStemmer.

    Args:
        words: Tokens to be stemmed.
        remove_stopwords: Whether stopwords should be removed or not.
        language: Identifier of stopwords' language.

    Returns:
        (List[str]): Stemmed tokens.

    """

    # Creates the RSLP stemmer
    stemmer = RSLPStemmer()

    if remove_stopwords:
        # Gathers the stopwords
        stop_words = stopwords.words(language)

        # Stems and removes the stopwords
        stemmed_words = [
            stemmer.stem(word) for word in words if word.lower() not in stop_words
        ]

    else:
        # Just stems the words
        stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words
