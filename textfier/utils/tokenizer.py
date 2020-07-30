"""Tokenization-based utilities, such as sentence- and word-level tokenizers.
"""

from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_to_sentences(text):
    """Tokenizes text into sentence-level.

    Args:
        text (str): String holding the text to be tokenized.

    Returns:
        List of sentence-level tokens.

    """

    # Applies the sentence tokenizer
    tokens = sent_tokenize(text, language='portuguese')

    return tokens


def tokenize_to_words(text):
    """Tokenizes text into word-level.

    Args:
        text (str): String holding the text to be tokenized.

    Returns:
        List of word-level tokens.

    """

    # Applies the word tokenizer
    tokens = word_tokenize(text, language='portuguese')

    return tokens
