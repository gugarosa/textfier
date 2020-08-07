"""Tokenization-based utilities, such as sentence- and word-level tokenizers.
"""

from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_to_sentences(text, language='portuguese'):
    """Tokenizes text into sentence-level.

    Args:
        text (str): String holding the text to be tokenized.
        language (str): Identifier of tokenizer's language.

    Returns:
        List of sentence-level tokens.

    """

    # Applies the sentence tokenizer
    tokens = sent_tokenize(text, language=language)

    return tokens


def tokenize_to_words(text, language='portuguese'):
    """Tokenizes text into word-level.

    Args:
        text (str): String holding the text to be tokenized.
        language (str): Identifier of tokenizer's language.

    Returns:
        List of word-level tokens.

    """

    # Applies the word tokenizer
    tokens = word_tokenize(text, language=language)

    return tokens
