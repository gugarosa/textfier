from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_to_sent(text, language='portuguese'):
    """Tokenizes text into sentence-level.

    Args:
        text (str): String holding the text to be tokenized.
        language (str): Tokenizer's language.

    Returns:
        List of sentence-level tokens.

    """

    # Applies the sentence tokenizer
    tokens = sent_tokenize(text, language=language)

    return tokens


def tokenize_to_word(text, language='portuguese'):
    """Tokenizes text into word-level.

    Args:
        text (str): String holding the text to be tokenized.
        language (str): Tokenizer's language.

    Returns:
        List of word-level tokens.

    """

    # Applies the word tokenizer
    tokens = word_tokenize(text, language=language)

    return tokens
