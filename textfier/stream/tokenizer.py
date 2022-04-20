"""Tokenization-based utilities, such as sentence- and word-level tokenizers.
"""

from typing import List, Optional

from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_to_sentences(
    text: str, language: Optional[str] = "portuguese"
) -> List[str]:
    """Tokenizes text into sentence-level.

    Args:
        text: String holding the text to be tokenized.
        language: Identifier of tokenizer's language.

    Returns:
        (List[str]): Sentence-level tokens.

    """

    tokens = sent_tokenize(text, language=language)

    return tokens


def tokenize_to_words(text: str, language: Optional[str] = "portuguese") -> List[str]:
    """Tokenizes text into word-level.

    Args:
        text: String holding the text to be tokenized.
        language: Identifier of tokenizer's language.

    Returns:
        (List[str]): Word-level tokens.

    """

    tokens = word_tokenize(text, language=language)

    return tokens
