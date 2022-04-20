import textfier.stream.cleaner as c
import textfier.stream.tokenizer as t

# Defines an input string
s = "The cakes costs about $12,00 in São Paulo. Please, buy two.\n\nThanks."

# Tokenizes, stems sentences and optionally remove stopwords
sentences = t.tokenize_to_sentences(s, language="english")
stemmed_sentences = c.clean_sentences(
    sentences, remove_stopwords=False, language="english"
)
print(f"Sentences: {stemmed_sentences}")

# Tokenizes, stems words and optionally remove stopwords
words = t.tokenize_to_words(s, language="english")
stemmed_words = c.clean_words(words, remove_stopwords=False, language="english")
print(f"Words: {stemmed_words}")
