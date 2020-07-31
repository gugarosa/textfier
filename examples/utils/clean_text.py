import textfier.utils.cleaner as c
import textfier.utils.tokenizer as t

# Defines an input string
s = 'Os bolos estão custando R$12,00 em São Paulo. Por favor, compre dois.\n\nObrigado.'

# Tokenizes, stems sentences and optionally remove stopwords
sentences = t.tokenize_to_sentences(s)
stemmed_sentences = c.clean_sentences(sentences, remove_stopwords=False)
print(f'Sentences: {stemmed_sentences}')

# Tokenizes, stems words and optionally remove stopwords
words = t.tokenize_to_words(s)
stemmed_words = c.clean_words(words, remove_stopwords=False)
print(f'Words: {stemmed_words}')
