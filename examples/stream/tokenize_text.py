import textfier.stream.tokenizer as t

# Defines an input string
s = 'Os bolos estão custando R$12,00 em São Paulo. Por favor, compre dois.\n\nObrigado.'

# Tokenizes to sentences
sentences = t.tokenize_to_sentences(s)
print(f'Sentences: {sentences}')

# Tokenizes to words
words = t.tokenize_to_words(s)
print(f'Words: {words}')
