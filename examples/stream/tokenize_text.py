import textfier.stream.tokenizer as t

# Defines an input string
s = 'The cakes costs about $12,00 in São Paulo. Please, buy two.\n\nThanks.'

# Tokenizes to sentences
sentences = t.tokenize_to_sentences(s, language='english')
print(f'Sentences: {sentences}')

# Tokenizes to words
words = t.tokenize_to_words(s, language='english')
print(f'Words: {words}')
