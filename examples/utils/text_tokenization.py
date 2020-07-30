import textfier.utils.tokenizer as t

# Input string
s = 'Os bolos estão custando R$12,00\nem São Paulo. Por favor, compre dois.\n\nObrigado.'


print(t.tokenize_to_sent(s))
print(t.tokenize_to_word(s))