import utils.tokenizer as t

s = 'Good muffins cost $3.88\nin New York. Please buy me two of them.\n\nThanks.'

print(t.tokenize_to_sent(s))
print(t.tokenize_to_word(s))