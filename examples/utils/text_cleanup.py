import utils.cleaner as c
import utils.tokenizer as t

#
s = 'Good muffins cost $3.88\nin New York. Please buy me two of them.\n\nThanks.'

#
tokens = t.tokenize_to_sent(s)

print(tokens)

#
tokens = c.stem_tokens(tokens)

print(tokens)