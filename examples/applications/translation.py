from textfier.tasks import Seq2SeqTask

# Creates a sequence-to-sequence task
task = Seq2SeqTask(model='t5-small')

# Defines the input text
text = 'translate English to German: My name is Textfier and I am able to help you.'

# Tokenizes the input
enc_text = task.tokenizer.encode(text, return_tensors='pt')

# Performs the translation
preds = task.model.generate(enc_text, max_length=40, num_beams=4, early_stopping=True)

# Decodes the translation outputs
decoded_preds = task.tokenizer.decode(preds[0])
print(decoded_preds)
