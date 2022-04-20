from textfier.tasks import Seq2SeqTask

# Creates a sequence-to-sequence task
task = Seq2SeqTask(model="t5-small")

# Defines the input text
text = """New York (CNN) When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County,
but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again."""

# Encodes the input
enc_text = task.tokenizer.encode(
    "summarize: " + text, return_tensors="pt", max_length=256
)

# Performs the translation
preds = task.model.generate(
    enc_text,
    max_length=25,
    min_length=10,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
)

# Decodes the translation outputs
decoded_preds = task.tokenizer.decode(preds[0])
print(decoded_preds)
