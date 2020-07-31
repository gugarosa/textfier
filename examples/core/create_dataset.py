from textfier.core.dataset import TextClassificationDataset
from textfier.core.task import TextClassificationTask

# Creates the task
task = TextClassificationTask(pretrained_model='neuralmind/bert-base-portuguese-cased')

# Defines the input sequences
sentences = ['quero andar por ai', 'olá, tudo bem?']

# Defines the input labels
labels = [0, 0]

# Encodes the input sequences using the model's tokenizer
encoded_sentences = task.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

# Creates the dataset
dataset = TextClassificationDataset(encoded_sentences['input_ids'], encoded_sentences['attention_mask'], labels)

print(dataset.data, dataset.mask, dataset.labels)
