import textfier.stream.tokenizer as t
import textfier.utils.loader as l
from textfier.core import Dataset, Runner
from textfier.tasks import SequenceClassificationTask

# Loading text from file
text = l.load_txt('data/sample.txt')

# Tokenizing into sentences
sentences = t.tokenize_to_sentences(text)

# Creating labels
labels = [0, 1, 0, 1, 0, 0]

# Creates the task
task = SequenceClassificationTask(model='neuralmind/bert-base-portuguese-cased', num_labels=2)

# Encodes the input sequences using the model's tokenizer
encoded_sentences = task.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

# Creates the dataset
train_dataset = Dataset(input_ids=encoded_sentences['input_ids'],
                        attention_mask=encoded_sentences['attention_mask'], labels=labels)

#
runner = Runner(task.model, train_dataset, num_train_epochs=5)

#
runner.train()

#
preds = runner.predict(train_dataset)

print(preds)

# encoded_sentences = task.tokenizer(sentences[3:], return_tensors='pt', padding=True, truncation=True)
# val_dataset = TextClassificationDataset(encoded_sentences['input_ids'], encoded_sentences['attention_mask'], labels[3:])

# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=50,              # total # of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )

# trainer = Trainer(
#     model=task.model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,             # training arguments, defined above
#     train_dataset=train_dataset,
#     compute_metrics=compute_metrics      # evaluation dataset
# )

# trainer.train()

# print(trainer.predict(val_dataset))
