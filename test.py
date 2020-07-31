import torch
from torch.utils.data import TensorDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from textfier.core.dataset import TextClassificationDataset
from textfier.utils.metrics import compute_metrics

# Using the community model
# BERT Base
config = AutoConfig.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', config=config)
print(model)


text_batch = ["quero andar por ai", "quero andar por ai"]
labels = [0, 0]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

train_data = TextClassificationDataset(input_ids, attention_mask, labels)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,             # training arguments, defined above
    train_dataset=train_data,
    compute_metrics=compute_metrics      # evaluation dataset
)

trainer.train()

print(trainer.predict(train_data))
