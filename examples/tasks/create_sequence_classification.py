from textfier.tasks import SequenceClassificationTask

# Creates a sequence classification task
task = SequenceClassificationTask(model='bert-base-cased-finetuned-mrpc', num_labels=2)
