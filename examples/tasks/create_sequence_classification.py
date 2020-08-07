from textfier.tasks import SequenceClassificationTask

# Creates sequence classification task
task = SequenceClassificationTask(model='neuralmind/bert-base-portuguese-cased', num_labels=2)
