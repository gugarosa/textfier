from textfier.tasks import NamedEntityRecognitionTask

# Creates a named entity recognition task
task = NamedEntityRecognitionTask(
    model="dbmdz/bert-base-cased-finetuned-conll03-english"
)
