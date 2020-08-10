"""Task-related classes and helpers with pre-defined tasks.
"""

from textfier.tasks.language_modeling import (CausalLanguageModelingTask,
                                              MaskedLanguageModelingTask)
from textfier.tasks.named_entity_recognition import NamedEntityRecognitionTask
from textfier.tasks.question_answering import QuestionAnsweringTask
from textfier.tasks.seq2seq import Seq2SeqTask
from textfier.tasks.sequence_classification import SequenceClassificationTask
