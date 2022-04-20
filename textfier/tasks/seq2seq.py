"""Sequence-to-Sequence pre-defined task.
"""

from transformers import AutoModelForSeq2SeqLM

from textfier.core import Task
from textfier.utils import logging

logger = logging.get_logger(__name__)


class Seq2SeqTask(Task):
    """Seq2SeqTask implements a pre-trained task used to
    handle sequence-to-sequence tasks (summarization and translation).

    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialization method.

        Args:
            model: Identifier of the pre-trained model to be loaded.

        """

        logger.debug("Task overridden: seq2seq.")

        super(Seq2SeqTask, self).__init__(model, **kwargs)

    def _build(self, model: str) -> None:
        """Builds up the pre-trained model according to the desired task.

        Args:
            model: Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, config=self.config)
