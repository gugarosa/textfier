"""Question answering pre-defined task.
"""

from transformers import AutoModelForQuestionAnswering

from textfier.core import Task
from textfier.utils import logging

logger = logging.get_logger(__name__)


class QuestionAnsweringTask(Task):
    """QuestionAnsweringTask implements a pre-trained task used to
    handle question answering.

    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialization method.

        Args:
            model: Identifier of the pre-trained model to be loaded.

        """

        logger.debug("Task overridden: question_answering.")

        super(QuestionAnsweringTask, self).__init__(model, **kwargs)

    def _build(self, model: str) -> None:
        """Builds up the pre-trained model according to the desired task.

        Args:
            model: Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model, config=self.config
        )
