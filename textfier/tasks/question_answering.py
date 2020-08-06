"""Question answering pre-defined task.
"""

from transformers import AutoModelForQuestionAnswering

import textfier.utils.logging as l
from textfier.core import Task

logger = l.get_logger(__name__)


class QuestionAnsweringTask(Task):
    """QuestionAnsweringTask implements a pre-trained task used to
    handle question answering.

    """

    def __init__(self, model, **kwargs):
        """Initialization method.

        Args:
            model (str): Identifier of the pre-trained model to be loaded.

        """

        logger.debug('Task overridden: question_answering.')

        # Override its parent class with inputted arguments
        super(QuestionAnsweringTask, self).__init__(model, **kwargs)

    def _build(self, model):
        """Builds up the pre-trained model according to the desired task.

        Args:
            model (str): Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForQuestionAnswering.from_pretrained(model, config=self.config)
