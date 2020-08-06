"""Sequence classification pre-defined task.
"""

from transformers import AutoModelForSequenceClassification

import textfier.utils.logging as l
from textfier.core import Task

logger = l.get_logger(__name__)


class SequenceClassificationTask(Task):
    """SequenceClassificationTask implements a pre-trained task used to
    handle text classification.

    """

    def __init__(self, model, **kwargs):
        """Initialization method.

        Args:
            model (str): Identifier of the pre-trained model to be loaded.

        """

        logger.debug('Task overridden: sequence_classification.')

        # Override its parent class with inputted arguments
        super(SequenceClassificationTask, self).__init__(model, **kwargs)

    def _build(self, model):
        """Builds up the pre-trained model according to the desired task.

        Args:
            model (str): Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=self.config)
