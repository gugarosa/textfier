"""Task-related classes and helpers with pre-defined tasks.
"""

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

import textfier.utils.logging as l

logger = l.get_logger(__name__)


class TextClassificationTask:
    """TextClassificationModel implements pre-trained tasks used to
    handle customizable text classification models.

    """

    def __init__(self, pretrained_model='neuralmind/bert-base-portuguese-cased', n_classes=2):
        """Initialization method.

        Args:
            pretrained_model (str): Identifier of the pre-trained model to be loaded.
            n_classes (int): Number of classes allowed in the classification task.

        """

        logger.debug('Instantiating from pre-trained model: %s ...', pretrained_model)

        # Overrides the model's configuration file
        self.config = AutoConfig.from_pretrained(pretrained_model, num_labels=n_classes)

        # Loads the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # Loads the pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=self.config)

        logger.debug('Task instantiated.')
