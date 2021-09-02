"""Default task class.
"""

from transformers import AutoConfig, AutoTokenizer

import textfier.utils.logging as l

logger = l.get_logger(__name__)


class Task:
    """Task implements a default class used to handle customizable tasks.

    """

    def __init__(self, model, **kwargs):
        """Initialization method.

        Args:
            model (str): Identifier of the pre-trained model to be loaded.

        """

        logger.debug('Creating task with: %s ...', model)

        # Overrides the model's configuration file with additional keywords
        self.config = AutoConfig.from_pretrained(model, **kwargs)

        # Loads the model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        try:
            self._build(model)

            logger.debug('Task created.')

        except NotImplementedError:
            logger.error('Private method `build` has not been overridden.')

    def _build(self, model):
        """Builds up the pre-trained model according to the desired task.

        Args:
            model (str): Identifier of the pre-trained model to be built.

        """

        raise NotImplementedError
