"""Language modeling pre-defined tasks.
"""

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

import textfier.utils.logging as l
from textfier.core import Task

logger = l.get_logger(__name__)


class CausalLanguageModelingTask(Task):
    """CausalLanguageModelingTask implements a pre-trained task used to
    handle causal language modeling.

    """

    def __init__(self, model, **kwargs):
        """Initialization method.

        Args:
            model (str): Identifier of the pre-trained model to be loaded.

        """

        logger.debug('Task overridden: causal_language_modeling.')

        # Override its parent class with inputted arguments
        super(CausalLanguageModelingTask, self).__init__(model, **kwargs)

    def _build(self, model):
        """Builds up the pre-trained model according to the desired task.

        Args:
            model (str): Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForCausalLM.from_pretrained(model, config=self.config)


class MaskedLanguageModelingTask(Task):
    """MaskedLanguageModelingTask implements a pre-trained task used to
    handle masked language modeling.

    """

    def __init__(self, model, **kwargs):
        """Initialization method.

        Args:
            model (str): Identifier of the pre-trained model to be loaded.

        """

        logger.debug('Task overridden: masked_language_modeling.')

        # Override its parent class with inputted arguments
        super(MaskedLanguageModelingTask, self).__init__(model, **kwargs)

    def _build(self, model):
        """Builds up the pre-trained model according to the desired task.

        Args:
            model (str): Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForMaskedLM.from_pretrained(model, config=self.config)
