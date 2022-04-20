"""Language modeling pre-defined tasks.
"""

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from textfier.core import Task
from textfier.utils import logging

logger = logging.get_logger(__name__)


class CausalLanguageModelingTask(Task):
    """CausalLanguageModelingTask implements a pre-trained task used to
    handle causal language modeling.

    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialization method.

        Args:
            model: Identifier of the pre-trained model to be loaded.

        """

        logger.debug("Task overridden: causal_language_modeling.")

        super(CausalLanguageModelingTask, self).__init__(model, **kwargs)

    def _build(self, model: str) -> None:
        """Builds up the pre-trained model according to the desired task.

        Args:
            model: Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForCausalLM.from_pretrained(model, config=self.config)


class MaskedLanguageModelingTask(Task):
    """MaskedLanguageModelingTask implements a pre-trained task used to
    handle masked language modeling.

    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialization method.

        Args:
            model: Identifier of the pre-trained model to be loaded.

        """

        logger.debug("Task overridden: masked_language_modeling.")

        super(MaskedLanguageModelingTask, self).__init__(model, **kwargs)

    def _build(self, model: str) -> None:
        """Builds up the pre-trained model according to the desired task.

        Args:
            model: Identifier of the pre-trained model to be built.

        """

        # Loads the pre-trained model according to the class' task
        self.model = AutoModelForMaskedLM.from_pretrained(model, config=self.config)
