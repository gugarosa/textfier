"""Default runner class, which overiddes and eases huggingface's trainer.
"""

from typing import Optional

from transformers import PreTrainedModel, Trainer, TrainingArguments

from textfier.core.dataset import Dataset
from textfier.utils import logging
from textfier.utils.metrics import compute_metrics

logger = logging.get_logger(__name__)


class Runner(Trainer):
    """Runner implements a default class used to handle customizable trainers."""

    def __init__(
        self,
        model: PreTrainedModel,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        **kwargs
    ) -> None:
        """Inialization method.

        Args:
            model: Pre-trained model.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.

        """

        logger.debug("Creating runner ...")

        args = TrainingArguments(output_dir="./results", logging_dir="./logs", **kwargs)

        super(Runner, self).__init__(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logger.debug("Runner created.")
