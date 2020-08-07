"""Default runner class, which overiddes and eases huggingface's trainer.
"""

from transformers import Trainer, TrainingArguments

import textfier.utils.logging as l
from textfier.utils.metrics import compute_metrics

logger = l.get_logger(__name__)


class Runner(Trainer):
    """Runner implements a default class used to handle customizable trainers.

    """

    def __init__(self, model, train_dataset=None, eval_dataset=None, **kwargs):
        """Inialization method.

        Args:
            model ():
            train_dataset ():
            eval_dataset ():

        """

        logger.debug('Creating runner ...')

        # Defines the arguments
        args = TrainingArguments(output_dir='./results', logging_dir='./logs', **kwargs)

        # Overrides its parent class with inputted arguments
        super(Runner, self).__init__(model, args, train_dataset=train_dataset,
                                     eval_dataset=eval_dataset, compute_metrics=compute_metrics)

        logger.debug('Runner created.')
