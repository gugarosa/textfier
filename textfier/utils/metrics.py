"""Metrics-based utilities for computing important information.
"""

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def compute_metrics(preds):
    """Computes a set of metrics (accuracy, f1-score, precision and recall).

    Args:
        preds (transformers.trainer_utils.PredictionOutput): Predictions and potential metrics.

    Returns:
        A dictionary containing a set of metrics.

    """

    # True labels
    y_true = preds.label_ids

    # Gathers the actual predictions
    y_preds = preds.predictions.argmax(-1)

    # Calculates the accuracy score
    accuracy = accuracy_score(y_true, y_preds)

    # Calculates the f1-score
    f1 = f1_score(y_true, y_preds, average='weighted')

    # Calculates the precision score
    precision = precision_score(y_true, y_preds, average='weighted')

    # Calculates the recall score
    recall = recall_score(y_true, y_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
