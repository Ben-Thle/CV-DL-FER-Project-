from typing import Union, Optional
import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score


def calculate_f1_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: str = 'macro',
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> Union[float, np.ndarray]:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}")
    
    return sklearn_f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        labels=labels,
        zero_division=zero_division)


def calculate_f1_scores_per_class(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> dict:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    f1_scores = calculate_f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        zero_division=zero_division)
    
    return dict(zip(labels, f1_scores))

def calculate_macro_f1_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> float:

    return calculate_f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average='macro',
        labels=labels,
        zero_division=zero_division)


def calculate_weighted_f1_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> float:

    return calculate_f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted',
        labels=labels,
        zero_division=zero_division)
