from typing import Union, Optional
import numpy as np
from sklearn.metrics import precision_score, recall_score


def calculate_precision(
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
    
    return precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        labels=labels,
        zero_division=zero_division)


def calculate_recall(
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
    
    return recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        labels=labels,
        zero_division=zero_division)


def calculate_precision_per_class(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> dict:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    precision_scores = calculate_precision(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        zero_division=zero_division)
    
    return dict(zip(labels, precision_scores))


def calculate_recall_per_class(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> dict:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    recall_scores = calculate_recall(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        zero_division=zero_division)
    
    return dict(zip(labels, recall_scores))


def calculate_macro_precision(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> float:

    return calculate_precision(
        y_true=y_true,
        y_pred=y_pred,
        average='macro',
        labels=labels,
        zero_division=zero_division)


def calculate_macro_recall(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> float:

    return calculate_recall(
        y_true=y_true,
        y_pred=y_pred,
        average='macro',
        labels=labels,
        zero_division=zero_division)
