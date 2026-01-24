from typing import Union, Optional
import numpy as np
from sklearn.metrics import precision_score, recall_score


def calculate_precision(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: str = 'macro',
    labels: Optional[Union[np.ndarray, list]] = None,
    zero_division: Union[str, float] = 0.0) -> Union[float, np.ndarray]:
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        average: Averaging strategy for multiclass problems:
            - 'micro': Calculate metrics globally by counting total true positives,
              false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their unweighted mean.
            - 'weighted': Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label).
            - 'binary': Only report results for the class specified by pos_label.
              This is applicable only if targets (y_{true,pred}) are binary.
            - None: Return precision scores for each class.
        labels: Optional list of labels to include when average != 'binary'.
        zero_division: Sets the value to return when there is a zero division.
            If set to "warn", this acts as 0, but warnings are also raised.
    
    Returns:
        Precision score(s). If average is None, returns array of precision scores
        for each class. Otherwise returns a single float.
"""
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
"""    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        average: Averaging strategy for multiclass problems:
            - 'micro': Calculate metrics globally by counting total true positives,
              false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their unweighted mean.
            - 'weighted': Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label).
            - 'binary': Only report results for the class specified by pos_label.
              This is applicable only if targets (y_{true,pred}) are binary.
            - None: Return recall scores for each class.
        labels: Optional list of labels to include when average != 'binary'.
        zero_division: Sets the value to return when there is a zero division.
            If set to "warn", this acts as 0, but warnings are also raised.
    
    Returns:
        Recall score(s). If average is None, returns array of recall scores
        for each class. Otherwise returns a single float.
"""
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
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to include. If None, uses unique labels
            from y_true and y_pred.
        zero_division: Sets the value to return when there is a zero division.
    
    Returns:
        Dictionary mapping class labels to their precision scores.
"""
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
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to include. If None, uses unique labels
            from y_true and y_pred.
        zero_division: Sets the value to return when there is a zero division.
    
    Returns:
        Dictionary mapping class labels to their recall scores.
"""
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
""" 
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to include. If None, uses all labels
            present in y_true and y_pred.
        zero_division: Sets the value to return when there is a zero division.
            If set to "warn", this acts as 0, but warnings are also raised.
    
    Returns:
        Macro-averaged precision score as a float.
"""
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
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to include. If None, uses all labels
            present in y_true and y_pred.
        zero_division: Sets the value to return when there is a zero division.
            If set to "warn", this acts as 0, but warnings are also raised.
    
    Returns:
        Macro-averaged recall score as a float.
"""
    return calculate_recall(
        y_true=y_true,
        y_pred=y_pred,
        average='macro',
        labels=labels,
        zero_division=zero_division)
