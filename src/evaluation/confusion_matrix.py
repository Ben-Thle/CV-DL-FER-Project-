from typing import Union, Optional
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    normalize: Optional[str] = None) -> np.ndarray:
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to index the matrix. If None, labels
            are inferred from y_true and y_pred, or use [0, 1, 2, ...] if
            both are empty.
        normalize: Normalization strategy for the confusion matrix:
            - None: Return counts (default)
            - 'true': Normalize over the true labels (rows)
            - 'pred': Normalize over the predicted labels (columns)
            - 'all': Normalize over the whole matrix
    
    Returns:
        2D numpy array representing the confusion matrix. If normalize is None,
        returns counts. Otherwise, returns normalized values.
        
        For a binary classification:
        [[TN, FP],
         [FN, TP]]
        
        Where:
        - TN: True Negatives
        - FP: False Positives
        - FN: False Negatives
        - TP: True Positives
"""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}" )
    
    
    cm = sklearn_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize=normalize)
    
    return cm


def calculate_confusion_matrix_normalized(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    normalize: str = 'true') -> np.ndarray:
"""
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Predicted targets returned by a classifier.
        labels: Optional list of labels to index the matrix.
        normalize: Normalization strategy:
            - 'true': Normalize over the true labels (rows) - default
            - 'pred': Normalize over the predicted labels (columns)
            - 'all': Normalize over the whole matrix
    
    Returns:
        2D numpy array representing the normalized confusion matrix.
"""
    return calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize=normalize)


def get_confusion_matrix_metrics(
    confusion_matrix: np.ndarray,
    class_index: Optional[int] = None) -> dict:
"""    
    Args:
        confusion_matrix: 2D numpy array representing the confusion matrix.
        class_index: Optional class index for multiclass problems. If None,
            assumes binary classification (2x2 matrix).
    
    Returns:
        Dictionary containing:
        - For binary (or when class_index specified):
          - 'TP': True Positives
          - 'TN': True Negatives
          - 'FP': False Positives
          - 'FN': False Negatives
        - For multiclass without class_index:
          - 'per_class_metrics': List of metrics dicts for each class
"""
    cm = np.asarray(confusion_matrix)
    
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"Confusion matrix must be a square 2D array. "
            f"Got shape {cm.shape}")
    
    n_classes = cm.shape[0]
    
    if class_index is not None:
        
        if class_index < 0 or class_index >= n_classes:
            raise ValueError(
                f"class_index must be between 0 and {n_classes-1}. "
                f"Got {class_index}")
        
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn }
    elif n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn}
    else:
        per_class_metrics = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            per_class_metrics.append({
                'class': i,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn })
        
        return {'per_class_metrics': per_class_metrics}
