from typing import Union
import numpy as np
from sklearn.metrics import accuracy_score as sklearn_accuracy_score


def calculate_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    normalize: bool = True) -> float:
 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}")
    
    return sklearn_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize)
