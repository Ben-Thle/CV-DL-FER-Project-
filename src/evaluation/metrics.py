from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd

from .accuracy import calculate_accuracy
from .f1_score import (
    calculate_macro_f1_score,
    calculate_weighted_f1_score,
    calculate_f1_scores_per_class)
from .confusion_matrix import calculate_confusion_matrix


def evaluate_all_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    class_names: Optional[list] = None,
    zero_division: Union[str, float] = 0.0) -> Dict[str, Any]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    
    if class_names is None:
        class_names = [f"Class {label}" for label in labels]
    elif len(class_names) != len(labels):
        class_names = {
            class_names[i] if i < len(class_names) else f"Class {labels[i]}"
            for i in range(len(labels))}
    
    accuracy = calculate_accuracy(y_true, y_pred)
    macro_f1 = calculate_macro_f1_score(y_true, y_pred, labels=labels, zero_division=zero_division)
    weighted_f1 = calculate_weighted_f1_score(y_true, y_pred, labels=labels, zero_division=zero_division)
    confusion_matrix = calculate_confusion_matrix(y_true, y_pred, labels=labels)
    per_class_f1 = calculate_f1_scores_per_class(y_true, y_pred, labels=labels, zero_division=zero_division)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': confusion_matrix,
        'per_class_f1': per_class_f1,
        'class_names': class_names,
        'labels': labels }


def get_per_class_f1_table(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    class_names: Optional[list] = None,
    zero_division: Union[str, float] = 0.0,
    return_dataframe: bool = True) -> Union[pd.DataFrame, Dict[str, Any]]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    
    per_class_f1 = calculate_f1_scores_per_class(y_true, y_pred, labels=labels, zero_division=zero_division)
    
    if class_names is None:
        class_names = [f"Class {label}" for label in labels]
    elif len(class_names) != len(labels):
        class_names = [
            class_names[i] if i < len(class_names) else f"Class {labels[i]}"
            for i in range(len(labels))]
    
    label_to_name = dict(zip(labels, class_names))
    
    if return_dataframe:
        data = [
            {
                'Class': label_to_name[label],
                'Label': label,
                'F1 Score': per_class_f1[label]}
            for label in labels]
        df = pd.DataFrame(data)
        df = df[['Class', 'F1 Score']]
        return df
    else:
        return {label_to_name[label]: per_class_f1[label] for label in labels}


def print_evaluation_summary(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[Union[np.ndarray, list]] = None,
    class_names: Optional[list] = None,
    zero_division: Union[str, float] = 0.0) -> None:

    results = evaluate_all_metrics(y_true, y_pred, labels=labels, class_names=class_names, zero_division=zero_division)
    
    print("=" * 52)
    print("Evaluation Summary")
    print("=" * 52)
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"Macro F1 (Main): {results['macro_f1']:.4f}")
    print(f"Weighted F1:     {results['weighted_f1']:.4f}")
    print()
    print("Per-Class F1 Scores:")
    print("-" * 52)
    df = get_per_class_f1_table(y_true, y_pred, labels=labels, class_names=class_names, zero_division=zero_division)
    print(df.to_string(index=False))
    print()
    print("Confusion Matrix:")
    print("-" * 52)
    print(results['confusion_matrix'])
    print("=" * 52)
