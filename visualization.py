"""
Visualization utilities for financial sentiment analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from typing import List


def plot_training_loss(training_loss: List[float], validation_loss: List[float]) -> None:
    """Plot training and validation loss over epochs"""
    plt.figure(figsize=(12, 6))
    plt.title('Loss over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.plot(training_loss, label='train')
    plt.plot(validation_loss, label='validation')
    
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true: List[int], y_predicted: List[int]) -> None:
    """Plot confusion matrix for predictions"""
    cm = metrics.confusion_matrix(y_true, y_predicted)
    print("Plotting the Confusion Matrix")
    
    labels = ["Negative", "Neutral", "Positive"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.yticks([0.5, 1.5, 2.5], labels, va='center')
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()


def display_training_stats(training_stats: List[dict]) -> None:
    """Display training statistics in a formatted table"""
    df_stats = pd.DataFrame(training_stats)
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(df_stats.to_string(index=False, float_format="%.4f"))
    print("="*60)


def print_classification_report(y_true: List[int], y_pred: List[int]) -> None:
    """Print detailed classification report"""
    label_names = ["Negative", "Neutral", "Positive"]
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(metrics.classification_report(y_true, y_pred, target_names=label_names))
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("="*60)


def process_predictions(predictions: List, true_labels: List) -> tuple:
    """
    Process raw predictions into final format
    
    Args:
        predictions: List of logits from model
        true_labels: List of true labels
    
    Returns:
        Tuple of (y_logits, y_true, y_preds)
    """
    # Flatten predictions
    y_logits, y_true, y_preds = [], [], []
    
    for chunk in predictions:
        for logits in chunk:
            y_logits.append(logits)
    
    for chunk in true_labels:
        for label in chunk:
            y_true.append(label)
    
    for logits in y_logits:
        y_preds.append(np.argmax(logits))
    
    return y_logits, y_true, y_preds
