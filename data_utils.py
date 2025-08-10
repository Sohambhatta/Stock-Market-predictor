"""
Data utilities for financial sentiment analysis
"""

import os
import pandas as pd
import requests
from typing import Tuple
import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import torch

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
RND_SEED = 2020


def download_finance_data(data_dir: str = "data") -> None:
    """Download financial sentiment datasets"""
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20NLP%2BFinance/"
    files = ["finance_test.csv", "finance_train.csv"]
    
    for file in files:
        url = base_url + file
        filepath = os.path.join(data_dir, file)
        
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file}")
        else:
            print(f"{file} already exists")


def load_finance_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets"""
    train_path = os.path.join(data_dir, "finance_train.csv")
    test_path = os.path.join(data_dir, "finance_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found. Downloading...")
        download_finance_data(data_dir)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test


def preprocess_sentences(sentences: list, tokenizer: BertTokenizer, max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess sentences for BERT input
    
    Args:
        sentences: List of sentences to process
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (input_ids, attention_masks)
    """
    # Add special tokens
    sentences_with_tokens = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    
    # Tokenize
    tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences_with_tokens]
    
    # Convert to IDs
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences]
    
    # Pad sequences
    input_ids = pad_sequences(input_ids,
                              maxlen=max_length,
                              dtype="long",
                              truncating="post",
                              padding="post")
    
    # Create attention masks
    attention_masks = []
    for sequence in input_ids:
        mask = [float(i > 0) for i in sequence]
        attention_masks.append(mask)
    
    return torch.tensor(input_ids), torch.tensor(attention_masks)


def prepare_train_val_data(input_ids: torch.Tensor, attention_masks: torch.Tensor, 
                          labels: np.ndarray, test_size: float = 0.15) -> Tuple:
    """
    Split data into training and validation sets
    
    Returns:
        Tuple of (train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels)
    """
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        input_ids.numpy(), labels, test_size=test_size, random_state=RND_SEED
    )
    
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks.numpy(), input_ids.numpy(), test_size=test_size, random_state=RND_SEED
    )
    
    # Convert to tensors
    train_inputs = torch.tensor(X_train)
    val_inputs = torch.tensor(X_val)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    
    return train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels


def flat_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy for predictions"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
