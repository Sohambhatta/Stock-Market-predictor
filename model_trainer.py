"""
Model training utilities for financial sentiment analysis
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from typing import List, Tuple, Dict
from data_utils import flat_accuracy


class FinancialSentimentTrainer:
    """Training class for financial sentiment analysis"""
    
    def __init__(self, num_labels: int = 3, model_name: str = "bert-base-uncased"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Initialize the BERT model"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    
    def create_data_loaders(self, train_inputs: torch.Tensor, train_masks: torch.Tensor, 
                           train_labels: torch.Tensor, val_inputs: torch.Tensor, 
                           val_masks: torch.Tensor, val_labels: torch.Tensor,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        
        # Training data loader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        
        # Validation data loader
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        
        return train_dataloader, val_dataloader
    
    def train_model(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                   epochs: int = 4, learning_rate: float = 2e-5) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary containing training statistics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        # Setup scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training tracking
        training_loss = []
        validation_loss = []
        training_stats = []
        
        for epoch_i in range(epochs):
            print(f'Epoch {epoch_i + 1} / {epochs} ========')
            print('Training the model')
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                if step % 20 == 0 and step != 0:
                    print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.')
                
                # Move batch to device
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                # Clear gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                
                loss = outputs[0]
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"  Average training loss: {avg_train_loss:.2f}")
            
            # Validation phase
            print("Evaluating on Validation Set")
            avg_val_loss, avg_val_accuracy = self._evaluate(val_dataloader)
            
            print(f"Validation Accuracy: {avg_val_accuracy:.2f}")
            print(f"Validation Loss: {avg_val_loss:.2f}")
            
            # Save stats
            training_loss.append(avg_train_loss)
            validation_loss.append(avg_val_loss)
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy
            })
        
        print("Training complete!")
        
        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'training_stats': training_stats
        }
    
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given dataloader"""
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        
        for batch in dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                
                loss = outputs[0]
                logits = outputs[1]
            
            total_eval_loss += loss.item()
            
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        avg_val_loss = total_eval_loss / len(dataloader)
        
        return avg_val_loss, avg_val_accuracy
    
    def predict(self, test_dataloader: DataLoader) -> Tuple[List, List]:
        """
        Make predictions on test data
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        print('Predicting labels for test data...')
        
        self.model.eval()
        predictions, true_labels = [], []
        
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)
        
        return predictions, true_labels
