"""
Main script for financial sentiment analysis and stock recommendations
Choose between training mode and live analysis mode
"""

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# Import our custom modules
from data_utils import (
    load_finance_data, 
    preprocess_sentences, 
    prepare_train_val_data
)
from model_trainer import FinancialSentimentTrainer
from visualization import (
    plot_training_loss, 
    plot_confusion_matrix, 
    display_training_stats,
    print_classification_report,
    process_predictions
)
from live_analyzer import LiveStockAnalyzer


def train_sentiment_model():
    """Train the sentiment analysis model on financial data"""
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    MODEL_NAME = "bert-base-uncased"
    
    print("="*60)
    print("FINANCIAL SENTIMENT ANALYSIS - TRAINING MODE")
    print("="*60)
    
    # Load data
    print("Loading data...")
    df_train, df_test = load_finance_data()
    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print("Sample training data:")
    print(df_train.head())
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    train_sentences = df_train["Sentence"].values
    train_labels = df_train["Label"].values
    
    train_input_ids, train_attention_masks = preprocess_sentences(
        train_sentences, tokenizer, MAX_LENGTH
    )
    
    # Split into train/validation
    print("Splitting data...")
    train_inputs, val_inputs, train_masks, val_masks, train_labels_tensor, val_labels = prepare_train_val_data(
        train_input_ids, train_attention_masks, train_labels
    )
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Initialize trainer
    print("\nInitializing model...")
    trainer = FinancialSentimentTrainer(num_labels=3, model_name=MODEL_NAME)
    trainer.setup_model()
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, val_dataloader = trainer.create_data_loaders(
        train_inputs, train_masks, train_labels_tensor,
        val_inputs, val_masks, val_labels,
        batch_size=BATCH_SIZE
    )
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    training_results = trainer.train_model(
        train_dataloader, val_dataloader,
        epochs=EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # Display training statistics
    display_training_stats(training_results['training_stats'])
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_loss(
        training_results['training_loss'], 
        training_results['validation_loss']
    )
    
    # Prepare test data
    print("\nPreparing test data...")
    test_sentences = df_test["Sentence"].values
    test_labels = df_test["Label"].values
    
    test_input_ids, test_attention_masks = preprocess_sentences(
        test_sentences, tokenizer, MAX_LENGTH
    )
    
    test_labels_tensor = torch.tensor(test_labels)
    
    # Create test data loader
    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    
    # Make predictions
    print("Making predictions on test set...")
    predictions, true_labels = trainer.predict(test_dataloader)
    
    # Process predictions
    y_logits, y_true, y_preds = process_predictions(predictions, true_labels)
    
    # Evaluate results
    print_classification_report(y_true, y_preds)
    plot_confusion_matrix(y_true, y_preds)
    
    print("\nModel training complete!")


def run_live_analysis():
    """Run live stock analysis and recommendations"""
    
    print("="*60)
    print("LIVE STOCK ANALYSIS MODE")
    print("="*60)
    
    print("📋 SETUP INSTRUCTIONS:")
    print("For better results, get free API keys from:")
    print("• NewsAPI: https://newsapi.org/")
    print("• Alpha Vantage: https://www.alphavantage.co/")
    print()
    
    # Initialize analyzer (you can add API keys here)
    api_keys = {
        # Add your API keys here:
        # 'newsapi': 'your_newsapi_key',
        # 'alpha_vantage': 'your_alpha_vantage_key'
    }
    
    analyzer = LiveStockAnalyzer(api_keys)
    
    # Run live analysis
    results = analyzer.run_live_analysis(
        max_articles=30,  # Number of news articles to analyze
        top_recommendations=10  # Number of stock recommendations to generate
    )
    
    # Display results
    analyzer.print_results(results)


def main():
    """Main function with mode selection"""
    
    print("="*60)
    print("🏢 FINANCIAL SENTIMENT ANALYSIS & STOCK RECOMMENDATIONS")
    print("="*60)
    print()
    print("Choose mode:")
    print("1. Train sentiment model (educational - learn how the model works)")
    print("2. Live stock analysis (practical - get stock recommendations)")
    print("3. Both (train model then run live analysis)")
    print()
    
    try:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            train_sentiment_model()
        elif choice == "2":
            run_live_analysis()
        elif choice == "3":
            print("Running both modes...")
            train_sentiment_model()
            print("\n" + "="*60)
            print("Now switching to live analysis...")
            print("="*60)
            run_live_analysis()
        else:
            print("Invalid choice. Running live analysis by default...")
            run_live_analysis()
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
