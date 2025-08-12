"""
Integrated live stock recommendation pipeline
Combines news collection, sentiment analysis, and stock recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import torch
from transformers import BertTokenizer
from datetime import datetime

# Import our modules
from live_news_collector import LiveNewsCollector
from data_utils import preprocess_sentences
from model_trainer import FinancialSentimentTrainer
from stock_recommender import StockRecommendationEngine
from visualization import plot_confusion_matrix, print_classification_report


class LiveStockAnalyzer:
    """
    Complete pipeline for live stock analysis and recommendations
    """
    
    def __init__(self, api_keys: Dict[str, str] = None, model_path: str = None):
        """
        Initialize the live analyzer
        
        Args:
            api_keys: Dictionary of API keys for data sources
            model_path: Path to saved model (if available)
        """
        self.api_keys = api_keys or {}
        self.model_path = model_path
        
        # Initialize components
        self.news_collector = LiveNewsCollector(self.api_keys)
        self.recommender = StockRecommendationEngine(self.api_keys)
        self.model_trainer = FinancialSentimentTrainer()
        self.tokenizer = None
        self.model_trained = False
        
        print("Live Stock Analyzer initialized")
        
    def setup_sentiment_model(self):
        """Initialize and setup the sentiment analysis model"""
        print("Setting up sentiment analysis model...")
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
        # Setup model
        self.model_trainer.setup_model()
        
        print("✅ Sentiment model ready")
        
    def train_model_if_needed(self):
        """Train the model if not already trained or loaded"""
        if self.model_trained:
            return
            
        print("Training sentiment model on financial data...")
        print("Note: This may take 30-60 minutes depending on your hardware")
        
        # Import training utilities
        from data_utils import load_finance_data, prepare_train_val_data
        
        # Load training data
        df_train, df_test = load_finance_data()
        
        # Preprocess data
        sentences = df_train["Sentence"].values
        labels = df_train["Label"].values
        
        input_ids, attention_masks = preprocess_sentences(sentences, self.tokenizer)
        
        # Split data
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = prepare_train_val_data(
            input_ids, attention_masks, labels
        )
        
        # Create data loaders
        train_dataloader, val_dataloader = self.model_trainer.create_data_loaders(
            train_inputs, train_masks, train_labels,
            val_inputs, val_masks, val_labels
        )
        
        # Train model
        training_results = self.model_trainer.train_model(train_dataloader, val_dataloader, epochs=2)  # Reduced for speed
        
        self.model_trained = True
        print("✅ Model training complete")
        
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment predictions added
        """
        if news_df.empty:
            return news_df
            
        print(f"Analyzing sentiment for {len(news_df)} articles...")
        
        # Prepare news text for analysis
        news_texts = news_df['full_text'].tolist()
        
        # Preprocess
        input_ids, attention_masks = preprocess_sentences(news_texts, self.tokenizer, max_length=128)
        
        # Create dataset
        from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
        
        # Create dummy labels (won't be used for prediction)
        dummy_labels = torch.zeros(len(input_ids))
        
        prediction_data = TensorDataset(input_ids, attention_masks, dummy_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=16)
        
        # Get predictions
        predictions, _ = self.model_trainer.predict(prediction_dataloader)
        
        # Process predictions
        processed_predictions = []
        for chunk in predictions:
            for logits in chunk:
                # Convert logits to probabilities
                probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
                processed_predictions.append(probabilities.tolist())
        
        # Add predictions to dataframe
        news_df = news_df.copy()
        news_df['sentiment_predictions'] = processed_predictions
        
        # Add readable sentiment labels
        sentiment_labels = []
        for pred in processed_predictions:
            sentiment_idx = np.argmax(pred)
            labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment_labels.append(labels_map[sentiment_idx])
        
        news_df['sentiment_label'] = sentiment_labels
        news_df['sentiment_confidence'] = [max(pred) for pred in processed_predictions]
        
        print("✅ Sentiment analysis complete")
        return news_df
        
    def prepare_stock_data(self, news_with_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for stock recommendations by expanding stock symbols
        
        Args:
            news_with_sentiment: DataFrame with news and sentiment data
            
        Returns:
            DataFrame with one row per stock symbol per article
        """
        stock_data = []
        
        for _, row in news_with_sentiment.iterrows():
            for symbol in row['stock_symbols']:
                stock_row = row.copy()
                stock_row['symbol'] = symbol
                stock_data.append(stock_row)
        
        return pd.DataFrame(stock_data)
    
    def run_live_analysis(self, max_articles: int = 50, top_recommendations: int = 10) -> Dict:
        """
        Run complete live analysis pipeline
        
        Args:
            max_articles: Maximum number of news articles to collect
            top_recommendations: Number of top stock recommendations to return
            
        Returns:
            Dictionary with analysis results
        """
        print("="*60)
        print("STARTING LIVE STOCK ANALYSIS PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Setup model
        if not self.tokenizer:
            self.setup_sentiment_model()
            
        if not self.model_trained:
            self.train_model_if_needed()
        
        # Step 2: Collect live news
        print("\n📰 Collecting live financial news...")
        news_df = self.news_collector.collect_live_news(max_articles=max_articles)
        
        if news_df.empty:
            print("❌ No news data collected. Check API keys or try again later.")
            return {'error': 'No news data available'}
        
        # Step 3: Analyze sentiment
        print("\n🧠 Analyzing news sentiment...")
        news_with_sentiment = self.analyze_news_sentiment(news_df)
        
        # Step 4: Prepare stock data
        print("\n📊 Preparing stock-specific data...")
        stock_data = self.prepare_stock_data(news_with_sentiment)
        
        if stock_data.empty:
            print("❌ No stock symbols found in news articles.")
            return {'error': 'No stock symbols detected'}
        
        # Step 5: Generate recommendations
        print("\n💡 Generating stock recommendations...")
        recommendations = self.recommender.generate_portfolio_recommendations(
            stock_data, top_n=top_recommendations
        )
        
        # Step 6: Generate report
        recommendations_report = self.recommender.format_recommendations_report(recommendations)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'timestamp': end_time.isoformat(),
            'processing_time_seconds': processing_time,
            'news_articles_analyzed': len(news_df),
            'stocks_identified': len(stock_data['symbol'].unique()),
            'recommendations': recommendations,
            'recommendations_report': recommendations_report,
            'news_summary': self._create_news_summary(news_with_sentiment),
            'sentiment_distribution': self._get_sentiment_distribution(news_with_sentiment)
        }
        
        print("\n✅ Analysis complete!")
        print(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        print(f"📰 Articles analyzed: {len(news_df)}")
        print(f"📈 Stocks identified: {len(stock_data['symbol'].unique())}")
        print(f"💼 Recommendations generated: {len(recommendations)}")
        
        return results
    
    def _create_news_summary(self, news_df: pd.DataFrame) -> Dict:
        """Create summary statistics of news data"""
        if news_df.empty:
            return {}
            
        return {
            'total_articles': len(news_df),
            'sources': news_df['source'].value_counts().to_dict(),
            'sentiment_breakdown': news_df['sentiment_label'].value_counts().to_dict(),
            'top_mentioned_stocks': news_df.explode('stock_symbols')['stock_symbols'].value_counts().head(10).to_dict()
        }
    
    def _get_sentiment_distribution(self, news_df: pd.DataFrame) -> Dict:
        """Get sentiment distribution statistics"""
        if news_df.empty:
            return {}
            
        sentiments = news_df['sentiment_predictions'].tolist()
        sentiment_scores = []
        
        for pred in sentiments:
            # Convert to sentiment score: positive - negative
            score = pred[2] - pred[0]
            sentiment_scores.append(score)
        
        return {
            'mean_sentiment': float(np.mean(sentiment_scores)),
            'sentiment_std': float(np.std(sentiment_scores)),
            'positive_articles': int(sum(1 for s in sentiment_scores if s > 0.1)),
            'negative_articles': int(sum(1 for s in sentiment_scores if s < -0.1)),
            'neutral_articles': int(sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1))
        }
    
    def print_results(self, results: Dict):
        """Print formatted analysis results"""
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
            
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic stats
        print(f"⏱️  Processing Time: {results['processing_time_seconds']:.1f} seconds")
        print(f"📰 Articles Analyzed: {results['news_articles_analyzed']}")
        print(f"📈 Stocks Identified: {results['stocks_identified']}")
        
        # Sentiment distribution
        if 'sentiment_distribution' in results:
            dist = results['sentiment_distribution']
            print(f"\n🧠 SENTIMENT OVERVIEW")
            print(f"   📈 Positive Articles: {dist.get('positive_articles', 0)}")
            print(f"   📊 Neutral Articles: {dist.get('neutral_articles', 0)}")
            print(f"   📉 Negative Articles: {dist.get('negative_articles', 0)}")
            print(f"   📊 Overall Sentiment: {dist.get('mean_sentiment', 0):+.2f}")
        
        # Top recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS")
            for i, rec in enumerate(recommendations[:5], 1):
                action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                print(f"   {i}. {rec.symbol} - {action_emoji.get(rec.action, '⚪')} {rec.action}")
                print(f"      Confidence: {rec.confidence:.1%}, Risk: {rec.risk_level}")
        
        # Print full report
        print(f"\n{results['recommendations_report']}")


# Example configuration
def create_api_config():
    """
    Create API configuration dictionary
    Get free API keys from:
    - NewsAPI: https://newsapi.org/
    - Alpha Vantage: https://www.alphavantage.co/
    """
    return {
        'newsapi': 'your_newsapi_key_here',
        'alpha_vantage': 'your_alphavantage_key_here'
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Live Stock Analyzer...")
    
    # Initialize analyzer (without API keys for demo)
    analyzer = LiveStockAnalyzer()
    
    # Run analysis
    results = analyzer.run_live_analysis(max_articles=10, top_recommendations=5)
    
    # Print results
    analyzer.print_results(results)
