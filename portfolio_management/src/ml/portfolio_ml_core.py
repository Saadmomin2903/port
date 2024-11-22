import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta


class MarketPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class PortfolioOptimizer:
    def __init__(self, risk_tolerance: float = 0.5):
        self.risk_tolerance = risk_tolerance

    def optimize(self, returns: pd.DataFrame, constraints: Dict = None) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        mu = returns.mean()
        sigma = returns.cov()

        n_assets = len(returns.columns)
        weights = self._solve_optimization(mu, sigma, n_assets, constraints)
        return weights

    def _solve_optimization(self, mu, sigma, n_assets, constraints):
        # Simplified optimization using Sharpe ratio maximization
        from scipy.optimize import minimize

        def objective(weights):
            portfolio_return = np.sum(mu * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            sharpe_ratio = portfolio_return / portfolio_risk
            return -sharpe_ratio  # Minimize negative Sharpe ratio

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negative weights
        ]

        result = minimize(
            objective,
            x0=np.array([1 / n_assets] * n_assets),
            method='SLSQP',
            constraints=constraints
        )
        return result.x


class FeatureEngine:
    def __init__(self):
        self.scaler = StandardScaler()

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators"""
        features = pd.DataFrame(index=df.index)

        # Moving averages
        features['SMA_20'] = df['Close'].rolling(window=20).mean()
        features['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Volatility
        features['Volatility'] = df['Close'].rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = exp1 - exp2

        return features.fillna(0)


class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def analyze_text(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        return {
            "positive": probabilities[0][0].item(),
            "negative": probabilities[0][1].item(),
            "neutral": probabilities[0][2].item()
        }


class PortfolioManager:
    def __init__(self):
        self.market_predictor = MarketPredictor(input_size=6)  # 6 features
        self.portfolio_optimizer = PortfolioOptimizer()
        self.feature_engine = FeatureEngine()
        self.sentiment_analyzer = SentimentAnalyzer()

    def get_market_data(self, symbols: List[str],
                        start_date: datetime,
                        end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch market data for given symbols"""
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            data[symbol] = df
        return data

    def generate_predictions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate predictions for each symbol"""
        predictions = {}
        for symbol, data in market_data.items():
            features = self.feature_engine.compute_technical_features(data)
            features_tensor = torch.FloatTensor(features.values).unsqueeze(0)
            with torch.no_grad():
                pred = self.market_predictor(features_tensor)
            predictions[symbol] = pred.numpy()
        return predictions

    def optimize_portfolio(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Optimize portfolio weights"""
        returns = pd.DataFrame({symbol: data['Close'].pct_change()
                                for symbol, data in market_data.items()})
        weights = self.portfolio_optimizer.optimize(returns.dropna())
        return dict(zip(market_data.keys(), weights))

    def calculate_risk_metrics(self, market_data: Dict[str, pd.DataFrame],
                               weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        returns = pd.DataFrame({symbol: data['Close'].pct_change()
                                for symbol, data in market_data.items()})

        portfolio_returns = returns.dropna().dot(pd.Series(weights))

        metrics = {
            "volatility": portfolio_returns.std() * np.sqrt(252),  # Annualized volatility
            "sharpe_ratio": portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            "var_95": portfolio_returns.quantile(0.05),  # 95% VaR
            "max_drawdown": (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
        }

        return metrics