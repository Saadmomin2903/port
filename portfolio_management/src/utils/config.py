# Configuration settings
API_BASE_URL = "http://localhost:8000"
DB_PATH = "data/portfolio.db"
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
MODEL_PATHS = {
    "market_predictor": "models/market_predictor.pth",
    "sentiment_analyzer": "models/finbert"
}