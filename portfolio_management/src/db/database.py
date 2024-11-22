# src/db/database.py

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from contextlib import contextmanager
import json


class DatabaseManager:
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.initialize_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row factory for dictionary-like access
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')

            # Portfolios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')

            # Portfolio holdings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
                )
            ''')

            # Transaction history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
                )
            ''')

            # ML Model predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date TIMESTAMP NOT NULL,
                    prediction_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Risk metrics history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    metric_date TIMESTAMP NOT NULL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    var_95 REAL,
                    max_drawdown REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
                )
            ''')

            conn.commit()

    # User management methods
    def create_user(self, username: str, password_hash: str, email: Optional[str] = None) -> int:
        """Create a new user and return user_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))
            conn.commit()
            return cursor.lastrowid

    def get_user(self, username: str) -> Optional[Dict]:
        """Get user details by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            return dict(result) if result else None

    # Portfolio management methods
    def create_portfolio(self, user_id: int, name: str) -> int:
        """Create a new portfolio and return portfolio_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolios (user_id, name)
                VALUES (?, ?)
            ''', (user_id, name))
            conn.commit()
            return cursor.lastrowid

    def add_holding(self, portfolio_id: int, symbol: str, quantity: float, entry_price: float) -> int:
        """Add a new holding to a portfolio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_holdings (portfolio_id, symbol, quantity, entry_price)
                VALUES (?, ?, ?, ?)
            ''', (portfolio_id, symbol, quantity, entry_price))
            conn.commit()
            return cursor.lastrowid

    def get_portfolio_holdings(self, portfolio_id: int) -> List[Dict]:
        """Get all holdings for a portfolio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM portfolio_holdings
                WHERE portfolio_id = ?
            ''', (portfolio_id,))
            return [dict(row) for row in cursor.fetchall()]

    # Transaction methods
    def record_transaction(self, portfolio_id: int, symbol: str,
                           transaction_type: str, quantity: float, price: float) -> int:
        """Record a new transaction"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions (portfolio_id, symbol, transaction_type, quantity, price)
                VALUES (?, ?, ?, ?, ?)
            ''', (portfolio_id, symbol, transaction_type, quantity, price))
            conn.commit()
            return cursor.lastrowid

    def get_transactions(self, portfolio_id: int,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict]:
        """Get transactions for a portfolio within date range"""
        query = 'SELECT * FROM transactions WHERE portfolio_id = ?'
        params = [portfolio_id]

        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ML Predictions methods
    def store_prediction(self, symbol: str, prediction_date: datetime, prediction_data: Dict) -> int:
        """Store ML model prediction"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_predictions (symbol, prediction_date, prediction_data)
                VALUES (?, ?, ?)
            ''', (symbol, prediction_date, json.dumps(prediction_data)))
            conn.commit()
            return cursor.lastrowid

    def get_latest_predictions(self, symbol: str, limit: int = 1) -> List[Dict]:
        """Get latest predictions for a symbol"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_predictions
                WHERE symbol = ?
                ORDER BY prediction_date DESC
                LIMIT ?
            ''', (symbol, limit))
            return [dict(row) for row in cursor.fetchall()]

    # Risk metrics methods
    def store_risk_metrics(self, portfolio_id: int, metric_date: datetime,
                           metrics: Dict[str, float]) -> int:
        """Store risk metrics for a portfolio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO risk_metrics (
                    portfolio_id, metric_date, volatility, sharpe_ratio, var_95, max_drawdown
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, metric_date, metrics.get('volatility'),
                  metrics.get('sharpe_ratio'), metrics.get('var_95'),
                  metrics.get('max_drawdown')))
            conn.commit()
            return cursor.lastrowid

    def get_risk_metrics_history(self, portfolio_id: int,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[Dict]:
        """Get risk metrics history for a portfolio"""
        query = 'SELECT * FROM risk_metrics WHERE portfolio_id = ?'
        params = [portfolio_id]

        if start_date:
            query += ' AND metric_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND metric_date <= ?'
            params.append(end_date)

        query += ' ORDER BY metric_date DESC'

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_portfolio_performance(self, portfolio_id: int) -> pd.DataFrame:
        """Calculate portfolio performance based on transactions"""
        with self.get_connection() as conn:
            # Get all transactions for the portfolio
            df = pd.read_sql_query('''
                SELECT symbol, transaction_type, quantity, price, timestamp
                FROM transactions
                WHERE portfolio_id = ?
                ORDER BY timestamp
            ''', conn, params=(portfolio_id,))

            if df.empty:
                return pd.DataFrame()

            # Calculate running balance and value for each symbol
            df['value'] = df.apply(lambda x:
                                   x['quantity'] * x['price'] * (1 if x['transaction_type'] == 'BUY' else -1),
                                   axis=1)

            df['running_value'] = df.groupby('symbol')['value'].cumsum()

            return df