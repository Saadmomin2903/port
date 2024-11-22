# src/scripts/init_database.py

import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.db.database import DatabaseManager
from src.utils.db_config import DB_CONFIG, ensure_data_directory
import hashlib

def hash_password(password: str) -> str:
    """Simple password hashing for demo purposes"""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_demo_data():
    """Initialize database with demo data"""
    # Ensure data directory exists
    ensure_data_directory()

    # Initialize database manager
    db = DatabaseManager(DB_CONFIG['DB_PATH'])

    try:
        # Create demo user
        demo_password = hash_password("demo123")
        user_id = db.create_user(
            username="demo_user",
            password_hash=demo_password,
            email="demo@example.com"
        )
        print(f"Created demo user with ID: {user_id}")

        # Create demo portfolio
        portfolio_id = db.create_portfolio(
            user_id=user_id,
            name="Demo Portfolio"
        )
        print(f"Created demo portfolio with ID: {portfolio_id}")

        # Add sample holdings
        holdings_data = [
            ("AAPL", 10, 150.00),
            ("GOOGL", 5, 2500.00),
            ("MSFT", 15, 280.00),
            ("AMZN", 8, 3200.00)
        ]

        for symbol, quantity, price in holdings_data:
            holding_id = db.add_holding(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=quantity,
                entry_price=price
            )
            print(f"Added holding {symbol} with ID: {holding_id}")

            # Record buy transaction
            transaction_id = db.record_transaction(
                portfolio_id=portfolio_id,
                symbol=symbol,
                transaction_type="BUY",
                quantity=quantity,
                price=price
            )
            print(f"Recorded transaction for {symbol} with ID: {transaction_id}")

        # Verify data
        holdings = db.get_portfolio_holdings(portfolio_id)
        print("\nCurrent Portfolio Holdings:")
        for holding in holdings:
            print(f"Symbol: {holding['symbol']}, "
                  f"Quantity: {holding['quantity']}, "
                  f"Entry Price: ${holding['entry_price']:.2f}")

        transactions = db.get_transactions(portfolio_id)
        print("\nTransaction History:")
        for tx in transactions:
            print(f"Symbol: {tx['symbol']}, "
                  f"Type: {tx['transaction_type']}, "
                  f"Quantity: {tx['quantity']}, "
                  f"Price: ${tx['price']:.2f}")

    except Exception as e:
        print(f"Error initializing demo data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Initializing database with demo data...")
    initialize_demo_data()
    print("\nDatabase initialization completed successfully!")