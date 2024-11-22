# src/utils/db_config.py

import os
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'DB_PATH': os.getenv('DB_PATH', 'data/portfolio.db'),
    'MAX_CONNECTIONS': int(os.getenv('DB_MAX_CONNECTIONS', 10)),
    'TIMEOUT': int(os.getenv('DB_TIMEOUT', 30)),

    # Table names
    'TABLES': {
        'USERS': 'users',
        'PORTFOLIOS': 'portfolios',
        'HOLDINGS': 'portfolio_holdings',
        'TRANSACTIONS': 'transactions',
        'PREDICTIONS': 'model_predictions',
        'RISK_METRICS': 'risk_metrics'
    },

    # Query timeouts (seconds)
    'QUERY_TIMEOUTS': {
        'DEFAULT': 30,
        'LONG_RUNNING': 120
    }
}


# Ensure data directory exists
def ensure_data_directory():
    data_dir = Path(DB_CONFIG['DB_PATH']).parent
    data_dir.mkdir(parents=True, exist_ok=True)