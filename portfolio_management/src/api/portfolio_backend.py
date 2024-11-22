from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Dict
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from portfolio_ml_core import PortfolioManager

app = FastAPI()
security = HTTPBasic()
portfolio_manager = PortfolioManager()


# Database initialization
def init_db():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (user_id TEXT, symbol TEXT, weight REAL)''')
    conn.commit()
    conn.close()


init_db()


class PortfolioRequest(BaseModel):
    symbols: List[str]
    weights: Dict[str, float]


class PredictionRequest(BaseModel):
    symbols: List[str]
    days: int = 30


@app.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioRequest):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        market_data = portfolio_manager.get_market_data(
            request.symbols, start_date, end_date
        )

        weights = portfolio_manager.optimize_portfolio(market_data)
        risk_metrics = portfolio_manager.calculate_risk_metrics(market_data, weights)

        return {
            "weights": weights,
            "risk_metrics": risk_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/market")
async def get_predictions(request: PredictionRequest):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        market_data = portfolio_manager.get_market_data(
            request.symbols, start_date, end_date
        )

        predictions = portfolio_manager.generate_predictions(market_data)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/risk_metrics")
async def get_risk_metrics(symbols: List[str]):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        market_data = portfolio_manager.get_market_data(symbols, start_date, end_date)
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols}  # Equal weights
        risk_metrics = portfolio_manager.calculate_risk_metrics(market_data, weights)

        return {"risk_metrics": risk_metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))