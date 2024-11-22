#!/bin/bash

# Start FastAPI backend
uvicorn src.api.portfolio_backend:app --reload --port 8000 &

# Start Streamlit frontend
streamlit run src.ui.portfolio_frontend:main --server.port 8501