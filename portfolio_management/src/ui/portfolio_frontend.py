import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Portfolio Management Assistant", layout="wide")

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False


def authenticate(username: str, password: str) -> bool:
    # Simple authentication for demo purposes
    return username == "demo" and password == "demo123"


def create_portfolio_chart(weights: Dict[str, float]):
    fig = go.Figure(data=[go.Pie(labels=list(weights.keys()),
                                 values=list(weights.values()))])
    fig.update_layout(title="Portfolio Composition")
    return fig


def create_prediction_chart(predictions: Dict[str, List[float]], dates: List[str]):
    fig = go.Figure()
    for symbol, values in predictions.items():
        fig.add_trace(go.Scatter(x=dates, y=values,
                                 mode='lines+markers',
                                 name=symbol))
    fig.update_layout(title="Price Predictions",
                      xaxis_title="Date",
                      yaxis_title="Predicted Price")
    return fig


def main():
    if not st.session_state.authenticated:
        st.title("Portfolio Management Assistant")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit and authenticate(username, password):
                st.session_state.authenticated = True
                st.experimental_rerun()
            elif submit:
                st.error("Invalid credentials")
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page",
                                ["Portfolio Overview",
                                 "ML Insights",
                                 "Risk Metrics"])

        if page == "Portfolio Overview":
            st.title("Portfolio Overview")

            # Portfolio Input
            symbols = st.multiselect(
                "Select Stocks",
                ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
            )

            if symbols:
                # Get optimized portfolio
                response = requests.post(
                    "http://localhost:8000/portfolio/optimize",
                    json={"symbols": symbols}
                )
                if response.status_code == 200:
                    data = response.json()
                    weights = data["weights"]
                    risk_metrics = data["risk_metrics"]

                    # Display portfolio composition
                    st.plotly_chart(create_portfolio_chart(weights))

                    # Display risk metrics
                    st.subheader("Risk Metrics")
                    metrics_df = pd.DataFrame(risk_metrics.items(),
                                              columns=["Metric", "Value"])
                    st.table(metrics_df)

        elif page == "ML Insights":
            st.title("ML Insights Dashboard")

            symbols = st.multiselect(
                "Select Stocks for Prediction",
                ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
            )

            days = st.slider("Prediction Days", 5, 30, 10)

            if symbols:
                response = requests.post(
                    "http://localhost:8000/predictions/market",
                    json={"symbols": symbols, "days": days}
                )

                if response.status_code == 200:
                    data = response.json()
                    dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                             for i in range(days)]
                    st.plotly_chart(create_prediction_chart(data["predictions"], dates))

        elif page == "Risk Metrics":
            st.title("Risk Metrics Analysis")

            symbols = st.multiselect(
                "Select Stocks for Risk Analysis",
                ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
            )

            if symbols:
                response = requests.get(
                    f"http://localhost:8000/portfolio/risk_metrics",
                    params={"symbols": symbols}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Portfolio Risk Metrics")
                    metrics_df = pd.DataFrame(data["risk_metrics"].items(),
                                              columns=["Metric", "Value"])
                    st.table(metrics_df)


if __name__ == "__main__":
    main()