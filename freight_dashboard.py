import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Function to calculate volatility over a specified period
def calculate_volatility(data, window=14):
    log_returns = np.log(data / data.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    if rolling_std.dropna().empty:
        return np.std(log_returns.dropna()) * np.sqrt(252)
    else:
        return rolling_std.dropna().iloc[-1] * np.sqrt(252)

# Improved mean reversion model with gradual sentiment effects
def predict_future_rates(mu, theta, sigma, last_rate, tonnage_effect, oil_effect, sentiment, future_days=7):
    future_rates = np.zeros(future_days)
    future_variances = np.zeros(future_days)
    dt = 1  # Daily steps
    
    future_rates[0] = last_rate  # Start with the last known rate
    future_variances[0] = sigma ** 2  # Initial variance based on current volatility

    # Gradual sentiment effect (affects days 1-2, peaks at day 4-5)
    for t in range(1, future_days):
        # Mean reversion component
        mean_reversion = theta * (mu - future_rates[t-1])
        
        # Gradual sentiment effect: takes 1-2 days to fully apply
        if t <= 2:
            sentiment_effect = 0.02 * sigma * t if sentiment == "Bullish" else -0.02 * sigma * t
        elif t <= 4:
            sentiment_effect = 0.04 * sigma if sentiment == "Bullish" else -0.04 * sigma
        else:
            sentiment_effect = 0
        
        # Adjust based on tonnage and oil prices
        future_rates[t] = future_rates[t-1] + mean_reversion + tonnage_effect + oil_effect + sentiment_effect
        
        # Variance keeps track of predicted uncertainty
        future_variances[t] = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

    return future_rates, future_variances

# Fetch real-time oil price data from Yahoo Finance
def get_historical_oil_data(start_date, end_date):
    oil_data = yf.download("CL=F", start=start_date, end=end_date)
    return oil_data['Close'].mean()

# Main Streamlit App
st.title("Freight Rate Prediction with Gradual Sentiment and Tonnage Availability")

# Input past 14 working days of freight rates from the user (from Excel, no commas)
st.write("Paste the past 14 working days' freight rates (separated by new lines or spaces):")
user_input = st.text_area("Paste rates here:")

if user_input:
    try:
        # Split the input by newline or spaces, then convert to float
        last_14_days_rates = np.array([float(rate.strip()) for rate in user_input.split() if rate.strip()])
        
        if len(last_14_days_rates) != 14:
            st.error("Please enter exactly 14 rates.")
        else:
            # Ask for market sentiment
            sentiment = st.selectbox("What's your market sentiment?", ["Bullish", "Bearish"])

            # Ask for tanker availability input
            availability = st.selectbox("What's the tanker availability?", ["High supply", "Low supply", "Average supply"])

            # Estimate model parameters based on recent data
            mu_est = np.mean(last_14_days_rates)  # Mean freight rate
            theta_est = 0.5  # Speed of mean reversion
            sigma_est = calculate_volatility(pd.Series(last_14_days_rates))  # Volatility estimate

            # Get today's oil price and calculate oil effect
            today = datetime.today()
            oil_price = get_historical_oil_data(today - timedelta(days=30), today)
            oil_effect = (oil_price - np.mean(last_14_days_rates)) * 0.05  # Calibrate oil effect

            # Tonnage effect based on availability input
            tonnage_effect = {
                "High supply": -0.05 * mu_est,   # Slight reduction in rates due to high supply
                "Low supply": 0.05 * mu_est,     # Slight increase in rates due to low supply
                "Average supply": 0              # No change for average supply
            }.get(availability, 0)

            # Predict the next 7 business days of freight rates
            predicted_future_rates, future_variances = predict_future_rates(mu_est, theta_est, sigma_est, last_14_days_rates[-1], tonnage_effect, oil_effect, sentiment, future_days=7)

            # Calculate confidence intervals for future predictions
            upper_band_future = predicted_future_rates + 1.96 * np.sqrt(future_variances)
            lower_band_future = predicted_future_rates - 1.96 * np.sqrt(future_variances)

            # Generate future dates for prediction (next 7 business days)
            future_days = 7
            start_date = today
            future_dates = pd.date_range(start=start_date + timedelta(days=1), periods=future_days)

            # Display the prediction results
            st.write("Predicted Freight Rates for the Next 7 Days (with Confidence Interval):")
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Rate': predicted_future_rates,
                'Upper Confidence Interval': upper_band_future,
                'Lower Confidence Interval': lower_band_future
            })
            st.write(predictions_df)

            # Plot the predictions
            plt.figure(figsize=(12, 6))
            plt.plot(future_dates, predicted_future_rates, label='Predicted Rates', color='blue')
            plt.fill_between(future_dates, lower_band_future, upper_band_future, color='gray', alpha=0.3, label='95% Confidence Interval')
            plt.title('Freight Rate Predictions for the Next 7 Days')
            plt.xlabel('Date')
            plt.ylabel('Freight Rate')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
    
    except ValueError:
        st.error("Invalid input. Please enter numeric rates separated by spaces or new lines.")

