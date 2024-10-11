# Adjust sentiment and tonnage effects based on document analysis
def predict_future_rates(mu, theta, sigma, last_rate, tonnage_effect, oil_effect, sentiment, future_days=7):
    future_rates = np.zeros(future_days)
    future_variances = np.zeros(future_days)
    dt = 1  # Daily steps
    
    future_rates[0] = last_rate  # Start with the last known rate
    future_variances[0] = sigma ** 2  # Initial variance based on current volatility

    for t in range(1, future_days):
        # Mean reversion component
        mean_reversion = theta * (mu - future_rates[t-1])
        
        # Sentiment effect: Affects initial days more significantly (Bearish: Sharp drop, Bullish: Sharp rise)
        if sentiment == "Bullish":
            sentiment_effect = 0.03 * sigma * min(t, 4)  # Strong positive effect peaking around day 4
        else:
            sentiment_effect = -0.03 * sigma * min(t, 4)  # Strong negative effect, sharper drop for bearish

        # Tonnage effect based on availability input (High Supply: sharp downward impact, Low Supply: upward impact)
        tonnage_adjustment = tonnage_effect * 1.5 if tonnage_effect < 0 else tonnage_effect * 2
        
        # Adjust based on tonnage and oil effects
        future_rates[t] = future_rates[t-1] + mean_reversion + tonnage_adjustment + oil_effect + sentiment_effect
        
        # Variance keeps track of predicted uncertainty
        future_variances[t] = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

    return future_rates, future_variances



