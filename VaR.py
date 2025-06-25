#so in this file I am going to mainly concentrate on the Value at Risk measure, it should be fairly easy to implement

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import datetime 
import numpy as np
import seaborn as sns
from scipy.stats import norm

st.header("Value at Risk calculations")
st.subheader("We have 3 main ways to determine VaR: 1. Historical Method 2. Variance-Covariance Method 3. Monte-Carlo method")
st.write("These are all widely used, and have been listed from least to most advanced")

st.subheader("1. Historical Method")
ticker = st.sidebar.text_input("Input the ticker of the stock that you would like to use:", value="MSFT")
start_date = st.sidebar.date_input("Input the starting date for downloading stock data:", value="2020-01-01")
end_date = st.sidebar.date_input("Input the ending date of the time window for downloading stock data:", value=datetime.date.today())

#downloading the stock data for the adjusted daily close prices
stock_df = yf.download(ticker, start=start_date, end=end_date, interval="1d")[['Close']]
#adding the logarithmic daily returns into the dataframe
log_returns = np.log(stock_df['Close']/ stock_df['Close'].shift(1))
log_returns.name = 'Log Return'
#combining the series data
stock_df = pd.concat([stock_df['Close'], log_returns], axis=1)
stock_df.columns = ['Adj Close', 'Log Return']
#Displaying
st.write(stock_df)

#getting the confidence level
confidence_level = st.sidebar.slider("Select a confidence interval (95% means 0.95)", min_value=0.0, max_value=1.0, step=0.05, value=0.95)
alpha = 1 - confidence_level

#calculating VaR
VaR = np.percentile(stock_df['Log Return'].dropna(), 100 * alpha)
#display
st.write(f"The {int(confidence_level*100)}% Historical VaR is: {VaR:.4f}")

#plotting all of this
# Create the plot
fig1, ax = plt.subplots(figsize=(10,6))

sns.histplot(stock_df['Log Return'].dropna(), bins=50, kde=True, color='skyblue', ax=ax)
ax.axvline(VaR, color='red', linestyle='--', label=f'{int(confidence_level*100)}% VaR: {VaR:.4f}')

ax.set_title(f'Distribution of Daily Log Returns for {ticker}')
ax.set_xlabel('Log Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig1)

#adjusting it for frequency
frequency = st.sidebar.selectbox("Select for the frequency for VaR and expected shortfall in trading days:", [1, 21, 63, 126, 252])
VaR_adjusted = VaR * (frequency ** 0.5)
st.write(f"Meaning that there is a(n) {confidence_level}% probability that the log return on a given day will be greater than {VaR * 100}%")
st.write(f"Value at Risk with {int((confidence_level) * 100)}% confidence level and {frequency} frequency: {VaR_adjusted*100:.2f}%")
##################################################################################################################################################################################################
st.header("2. Variance-Covariance Method")
st.write("Assumptions: 1. Returns are normally distributed: they follow a bell-shaped curve, fully described by mean and std dev. 2. Each day's return is independent of the previous, and drawn from the same distribution. 3. The mean and standard deviation are constant over time. 4. The portfolio’s value changes linearly with asset returns. 5. The normal distribution has 0 skew and kurtosis = 3. 6. Past returns do not predict future returns.")

# Mean and Std Dev of daily log returns
mu = stock_df['Log Return'].mean()
sigma = stock_df['Log Return'].std()
st.write("Mean:", mu)
st.write("Variance: ", sigma)
# Calculate z-score for the confidence level
z_score = norm.ppf(alpha)

# Daily VaR
VaR_vc = mu + z_score * sigma

# Adjusted VaR for selected frequency
VaR_vc_adjusted = VaR_vc * (frequency ** 0.5)

# Display results
st.write(f"Variance-Covariance Daily VaR at {int(confidence_level*100)}% confidence level: {VaR_vc*100:.4f}%")
st.write(f"Variance-Covariance VaR over {frequency} trading days: {VaR_vc_adjusted*100:.2f}%")

#plotting
# Theoretical Normal Distribution Plot for Variance-Covariance VaR
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)  # range of possible returns
pdf = norm.pdf(x, mu, sigma)  # normal PDF based on mean and std dev

fig2, ax = plt.subplots(figsize=(10,6))

ax.plot(x, pdf, color='blue', label='Normal Distribution of Returns')
ax.axvline(VaR_vc, color='red', linestyle='--', label=f'{int(confidence_level*100)}% VaR: {VaR_vc*100:.4f}%')

ax.set_title(f'Variance-Covariance Method: Theoretical Return Distribution for {ticker}')
ax.set_xlabel('Log Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True)

st.pyplot(fig2)
##################################################################################################################################################################################################
st.subheader("3. Monte Carlo Method")

# Number of simulations
num_simulations = st.sidebar.slider("Number of Monte Carlo simulations:", min_value=1000, max_value=50000, step=1000, value=10000)

# Simulate returns
simulated_returns = np.random.normal(mu, sigma, num_simulations)

# Monte Carlo VaR (left-tail alpha quantile)
VaR_mc = np.percentile(simulated_returns, 100 * alpha)

# Adjust for frequency
VaR_mc_adjusted = VaR_mc * (frequency ** 0.5)

# Display results
st.write(f"Monte Carlo Daily VaR at {int(confidence_level*100)}% confidence level: {VaR_mc*100:.4f}%")
st.write(f"Monte Carlo VaR over {frequency} trading days: {VaR_mc_adjusted*100:.2f}%")

# Plot Monte Carlo simulated return distribution
fig3, ax = plt.subplots(figsize=(10,6))
sns.histplot(simulated_returns, bins=50, kde=True, color='green', label='Simulated Returns', ax=ax)
ax.axvline(VaR_mc, color='red', linestyle='--', label=f'{int(confidence_level*100)}% VaR: {VaR_mc*100:.4f}%')

ax.set_title(f'Monte Carlo Method: Simulated Return Distribution for {ticker}')
ax.set_xlabel('Simulated Log Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True)

st.pyplot(fig3)

st.subheader("Explanation of the Monte Carlo Method")
st.write("1. Assume that the returns follow a normal distribution 2.We then simulate many outcomes, and generate random samples from the normal simulation based on the mu and the variance and the number of simulations 3. Then this is the distribution, and then we find the left tail percentile of this distribution based on the confidence level 4. This is daily, and then we use the adjusted as well Final: the daily VaR refreshes because of the simulation every time, that is why it changes")
