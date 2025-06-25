#so in this the main objective is going to be to
# 1. retrieve daily stock price data with yfinance
# 2. plot the stock data on a graph
# 3. calculate some basic risk measures

import streamlit as st
import yfinance as yf
import plotly.express as px
import quantstats as qs

st.title("Plotting and basic risk measures")

############################################################################################################################################
#so here we determine the time window and the ticker we want to retrieve
ticker = st.sidebar.text_input('Please input the ticker symbol for the stock (as found on yahoofinance):', value='MSFT')
start_date = st.sidebar.date_input('Start date of close price retrievals:',value="2020-01-01")
end_date = st.sidebar.date_input('End date of close price retrievals:')

data = yf.download(ticker, start = start_date, end = end_date)
fig = px.line(data, x = data.index, y = data['Close'].squeeze(), title = ticker, labels={'y': 'Close'})
    
st.subheader("Daily stock close price graph:")
st.plotly_chart(fig)
############################################################################################################################################

#now here we are going to calculate the risk measures for the stock to see how it is doing
st.subheader("We will need a period to determine for how long we will go back into the past, and also a frequency for the VaR calculations as the library we are using requires it")

period = st.sidebar.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
confidence_interval = st.sidebar.selectbox("Select confidence interval for expected shortfall and VaR:", [0.9, 0.95, 0.99])
frequency = st.sidebar.selectbox("Select for the frequency for VaR and expected shortfall in trading days:", [1, 21, 63, 126, 252])

stock = qs.utils.download_returns(ticker, period = period)
stock2 = qs.utils.download_returns(ticker, period= "5y")

st.subheader(f"Key Metrics for {ticker}:")
sharpe = qs.stats.sharpe(stock)
calmar = qs.stats.calmar(stock)
cagr = qs.stats.cagr(stock)
volatility = qs.stats.volatility(stock)
kurtosis = qs.stats.kurtosis(stock)
skewness = qs.stats.skew(stock)
av_return = qs.stats.avg_return(stock)
av_loss = qs.stats.avg_loss(stock)
av_win = qs.stats.avg_win(stock)
win_loss_ratio = qs.stats.win_loss_ratio(stock)
es = qs.stats.expected_shortfall(stock2, confidence=confidence_interval)
es_adjusted = es * (frequency ** 0.5)
VaR = qs.stats.value_at_risk(stock2, confidence=confidence_interval)
VaR_adjusted = VaR * (frequency ** 0.5)
max_drawdown = qs.stats.max_drawdown(stock)
st.write(f"Sharpe ratio for the past {period} period:", sharpe)
st.write(f"Calmar ratio for the past {period} period:", calmar)
st.write(f"The compound annual growth rate for the past {period} period: {cagr*100:.2f}%")
st.write(f"The annualized volatility of the stock based on the past {period} period:{volatility*100:.2f}%")
st.write(f"Kurtosis of daily returns for a {period} period: {kurtosis}")
st.write(f"The skewness of daily returns for a {period} period: {skewness}")
st.write(f"Average daily return for the past {period} period: {av_return*100:.2f}%")
st.write(f"Average daily loss of losing days for the past {period} period: {av_loss*100:.2f}%")
st.write(f"Average daily gain of winning days for the past {period} period: {av_win*100:.2f}%")
st.write(f"The daily win/loss ratio for the past {period} period: {win_loss_ratio}")
st.write(f"Maximum drawdown for the past {period} period: {max_drawdown*100:.2f}%")
st.write(f"Expected shortfall with {int((confidence_interval) * 100)}% confidence level and{frequency} frequency: {es_adjusted*100:.2f}%")
st.write(f"Value at Risk with {int((confidence_interval) * 100)}% confidence level and{frequency} frequency: {VaR_adjusted*100:.2f}%")
############################################################################################################################################
