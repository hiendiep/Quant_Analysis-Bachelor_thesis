Introduction
+ This repository contains a Python-based backtesting engine designed to evaluate the effectiveness of trend-following trading strategies in the cryptocurrency market. 
+ The code was developed as part of my graduation thesis, "The Effectiveness of Trend-Following Strategy in Cryptocurrency Market", in May 2024. 
+ The thesis investigates the performance of 3,200 trend-following trading rules across four strategy classes
___Moving Average (MA)
___Resistance and Support (R&S)
___Relative Strength Index (RSI)
___and Moving Average Convergence Divergence (MACD)—applied to 12 major cryptocurrencies from January 2017 to April 2024.
+ The engine:
___Fetches historical cryptocurrency data
___Implements the specified trading strategies
___Constructs equally-weighted portfolios
___Assesses their performance using key metrics such as cumulative return, annualized return, Sharpe ratio, Sortino ratio, and Calmar ratio.
+ It also incorporates robust statistical techniques, including stationary bootstrapping for hypothesis testing and adjustments for data snooping biases, to ensure reliable results.
+ This codebase aligns with the methodology outlined in the thesis, providing a practical tool to replicate and extend the empirical analysis.


Purpose: The primary goal of this code is to:
+ Simulate and backtest trend-following strategies on cryptocurrency price data.
+ Evaluate strategy performance at both ticker and portfolio levels across multiple time periods.
+ Provide actionable insights for investors by comparing strategy outcomes against a Buy-and-Hold benchmark, while accounting for transaction costs and market conditions.
=> This repository serves as a companion to the thesis, enabling researchers, traders, and enthusiasts to explore the effectiveness of trend-following strategies in the dynamic and volatile cryptocurrency market.

Features
+ Data Loading: Retrieves historical price data for specified cryptocurrency tickers from a local Excel file (adaptable to APIs like yfinance).
+ Strategy Formulation: Implements MA, R&S, RSI, and MACD strategies with customizable parameters.
+ Portfolio Construction: Forms equally-weighted portfolios rebalanced annually.
+ Performance Measurement: Calculates risk-adjusted returns and conducts statistical inference.
Reporting: Generates detailed performance reports at ticker, strategy, and portfolio levels.


File:
+ Thesis pdf file
+ Complete code file (Final Code)
+ Excel file: https://docs.google.com/spreadsheets/d/1h-Hgyuglj80sSApsC6JmyxRsRJNDnqgV/edit?usp=sharing&ouid=101905111734653218108&rtpof=true&sd=true

