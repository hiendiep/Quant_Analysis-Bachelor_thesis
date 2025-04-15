import yfinance as yf
import pandas as pd
import numpy as np
from playhouse.signals import Signal
from tabulate import tabulate
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import shapiro
from arch import arch_model
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from statsmodels.stats.diagnostic import het_arch
from arch.bootstrap import StationaryBootstrap
from scipy.stats import norm


# CODE PART 1: GENERATE PORTFOLIOS AND REPORT THEIR PERFORMANCE
# A. DATA LOADING
# Input: Ticker price data in Excel sheet
# Output: Ticker return series with proper setting

class LoadTicker:
    def __init__(self, ticker_symbol, start_date_download, end_date_download):
        self.ticker_symbol = ticker_symbol
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.data = None
        self.file_path = r"C:\Users\ASUS\Desktop\Thesis\Data\Day\Data.xlsx"
        self.load()

    def load(self):
        self.data = pd.read_excel(self.file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
        self.data.set_index('Date', inplace=True)
        self.data = self.data.sort_index()
        self.data = self.data[[self.ticker_symbol]]
        self.data['Adj Close'] = self.data[self.ticker_symbol]

        self.data = self.data.loc[self.start_date_download:self.end_date_download]
        self.data.drop(self.data.index[-1], inplace = True)
        self.data['ret'] = self.data['Adj Close'].pct_change()


# ----------------------------------------------------------------------------------------------------------------------


# B. STRATEGY LOADING
# Input: Section A
# Output: creates trading rule under 4 classes of Trend-following strategy with both long and short order allowed
# To create long-only rule, expand the functions and follow instruction

#   1. Moving Average Strategy
class LoadStrategy_MA():
    def __init__(self, load_ticker, short_window, long_window, x, d):
        self.data = load_ticker.data
        self.short_window = short_window
        self.long_window = long_window
        self.x = x
        self.d = d

        self.name = 'Moving Average'
        self.parameterization = "short-%i, long-%i, x-%f, d-%i"  %(self.short_window, self.long_window, self.x, self.d)
        self.ticker_symbol = load_ticker.ticker_symbol
        self.gen_signal()

    def gen_moving_average(self, window):
        return self.data['Adj Close'].rolling(window=window).mean()

    def gen_signal(self):
        self.data['MA_short'] = self.gen_moving_average(self.short_window)
        self.data['MA_long'] = self.gen_moving_average(self.long_window)
        self.data['MA_diff'] = self.data['MA_short'] / self.data['MA_long'] - 1
        self.data['Condition_x'] = np.select([self.data['MA_diff'] >= self.x, self.data['MA_diff'] <= -self.x], [1,-1], 0)
        self.data['Condition_d'] = self.data['Condition_x'].rolling(window = self.d).sum()

        self.data['Signal'] = np.select([self.data['Condition_d'] == self.d, self.data['Condition_d'] == -self.d], [1,-1], np.nan)
        self.data['Signal'] = self.data['Signal'].ffill()
        self.data['Signal_lagged'] = self.data['Signal'].shift(1)

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0, self.data['Signal_lagged'] ==0 ],
                                        [self.data['ret'], -self.data['ret'], 0], default=np.nan)
        # Switch the 2 lines code above to the 2 lines code below for Long-only Moving Average strategy
        # self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] <= 0 ],
        #                                 [self.data['ret'], 0], default=np.nan)

#   2. Resistance and Support Strategy
class LoadStrategy_RnS():
    def __init__(self, load_ticker, window, x, d):
        self.data = load_ticker.data
        self.window = window
        self.x = x
        self.d = d

        self.name = 'Resistance and Support'
        self.parameterization = "window-%i, x-%f, d-%i"  %(self.window, self.x, self.d)
        self.ticker_symbol = load_ticker.ticker_symbol
        self.gen_signal()

    def gen_signal(self):
        self.data['Moving_min'] = self.data['Adj Close'].shift(1).rolling(window=self.window).min()
        self.data['Cross_min'] = np.nan
        self.data['Testing_min'] = np.nan

        self.data['Moving_max'] = self.data['Adj Close'].shift(1).rolling(window=self.window).max()
        self.data['Cross_max'] = np.nan
        self.data['Testing_max'] = np.nan

        self.data['Min_diff'] = np.nan
        self.data['Max_diff'] = np.nan
        self.data['Condition_x'] = np.nan
        self.data['Condition_d'] = np.nan
        self.data['Signal'] = 1            #
        self.data.reset_index(drop=False, inplace=True)

        for i in range(0, len(self.data)):
            if i < self.window:
                continue
            else:

                self.data.loc[i, 'Cross_min'] = np.where((self.data.loc[i, 'Adj Close'] <= self.data.loc[i, 'Moving_min']) & (self.data.loc[i-1, 'Adj Close'] > self.data.loc[i-1, 'Moving_min']),
                    self.data.loc[i, 'Moving_min'], np.nan)
                self.data.loc[i, 'Cross_max'] = np.where((self.data.loc[i, 'Adj Close'] >= self.data.loc[i, 'Moving_max']) & (self.data.loc[i - 1, 'Adj Close'] < self.data.loc[i - 1, 'Moving_max']),
                    self.data.loc[i, 'Moving_max'], np.nan)

                self.min_x = (self.data.loc[i-self.d+1 : i-1, 'Condition_x']==-1).sum()
                self.max_x = (self.data.loc[i-self.d+1 : i-1, 'Condition_x']== 1).sum()

                self.data.loc[i, 'Testing_min'] = np.where((self.min_x>0) & (self.min_x<=self.d),
                                                           np.where(pd.isna(self.data.loc[i - 1, 'Testing_min']),
                                                                    self.data.loc[i - 1, 'Cross_min'],
                                                                    self.data.loc[i - 1, 'Testing_min']), np.nan)

                self.data.loc[i, 'Testing_max'] = np.where((self.max_x>0) & (self.max_x<=self.d),
                                                           np.where(pd.isna(self.data.loc[i-1,'Testing_max']),
                                                             self.data.loc[i - 1, 'Cross_max'],
                                                             self.data.loc[i - 1,'Testing_max']), np.nan)

                self.data.loc[i, 'Min_diff'] = np.where(self.data.loc[i-1, 'Signal'] == -1, np.nan,
                                                        np.where(pd.notna(self.data.loc[i, 'Testing_min']), self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Testing_min']-1,
                                                                 np.where(pd.notna(self.data.loc[i, 'Cross_min']), self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Cross_min']-1,
                                                                          self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Moving_min']-1)))
                self.data.loc[i, 'Max_diff'] = np.where(self.data.loc[i-1, 'Signal'] == 1, np.nan,
                                                        np.where(pd.notna(self.data.loc[i, 'Testing_max']), self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Testing_max']-1,
                                                                 np.where(pd.notna(self.data.loc[i, 'Cross_max']), self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Cross_max']-1,
                                                                          self.data.loc[i, 'Adj Close']/self.data.loc[i, 'Moving_max']-1)))

                self.data.loc[i, 'Condition_x'] = np.where(pd.notna(self.data.loc[i, 'Max_diff']),
                                                           np.where(self.data.loc[i, 'Max_diff'] >= self.x,1, np.nan),
                                                           np.where(self.data.loc[i, 'Min_diff'] <= -self.x, -1, np.nan))

                self.data.loc[i, 'Condition_d'] = self.data.loc[i-self.d+1:i, 'Condition_x'].sum()
                self.data.loc[i, 'Signal'] = np.where(self.data.loc[i, 'Condition_d'] == self.d, 1,
                                                      np.where(self.data.loc[i, 'Condition_d'] == -self.d, -1,
                                                               self.data.loc[i-1, 'Signal']))

        self.data['Signal_lagged'] = self.data['Signal'].shift(1)
        self.data.set_index('Date', inplace=True)

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0, self.data['Signal_lagged'] ==0 ],
                                        [self.data['ret'], -self.data['ret'], 0], default=np.nan)
        # Switch the 2 lines code above to the 2 lines code below for Long-only Support & Resistance strategy
        # self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] <= 0 ],
        #                                 [self.data['ret'], 0], default=np.nan)

#   3. RSI Strategy
class LoadStrategy_RSI():
    def __init__(self, load_ticker, window, v, d):
        self.data = load_ticker.data
        self.window = window
        self.v = v
        self.d = d

        self.name = 'RSI Oscillator'
        self.parameterization = "window-%i, v-%f, d-%i"  %(self.window, self.v, self.d)
        self.ticker_symbol = load_ticker.ticker_symbol
        self.gen_signal()

    def gen_signal(self):
        self.delta = self.data['Adj Close'].diff(1)
        self.data['gain'] = np.where(self.delta>0, self.delta, 0)
        self.data['loss'] = np.where(self.delta<0, -self.delta, 0)
        self.avg_gain = self.data['gain'].rolling(window=self.window).mean()
        self.avg_loss = self.data['loss'].rolling(window=self.window).mean()
        self.RS = self.avg_gain/self.avg_loss
        self.data['raw RSI'] = 100 - (100/(1+self.RS))
        self.data['RSI'] = self.data['raw RSI'].rolling(window=3).mean()

        self.data['Condition_x'] = np.select([self.data['RSI'] >= 50+self.v, self.data['RSI'] <= 50-self.v], [1, -1], default=0)
        self.data['Condition_d'] = self.data['Condition_x'].rolling(window = self.d).sum()

        self.data['Signal'] = np.select([(self.data['Condition_d'] ==  self.d) & (self.data['RSI'].shift(self.d) < 50+self.v),
                                                 (self.data['Condition_d'] == -self.d) & (self.data['RSI'].shift(self.d) > 50-self.v)],
                                        [1,-1], np.nan)
        self.data['Signal'] = self.data['Signal'].ffill()
        self.data['Signal_lagged'] = self.data['Signal'].shift(1)

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] <= 0 ],
                                        [self.data['ret'], 0], default=np.nan)

        # Switch the 2 lines code above to the 2 lines code below for Long-only RSI strategy
        # self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] <= 0 ],
        #                                 [self.data['ret'], 0], default=np.nan)

#   4. MACD Strategy
class LoadStrategy_MACD():
    def __init__(self, load_ticker, short_window, long_window, signal_window, x, d):
        self.data = load_ticker.data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.x = x
        self.d = d

        self.name = 'MACD'
        self.parameterization = "short-%i, long-%i, sigal- %i, x-%f, d-%i"  %(self.short_window, self.long_window, self.signal_window, self.x, self.d)
        self.ticker_symbol = load_ticker.ticker_symbol
        self.gen_signal()

    def gen_exp_average(self, window):
        return self.data['Adj Close'].ewm(span=window, adjust=False).mean()

    def gen_signal(self):
        self.data['EMA_short'] = self.gen_exp_average(self.short_window)
        self.data['EMA_long'] = self.gen_exp_average(self.long_window)
        self.data['MACD'] = self.data['EMA_short'] - self.data['EMA_long']
        self.data['MACD_signal'] = self.data['MACD_signal'].ewm(span=self.signal_window).mean()

        self.data['MACD_diff'] = (self.data['MACD'] - self.data['MACD_signal']) / abs(self.data['MACD_signal'])
        self.data['Condition_x'] = np.select([self.data['MACD_diff'] >= self.x, self.data['MACD_diff'] <= -self.x], [1,-1], 0)
        self.data['Condition_d'] = self.data['Condition_x'].rolling(window = self.d).sum()

        self.data['Signal'] = np.select([self.data['Condition_d'] == self.d, self.data['Condition_d'] == -self.d], [1,-1], np.nan)
        self.data['Signal'] = self.data['Signal'].ffill()
        self.data['Signal_lagged'] = self.data['Signal'].shift(1)

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0, self.data['Signal_lagged'] ==0 ],
                                        [self.data['ret'], -self.data['ret'], 0], default=np.nan)
        # Switch the 2 lines code above to the 2 lines code below for Long-only RSI strategy
        # self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] <= 0 ],
        #                                 [self.data['ret'], 0], default=np.nan)


# ----------------------------------------------------------------------------------------------------------------------


# C. PORTFOLIO FORMATION: YEAR BEGIN EQUALLY DISTRIBUTED
# Input: Loaded data of portfolio ticker's component from section B
# Ouput: Strategy-specific portfolio objects with portfolio detail properties and portfolio return, price index in exact testing time range

# 1. Buy and Hold Portfolio
class BnH_Portfolio:
    def __init__(self, ticker_list, start_date_download, end_date_download, start_date_testing, end_date_testing):
        self.ticker_list = ticker_list
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.start_date_testing = start_date_testing
        self.end_date_testing = end_date_testing

        self.data = pd.DataFrame()
        self.create_port()
        self.filter_data()

    def create_port(self):
        for i in self.ticker_list:
            load_tickers = LoadTicker(i, self.start_date_download, self.end_date_download)
            self.data[i] = load_tickers.data['Adj Close']
            self.data[i+' ret'] = load_tickers.data['ret']
            self.data[i+' weight'] = 1/len(self.ticker_list)

        self.data['port_ret'] = 0.0                       # Generate portfolio return and ticker's weight
        for j in self.data.loc[self.start_date_testing:self.end_date_testing].index:
            for i in self.ticker_list:
                self.data.loc[j, 'port_ret'] += (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] - 1/len(self.ticker_list)
            for i in self.ticker_list:
                self.data.loc[j, i+' weight'] = (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] / (1+self.data.loc[j, 'port_ret'])

        self.data['Adj Close'] = 1000.0                   # Generate portfolio price index as output for Performance class
        self.data.loc[self.start_date_testing:self.end_date_testing, 'Adj Close'] = (1 + self.data['port_ret']).cumprod()*1000
        self.data['Strategy class'] = 'Buy and Hold'
        self.data['Parameterization'] = 'NA'
    def filter_data(self):
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()

# 2. Strategy_MA Portfolio
class Strategy_MA_Portfolio:
    def __init__(self, ticker_list, start_date_download, end_date_download, start_date_testing, end_date_testing, short_window, long_window, x, d):
        self.ticker_list = ticker_list
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.start_date_testing = start_date_testing
        self.end_date_testing = end_date_testing
        self.long_window = long_window
        self.short_window = short_window
        self.x = x
        self.d = d

        self.data = pd.DataFrame()
        self.create_port()
        self.filter_data()

    def create_port(self):
        for i in self.ticker_list:
            load_tickers = LoadTicker(i, self.start_date_download, self.end_date_download)
            load_strategy = LoadStrategy_MA(load_tickers, self.short_window, self.long_window, self.x, self.d)
            self.data[i+' ret'] = load_strategy.data['strategy_ret']
            self.data[i+' weight'] = 1/len(self.ticker_list)

        self.data['port_ret'] = 0.0                       # Generate portfolio return and ticker's weight
        for j in self.data.loc[self.start_date_testing:self.end_date_testing].index:
            for i in self.ticker_list:
                self.data.loc[j, 'port_ret'] += (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] - 1/len(self.ticker_list)
            for i in self.ticker_list:
                self.data.loc[j, i+' weight'] = (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] / (1+self.data.loc[j, 'port_ret'])

        self.data['Adj Close'] = 1000.0                   # Generate portfolio price index as output for Performance class
        self.data.loc[self.start_date_testing:self.end_date_testing, 'Adj Close'] = (1 + self.data['port_ret']).cumprod()*1000
        self.data['Strategy class'] = 'Moving Average'
        self.data['Parameterization'] = "short-%i, long-%i, x-%f, d-%i"  %(self.short_window, self.long_window, self.x, self.d)
    def filter_data(self):
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()

# 3. Strategy_RnS Portfolio
class Strategy_RnS_Portfolio:
    def __init__(self, ticker_list, start_date_download, end_date_download, start_date_testing, end_date_testing, window, x, d):
        self.ticker_list = ticker_list
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.start_date_testing = start_date_testing
        self.end_date_testing = end_date_testing
        self.window = window
        self.x = x
        self.d = d

        self.data = pd.DataFrame()
        self.create_port()
        self.filter_data()

    def create_port(self):
        for i in self.ticker_list:
            load_tickers = LoadTicker(i, self.start_date_download, self.end_date_download)
            load_strategy = LoadStrategy_RnS(load_tickers, self.window, self.x, self.d)
            self.data[i+' ret'] = load_strategy.data['strategy_ret']
            self.data[i+' weight'] = 1/len(self.ticker_list)

        self.data['port_ret'] = 0.0                       # Generate portfolio return and ticker's weight
        for j in self.data.loc[self.start_date_testing:self.end_date_testing].index:
            for i in self.ticker_list:
                self.data.loc[j, 'port_ret'] += (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] - 1/len(self.ticker_list)
            for i in self.ticker_list:
                self.data.loc[j, i+' weight'] = (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] / (1+self.data.loc[j, 'port_ret'])

        self.data['Adj Close'] = 1000.0                   # Generate portfolio price index as output for Performance class
        self.data.loc[self.start_date_testing:self.end_date_testing, 'Adj Close'] = (1 + self.data['port_ret']).cumprod()*1000
        self.data['Strategy class'] = 'Support & Resistance'
        self.data['Parameterization'] = "window-%i, x-%f, d-%i"  %(self.window, self.x, self.d)
    def filter_data(self):
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()

# #4. Strategy RSI Portfolio
class Strategy_RSI_Portfolio:
    def __init__(self, ticker_list, start_date_download, end_date_download, start_date_testing, end_date_testing, window, v, d):
        self.ticker_list = ticker_list
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.start_date_testing = start_date_testing
        self.end_date_testing = end_date_testing
        self.window = window
        self.v = v
        self.d = d

        self.data = pd.DataFrame()
        self.create_port()
        self.filter_data()

    def create_port(self):
        for i in self.ticker_list:
            load_tickers = LoadTicker(i, self.start_date_download, self.end_date_download)
            load_strategy = LoadStrategy_RSI(load_tickers, self.window, self.v, self.d)
            self.data[i+' ret'] = load_strategy.data['strategy_ret']
            self.data[i+' weight'] = 1/len(self.ticker_list)

        self.data['port_ret'] = 0.0                       # Generate portfolio return and ticker's weight
        for j in self.data.loc[self.start_date_testing:self.end_date_testing].index:
            for i in self.ticker_list:
                self.data.loc[j, 'port_ret'] += (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] - 1/len(self.ticker_list)
            for i in self.ticker_list:
                self.data.loc[j, i+' weight'] = (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] / (1+self.data.loc[j, 'port_ret'])

        self.data['Adj Close'] = 1000.0                   # Generate portfolio price index as output for Performance class
        self.data.loc[self.start_date_testing:self.end_date_testing, 'Adj Close'] = (1 + self.data['port_ret']).cumprod()*1000
        self.data['Strategy class'] = 'RSI Oscillator'
        self.data['Parameterization'] = "window-%i, v-%f, d-%i"  %(self.window, self.v, self.d)
    def filter_data(self):
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()


# 5. Strategy_MACD Portfolio
class Strategy_MACD_Portfolio:
    def __init__(self, ticker_list, start_date_download, end_date_download, start_date_testing, end_date_testing, short_window, long_window, signal_window, x, d):
        self.ticker_list = ticker_list
        self.start_date_download = start_date_download
        self.end_date_download = end_date_download
        self.start_date_testing = start_date_testing
        self.end_date_testing = end_date_testing
        self.long_window = long_window
        self.short_window = short_window
        self.signal_window = signal_window
        self.x = x
        self.d = d

        self.data = pd.DataFrame()
        self.create_port()
        self.filter_data()

    def create_port(self):
        for i in self.ticker_list:
            load_tickers = LoadTicker(i, self.start_date_download, self.end_date_download)
            load_strategy = LoadStrategy_MACD(load_tickers, self.short_window, self.long_window, self.signal_window, self.x, self.d)
            self.data[i+' ret'] = load_strategy.data['strategy_ret']
            self.data[i+' weight'] = 1/len(self.ticker_list)

        self.data['port_ret'] = 0.0                       # Generate portfolio return and ticker's weight
        for j in self.data.loc[self.start_date_testing:self.end_date_testing].index:
            for i in self.ticker_list:
                self.data.loc[j, 'port_ret'] += (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] - 1/len(self.ticker_list)
            for i in self.ticker_list:
                self.data.loc[j, i+' weight'] = (1 + self.data[i + ' ret'].loc[j]) * self.data[i + ' weight'].shift(1).loc[j] / (1+self.data.loc[j, 'port_ret'])

        self.data['Adj Close'] = 1000.0                   # Generate portfolio price index as output for Performance class
        self.data.loc[self.start_date_testing:self.end_date_testing, 'Adj Close'] = (1 + self.data['port_ret']).cumprod()*1000
        self.data['Strategy class'] = 'MACD'
        self.data['Parameterization'] = "short-%i, long-%i, signal-%i, x-%f, d-%i"  %(self.short_window, self.long_window, self.signal_window, self.x, self.d)
    def filter_data(self):
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()

# ----------------------------------------------------------------------------------------------------------------------


# D. PERFORMANCE
# Input: Section B
# Output: Generate performance report for any trading rule portfolio
class Portfolio_Performance:
    def __init__(self, load_portfolio, risk_free_rate):
        self.data = load_portfolio
        # self.ticker_symbol = load_portfolio.name
        # self.start_date_testing = pd.to_datetime(start_date_testing)
        # self.end_date_testing = pd.to_datetime(end_date_testing)
        self.daily_mean_risk_free_rate = (1+risk_free_rate) ** (1/365) - 1
        self.data['Return'] = self.data['port_ret']

        self.sharpe_ratio()
        self.sortino_ratio()
        self.calmar_ratio()
        self.inference()
        self.portfolio_report()


    # 1. Accumulative return of Testing period, and Interval in Testing Period
    def accum_return_properties(self, return_series = None, **options):
        if return_series is None:
            return_series = self.data['Return']     # other options: self.data['Long_return'], self.data['Short_return']

        interval = options.get('interval', 'None')          # other options: 'YE', 'ME',...
        properties = options.get('properties', 'Normal')    # other options: 'max drawdown', 'running max', ' drawdown'

        if interval == 'None':
            self.data['Accum_return'] = (1 + return_series).cumprod() - 1
            self.running_max = (1+self.data['Accum_return']).cummax()
            self.drawdown = ((1+self.data['Accum_return']) - self.running_max) / self.running_max
            self.max_drawdown = self.drawdown.min()
            if properties == 'Normal':
                return self.data['Accum_return']
            if properties == 'max drawdown':
                return self.max_drawdown
            if properties == 'running max':
                return self.running_max
            if properties == 'drawdown':
                return self.drawdown

        if interval == 'YE':
            self.data['Accum_return_yearly'] = (1 + return_series).groupby(self.data.index.year).cumprod() - 1
            self.running_max = (1 + self.data['Accum_return_yearly']).groupby(self.data.index.year).cummax()
            self.drawdown = ((1 + self.data['Accum_return_yearly']) - self.running_max) / self.running_max
            self.max_drawdown = self.drawdown.resample(interval).min()
            if properties == 'Normal':
                return self.data['Accum_return_yearly']
            if properties == 'max drawdown':
                return self.max_drawdown
            if properties == 'running max':
                return self.running_max
            if properties == 'drawdown':
                return self.drawdown

        if interval == 'ME':
            self.data['Accum_return_monthly'] = (1 + return_series).groupby(self.data.index.month).cumprod() - 1
            self.running_max = (1 + self.data['Accum_return_monthly']).groupby(self.data.index.month).cummax()
            self.drawdown = ((1 + self.data['Accum_return_monthly']) - self.running_max) / self.running_max
            self.max_drawdown = self.drawdown.resample(interval).min()
            if properties == 'Normal':
                return self.data['Accum_return_monthly']
            if properties == 'max drawdown':
                return self.max_drawdown
            if properties == 'running max':
                return self.running_max
            if properties == 'drawdown':
                return self.drawdown


    # 2. Final return: Testing period, Interval in Testing Period
    def final_return(self, return_series = None, **options):
        if return_series is None:
            return_series = self.data['Return']     # other options: self.data['Long_return'], self.data['Short_return']

        interval = options.get('interval', 'None')  # other options: 'YE', 'ME',...
        form = options.get('form', 'final return')  # other options: 'daily mean', 'annualized'

        if interval == 'None':
            self.result = (1 + return_series).product() - 1
            if form == 'final return':
                return self.result
            if form == 'daily mean':
                return (1 + self.result) ** (1 / len(return_series)) - 1
            if form == 'annualized':
                return (1 + self.result) ** (365 / len(return_series)) - 1

        else:
            self.result = return_series.resample(interval).apply(lambda x: (1+x).prod()-1)
            self.data['index'] = 1
            self.interval_len = self.data['index'].resample(interval).sum()
            if form == 'final return':
                return self.result
            if form == 'daily mean':
                return (1+self.result) ** (1/self.interval_len) - 1
            if form == 'annualized':
                return (1+self.result) ** (365/self.interval_len) - 1


    # 3. Standard Deviation
    def std_dev(self, return_series = None, **options):
        if return_series is None:
            return_series = self.data['Return']     # other options: self.data['Long_return'], self.data['Short_return']

        interval = options.get('interval', 'None')  # other options: 'YE', 'ME',...
        side = options.get('side', 'all side')      # other options: 'downside', 'upside'

        if side == 'all side':
            self.return_series = return_series
        elif side == 'downside':
            self.return_series = self.daily_downside_return = return_series[return_series < self.daily_mean_risk_free_rate]
        else:
            self.return_series = self.daily_upside_return = return_series[return_series > self.daily_mean_risk_free_rate]

        if interval == 'None':
            return self.return_series.std()
        else:
            return self.return_series.resample(interval).std()


    # 4. Sharpe ratio
    def sharpe_ratio(self, **options):
        interval = options.get('interval', 'None')      # other options: 'YE', 'ME',..
        form = options.get('form', 'daily')             # other options: 'annualized'

        self.daily_mean_return = self.final_return(interval=interval, form='daily mean')
        self.daily_std_dev = self.std_dev(interval = interval)
        self.daily_sharpe_ratio = (self.daily_mean_return - self.daily_mean_risk_free_rate) / self.daily_std_dev

        if form == 'daily':
            return self.daily_sharpe_ratio
        if form == 'annualized':
            return self.daily_sharpe_ratio * np.sqrt(365)


    # 5. Sortino ratio
    def sortino_ratio(self, **option):
        interval = option.get('interval', 'None')
        form = option.get('form', 'daily')

        self.daily_mean_return = self.final_return(interval=interval, form='daily mean')
        self.daily_downside_std_dev = self.std_dev(interval=interval, side='downside')
        self.daily_sortino = (self.daily_mean_return - self.daily_mean_risk_free_rate) / self.daily_downside_std_dev

        if form == 'daily':
            return self.daily_sortino
        if form == 'annualized':
            return self.daily_sortino * np.sqrt(365)


    # 6. Calmar ratio
    def calmar_ratio(self, **option):
        interval = option.get('interval', 'None')

        self.annualized_return = self.final_return(interval = interval, form = 'annualized')
        self.accumulative_return = self.accum_return_properties(interval = interval, properties = 'Normal')
        self.maximum_drawdown = self.accum_return_properties(interval = interval, properties = 'max drawdown')
        return self.annualized_return / abs(self.maximum_drawdown)


    # 7. Stationary bootstrap
    def inference(self):
        self.data['Return_HO'] = self.data['Return'] - self.data['Return'].mean()
        def mean_return(series):
            return series.mean()
        self.block_len = int(len(self.data['Return_HO'].dropna())*0.1)
        self.bs = StationaryBootstrap(self.block_len, self.data['Return_HO'].dropna())
        self.results = self.bs.apply(mean_return, 100000)
        self.observed_mean = self.data['Return'].mean()

        self.p_value = np.sum(self.results >= self.observed_mean) / len(self.results)
        return self.p_value


    # 8. Reporting
    def portfolio_report(self):
        data = {
            ("Strategy class")   : self.data['Strategy class'].iloc[1],
            ("Parameterization") : self.data['Parameterization'].iloc[1],
            ("Time range")       : '%s - %s' %(self.data.index[0].date(), self.data.index[-1].date()),

            ("Avg. Daily return"): self.final_return(interval = 'None', form ='daily mean'),
            ("Avg. Annual")      : self.final_return(interval = 'None', form ='annualized'),
            ("P_value")          : self.p_value,

            ("Sharpe")           : self.sharpe_ratio(interval = 'None', form = 'daily').mean(),
            ("Ann. Sharpe")      : self.sharpe_ratio(interval = 'None', form = 'annualized').mean(),
            ("Sortino")          : self.sortino_ratio(interval = 'None', form = 'daily').mean(),
            ("Ann. Sortino")     : self.sortino_ratio(interval = 'None', form = 'annualized').mean(),
            ("Calmar")           : self.calmar_ratio(interval = 'None')
        }
        return data


# ----------------------------------------------------------------------------------------------------------------------


# E. IMPLEMENTATION
# Input: Section D
# Output: Execute the Section D code for 3200 different trading rules portfolio

# 1. Execution function
def load_port_BnH(ticker_group, date_list, year = 0):
    def generator0():
        for i in range(0, 8):
            load_port1 = BnH_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2])
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]
    def generator1():
        for i in range(0, 4):
            load_port1 = BnH_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2])
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    def generator2():
        for i in range(4, 8):
            load_port1 = BnH_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2])
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    if year == 0:
        data = pd.concat(generator0())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data

    elif year == 1:
        data = pd.concat(generator1())
        data['Adj Close'] = (1 + data['port_ret']).cumprod() * 1000
        return data
    elif year == 2:
        data = pd.concat(generator2())
        data['Adj Close'] = (1 + data['port_ret']).cumprod() * 1000
        return data

def load_port_MA(ticker_group, date_list, year, short_MA, long_MA, x_list, d_list):
    def generator0():
        for i in range(0, 8):
            load_port1 = Strategy_MA_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MA, long_MA, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]
    def generator1():
        for i in range(0, 4):
            load_port1 = Strategy_MA_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MA, long_MA, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    def generator2():
        for i in range(4, 8):
            load_port1 = Strategy_MA_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MA, long_MA, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    if year == 0:
        data = pd.concat(generator0())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 1:
        data = pd.concat(generator1())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 2:
        data = pd.concat(generator2())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data


def load_port_RnS(ticker_group, date_list, year, window, x_list, d_list):
    def generator0():
        for i in range(0, 8):
            load_port1 = Strategy_RnS_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]
    def generator1():
        for i in range(0, 4):
            load_port1 = Strategy_RnS_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    def generator2():
        for i in range(4, 8):
            load_port1 = Strategy_RnS_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    if year == 0:
        data = pd.concat(generator0())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 1:
        data = pd.concat(generator1())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 2:
        data = pd.concat(generator2())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data

def load_port_RSI(ticker_group, date_list, year, window, v_list, d_list):
    def generator0():
        for i in range(0, 8):
            load_port1 = Strategy_RSI_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, v_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]
    def generator1():
        for i in range(0, 4):
            load_port1 = Strategy_RSI_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, v_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    def generator2():
        for i in range(4, 8):
            load_port1 = Strategy_RSI_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], window, v_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    if year == 0:
        data = pd.concat(generator0())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data

    elif year == 1:
        data = pd.concat(generator1())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data

    elif year == 2:
        data = pd.concat(generator2())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data

def load_port_MACD(ticker_group, date_list, year, short_MACD, long_MACD, signal_MACD,  x_list, d_list):
    def generator0():
        for i in range(0, 8):
            load_port1 = Strategy_MACD_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MACD, long_MACD, signal_MACD, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]
    def generator1():
        for i in range(0, 4):
            load_port1 = Strategy_MACD_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MACD, long_MACD, signal_MACD, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    def generator2():
        for i in range(4, 8):
            load_port1 = Strategy_MACD_Portfolio(ticker_group[i], date_list[i], date_list[i + 2], date_list[i + 1], date_list[i + 2], short_MACD, long_MACD, signal_MACD, x_list, d_list)
            yield load_port1.data[['Strategy class', 'Parameterization', 'Adj Close', 'port_ret']]

    if year == 0:
        data = pd.concat(generator0())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 1:
        data = pd.concat(generator1())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data
    elif year == 2:
        data = pd.concat(generator2())
        data['Adj Close'] = (1 + data['port_ret']).cumprod()*1000
        return data


# 2. Execution settting
ticker_2017 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'XMR-USD']
ticker_2018 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD']
ticker_2019 = ['BTC-USD', 'XRP-USD', 'ETH-USD', 'BCH-USD', 'EOS-USD']
ticker_2020 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'BSV-USD']
ticker_2020_June = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'BSV-USD']
ticker_2021 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'LINK-USD']
ticker_2022 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
ticker_2023 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD']
ticker_group = [ticker_2017, ticker_2018, ticker_2019, ticker_2020, ticker_2020_June, ticker_2021, ticker_2022, ticker_2023]

year_2016 = '2016-01-01'
year_2017 = '2017-01-01'
year_2018 = '2018-01-01'
year_2019 = '2019-01-01'
year_2020 = '2020-01-01'
year_2020_June = '2020-07-01'
year_2021 = '2021-01-01'
year_2022 = '2022-01-01'
year_2023 = '2023-01-01'
year_2024 = '2024-04-30'
date_list = [year_2016, year_2017, year_2018, year_2019, year_2020, year_2020_June, year_2021, year_2022, year_2023, year_2024]

year = [0, 1, 2]



# 3. Execution
#     Buy and Hold PORTFOLIO
def BnH_portfolio_report(year):
    for y in year:
        port = load_port_BnH(ticker_group, date_list, y)
        performance = Portfolio_Performance(port, 0.06)
        yield performance.portfolio_report()

df_BnH_port = pd.DataFrame(BnH_portfolio_report(year))
display = tabulate(df_BnH_port, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)


    # Moving Average PORTFOLIO

long_MA = [2, 5, 10, 20, 25, 50, 100, 150, 200]
short_MA = [1, 2, 5, 10, 20, 25, 50, 100, 150]
x_MA = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 10]
d_MA = [1, 2, 3, 4, 5]

def MA_portfolio_report(year, short_MA, long_MA, x_list, d_list):
    for y in year:
        for s in short_MA:
            for l in long_MA:
                if l <= s:
                    continue
                else:
                    for x in x_list:
                        for d in d_list:
                            try:
                                port = load_port_MA(ticker_group, date_list, y, s, l, x, d)
                                performance = Portfolio_Performance(port, 0.06)
                                yield performance.portfolio_report()
                            except ZeroDivisionError:
                                print(f"Strategy year: {y}, short: {s}, long: {l}, x: {x}, d: {d} encountered a ZeroDivisionError. Skipping to the next strategy.")
                            except Exception as err:
                                print(
                                    f"Strategy year: {y}, short: {s}, long: {l}, x: {x}, d: {d} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

df_MA_port = pd.DataFrame(MA_portfolio_report(year, short_MA, long_MA, x_MA, d_MA))
display = tabulate(df_MA_port, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)

    # MACD PORTFOLIO

long_MACD = [2, 5, 10, 20, 25, 50, 100]
short_MACD = [1, 2, 5, 10, 20, 25, 50]
signal_MACD = [2, 5, 10, 20,25]
x_MACD = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
d_MACD = [1, 2, 3, 4, 5]

def MACD_portfolio_report(year, short_MACD, long_MACD, signal_MACD, x_list, d_list):
    for y in year:
        for s in short_MACD:
            for l in long_MACD:
                if l <= s:
                    continue
                else:
                    for si in signal_MACD:
                        if si >= s:
                            continue
                        else:
                            for x in x_list:
                                for d in d_list:
                                    try:
                                        port = load_port_MACD(ticker_group, date_list, y, s, l, si, x, d)
                                        performance = Portfolio_Performance(port, 0.06)
                                        yield performance.portfolio_report()
                                    except ZeroDivisionError:
                                        print(f"Strategy year: {y}, short: {s}, long: {l}, signal {si}, x: {x}, d: {d} encountered a ZeroDivisionError. Skipping to the next strategy.")
                                    except Exception as err:
                                        print(f"Strategy year: {y}, short: {s}, long: {l}, signal {si}, x: {x}, d: {d} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

df_MACD_port = pd.DataFrame(MACD_portfolio_report(year, short_MACD, long_MACD, signal_MACD, x_MACD, d_MACD))
display = tabulate(df_MACD_port, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)


    # RSI PORTFOLIO

window_RSI = [2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
v_RSI = [2, 5, 10, 15, 20, 25, 30, 40]
d_RSI = [1, 2, 3, 4, 5]

def RSI_portfolio_report(year, window, v_list, d_list):
    for y in year:
        for w in window:
            for v in v_list:
                for d in d_list:
                    try:
                        port = load_port_RSI(ticker_group, date_list, y, w, v, d)
                        performance = Portfolio_Performance(port, 0.06)
                        yield performance.portfolio_report()
                    except ZeroDivisionError:
                        print(f"Strategy year: {y}, window: {w}, v: {v}, d: {d} encountered a ZeroDivisionError. Skipping to the next strategy.")
                    except Exception as err:
                        print(f"Strategy ticker: {y}, window: {w}, v: {v}, d: {d} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

df_RSI_port = pd.DataFrame(RSI_portfolio_report(year, window_RSI, v_RSI, d_RSI))
display = tabulate(df_RSI_port, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)


    # RnS PORTFOLIO

window_RnS = [2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
x_RnS = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 10]
d_RnS = [1, 2, 3, 4, 5]

def RnS_portfolio_report(year, window, x_list, d_list):
    for y in year:
        for w in window:
            for x in x_list:
                for d in d_list:
                    try:
                        port = load_port_RnS(ticker_group, date_list, y, w, x, d)
                        performance = Portfolio_Performance(port, 0.06)
                        yield performance.portfolio_report()
                    except ZeroDivisionError:
                        print(f"Strategy year: {y}, window: {w}, x: {x}, d: {d} encountered a ZeroDivisionError. Skipping to the next strategy.")
                    except Exception as err:
                        print(f"Strategy ticker: {y}, window: {w}, x: {x}, d: {d} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

df_RnS_port = pd.DataFrame(RnS_portfolio_report(year, window_RnS, x_RnS, d_RnS))
display = tabulate(df_RnS_port, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)


# ***********************************************************************************************************************


# CODE PART 2: GENERATE PORTFOLIOS AND REPORT THEIR PERFORMANCE
