import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
from arch.bootstrap import StationaryBootstrap
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


# A. DATA LOADING
#       Input: Ticker symbol, download date range
#       Output: Dataframe downloaded

# A. DATA LOADING
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


# B. STRATEGY FORMULATION

#   1. Moving Average
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

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0],
                                        [self.data['ret'], -self.data['ret']], default=np.nan)


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

        self.data['strategy_ret'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0],
                                        [self.data['ret'], -self.data['ret']], default=np.nan)


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

# ----------------------------------------------------------------------------------------------------------------------


# C. TICKER PERFORMANCE
class Ticker_Performance:
    def __init__(self, load_ticker, start_date_testing, end_date_testing, risk_free_rate):
        self.data = load_ticker.data
        self.ticker_symbol = load_ticker.ticker_symbol
        self.start_date_testing = pd.to_datetime(start_date_testing)
        self.end_date_testing = pd.to_datetime(end_date_testing)
        self.daily_mean_risk_free_rate = (1+risk_free_rate) ** (1/365) - 1

        self.filter_data()
        self.sharpe_ratio()
        self.sortino_ratio()
        self.calmar_ratio()
        self.normality_test()
        self.inference()
        self.ticker_report()

    def filter_data(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.data['Return_dataset'] = self.data['Adj Close'].pct_change()
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()
        self.data['Return'] = self.data['Return_dataset']


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
            self.result = (1 + return_series.loc[self.start_date_testing:self.end_date_testing]).product() - 1
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


#   3. Standard Deviation
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


    #   6. Calmar ratio
    def calmar_ratio(self, **option):
        interval = option.get('interval', 'None')

        self.annualized_return = self.final_return(interval = interval, form = 'annualized')
        self.accumulative_return = self.accum_return_properties(interval = interval, properties = 'Normal')
        self.maximum_drawdown = self.accum_return_properties(interval = interval, properties = 'max drawdown')
        return self.annualized_return / abs(self.maximum_drawdown)


    #   7. Time series test
    def normality_test(self):
        self.stat, self.p_value = stats.shapiro(self.data['Return'].dropna())
        return self.p_value


    #   8. Significant mean return inference
    def inference(self):
        self.data['Return_HO'] = self.data['Return'] - self.data['Return'].mean()

        def mean_return(series):
            return series.mean()

        self.block_len = int(len(self.data['Return_HO'].dropna()) * 0.1)
        self.bs = StationaryBootstrap(self.block_len, self.data['Return_HO'].dropna())
        self.results = self.bs.apply(mean_return, 100000)
        self.observed_mean = self.data['Return'].mean()

        self.p_value = np.sum(self.results >= self.observed_mean) / len(self.results)
        return self.p_value


    # 9. Reporting
    def ticker_report(self):
        table_name = ['ID', 'Descriptive statistic', 'Risk and Return']
        data = {
            ("Cryptocurrency"): self.ticker_symbol,
            ("Strategy")     : "Buy and Hold",
            ("Time range")   : '%s - %s' %(self.data.index[0].date(), self.data.index[-1].date()),
            ("Obs"): self.data['Return'].count(),

            ("Mean")         : self.data['Return'].mean(),
            ("Std.Dev")      : self.std_dev(interval = 'None', side = 'all side'),
            ("Max")          : self.data['Return'].max(),
            ("Min")          : self.data['Return'].min(),
            ("Skewness")     : self.data['Return'].skew(),
            ("Kurtosis")     : self.data['Return'].kurtosis(),
            ("Shapiro p_value"):self.normality_test(),

            ("Avg. Daily return"): self.final_return(interval = 'None', form ='daily mean'),
            ("Avg. Annual")      : self.final_return(interval = 'None', form ='annualized'),
            ("P_value"): self.p_value,
            ("Daily mean return"): self.data['Return'].mean(),

            ("Sharpe")           : self.sharpe_ratio(interval = 'None', form = 'daily').mean(),
            ("Ann. Sharpe")      : self.sharpe_ratio(interval = 'None', form = 'annualized').mean(),
            ("Sortino")          : self.sortino_ratio(interval = 'None', form = 'daily').mean(),
            ("Ann. Sortino")     : self.sortino_ratio(interval = 'None', form = 'annualized').mean(),
            ("Calmar")           : self.calmar_ratio(interval = 'None')
        }
        return data


# ----------------------------------------------------------------------------------------------------------------------


# D. STRATEGY PERFORMANCE
class Strategy_Performance(Ticker_Performance):
    def __init__(self, load_strategy, load_ticker, start_date_testing, end_date_testing, risk_free_rate):
        super().__init__(load_ticker, start_date_testing, end_date_testing, risk_free_rate)
        self.data = load_strategy.data
        self.name = load_strategy.name
        self.parameterization = load_strategy.parameterization
        self.ticker_symbol = load_strategy.ticker_symbol

        self.filter_data()
        self.trade_accum_return()
        self.no_trade_performance()
        self.avg_return_trade()
        self.strategy_report()


    def filter_data(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.ticker_return = self.data['Adj Close'].pct_change()
        self.data = self.data.loc[self.start_date_testing:self.end_date_testing].copy()

        self.data['Ticker_return'] = self.ticker_return
        self.data['Return'] = np.select([self.data['Signal_lagged'] > 0, self.data['Signal_lagged'] < 0],[self.data['Ticker_return'], -self.data['Ticker_return']], default=np.nan)
        self.data['Long_return'] = self.data['Return'].loc[self.data['Signal_lagged'] > 0]
        self.data['Short_return'] = self.data['Return'].loc[self.data['Signal_lagged'] < 0]

        self.data['Trade'] = np.select([(self.data['Signal_lagged'] != self.data['Signal_lagged'].shift(1)) & (~self.data['Signal_lagged'].isna())], [1], default = np.nan)
        self.data['Long'] = self.data['Trade'].loc[self.data['Signal_lagged'] > 0]
        self.data['Short'] = self.data['Trade'].loc[self.data['Signal_lagged'] < 0]


#   1. Accum return, Final return, Risk-adjusted return already inherited from Ticker_Performance class

#   2. Trade analysis
    def trade_accum_return(self):                   # Accumulative return of every single trade
        self.data['Trade_accum_return'] = np.nan
        self.accum = 1
        self.trade_id = 0
        for i, row in self.data.iterrows():
            if row['Trade'] == 1:
                self.accum = 1
                self.trade_id += 1
            if not np.isnan(row['Return']):
                self.accum *= (1+row['Return'])
                self.data.at[i, 'Trade_accum_return'] = self.accum - 1
            else:
                self.data.at[i, 'Trade_accum_return'] = np.nan

    def no_trade_performance(self, trade_series = None, **options):     # Count number of trade / long / short
        if trade_series is None:
            trade_series = self.data['Trade']     # other options: self.data['Long'], self.data['Short']

        interval = options.get('interval', 'None')  # other options: 'YE', 'ME',...
        perf = options.get('perf', 'all')           # other options: 'win', 'loss'

        self.trade_result = self.data['Trade_accum_return'].loc[trade_series.shift(-1) == 1]
        if interval == 'None':
            if perf == 'all':
                return trade_series.sum()
            if perf == 'win':
                return (self.trade_result > 0).sum()
            if perf == 'loss':
                return (self.trade_result < 0).sum()

        else:
            if perf == 'all':
                return trade_series.resample(interval).sum()
            if perf == 'win':
                return self.trade_result.resample(interval).apply(lambda x: (x > 0).sum())
            if perf == 'loss':
                return self.trade_result.resample(interval).apply(lambda x: (x < 0).sum())

    def avg_return_trade(self, return_series = None, trade_series = None, **options):
        if return_series is None:
            return_series = self.data['Return']            # other options: self.data['Long_return'], self.data['Short_return']
            trade_series = self.data['Trade']              # other options: self.data['Long'], self.data['Short']

        interval = options.get('interval', 'None')                     # other options: 'YE', 'ME',...

        return self.final_return(return_series, interval = interval) / self.no_trade_performance(trade_series, interval = interval)

    def strategy_report(self):
        table_name = ['ID', 'Trade analysis', 'Risk-adjusted return']
        data = {
            ("Cryptocurrency")           : self.ticker_symbol,
            ("Time range")               : '%s - %s' % (self.data.index[0].date(), self.data.index[-1].date()),
            ("Strategy class")           : self.name,
            ("Parameterization")         : self.parameterization,

            ("No. Trade")                : self.no_trade_performance(self.data['Trade'], perf='all'),
            ("No. Long")                 : self.no_trade_performance(self.data['Long'], perf='all'),
            ("No. Short")                : self.no_trade_performance(self.data['Short'], perf='all'),
            ("Avg. Trade return")        : self.avg_return_trade(self.data['Return'], self.data['Trade']),
            ("Avg. Long return")         : self.avg_return_trade(self.data['Long_return'], self.data['Long']),
            ("Avg. Short return")        : self.avg_return_trade(self.data['Short_return'], self.data['Short']),
            ("Avg. Daily return")        : self.final_return(self.data['Return'], form = 'daily mean'),
            ("Avg. Annual")              : self.final_return(self.data['Return'], form = 'annualized'),
            ("P_value")                  : self.p_value,
            ("Daily mean return")        : self.data['Return'].mean(),

            ("Sharpe")                   : (self.sharpe_ratio(interval = 'None', form = 'daily')).mean(),
            ("Ann. Sharpe")              : (self.sharpe_ratio(interval = 'None', form = 'annualized')).mean(),
            ("Sortino")                  : (self.sortino_ratio(interval = 'None', form = 'daily')).mean(),
            ("Ann. Sortino")             : (self.sortino_ratio(interval = 'None', form = 'annualized')).mean(),
            ("Calmar")                   : self.calmar_ratio(interval = 'None')
        }
        return data


# ----------------------------------------------------------------------------------------------------------------------


# E. REPORTING SETTING
pd.set_option('display.max_columns', None)

        # 1. Ticker Buy and Hold performance
def ticker_report(ticker_list, start_date, end_date):
    for i in range(0, 8):
        for j in range(0, 5):
            load_ticker = LoadTicker(ticker_list[i][j], '2016-01-01', '2024-04-30')
            performance = Ticker_Performance(load_ticker, start_date[i], end_date[i], 0.06)
            yield performance.ticker_report()

        # 2. Moving Average Strategy performance
def strategy_MA_report(ticker_list, start_date, end_date, long, short, x, d):
    for i in range(0,8):
        for j in range(0,5):
            load_ticker = LoadTicker(ticker_list[i][j],'2016-01-01', '2024-04-30')
            for a in long:
                for b in short:
                    if b >= a:
                        continue
                    else:
                        for c in x:
                            for e in d:
                                try:
                                    load_strategy = LoadStrategy_MA(load_ticker, b, a, c, e)
                                    performance = Strategy_Performance(load_strategy, load_ticker, start_date[i], end_date[i], 0.06)
                                    yield performance.strategy_report()
                                except ZeroDivisionError:
                                    print(f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, long: {a}, short: {b}, x: {c}, d: {e} encountered a ZeroDivisionError. Skipping to the next strategy.")
                                except Exception as err:
                                    print(f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, long: {a}, short: {b}, x: {c}, d: {e} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

        # 3. Resistance and Support Strategy performance
def strategy_RnS_report(ticker_list, start_date, end_date, window, x, d):
    for i in range(0,8):
        for j in range(0,5):
            load_ticker = LoadTicker(ticker_list[i][j],'2016-01-01', '2024-04-30')
            for a in window:
                for c in x:
                    for e in d:
                        try:
                            load_strategy = LoadStrategy_RnS(load_ticker, a, c, e)
                            performance = Strategy_Performance(load_strategy, load_ticker, start_date[i], end_date[i], 0.06)
                            yield performance.strategy_report()
                        except ZeroDivisionError:
                            print(f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, window: {a}, x: {c}, d: {e} encountered a ZeroDivisionError. Skipping to the next strategy.")
                        except Exception as err:
                            print(
                                f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, window: {a}, x: {c}, d: {e} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")

        # 4. RSI strategy performance
def strategy_RSI_report(ticker_list, start_date, end_date, window, v, d):
    for i in range(0,8):
        for j in range(0,5):
            load_ticker = LoadTicker(ticker_list[i][j],'2016-01-01', '2024-04-30')
            for a in window:
                for c in v:
                    for e in d:
                        try:
                            load_strategy = LoadStrategy_RSI(load_ticker, a, c, e)
                            performance = Strategy_Performance(load_strategy, load_ticker, start_date[i], end_date[i], 0.06)
                            yield performance.strategy_report()
                        except ZeroDivisionError:
                            print(f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, window: {a}, v: {c}, d: {e} encountered a ZeroDivisionError. Skipping to the next strategy.")
                        except Exception as err:
                            print(f"Strategy ticker: {ticker_list[i][j]}, start date {start_date[i]}, window: {a}, v: {c}, d: {e} encountered an error: {err}. Skipping to the next strategy.")
    print("All strategies have been tested.")



# ----------------------------------------------------------------------------------------------------------------------


# F. IMPLEMENTATION

# 0. Ticker performance and general setting
ticker_2017 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'XMR-USD']
ticker_2018 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD']
ticker_2019 = ['BTC-USD', 'XRP-USD', 'ETH-USD', 'BCH-USD', 'EOS-USD']
ticker_2020 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'BSV-USD']
ticker_2020_June = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'BSV-USD']
ticker_2021 = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'LINK-USD']
ticker_2022 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
ticker_2023 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD']
ticker = [ticker_2017, ticker_2018, ticker_2019, ticker_2020, ticker_2020_June, ticker_2021, ticker_2022, ticker_2023]

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
start_date = [year_2017, year_2018, year_2019, year_2020, year_2020_June, year_2021, year_2022, year_2023]
end_date = ['2017-12-31', '2018-12-31', '2019-12-31', '2020-06-30', '2020-12-31', '2021-12-31', '2022-12-31', '2024-04-29']

x = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
d = [1, 2 , 3, 4, 5]


df_ticker = pd.DataFrame(ticker_report(ticker, start_date, end_date))
display = tabulate(df_ticker, headers = 'keys', tablefmt='pretty', showindex=True)
print(display)
print(df_ticker.iloc[:,3:].sum(axis=1))


# 1. Moving Average Strategy setting
long_MA = [2, 5, 10, 20, 25, 50, 100, 150, 200]
short_MA = [1, 2, 5, 10, 20, 25, 50, 100, 150]

df_MA = pd.DataFrame(strategy_MA_report(ticker, start_date, end_date, long_MA, short_MA, x, d))
display_MA = tabulate(df_MA, headers = 'keys', tablefmt='pretty', showindex=True)
print(display_MA)
check_MA = pd.DataFrame(df_MA.iloc[:,4:].sum(axis=1))


# 2. Resistance and Support Strategy setting
windows = [2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
df_RnB = pd.DataFrame(strategy_RnS_report(ticker, start_date, end_date, windows, x, d))
display_RnB = tabulate(df_RnB, headers = 'keys', tablefmt='pretty', showindex=True)
print(display_RnB)
check_RnB = pd.DataFrame(df_RnB.iloc[:,4:].sum(axis=1))

# 3. RSI strategy setting
window_RSI = [2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
v = [2, 5, 10, 15, 20, 25, 30, 40]
df_RSI = pd.DataFrame(strategy_RSI_report(ticker, start_date, end_date, window_RSI, v, d))
display_RSI = tabulate(df_RSI, headers = 'keys', tablefmt='pretty', showindex=True)
print(display_RSI)
check_RSI = pd.DataFrame(df_RSI.iloc[:,4:].sum(axis=1))


