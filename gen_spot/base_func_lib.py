
from time import time
import numpy as np
import pandas as pd
class O:
    # @staticmethod
    # def ts_weighted_mean(df, window=10):
    #     # noinspection PyUnresolvedReferences
    #     from talib import WMA

    #     wma = WMA(df,timeperiod=window)
    #     return wma
    @staticmethod
    def calculate_rsi(series: pd.Series, d: int = 14) -> pd.Series:
        """Tính chỉ số sức mạnh tương đối (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(d).mean() # Dùng SMA/rolling(d).mean() cho RSI ban đầu
        avg_loss = loss.rolling(d).mean()

        # Để tính RSI chính xác, cần dùng EMA (wilder's method) sau d ngày đầu.
        # Tuy nhiên, ta dùng EWM cho tính toán đơn giản và tốc độ.
        avg_gain = gain.ewm(span=d, adjust=False).mean()
        avg_loss = loss.ewm(span=d, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def ts_rank_normalized(df: pd.DataFrame, window=10):
        df2 = df.rolling(window).rank()
        
        return (df2-1) / (window-1)
    
    @staticmethod
    def ts_weighted_mean(df, window=10):
        weights = np.arange(1, window + 1)

        def weighted_ma(x):
            return np.dot(x, weights) / weights.sum()

        return df.rolling(window).apply(weighted_ma, raw=True)

    @staticmethod
    def decay_linear(df, d):
        """
        weighted moving average over the past d days with linearly decaying
        weights d, d – 1, …, 1 (rescaled to sum up to 1)
        :param df: data frame containing prices
        :param d: number of days to look back (rolling window)
        :return: Pandas series
        """

        return df.ewm(d).mean()

    @staticmethod
    def rank(df):
        """Return the cross-sectional percentile rank

         Args:
             :param df: tickers in columns, sorted dates in rows.

         Returns:
             pd.DataFrame: the ranked values
         """
        return df.rank(axis=1, pct=True)

    # noinspection PyIncorrectDocstring,PyUnresolvedReferences
    @staticmethod
    def scale(df):
        """
        Scaling time serie.
        :param df: a pandas DataFrame.
        :param k: scaling factor.
        :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
        """
        return df.div(df.abs().sum(axis=1), axis=0)

    @staticmethod
    def log(df):
        return np.log1p(df)

    @staticmethod
    def sign(df):
        return np.sign(df)

    @staticmethod
    def power(df, exp):
        return df.pow(exp)

    @staticmethod
    def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """Return the lagged values t periods ago.

        Args:
            :param df: tickers in columns, sorted dates in rows.
            :param t: lag

        Returns:
            pd.DataFrame: the lagged values
        """
        return df.shift(t)

    @staticmethod
    def ts_delta(df, period=1):
        """
        Wrapper function to estimate difference.
        :param df: a pandas DataFrame.
        :param period: the difference grade.
        :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
        """
        return df.diff(period)

    @staticmethod
    def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Computes the rolling ts_sum for the given window size.

        Args:
            df (pd.DataFrame): tickers in columns, dates in rows.
            window      (int): size of rolling window.

        Returns:
            pd.DataFrame: the ts_sum over the last 'window' days.
        """
        return df.rolling(window).sum()

    @staticmethod
    def ts_mean(df, window=10):
        """Computes the rolling mean for the given window size.

        Args:
            df (pd.DataFrame): tickers in columns, dates in rows.
            window      (int): size of rolling window.

        Returns:
            pd.DataFrame: the mean over the last 'window' days.
        """
        return df.rolling(window).mean()

    @staticmethod
    def ts_std(df, window=10):
        """
        Wrapper function to estimate rolling standard deviation.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series min over the past 'window' days.
        """
        return (df
                .rolling(window)
                .std())
    
    @staticmethod
    def ts_median(df, window = 10):
        """
        Rolling median theo time-series
        """
        return df.rolling(
            window=window,
            min_periods=window
        ).median()

    @staticmethod
    def ts_rank(df, window=10):
        """
        Wrapper function to estimate rolling rank.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series rank over the past window days.
        """
        return df.rolling(window).rank()

    @staticmethod
    def ts_product(df, window=10):
        """
        Wrapper function to estimate rolling ts_product.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series ts_product over the past 'window' days.
        """
        return (df
                .rolling(window)
                .apply(np.prod))

    @staticmethod
    def ts_min(df, window=10):
        """
        Wrapper function to estimate rolling min.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series min over the past 'window' days.
        """
        return df.rolling(window).min()

    @staticmethod
    def ts_max(df, window=10):
        """
        Wrapper function to estimate rolling min.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series max over the past 'window' days.
        """
        return df.rolling(window).max()

    @staticmethod
    def ts_argmax(df, window=10):
        """
        Wrapper function to estimate which day ts_max(df, window) occurred on
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: well.. that :)
        """
        return df.rolling(window).apply(np.argmax).add(1)

    @staticmethod
    def ts_argmin(df, window=10):
        """
        Wrapper function to estimate which day ts_min(df, window) occurred on
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: well.. that :)
        """
        return (df.rolling(window)
                .apply(np.argmin)
                .add(1))

    # noinspection PyIncorrectDocstring
    @staticmethod
    def ts_corr(x, y, window=10):
        """
        Wrapper function to estimate rolling correlations.
        :param x, y: pandas DataFrames.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series min over the past 'window' days.
        """
        return x.rolling(window).corr(y)

    # noinspection PyIncorrectDocstring,PyUnresolvedReferences
    @staticmethod
    def ts_cov(x, y, window=10):
        """
        Wrapper function to estimate rolling covariance.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series min over the past 'window' days.
        """
        return x.rolling(window).cov(y)

    @staticmethod
    def zscore(x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x-m)/s
        return z

    @staticmethod
    def compute_vwap(df, window=200):
        df['average_price'] = (df['low'] + df['close'] + df['open'] + df['high'])/4
        df['vwap'] = \
        (
            df['average_price']
            *
            df['matchingVolume']
        ).rolling(window).sum() \
        /  \
        df['matchingVolume'].rolling(window).sum()
        return df


    @staticmethod
    def linear_weighted_moving_average(series, window_size):
        weights = np.arange(1, window_size+1)
        weights_sum = np.sum(weights)
        conv_weights = np.flip(weights) / weights_sum
        conv_series = np.convolve(series, conv_weights, mode='valid')
        return pd.Series(conv_series, index=series.iloc[window_size-1:].index)


class Base:
    @staticmethod
    def base_001(df, window=20, factor = 0):

        # Tính toán Đường trung bình trượt (Moving Average - MA)
        ma = df["based_col"].rolling(window=window).mean()
        
        # Tính toán Độ lệch chuẩn trượt (Standard Deviation - StdDev)
        stddev  = df["based_col"].rolling(window=window).std()
        
        # Tính toán Z-Score: (OFP - MA) / StdDev
        z_score = (df["based_col"] - ma) / stddev.replace(0, np.nan)
        
        if factor > 0:
            z_score = z_score.ewm(halflife=factor).mean()
            
        signal = z_score.rolling(window).apply(
            lambda x: (x.argsort().argsort()[-1] + 1) / len(x),
            raw=True
        )
        
        signal = 2 * signal - 1
        
        return signal
    
    @staticmethod
    def base_002(df, window_fast, window_slow, window_rank):
        df['acc_ofp'] = df["based_col"].cumsum()
        df['ema_fast'] = df.groupby('day')['acc_ofp'].transform(lambda x: x.ewm(span=window_fast).mean())
        df['ema_slow'] = df.groupby('day')['acc_ofp'].transform(lambda x: x.ewm(span=window_slow).mean())
        raw_sig = df['ema_fast'] - df['ema_slow']
        rank_pct = raw_sig.rolling(window_rank).rank(pct=True)

        signal = 2 * rank_pct - 1

        return signal
    
    @staticmethod
    def base_003(df, window):
        close_diff = df["based_col"].diff(1)
        signal = close_diff
        flt_min = 0 >= close_diff.rolling(window).min()
        flt_max = close_diff.rolling(window).max() < 0
        signal.loc[flt_min & flt_max] = close_diff
        signal.loc[flt_min & (~flt_max)] = -close_diff

        lower, upper = signal.quantile(0.05), signal.quantile(0.95)
        normalized_signal = signal / (upper - lower)
        signal = -normalized_signal

        return signal
     
    @staticmethod
    def alpha_009( df, window):
        close_diff = df['based_col'].diff(1)
        signal = close_diff
        flt_min = 0 >= close_diff.rolling(window).min()
        flt_max = close_diff.rolling(window).max() < 0
        signal.loc[flt_min & flt_max] = close_diff
        signal.loc[flt_min & (~flt_max)] = -close_diff

        lower, upper = signal.quantile(0.05), signal.quantile(0.95)
        normalized_signal = signal / (upper - lower)
        signal = -normalized_signal

        return signal
    
    @staticmethod
    def alpha_new_018(df,window,factor):
        ofp = df['based_col']

        rolling_mean = O.ts_mean(ofp, window)
        rolling_std = O.ts_std(ofp, window)
        normalized_vol = 4.0 * rolling_std / (rolling_mean.abs() + 1e-9)
        vol_expanding = O.ts_delta(normalized_vol, factor) > 0
        mean_reversion = -O.ts_delta(ofp, factor)
        conditional_signal = mean_reversion.where(vol_expanding, 0)
        ts_rank = O.ts_rank_normalized(conditional_signal, window=window)
        signal = -(2 * ts_rank - 1)
        return signal
    
    @staticmethod
    def alpha_049(df):
        delay = lambda df, window: df.shift(window)
        based_col = df['based_col']
        flt = ((((delay(based_col, 20) - delay(based_col, 10)) / 10) - ((delay(based_col, 10) - based_col) / 10)) < (-1 * 0.1))
        signal = pd.DataFrame(
            {
                'signal': np.ones_like(based_col)
            },
            index=based_col.index)
        based_col = based_col.loc[~flt]
        signal.loc[~flt, 'signal'] = ((-1 * 1) * (based_col - delay(based_col, 1)))
        signal = signal / 30
        
        return - signal
    
    def alpha_004(df, window):

        raw_signal = O.ts_rank(O.ts_rank(df['based_col'], window=window), window=window)
        signal = (raw_signal / window) * 2 - 1
        return signal
    
    def alpha_010(df, window, factor):

        close_diff = df['based_col'].diff(1)
        signal = close_diff.copy() 

        flt_min = 0 >= close_diff.rolling(window = window).min()
        flt_max = close_diff.rolling(window = window).max() >= 0

        signal.loc[flt_min & flt_max] = -close_diff
        signal = (O.ts_rank(signal, window=factor) / (factor / 2)) - 1

        return -signal
    
class Domains:

    @staticmethod
    def get_list_of_bases(verbosity=1):
        dic_bases = {}

        # duyệt tất cả attribute của Base
        for attr_name in dir(Base):
            if not attr_name.startswith("base_"):
                continue

            attr = getattr(Base, attr_name)

            if callable(attr):
                dic_bases[attr_name] = attr

        return dic_bases





