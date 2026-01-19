
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
    
    # @staticmethod
    # def base_002(df, window_fast, window_slow, window_rank):
    #     df['acc_ofp'] = df["based_col"].cumsum()
    #     df['ema_fast'] = df.groupby('day')['acc_ofp'].transform(lambda x: x.ewm(span=window_fast).mean())
    #     df['ema_slow'] = df.groupby('day')['acc_ofp'].transform(lambda x: x.ewm(span=window_slow).mean())
    #     raw_sig = df['ema_fast'] - df['ema_slow']
    #     rank_pct = raw_sig.rolling(window_rank).rank(pct=True)

    #     signal = 2 * rank_pct - 1

    #     return signal

    @staticmethod
    def base_003(df, window, window_rank):
        busd_min = df['based_col'].rolling(window=window).min()
        busd_max = df['based_col'].rolling(window=window_rank).max()
        range_busd = busd_max - busd_min

        ma_busd = df['based_col'].rolling(window=window).mean()
        
        spring_force = (df['based_col'] - ma_busd) / (range_busd + 1e-6)
        signal = 2 * spring_force.rolling(window=window_rank).rank(pct=True) - 1

        return signal
    
    @staticmethod
    def base_004(df, window, factor):
        raw_signal = O.ts_rank(O.ts_rank(df['based_col'], window=window), window=window)
        signal = (raw_signal / window) * 2 - 1
        if factor > 0: 
            signal = signal.ewm(halflife=factor).mean()
            
        return signal
    
    @staticmethod
    def base_005(df, window, factor, window_rank):
        roll_busd = df['based_col'].rolling(window)
        avg_flow, std_flow = roll_busd.mean(), roll_busd.std()
        flow_min, flow_max = roll_busd.min(), roll_busd.max()
    
        spring_force = (df['based_col'] - avg_flow) / (flow_max - flow_min + 1e-6)
        flow_intensity = (df['based_col'] - avg_flow) / (std_flow + 1e-6)
        is_overloaded = flow_intensity.abs() > factor
        trade_direction = np.where(is_overloaded, -1, 1)
        
        raw_signal = spring_force * trade_direction
        signal = 2 * raw_signal.rolling(window_rank).rank(pct=True) - 1

        return signal
    
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





