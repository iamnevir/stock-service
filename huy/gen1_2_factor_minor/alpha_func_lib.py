from blinker import signal
import pandas as pd
import json
# from redis import StrictRedis
from datetime import datetime as dt
from datetime import timedelta
from time import time
import numpy as np
import pickle
from matplotlib import pyplot as plt
import psutil
import datetime

class RESOURCES:
    THE_DIC_FREQS_FN = '/home/ubuntu/nevir/gen/alpha.pkl'
    DIR = '/home/ubuntu/cache/df_ohlc_ps_1m/'
    FN_DIC_F1 = f'{DIR}/dic_df_f1.pickle'
    PLOT_DIR = '/tmp/alpha_func_lib2_plots'
    FAKE_REDIS_DB = 1
    SCAN_SHARPE_DIR = '/d/data/alpha_scans'


    @staticmethod
    def compute_f1_redis_key(day):
        return f"realtime_{day}_F1"

    @staticmethod
    def get_fn_dic_df1_resample2():
        return RESOURCES.FN_DIC_F1 + '2'


class TIME_RESOURCES:
    PUBSUB_KEY = 'redis.pubsub.redis_global_time'
    FAKE_REDIS_DB = 1

"""CONSTANTS"""
class C:
    TIMES = [
       '09:00:00', '09:01:00', '09:02:00', '09:03:00', '09:04:00',
       '09:05:00', '09:06:00', '09:07:00', '09:08:00', '09:09:00',
       '09:10:00', '09:11:00', '09:12:00', '09:13:00', '09:14:00',
       '09:15:00', '09:16:00', '09:17:00', '09:18:00', '09:19:00',
       '09:20:00', '09:21:00', '09:22:00', '09:23:00', '09:24:00',
       '09:25:00', '09:26:00', '09:27:00', '09:28:00', '09:29:00',
       '09:30:00', '09:31:00', '09:32:00', '09:33:00', '09:34:00',
       '09:35:00', '09:36:00', '09:37:00', '09:38:00', '09:39:00',
       '09:40:00', '09:41:00', '09:42:00', '09:43:00', '09:44:00',
       '09:45:00', '09:46:00', '09:47:00', '09:48:00', '09:49:00',
       '09:50:00', '09:51:00', '09:52:00', '09:53:00', '09:54:00',
       '09:55:00', '09:56:00', '09:57:00', '09:58:00', '09:59:00',
       '10:00:00', '10:01:00', '10:02:00', '10:03:00', '10:04:00',
       '10:05:00', '10:06:00', '10:07:00', '10:08:00', '10:09:00',
       '10:10:00', '10:11:00', '10:12:00', '10:13:00', '10:14:00',
       '10:15:00', '10:16:00', '10:17:00', '10:18:00', '10:19:00',
       '10:20:00', '10:21:00', '10:22:00', '10:23:00', '10:24:00',
       '10:25:00', '10:26:00', '10:27:00', '10:28:00', '10:29:00',
       '10:30:00', '10:31:00', '10:32:00', '10:33:00', '10:34:00',
       '10:35:00', '10:36:00', '10:37:00', '10:38:00', '10:39:00',
       '10:40:00', '10:41:00', '10:42:00', '10:43:00', '10:44:00',
       '10:45:00', '10:46:00', '10:47:00', '10:48:00', '10:49:00',
       '10:50:00', '10:51:00', '10:52:00', '10:53:00', '10:54:00',
       '10:55:00', '10:56:00', '10:57:00', '10:58:00', '10:59:00',
       '11:00:00', '11:01:00', '11:02:00', '11:03:00', '11:04:00',
       '11:05:00', '11:06:00', '11:07:00', '11:08:00', '11:09:00',
       '11:10:00', '11:11:00', '11:12:00', '11:13:00', '11:14:00',
       '11:15:00', '11:16:00', '11:17:00', '11:18:00', '11:19:00',
       '11:20:00', '11:21:00', '11:22:00', '11:23:00', '11:24:00',
       '11:25:00', '11:26:00', '11:27:00', '11:28:00', '11:29:00',
       '11:30:00', '13:00:00', '13:01:00', '13:02:00', '13:03:00',
       '13:04:00', '13:05:00', '13:06:00', '13:07:00', '13:08:00',
       '13:09:00', '13:10:00', '13:11:00', '13:12:00', '13:13:00',
       '13:14:00', '13:15:00', '13:16:00', '13:17:00', '13:18:00',
       '13:19:00', '13:20:00', '13:21:00', '13:22:00', '13:23:00',
       '13:24:00', '13:25:00', '13:26:00', '13:27:00', '13:28:00',
       '13:29:00', '13:30:00', '13:31:00', '13:32:00', '13:33:00',
       '13:34:00', '13:35:00', '13:36:00', '13:37:00', '13:38:00',
       '13:39:00', '13:40:00', '13:41:00', '13:42:00', '13:43:00',
       '13:44:00', '13:45:00', '13:46:00', '13:47:00', '13:48:00',
       '13:49:00', '13:50:00', '13:51:00', '13:52:00', '13:53:00',
       '13:54:00', '13:55:00', '13:56:00', '13:57:00', '13:58:00',
       '13:59:00', '14:00:00', '14:01:00', '14:02:00', '14:03:00',
       '14:04:00', '14:05:00', '14:06:00', '14:07:00', '14:08:00',
       '14:09:00', '14:10:00', '14:11:00', '14:12:00', '14:13:00',
       '14:14:00', '14:15:00', '14:16:00', '14:17:00', '14:18:00',
       '14:19:00', '14:20:00', '14:21:00', '14:22:00', '14:23:00',
       '14:24:00', '14:25:00', '14:26:00', '14:27:00', '14:28:00',
       '14:29:00', '14:30:00', '14:45:00']
    NUM_TRADING_DAYS_PER_YEAR = 250
    FIG_HEIGHT = 7
    FIG_WIDTH = 15
    DEFAULT_FEE = 0.3
    DEFAULT_FREQUENCIES = range(8, 65)
    DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0, 75, 0.8, 0.85)
    DEFAULT_WINDOWS = [np.nan]
    DEFAULT_ATC_POSITIONS = [0]
    DEFAULT_MEDIAN_VALUES = [np.nan]
    DEFAULT_SIZING_FUNCS = [np.sign]
    DEFAULT_OTHERS = {}
    BAR = '*' * 50

    FLAT_ATC = 'unconditionalATC'
    PRE_ATC = 'preATC'
    LUNCH = 'lunch'
    QUICK_FIX_COMPUTE_RETURNS = 'bfill'

    EXCLUDED_ALPHAS = ['001', '041', '042']
    COLS1 = ['day', 'signal', 'position', 'action', 'cumVolume',
             'profit', 'pctChange', 'pnl', 'ppc']
    COLS2 = COLS1 + ['executable', 'executionTime']
    DUMMY_VALUE = -99

    PRINT_FUNC = print


class Fake_Dt:
    @staticmethod
    def fromtimestamp(*args, **kwargs):
        from datetime import datetime as real_dt
        return real_dt.fromtimestamp(*args, **kwargs)

    @staticmethod
    def now():
        # noinspection PyUnresolvedReferences
        from redis_time import REDIS_GLOBAL_TIME
        return REDIS_GLOBAL_TIME.GET_CURRENT_TIME()

    @staticmethod
    def strptime(*args, **kwargs):
        from datetime import datetime as dt
        return dt.strptime(*args, **kwargs)


"""UTILITIES"""
class U:
    TIK_TOK = {}
    glob_threads = []

    @staticmethod
    def get_original_dt():
        from datetime import datetime as dt
        return dt

    @staticmethod
    def compute_third_thursdays(start=2017, end=2024):
        import calendar
        def find_third_thursday(year, month):
            # Get the weekday of the first day of the month (0 - Monday, 1 - Tuesday, ..., 6 - Sunday)
            first_day_weekday = calendar.weekday(year, month, 1)

            # Calculate the day of the third Thursday (Thursday is 3)
            third_thursday = (3 - first_day_weekday + 7) % 7 + 1 + 14

            # Return the date of the third Thursday
            return third_thursday

        lst = []
        for year in range(start, end):
            for month in range(1, 13):
                try:
                    third_thursday = find_third_thursday(year, month)
                    date = f"{year}_{month:02d}_{third_thursday:02d}"
                    lst.append(date)

                except ValueError:
                    # Ignore the exception when there is no third Thursday in the month
                    pass
        return lst

    @staticmethod
    def get_process_creation_time(pid):
        try:
            process = psutil.Process(pid)
            return datetime.datetime.fromtimestamp(process.create_time()).strftime('%H:%M:%S')
        except psutil.NoSuchProcess:
            return '00:00:00'

    @staticmethod
    def convert_float_to_string(number):
        string_representation = str(number)
        if '.' in string_representation:
            # Remove trailing zeros
            string_representation = string_representation.rstrip('0')
            # Remove the decimal point if there are no decimal places
            if string_representation.endswith('.'):
                string_representation = string_representation[:-1]
        return string_representation

    @staticmethod
    def set_pd_display_options():
        pd.options.display.width = 500
        pd.options.display.max_columns = 20
        pd.options.display.max_rows = 100

    @staticmethod
    def threading_func_wrapper(func, delay=0.001, args=None, start=True):
        import threading
        threads = U.glob_threads
        if args is None:
            func_thread = threading.Timer(delay, func)
        else:
            func_thread = threading.Timer(delay, func, (args,))
        if start: func_thread.start()
        threads.append(func_thread)
        threads = threads[-5:]
        return func_thread

    @staticmethod
    def tik(label=None):
        title = '\x1b[90mtik: started...\x1b[0m '
        if label is None: label = ' '
        # noinspection PyGlobalUndefined
        print(f'{title} \x1b[90m{label}\x1b[0m')
        if len(U.TIK_TOK) > 1000:
            print('Trunking TIK_TOK')
            keys = list(U.TIK_TOK.keys())
            for key in keys[: (len(keys) - 1000)]:
                del U.TIK_TOK[key]
        U.TIK_TOK[label] = time()

    @staticmethod
    def tok(label=None):
        title = '\x1b[90mtok\x1b[0m: '
        if label is None: label = ' '
        if label in U.TIK_TOK:
            elapsed = time() - U.TIK_TOK[label]
            print(f'{title} \x1b[96m{label}\x1b[0m: \x1b[93m{elapsed * 1000:,.1f} ms\x1b[0m')
            del U.TIK_TOK[label]
        else:
            elapsed = time() - list(U.TIK_TOK.values())[-1]
            print(f'{title} \x1b[96m{label}\x1b[0m2: \x1b[93m{elapsed * 1000:,.1f} ms\x1b[0m')

    @staticmethod
    def lshift_columns(df, cols):
        if cols is None:
            cols = ['time', 'last', 'matchingVolume', 'totalMatchVolume']
        flt = df.columns.isin(cols)
        new_cols = cols + df.columns[~flt].to_list()
        return df[new_cols]

    @staticmethod
    def walk_through_files(TARGET, ext="*"):
        import os
        lst = []
        for root, dirs, files in os.walk(TARGET):
            for file in files:
                if ext == "*" or file.endswith(ext):
                    lst.append(os.path.join(root, file))
        return lst

    @staticmethod
    def dummy_function():
        return None

    @staticmethod
    def run_with_disabled_warnings(func):
        import warnings
        def wrapper(*args, **kwargs):
            warnings.filterwarnings("ignore")
            result = func(*args, **kwargs)
            warnings.filterwarnings("default")
            return result

        return wrapper

    @staticmethod
    def yield_rows(cursor, chunk_size):
        chunk = []
        for i, row in enumerate(cursor):
            if i % chunk_size == 0 and i > 0:
                yield chunk
                del chunk[:]
            chunk.append(row)
        yield chunk

    @staticmethod
    def ignore_warnings():
        import warnings
        warnings.filterwarnings("ignore")

    @staticmethod
    def check(df_1min):
        print(df_1min['day'].value_counts()
              .reset_index()
              .sort_values('index',
                           ignore_index=True))

    @staticmethod
    def remove_one_candle_days(df_1min):
        one_candle_days = \
            (lambda df: df[df == 1].reset_index()['index'])\
                (df_1min['day'].value_counts())
        df = df_1min[~df_1min['day'].isin(one_candle_days)].copy()
        return df
    @staticmethod
    def execute_cmd(cmd, print_result=True, wait_til_finish=True):
        import subprocess, re, shlex
        cmd = cmd.replace("\n", "").strip().replace("\t", "")
        cmd = cmd.replace("\n", "").strip().replace("\t", "")
        cmd = re.sub(" +", " ", cmd)
        args = shlex.split(cmd)
        process = subprocess.Popen(
            args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        if not wait_til_finish: return process
        stdout, stderr = process.communicate()
        err = stderr.decode("utf-8")
        if len(err) > 0: print(f"\x1b[93m{err}\x1b[90m")
        res = stdout.decode("utf-8")
        if print_result: print(f"\x1b[93m{res}\x1b[0m")
        return re.sub(" +", " ", res)

    @staticmethod
    def maybe_create_dir(DIR, verbosity=1):
        import os
        if not os.path.isdir(DIR):
            U.execute_cmd(f"mkdir -p {DIR}")
            if verbosity >= 1: print(f"Created folder {DIR}")

    @staticmethod
    def report_error(e, function_name="unnamed_foo"):
        from datetime import datetime
        from traceback import print_exc
        print(datetime.now().strftime("%H:%M:%S"), end=" ")
        print(f"GREEN{function_name}()ENDC Có lỗi xảy ra: REDBG{e}"
              f"ENDC type: REDBG{type(e).__name__}ENDC"
              f"\nArguments:BLUE{e.args}ENDC"   \
                  .replace("REDBG", '\33[41m')  \
                  .replace("ENDC", '\033[0m')   \
                  .replace("GREEN", '\33[32m')  \
                  .replace("BLUE", '\33[34m'))
        print_exc()
        print(f"\x1b[95m{dt.now().strftime('%H:%M:%S')} "
              f"\x1b[91mError ({e}) handled (hopefully), "
              f"continuing as if nothing happened...\x1b[90m")

    @staticmethod
    def get_stock_daily(symbol='HPG', num_gap=0, resolution='1D'):
        import pandas as pd
        import requests
        TIME_GAP = 34128060  # 395 days
        headers = {
            'Connection': 'keep-alive',
            'sec-ch-ua': '"Google Chrome";v="93", '
                         '" Not;A Brand";v="99", "Chromium";v="93"',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/93.0.4577.82 Safari/537.36',
            'sec-ch-ua-platform': '"Linux"',
            'Accept': '*/*',
            'Origin': 'https://chart.vps.com.vn',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Referer': 'https://chart.vps.com.vn/',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        timestamp = int(dt.strptime(
            dt.now().strftime('%Y_%m_%d'), '%Y_%m_%d'
        ).timestamp())
        params = (
            ('symbol', symbol),
            ('resolution', resolution),
            ('from', str(timestamp - TIME_GAP * (num_gap + 1))),
            ('to', str(timestamp - TIME_GAP * num_gap)),
        )
        response = requests.get(
            'https://histdatafeed.vps.com.vn/tradingview/history',
            headers=headers,
            params=params)
        df = pd.DataFrame(response.json())
        df['t'] = df['t'].map(lambda x: dt.fromtimestamp(x - 3600 * 7))
        df.rename(columns={
            'c': 'close',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'v': 'volume'}, inplace=True)
        df['day'] = df['t'].map(lambda x: x.strftime('%Y_%m_%d'))
        del df['s']
        df = df.reset_index(drop=True)
        df['stock'] = symbol
        return df

    @staticmethod
    def load_time_to_index():
        with open('time_to_index.pickle', 'rb') as file:
            dic_time_to_index = pickle.load(file)
            dic_index_to_time = {v: k for k, v in dic_time_to_index.items()}

        return dic_time_to_index, dic_index_to_time

    @staticmethod
    def get_df_last_day(df):
        return (lambda df: df[df.index.strftime('%Y_%m_%d') == df.index[-1].strftime('%Y_%m_%d')])(df)


"""OPERATORS"""
class O:
    # @staticmethod
    # def ts_weighted_mean(df, window=10):
    #     # noinspection PyUnresolvedReferences
    #     from talib import WMA

    #     wma = WMA(df,timeperiod=window)
    #     return wma
    
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

"""DOMAINS (DATA AGGREGATION & TRANSFORMATION)"""
class Resampling_Domain:
    @staticmethod
    def compute_returns(df):
        df1 = df.merge(
            df.groupby('day')
                .agg({'close': 'last'})
                .rename(columns={'close': 'prevClose'})
                .shift(1).fillna(method=C.QUICK_FIX_COMPUTE_RETURNS),
            left_on='day',
            right_index=True,
            how='left'
        )
        df1['returns'] = df1['close'] / df1['prevClose'] - 1
        return df1['returns'].copy()

    @staticmethod
    def compute_df_resampled_from_df_1min(freq, df1, compute_returns=True):

        def compute_df1_grouped(df1, freq):
            INPUT_COLS = [
                'day', 'open', 'high', 'low', 'close', 'matchingVolume',
                'totalMatchVolume', 'time', 'timeFirst', 'timeLast']
            df1['t'] = df1.index
            df1 = df1.set_index('t')
            df1 = df1[INPUT_COLS] \
                .reset_index() \
                .rename(columns={'t': 'idx'}) \
                .set_index('idx')

            ds_ref_time = df1.index.map(lambda x: x.replace(
                hour=9,
                minute=0,
                second=0))

            df1['group'] = (
                               df1.index -
                               ds_ref_time
                           ) \
                               .map(lambda x: x.seconds) / 60
            df1['group'] = np.floor_divide(df1['group'], freq) * freq
            df1['groupTime'] = ds_ref_time + \
                               (df1['group']) \
                                   .map(lambda x: timedelta(minutes=x))
            df1['timeFirst'] = df1['timeFirst']
            df1['timeLast'] = df1['timeLast']
            df1_grouped = df1 \
                .groupby('groupTime') \
                .agg(
                {
                    'day': 'last',
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'matchingVolume': 'sum',
                    'totalMatchVolume': 'last',
                    'group': 'first',
                    'timeFirst': 'first',
                    'timeLast': 'last'
                })

            td = timedelta
            delta = td(minutes=freq) - td(seconds=0.001)

            df1_grouped['theoreticalTimeLast'] = (
                df1_grouped.index + delta
            )\
            .map(lambda x: x.strftime('%H:%M:%S'))

            df1_grouped['executionTime'] = (
                df1_grouped.index
                +
                td(minutes=freq)
            )\
            .map(lambda x: x.strftime('%H:%M:%S'))
            flt = df1_grouped['executionTime'] > '14:45:00'
            df1_grouped.loc[flt, 'executionTime'] = '14:45:00'

            return df1_grouped

        def compute_executable_times(df1_grouped, compute_returns=True):
            if 'time' not in df1_grouped:
                df1_grouped['time'] = \
                    df1_grouped.index\
                               .map(lambda x: x.strftime('%H:%M:%S'))
            ##########
            df1_grouped['executable'] = True
            data_time = df1_grouped['theoreticalTimeLast']
            execution_time = df1_grouped['executionTime']
            ##########
            """1. Lunch"""
            flt_lunch_non_executable = \
                (data_time >= '11:30:00') & \
                (data_time < '13:00:00') & \
                (execution_time > '11:30:00') & \
                (execution_time < '13:00:00')

            df1_grouped.loc[flt_lunch_non_executable, 'executable'] = False
            df1_grouped.loc[flt_lunch_non_executable, 'session'] = 'lunch'

            #####
            """2. preATC"""
            flt_pre_atc_non_executable = \
                (data_time >= '14:30:00') & \
                (data_time < '14:45:00') & \
                (execution_time > '14:30:00') & \
                (execution_time < '14:45:00')

            df1_grouped.loc[flt_pre_atc_non_executable, 'executable'] = False
            df1_grouped.loc[flt_pre_atc_non_executable, 'session'] = 'preATC'


            #####
            """3. unconditionalATC"""
            flt_atc_executable = \
                (data_time >= '14:45:00') & \
                (execution_time >= '14:45:00')
            df1_grouped.loc[flt_atc_executable, 'executable'] = True
            df1_grouped.loc[flt_atc_executable, 'session'] = C.FLAT_ATC
            df1_grouped.loc[flt_atc_executable, 'executionTime'] = '14:45:00'

            #####
            # """4. Entry, exit & price change"""
            df1_grouped['entryPrice'] = df1_grouped['open'].shift(-1)
            df1_grouped.loc[flt_atc_executable, 'entryPrice'] = \
                df1_grouped.loc[flt_atc_executable, 'close']
            df1_grouped['exitPrice'] = df1_grouped['entryPrice'].shift(-1)
            df1_grouped['priceChange'] = \
                df1_grouped['exitPrice'] - df1_grouped['entryPrice']
            #####
            """5. Extra modifications needed for 5m, 23m ..."""
            flt_is_atc_time = (df1_grouped['executionTime'] == '14:45:00')
            flt_not_unconditional_atc = df1_grouped['session'] != C.FLAT_ATC
            flt = flt_is_atc_time & flt_not_unconditional_atc
            if np.sum(flt) != 0:
                df1_grouped.loc[flt, 'executable'] = False
                df1_grouped.loc[flt, 'session'] = C.PRE_ATC


            #####
            """6. Entering trades at 14:30:00 must be prevented """
            df1_grouped.loc[df1_grouped['executionTime'] == '14:30:00', 'executable'] = False
            #####
            if 'return' not in df1_grouped and compute_returns:
                df1_grouped['return'] = Resampling_Domain.compute_returns(df1_grouped)

            return df1_grouped

        df1_grouped = compute_df1_grouped(
            df1,
            freq)
        df1_grouped = compute_executable_times(
            df1_grouped, compute_returns=compute_returns)

        return df1_grouped

    @staticmethod
    def resample_all(freqs, dic_freqs, to_pickle=True):
        header = '\x1b[90mResampling_Domain.resample_all\x1b[0m: '
        dic = {1: dic_freqs[1]}
        if freqs is None:
            freqs = range(2, 101)
        for freq in freqs:
            df2 = Resampling_Domain.compute_df_resampled_from_df_1min(
                freq=freq, df1=dic[1])
            print(header, freq)
            dic[freq] = df2

        if to_pickle:
            dic[1] = Resampling_Domain.compute_df_resampled_from_df_1min(
                freq=1,
                df1=dic[1])
            Resampling_Domain.to_pickle(dic)

        return dic

    @staticmethod
    def to_pickle(dic_freqs):
        header = '\x1b[90mResampling_Domain.to_pickle\x1b[0m: '
        print(header, end='')
        fn = RESOURCES.get_fn_dic_df1_resample2()

        with open(fn, 'wb') as file:
            pickle.dump(dic_freqs, file)
        U.execute_cmd(f'ls -lahtr {fn}')

    @staticmethod
    def from_pickle():
        header = '\x1b[90mResampling_Domain.from_pickle\x1b[0m: '
        print(header, end='')
        fn = RESOURCES.get_fn_dic_df1_resample2()
        U.execute_cmd(f'ls -lahtr {fn}')
        with open(fn, 'rb') as file:
            dic_freqs = pickle.load(file)
        return dic_freqs

    @staticmethod
    def validate():
        fn = RESOURCES.get_fn_dic_df1_resample2()

        with open(fn, 'rb') as file:
            dic_freqs = pickle.load(file)

        lst = []
        for freq in dic_freqs:
            if freq == 1: continue
            print(freq)
            df = dic_freqs[freq]
            df['freq'] = freq
            lst.append(df[['freq', 'time', 'entryPrice', 'exitPrice',
                           'executionTime', 'session', 'executable',
                           'theoreticalTimeLast',
                           ]].copy())
        df = pd.concat(lst)


        # Filter out trades that execute at lunch break
        df_lunch = df[(df['executionTime'] > '11:30:00') &
                      (df['executionTime'] <= '13:00:00')]
        print(df_lunch, '\n^^^ df_lunch')
        assert int(np.sum(df_lunch['executable'])) == 0, \
               'Sai! Vẫn có executable candle trong lunch break'

        # Filter out trades that execute at preATC (14:30 => 14:45)
        df_preatc = df[(df['executionTime'] >= '14:30:00') &
                       (df['executionTime'] < '14:45:00')]
        df_preatc_executable = df_preatc[df_preatc['executable']].copy()
        assert len(df_preatc_executable) == 0, \
               'Sai! Vẫn có executable candle preATC'

    @staticmethod
    def resample_df_to_1min(df, freq=1, verbosity=1):
        if verbosity >= 2:
            U.tik('Resampling_Domains.resample_to_1min')

        df = df.copy()
        df['open'] = df['high'] = df['low'] = df['close'] = df['last']
        df['timeFirst'] = df['timeLast'] = df['time']
        INPUT_COLS = ['day', 'open', 'high', 'low', 'close', 'matchingVolume',
                      'totalMatchVolume', 'time', 'timeFirst', 'timeLast', 't']

        df1 = df[INPUT_COLS].set_index('t')
        df1['timeRef'] = df1.index.map(lambda x: x.replace(
            hour=9, minute=0, second=0))
        df1 = df1[df1.index >= df1['timeRef']]
        df1['group'] = (df1.index - df1['timeRef']).map(lambda x: x.seconds / 60).astype('int')
        df1['groupTime'] = df1['timeRef'] + \
                           df1['group'] \
                               .map(lambda x: timedelta(minutes=x))
        df1['timeFirst'] = df1['timeFirst']
        df1['timeLast'] = df1['timeLast']
        df1_grouped = df1 \
            .groupby('groupTime') \
            .agg(
            {
                'day': 'last',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'matchingVolume': 'sum',
                'totalMatchVolume': 'last',
                'group': 'first',
                'timeFirst': 'first',
                'timeLast': 'last'
            })


        def compute_deltas():
            from datetime import timedelta as td
            delta = td(minutes=freq) - td(seconds=0.001)
            delta_freq = td(minutes=freq)
            return delta, delta_freq


        delta, delta_freq = compute_deltas()
        df1_grouped['theoreticalTimeLast'] = (
                df1_grouped.index + delta
        ) \
            .map(lambda x: x.strftime('%H:%M:%S'))
        df1_grouped['executionTime'] = (
                df1_grouped.index
                +
                delta_freq
        ) \
            .map(lambda x: x.strftime('%H:%M:%S'))
        flt = df1_grouped['executionTime'] > '14:45:00'
        df1_grouped.loc[flt, 'executionTime'] = '14:45:00'


        def time_is_valid(x):
            res = not (
                    '14:31:00' <= x < '14:45:00' or
                    x > '14:45:59' or
                    x < '09:00:00')
            return res


        df2 = df1_grouped[
            df1_grouped['timeFirst'].map(time_is_valid)].copy()
        df2['time'] = df2.index

        if verbosity >= 2:
            print(f"\x1b[93m{df2}")
            U.tok('Resampling_Domains.resample_to_1min')

        return df2

    @staticmethod
    def compute_df_input(df0, verbosity=1):
        if verbosity >= 2: U.tik('step 1')
        df = df0[['timestamp', 'code', 'best1Bid', 'lastPrice', 'best1Offer',
                  'lastVol', 'totalMatchVol', 'totalMatchValue', 'refPrice']]

        df = df.rename(columns={
            'best1Bid': 'bestBid1',
            'best1Offer': 'bestOffer1',
            'totalMatchVol': 'totalMatchVolume',
            'lastPrice': 'last',
            'code': 'stock',
        })
        fillna_cols = ['bestBid1', 'bestOffer1', 'totalMatchVolume']
        df[fillna_cols] = df[fillna_cols].fillna(method='ffill')
        df = df.dropna(subset=fillna_cols).copy()
        df[fillna_cols] = df[fillna_cols].astype('int')
        df['matchingVolume'] = df['totalMatchVolume'].diff()
        df = df[df['matchingVolume'] > 0].copy()
        df['t'] = (df['timestamp'] / 1000).astype('int').map(dt.fromtimestamp)
        df['time'] = df['t'].map(lambda x: x.strftime('%H:%M:%S'))

        if verbosity >= 2: U.tok('step 1')
        return df


class Alphas:

    @staticmethod
    def alpha_custom_resid(df: pd.DataFrame, window_corr=15, window_resid=20):
        """
        Gốc: Neg(Neg(CsZScore(Resid(Corr($close, $amt, 15), $volume, 20))))
        Mô tả: Tương quan Close-Amount sau khi khử nhiễu từ Volume.
        """
        # 1. Tính Amount (giá trị giao dịch) nếu chưa có
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume'))
        amt = df.get('amt', close * volume)

        # 2. Tính Rolling Correlation (Tương quan động)
        correlation = O.ts_corr(close, amt, window=window_corr)

        # 3. Tính Resid (Phần dư từ hồi quy tuyến tính đơn giản: corr = a + b*volume + error)
        # Để đơn giản và bao quát, ta sử dụng phương pháp trừ đi thành phần có tương quan với volume
        def get_resid(y, x):
            # Cửa sổ trượt cho hồi quy
            rolling_corr_xy = y.rolling(window_resid).corr(x)
            rolling_std_y = y.rolling(window_resid).std()
            rolling_std_x = x.rolling(window_resid).std()
            
            # Beta = corr(x,y) * (std_y / std_x)
            beta = rolling_corr_xy * (rolling_std_y / (rolling_std_x + 1e-9))
            alpha = y.rolling(window_resid).mean() - beta * x.rolling(window_resid).mean()
            
            return y - (alpha + beta * x)

        residual = get_resid(correlation.fillna(0), volume.fillna(0))

        # 4. Cross-sectional Rank (Thay cho CsZScore để khớp với class O và output [-1, 1])
        # Rank giúp mapping dữ liệu về khoảng [0, 1] ổn định hơn Z-score khi có outlier
        ranked_signal = O.rank(residual.to_frame().T).T[0] # Giả lập cross-section nếu df chỉ có 1 mã
        # Nếu áp dụng cho cả bảng (nhiều tickers):
        # ranked_signal = residual.rank(axis=1, pct=True) 

        # 5. Normalize về [-1, 1]
        # Neg(Neg(x)) = x, nên hướng giữ nguyên
        signal = 2 * ranked_signal - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_095(df: pd.DataFrame, delta_window=5, ma_window=60):
        """
        Name: confirmed_momentum_signal
        Formula: RANK(SIGN(DELTA($close, 5)/DELAY($close, 5)) * 
                 SIGN(($close - TS_MEAN($close, 60))/(TS_MEAN($close, 60) + 1e-8)))
        Logic: Kết hợp hướng thay đổi giá ngắn hạn (5 phiên) với vị thế giá so với MA60.
        """
        close = df['close']
        
        # 1. Thành phần 1: Hướng biến động 5 phiên (Short-term Direction)
        # Delta(5) / Delay(5) chính là tỷ suất sinh lời trong 5 phiên
        ret_5 = (close.diff(delta_window) / close.shift(delta_window).replace(0, 1e-8))
        direction_5 = np.sign(ret_5.fillna(0))
        
        # 2. Thành phần 2: Vị thế so với xu hướng dài hạn (Long-term Regime)
        # Nếu Close > MA60 -> Sign = 1 (Bullish)
        # Nếu Close < MA60 -> Sign = -1 (Bearish)
        ma_60 = close.rolling(ma_window).mean()
        regime_sign = np.sign((close - ma_60) / (ma_60 + 1e-8))
        
        # 3. Kết hợp bằng phép nhân
        # Ý nghĩa: 
        # (1 * 1) hoặc (-1 * -1) -> Tín hiệu dương (Giá đang hồi phục về MA hoặc bùng nổ theo trend)
        # (1 * -1) hoặc (-1 * 1) -> Tín hiệu âm (Phân kỳ giữa ngắn hạn và dài hạn)
        combined_signal = direction_5 * regime_sign
        
        # 4. Áp dụng Rank và đưa về biên độ [-1, 1]
        # Sử dụng ts_rank_normalized để giả lập RANK trong môi trường Single Asset
        ranked_signal = O.ts_rank_normalized(combined_signal.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_096(df: pd.DataFrame, window=15):
        """
        Name: rolling_risk_adjusted_return
        Formula: RANK(TS_MEAN($return, 15) / (TS_STD($return, 15) + 1e-8))
        Logic: Tỷ lệ Sharpe nội tại trong 15 phiên. 
               Ưu tiên các tài sản có xu hướng tăng ổn định, ít biến động giật cục.
        """
        # 1. Tính Return (Lợi nhuận theo phiên)
        # Nếu df chưa có cột 'return', tính bằng pct_change
        returns = df['close'].pct_change(1).fillna(0)
        
        # 2. Tính trung bình động của Return (Lợi nhuận kỳ vọng ngắn hạn)
        mean_ret = returns.rolling(window).mean()
        
        # 3. Tính độ lệch chuẩn của Return (Rủi ro/Biến động ngắn hạn)
        std_ret = returns.rolling(window).std()
        
        # 4. Tính tỷ lệ hiệu suất (Efficiency Ratio)
        # Thêm 1e-8 để tránh chia cho 0 khi giá đứng yên (limit hoặc đóng cửa không đổi)
        efficiency_ratio = mean_ret / (std_ret + 1e-8)
        
        # 5. Áp dụng Rank và đưa về biên độ [-1, 1]
        # Sử dụng ts_rank_normalized để chuẩn hóa tín hiệu
        ranked_signal = O.ts_rank_normalized(efficiency_ratio.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_098(df: pd.DataFrame, window=8):
        """
        Name: volume_shock_price_position
        Formula: (DELTA($volume, 1) - TS_MEAN(DELTA($volume, 1), 8)) / (TS_STD(DELTA($volume, 1), 8) + 1e-8) * 
                 (($close - TS_MIN($close, 8)) / (TS_MAX($close, 8) - TS_MIN($close, 8) + 1e-8)) * 
                 (1 - ABS(TS_CORR(DELTA($volume, 1), DELTA($close, 1), 8)))
        Logic: Kết hợp Cú sốc khối lượng, Vị thế giá và Độ độc lập giữa Biến động Giá-Khối lượng.
        """
        # 1. Thành phần 1: Volume Delta Z-Score
        # Đo lường xem thay đổi khối lượng hiện tại có phải là một "cú sốc" so với 8 phiên qua không.
        vol_delta = df['matchingVolume'].diff(1)
        mean_vol_delta = vol_delta.rolling(window).mean()
        std_vol_delta = vol_delta.rolling(window).std()
        vol_zscore = (vol_delta - mean_vol_delta) / (std_vol_delta + 1e-8)
        
        # 2. Thành phần 2: Price Range Position (Stochastic Oscillator Logic)
        # Giá đang nằm ở đâu trong biên độ High-Low của 8 phiên gần nhất.
        # 0 = Đáy 8 phiên, 1 = Đỉnh 8 phiên.
        ts_min = df['close'].rolling(window).min()
        ts_max = df['close'].rolling(window).max()
        price_pos = (df['close'] - ts_min) / (ts_max - ts_min + 1e-8)
        
        # 3. Thành phần 3: Price-Volume Independence (Phân kỳ)
        # Lấy (1 - Trị tuyệt đối Tương quan). 
        # Nếu Giá và Khối lượng chạy cực kỳ đồng thuận hoặc nghịch biến (Corr gần 1 hoặc -1), thành phần này tiến về 0.
        # Nếu hướng đi của chúng hỗn loạn/phân kỳ (Corr gần 0), thành phần này tiến về 1 (tăng sức mạnh tín hiệu).
        price_delta = df['close'].diff(1)
        ts_corr = vol_delta.rolling(window).corr(price_delta)
        pv_independence = 1 - ts_corr.abs().fillna(0)
        
        # 4. Kết hợp 3 thành phần (Nhân lại với nhau)
        combined_raw = vol_zscore * price_pos * pv_independence
        
        # 5. Áp dụng Rank và đưa về biên độ [-1, 1]
        final_signal = -O.ts_rank_normalized(combined_raw.fillna(0), 20)
        return (2 * final_signal) - 1

    @staticmethod
    def alpha_factor_miner_099(df: pd.DataFrame, corr_window=20, mean_window=5):
        """
        ID: 23
        Name: money_flow_return_correlation
        Formula: TS_CORR((DELTA($close, 1)/$close) * $volume, TS_MEAN($return, 5), 20)
        Logic: Sự đồng thuận giữa dòng tiền biến động (Money Flow) và xu hướng lợi nhuận ngắn hạn.
        """
        # 1. Thành phần 1: Instantaneous Money Flow (Dòng tiền tức thời)
        # (Delta(Close)/Close) * Volume => Phần trăm thay đổi giá nhân với khối lượng.
        # Đây là biến thể của Money Flow Index (MFI) nhưng ở mức độ nến đơn lẻ.
        price_pct_change = df['close'].diff(1) / (df['close'] + 1e-8)
        money_flow = price_pct_change * df['matchingVolume']
        
        # 2. Thành phần 2: Short-term Mean Return (Lợi nhuận trung bình 5 phiên)
        returns = df['close'].pct_change(1).fillna(0)
        mean_return_5 = returns.rolling(mean_window).mean()
        
        # 3. Tính Tương quan (Correlation) trong 20 phiên
        # Nếu tương quan dương cao: Dòng tiền đang đẩy giá đi đúng theo xu hướng trung bình.
        # Nếu tương quan âm: Dòng tiền đang chống lại xu hướng (dấu hiệu đảo chiều).
        ts_corr = money_flow.rolling(corr_window).corr(mean_return_5)
        
        # 4. Đưa về biên độ chuẩn [-1, 1]
        # Vì TS_CORR vốn dĩ nằm trong khoảng [-1, 1], ta chỉ cần Rank để làm mượt.
        ranked_signal = O.ts_rank_normalized(ts_corr.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_100(df: pd.DataFrame, accel_window=3, std_window=15):
        """
        ID: 24 (Alpha Thế Kỷ)
        Name: conditional_volatility_acceleration
        Formula: (TS_PCTCHANGE($high - $low, 3) * SIGN(TS_PCTCHANGE($volume, 3))) / 
                 (TS_STD($high - $low, 15) + 1e-8)
        Logic: Sự gia tốc của biên độ nến được xác nhận bởi hướng đi của khối lượng.
               Tìm kiếm các điểm bùng nổ (Breakout) hoặc kiệt sức (Climax).
        """
        # 1. Tính biên độ nến (Range)
        candle_range = df['high'] - df['low']
        
        # 2. Gia tốc biên độ (% thay đổi trong 3 phiên)
        # Nếu biên độ mở rộng nhanh -> range_accel dương lớn.
        range_accel = candle_range.pct_change(accel_window).fillna(0)
        
        # 3. Hướng thay đổi khối lượng (Volume Direction)
        # Sign = 1 nếu Vol tăng, -1 nếu Vol giảm so với 3 phiên trước.
        vol_accel_sign = np.sign(df['matchingVolume'].pct_change(accel_window).fillna(0))
        
        # 4. Chuẩn hóa bằng Volatility của chính biên độ nến (15 phiên)
        # Giúp làm mượt tín hiệu trong các môi trường thị trường khác nhau.
        range_std = candle_range.rolling(std_window).std().fillna(0)
        
        # 5. Kết hợp: (Gia tốc Range * Hướng Vol) / Độ biến động Range
        # Logic: Tín hiệu bùng nổ khi cả Range và Vol cùng tăng tốc đồng thuận.
        raw_signal = (range_accel * vol_accel_sign) / (range_std + 1e-8)
        
        # 6. Đưa về biên độ chuẩn [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_trend_efficiency(df: pd.DataFrame, window=20, vol_window=10):
        """
        Ý tưởng: 
        1. Tính toán độ dốc giá (Delta) so với mức trung bình (Mean).
        2. Chuẩn hóa bằng độ lệch chuẩn (Std) để tránh các cú sốc giá ảo (Z-Score).
        3. Kết hợp với tương quan giữa Giá và Khối lượng để xác nhận dòng tiền.
        4. Ranking để đưa về tín hiệu -1 đến 1.
        """
        # 1. Tính VWAP và giá trung bình
        df = O.compute_vwap(df, window=window)
        
        # 2. Tính sự lệch pha giữa giá hiện tại và VWAP (Resid)
        # Nếu giá > VWAP và đang tăng nhanh hơn mức trung bình 20 phiên
        price_diff = df['close'] - df['vwap']
        
        # 3. Tính độ biến động (Volatility) để chuẩn hóa
        volatility = O.ts_std(df['close'], window=window)
        
        # 4. Tính Z-Score của sự lệch giá (đo lường độ bất thường)
        z_score = (price_diff / (volatility + 1e-6))
        
        # 5. Xác nhận bằng tương quan Giá - Khối lượng (Price-Volume Correlation)
        # Một xu hướng tăng bền vững cần có sự đồng thuận của Volume
        pv_corr = O.ts_corr(df['close'], df['matchingVolume'], window=vol_window)
        
        # 6. Kết hợp: Tín hiệu mạnh khi Z-score cao và tương quan PV cao
        raw_signal = z_score * pv_corr
        
        # 7. Xử lý Ranking và đưa về dải [-1, 1]
        # Rank theo thời gian (Time-series Rank)
        ts_ranked = O.ts_rank_normalized(raw_signal.fillna(0), window=window)
        
        # Chuyển đổi từ [0, 1] sang [-1, 1]
        signal = (2 * ts_ranked) - 1
        
        return -signal.fillna(0)

     # factor
    @staticmethod
    def alpha_factor_miner_000(df: pd.DataFrame, window=20):
        """
        Logic: Volume-Adjusted Efficiency Ratio.
        Mục đích: Tìm kiếm các cây nến có thân dài nhưng 'sạch' (ít nhiễu) và có thanh khoản xác nhận.
        """
        # 1. Tính thân nến (Body)
        body = df['close'] - df['open']
        
        # 2. Tính độ lệch chuẩn của thân nến trong 20 phiên
        volatility = O.ts_std(body, 20)
        
        # 3. Lấy căn bậc hai của khối lượng
        volume = df.get('volume', df.get('matchingVolume'))
        sqrt_vol = np.sqrt(volume + 1e-8)
        
        # 4. Tính toán tỷ lệ hiệu quả (Efficiency Ratio)
        # Ratio = (Close - Open) / (Std * Sqrt(Vol))
        raw_ratio = body / (volatility * sqrt_vol + 1e-8)
        
        # 5. Chuẩn hóa về dải -1 đến 1
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
    

    @staticmethod
    def alpha_factor_miner_001(df: pd.DataFrame, window=20):
        """
        Logic: Liquidity-Normalized VWAP/EMA Spread.
        Mục đích: Tìm điểm bùng nổ xu hướng dựa trên độ lệch giá và khối lượng trung bình.
        """
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        volume = df.get('volume', df.get('matchingVolume'))
        
        # 1. Tính EMA 15 phiên cho Close và Volume
        ema_close_15 = df['close'].ewm(span=15, adjust=False).mean()
        ema_vol_15 = volume.ewm(span=15, adjust=False).mean()
        
        # 2. Tính độ lệch (Spread)
        spread = vwap - ema_close_15
        
        # 3. Chuẩn hóa Spread theo quy mô Volume trung bình
        # (Spread / EMA_Vol)
        raw_ratio = spread / (ema_vol_15 + 1e-8)
        
        # 4. Chuẩn hóa về dải -1 đến 1
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_002(df: pd.DataFrame, window=30):
        
        returns = (df['close'] / O.ts_lag(df['close'], 1)) - 1
        
        amt = df.get('amount', df['close'] * df.get('volume', df.get('matchingVolume')))
        log_amt = np.log(amt + 1e-8)
        
        correlation = O.ts_corr(log_amt, returns, window)
        
        raw_signal = -1 * correlation
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_003(df: pd.DataFrame, window=40, delay_step=5):
        def get_kama(price, n=40):
            change = abs(price - price.shift(n))
            volatility = abs(price - price.shift(1)).rolling(n).sum()
            er = change / (volatility + 1e-8)
            
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
            
            kama = np.zeros_like(price)
            kama[n-1] = price[n-1] 
            for i in range(n, len(price)):
                kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
            return pd.Series(kama, index=price.index)

        kama_val = get_kama(df['close'], window)
        
        kama_diff = kama_val - kama_val.shift(delay_step)

        raw_signal = -1 * kama_diff
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_004(df: pd.DataFrame, window=50, delay_step=10, z_window=60):
       
        volume = df.get('volume', df.get('matchingVolume'))
        
        vol_skew = volume.rolling(window).skew()
        
        skew_delta = vol_skew - vol_skew.shift(delay_step)
        
        mean_delta = skew_delta.rolling(z_window).mean()
        std_delta = skew_delta.rolling(z_window).std()
        z_score = (skew_delta - mean_delta) / (std_delta + 1e-8)
       
        raw_signal = -1 * z_score
        
        ranked_signal = O.ts_rank_normalized(raw_signal, z_window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_005(df: pd.DataFrame, window=20):
        body = df['close'] - df['open']
        
        full_range = df['high'] - df['low']
      
        raw_efficiency = body / (full_range + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_efficiency, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_006(df: pd.DataFrame, window=20):
        
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        volume = df.get('volume', df.get('matchingVolume'))
        
        vwap_std = vwap.rolling(window).std()
        
        avg_vol = volume.rolling(window).mean()
        
        raw_ratio = vwap_std / (avg_vol + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_007(df: pd.DataFrame, reg_window=15, corr_window=10):
    
        amt = df.get('amount', df['close'] * df.get('volume', df.get('matchingVolume')))
        
     
        def get_slope(series, n):
            y = series.values
            x = np.arange(n)
            def calc_slope(y_window):
                if len(y_window) < n: return np.nan
                slope, _ = np.polyfit(x, y_window, 1)
                return slope
            return series.rolling(window=n).apply(calc_slope, raw=True)

        slope_amt = get_slope(amt, reg_window)
        slope_close = get_slope(df['close'], reg_window)
        
        correlation = slope_amt.rolling(corr_window).corr(slope_close)
        
       
        raw_signal = -1 * correlation
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_008(df: pd.DataFrame, kama_window=30, std_window=10, delay_step=5):
        
        close_values = df['close'].values
        returns = np.diff(close_values, prepend=close_values[0]) / (close_values + 1e-8)
        
        def get_kama_fast(arr, n=30):
            abs_diff = np.abs(arr - np.roll(arr, n))
            vol = pd.Series(np.abs(arr - np.roll(arr, 1))).rolling(n).sum().values
            
            er = np.where(vol > 1e-8, abs_diff / vol, 0)
            
            sc = (er * (2/3 - 2/31) + 2/31)**2
            
            kama = np.zeros_like(arr)
            kama[n-1] = arr[n-1]
            
            for i in range(n, len(arr)):
                kama[i] = kama[i-1] + sc[i] * (arr[i] - kama[i-1])
            return kama

        kama_ret = get_kama_fast(returns, kama_window)
        kama_ret_series = pd.Series(kama_ret, index=df.index)
        
        kama_volatility = kama_ret_series.rolling(std_window).std(ddof=0)
        
        vol_change = kama_volatility.diff(delay_step)
        
        raw_signal = -1 * vol_change.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_009(df: pd.DataFrame, window=60):
      
        volume = df.get('volume', df.get('matchingVolume')).values
        cumsum_vol = np.cumsum(volume)
        
        cumsum_ser = pd.Series(cumsum_vol, index=df.index)
        
        ts_max_vol = cumsum_ser.rolling(window).max()
       
        ratio = cumsum_ser / (ts_max_vol + 1e-8)
        
   
        raw_signal = -1 * ratio
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_010(df: pd.DataFrame, delta_window=5, std_window=10):
       
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        delta_ret = returns_ser.diff(delta_window)
        
        std_ret = returns_ser.rolling(std_window).std(ddof=0)
       
        raw_ratio = delta_ret / (std_ret + 1e-8)
        
      
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_011(df: pd.DataFrame, short_window=5, long_window=20):
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        std_10 = returns_ser.rolling(short_window).std(ddof=0)
        std_25 = returns_ser.rolling(long_window).std(ddof=0)
        
        vol_ratio = std_10 / (std_25 + 1e-8)
        
        signed_ratio = np.where(returns > 0, vol_ratio, -vol_ratio)
        signed_ratio_ser = pd.Series(signed_ratio, index=df.index)
        
        
        raw_signal = -1 * signed_ratio_ser.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_012(df: pd.DataFrame, short_window=5, long_window=20):
       
        range_hl = df['high'] - df['low']
        
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        
        std_range_5 = range_hl.rolling(short_window).std(ddof=0)
 
        std_ret_20 = returns_ser.rolling(long_window).std(ddof=0)
     
        raw_ratio = std_range_5 / (std_ret_20 + 1e-8)
        
        
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_013(df: pd.DataFrame, ema_window=12, delay_step=3):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        price_vwap_ratio = df['close'] / (vwap + 1e-8)
        
        
        smoothed_ratio = price_vwap_ratio.ewm(span=ema_window, adjust=False).mean()
        
        
        delta_ratio = smoothed_ratio - smoothed_ratio.shift(delay_step)
        
        ranked_signal = O.ts_rank_normalized(delta_ratio.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_014(df: pd.DataFrame, rank_window=15, reg_window=8):
        price_rank = df['close'].rolling(rank_window).rank(pct=True)
        
        x = np.arange(reg_window)
        x_var = np.var(x)
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        
        numerator = price_rank.rolling(reg_window).cov(ser_index)
        slope_rank = numerator / x_var
        
        std_rank = price_rank.rolling(reg_window).std(ddof=0)
        
        raw_ratio = slope_rank / (std_rank + 1e-8)
        
        raw_signal = -1 * raw_ratio.fillna(0)
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_015(df: pd.DataFrame, kurt_window=30, delta_window=3, slope_window=20):
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        kurtosis = pd.Series(returns).rolling(kurt_window).kurt().values
        
        delta_close = df['close'].diff(delta_window)
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(slope_window))
        slope_close = df['close'].rolling(slope_window).cov(ser_index) / (x_var + 1e-8)
        
        rank_delta = O.ts_rank_normalized(delta_close, 20)
        rank_slope = O.ts_rank_normalized(slope_close, 20)
        
        condition_signal = np.where(kurtosis > 0, -rank_delta, rank_slope)
        
        raw_signal = -1 * pd.Series(condition_signal, index=df.index)
        
        signal = (2 * raw_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_016(df: pd.DataFrame, short_window=5, long_window=10):
        range_hl = df['high'] - df['low']
        
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        # 2. Tính Std Range (5 phiên) và Std Returns (10 phiên)
        std_range_5 = range_hl.rolling(short_window).std(ddof=0)
        std_ret_10 = returns_ser.rolling(long_window).std(ddof=0)
        
        # 3. Tính Volatility Ratio
        vol_ratio = std_range_5 / (std_ret_10 + 1e-8)
        
        
        condition_signal = np.where(returns < 0, vol_ratio, -vol_ratio)
        
        
        raw_signal = pd.Series(condition_signal, index=df.index).fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_017(df: pd.DataFrame, delta_short=1, delta_mid=3, rank_window=15):
        
        delta_1 = df['close'].diff(delta_short)
        delta_3 = df['close'].diff(delta_mid)
        
        rank_delta_1 = delta_1.rolling(rank_window).rank(pct=True)
        rank_delta_3 = delta_3.rolling(rank_window).rank(pct=True)
        
      
        rank_diff = rank_delta_3 - rank_delta_1
        
        raw_signal = -1 * rank_diff.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_018(df: pd.DataFrame, ema_window=10, delay_step=2, std_window=20):
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        ema_ret = returns_ser.ewm(span=ema_window, adjust=False).mean()
        
        delta_ema = ema_ret - ema_ret.shift(delay_step)
        
        std_ema = ema_ret.rolling(std_window).std(ddof=0)
        
        raw_ratio = delta_ema / (std_ema + 1e-8)
        
        
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_019(df: pd.DataFrame, ema_window=8, slope_window=5, corr_window=10):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        price_vwap_ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(slope_window))
        price_slope = df['close'].rolling(slope_window).cov(ser_index) / (x_var + 1e-8)
       
        correlation = price_vwap_ratio.rolling(corr_window).corr(price_slope)
        
        raw_signal = correlation.fillna(0)
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_020(df: pd.DataFrame, ema_window=5, ret_window=2, corr_window=8):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        price_vwap_ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
       
        returns_2 = df['close'].pct_change(ret_window)
        
        correlation = price_vwap_ratio.rolling(corr_window).corr(returns_2)
        
        raw_signal = -1 * correlation.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_021(df: pd.DataFrame, reg_window=15, delay_step=3, std_window=20):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
      
        delta_slope = price_slope - price_slope.shift(delay_step)
        
        std_slope = price_slope.rolling(std_window).std(ddof=0)
        
        raw_ratio = delta_slope / (std_slope + 1e-8)
        
        
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_022(df: pd.DataFrame, std_window=8, mean_window=32, moment_window=15):
       
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        vola_8 = returns_ser.rolling(std_window).std(ddof=0)
        vola_mean_32 = vola_8.rolling(mean_window).mean()
        
        skew_15 = returns_ser.rolling(moment_window).skew()
        kurt_15 = returns_ser.rolling(moment_window).kurt()
        
        rank_skew = O.ts_rank_normalized(skew_15, 20)
        rank_kurt = O.ts_rank_normalized(kurt_15, 20)
        
      
        condition = vola_8 > vola_mean_32
        signal_raw = np.where(condition, -rank_skew, rank_kurt)
        
        raw_series = pd.Series(signal_raw, index=df.index).fillna(0)
        ranked_signal = O.ts_rank_normalized(raw_series, 20)
        
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_023(df: pd.DataFrame, ret_short=1, ret_mid=3, rank_window=12):
        ret_1 = df['close'].pct_change(ret_short)
        ret_3 = df['close'].pct_change(ret_mid)
      
        rank_ret_1 = ret_1.rolling(rank_window).rank(pct=True)
        rank_ret_3 = ret_3.rolling(rank_window).rank(pct=True)
        
        
        rank_diff = rank_ret_3 - rank_ret_1
        
        raw_signal = rank_diff.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_024(df: pd.DataFrame, ema_window=6, amt_reg_window=8, corr_window=10):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        price_vwap_ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
        
        amt = df.get('amount', df['close'] * df['matchingVolume'])
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(amt_reg_window))
        amt_slope = amt.rolling(amt_reg_window).cov(ser_index) / (x_var + 1e-8)
        
        correlation = price_vwap_ratio.rolling(corr_window).corr(amt_slope)
        
       
        raw_signal = correlation.fillna(0)
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_025(df: pd.DataFrame, range_std_short=5, range_std_long=20):
        
        range_hl = df['high'] - df['low']
        std_range_5 = range_hl.rolling(range_std_short).std(ddof=0)
        mean_std_range_20 = std_range_5.rolling(range_std_long).mean()
        
        rev_delta = -1 * df['close'].diff(2)
        
        trend_ret = df['close'].pct_change(1).rolling(10).rank(pct=True)
        
        
        condition = std_range_5 < mean_std_range_20
        
        rank_rev = O.ts_rank_normalized(rev_delta.fillna(0), 20)
        rank_trend = O.ts_rank_normalized(trend_ret.fillna(0), 20)
        
        raw_signal = np.where(condition, rank_rev, rank_trend)
        
        ranked_signal = O.ts_rank_normalized(pd.Series(raw_signal, index=df.index), 20)
        
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_026(df: pd.DataFrame, reg_window=5, delay_step=1, std_window=20):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        delta_slope = price_slope - price_slope.shift(delay_step)
        
        std_slope = price_slope.rolling(std_window).std(ddof=0)
        
        raw_ratio = delta_slope / (std_slope + 1e-8)
        
       
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_factor_miner_027(df: pd.DataFrame, ema_short=5, ema_long=10, rank_window=12):
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        ema_5 = returns_ser.ewm(span=ema_short, adjust=False).mean()
        ema_10 = returns_ser.ewm(span=ema_long, adjust=False).mean()
        
        
        rank_ema_5 = ema_5.rolling(rank_window).rank(pct=True)
        rank_ema_10 = ema_10.rolling(rank_window).rank(pct=True)
        
       
        raw_signal = rank_ema_5 - rank_ema_10
        
        ranked_signal = O.ts_rank_normalized(raw_signal.fillna(0), 20)
        
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_028(df: pd.DataFrame, vwap_delta=1, reg_window=10):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        ratio = df['close'] / (vwap + 1e-8)
        
        vwap_delta_val = vwap.diff(vwap_delta)
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        ratio_slope = ratio.rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
      
        rank_slope = O.ts_rank_normalized(ratio_slope.fillna(0), reg_window)
        
        condition_signal = np.where(vwap_delta_val < 0, -rank_slope, rank_slope)
        
        
        raw_signal = -1 * pd.Series(condition_signal, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, reg_window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_029(df: pd.DataFrame, range_std=10, ret_std=15, threshold=0.8):
        range_hl = df['high'] - df['low']
        std_range_3 = range_hl.rolling(range_std).std(ddof=0)
        
        # Dùng pct_change để giữ nguyên index và độ dài
        returns_ser = df['close'].pct_change(1).fillna(0)
        std_ret_10 = returns_ser.rolling(ret_std).std(ddof=0)
        
        # Tránh chia cho 0
        vol_ratio = std_range_3 / (std_ret_10 + 1e-8)
        
       
        rev_delta = -df['close'].diff(1).fillna(0)
        
        # Thuận xu hướng VWAP Ratio (8 phiên)
        vwap = (df['high'] + df['low'] + df['close']) / 3
        trend_vwap_ratio = df['close'] / (vwap + 1e-8)
        
        
        rank_rev = O.ts_rank_normalized(rev_delta, ret_std)
        rank_trend = O.ts_rank_normalized(trend_vwap_ratio, range_std)
        
        condition = vol_ratio < threshold
        
        raw_signal = rank_trend.copy()
        raw_signal[condition] = rank_rev[condition]
        
        ranked_signal = O.ts_rank_normalized(raw_signal, ret_std)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_030(df: pd.DataFrame, vwap_delta=3, reg_window=12):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        ratio = df['close'] / (vwap + 1e-8)
        
        vwap_trending_up = vwap.diff(vwap_delta) > 0
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        ratio_slope = ratio.rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
   
        conditional_slope = np.where(vwap_trending_up, ratio_slope, -ratio_slope)
        
       
        raw_signal = -1 * pd.Series(conditional_slope, index=df.index).fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_031(df: pd.DataFrame, reg_window=18, delay_step=3):
        def get_last_resid(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            line = slope * (len(y) - 1) + intercept
            return y[-1] - line

        resid = df['close'].rolling(window=reg_window).apply(get_last_resid, raw=True)
        
        
        delta_resid = resid - resid.shift(delay_step)
        
        
        raw_signal = -1 * delta_resid.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_032(df: pd.DataFrame, corr_window=15, beta_window=12):
        volume = df['matchingVolume']
        returns = df['close'].pct_change(1).fillna(0)
        
        pv_corr = returns.rolling(corr_window).corr(volume)
        
        
        amt = df.get('amount', df['close'] * volume)
        
        avg_price = amt / (volume + 1e-8)
        avg_price_ret = avg_price.pct_change(1).fillna(0)
        
        covariance = returns.rolling(beta_window).cov(avg_price_ret)
        variance = avg_price_ret.rolling(beta_window).var()
        beta_val = covariance / (variance + 1e-8)
        
        condition = pv_corr >= 0
        raw_signal = np.where(condition, -beta_val, beta_val)
        
        raw_series = pd.Series(raw_signal, index=df.index).fillna(0)
        ranked_signal = O.ts_rank_normalized(raw_series, 20)
        
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_033(df: pd.DataFrame, reg_window=12, delay_step=2, std_window=20):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        delta_slope = price_slope - price_slope.shift(delay_step)
        
        std_slope = price_slope.rolling(std_window).std(ddof=0)
        
        raw_ratio = delta_slope / (std_slope + 1e-8)
        
        
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_034(df: pd.DataFrame, reg_window=16, delay_step=4):
        def get_last_resid(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            # Tính giá trị dự báo tại điểm cuối cùng
            line_val = slope * (len(y) - 1) + intercept
            return y[-1] - line_val

        # Sử dụng rolling apply để tính phần dư tại mỗi bước
        resid = df['close'].rolling(window=reg_window).apply(get_last_resid, raw=True)
        
        # 2. Tính sự thay đổi của Residual sau 4 phiên
        delta_resid = resid - resid.shift(delay_step)
        
        
        raw_signal = -1 * delta_resid.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_035(df: pd.DataFrame, reg_window=14, delay_step=4, std_window=18):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        delta_slope = price_slope - price_slope.shift(delay_step)
        
        std_slope = price_slope.rolling(std_window).std(ddof=0)
        
        raw_ratio = delta_slope / (std_slope + 1e-8)
        
       
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_036(df: pd.DataFrame, ema_window=10, reg_window=8, corr_window=12):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
        
        # 2. Tính Slope của giá (8 phiên)
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        
        correlation = ratio.rolling(corr_window).corr(price_slope)
        
        raw_signal = correlation.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_037(df: pd.DataFrame, ema_window=8, delay_step=3, std_fast=15, std_slow=30):
        
        returns = df['close'].pct_change(1).fillna(0)
        ema_ret = returns.ewm(span=ema_window, adjust=False).mean()
        
        mom_ema_ret = ema_ret - ema_ret.shift(delay_step)
        
       
        market_vol = returns.rolling(5).std(ddof=0)
        
        std_ema_fast = ema_ret.rolling(std_fast).std(ddof=0)
        std_ema_slow = ema_ret.rolling(std_slow).std(ddof=0)
        
        dynamic_std = np.where(market_vol > 0.02, std_ema_fast, std_ema_slow)
        
        raw_ratio = mom_ema_ret / (dynamic_std + 1e-8)
        
        
        raw_signal = -1 * pd.Series(raw_ratio, index=df.index).fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_038(df: pd.DataFrame, ema_window=8, reg_window=12, delay_step=3, std_window=18):
        smoothed_price = df['close'].ewm(span=ema_window, adjust=False).mean()
        
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        slope_smoothed = smoothed_price.rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        accel_slope = slope_smoothed - slope_smoothed.shift(delay_step)
        
        std_slope = slope_smoothed.rolling(std_window).std(ddof=0)
        
        raw_ratio = accel_slope / (std_slope + 1e-8)
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_039(df: pd.DataFrame, ema_window=6, delta_step=2, corr_window=10):
        
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
        
        # 2. Tính Delta của Returns (2 phiên)
        returns = df['close'].pct_change(1).fillna(0)
        delta_ret = returns.diff(delta_step).fillna(0)
        
        
        correlation = ratio.rolling(corr_window).corr(delta_ret)
        
        
        raw_signal = -1 * correlation.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_040(df: pd.DataFrame, vol_window=5, reg_window=12, delay_step=3, std_window=18):
      
        returns = df['close'].pct_change(1).fillna(0)
        vol = returns.rolling(vol_window).std(ddof=0)
        
        # 2. Tính Slope của Biến động (12 phiên)
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        vol_slope = vol.rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        # 3. Tính Gia tốc Biến động (Sự thay đổi Slope sau 3 phiên)
        accel_vol = vol_slope - vol_slope.shift(delay_step)
        
        # 4. Chuẩn hóa rủi ro bằng Std của chính Vol Slope (18 phiên)
        std_vol_slope = vol_slope.rolling(std_window).std(ddof=0)
        
        # 5. Tính tỷ lệ chuẩn hóa và áp dụng Neg
        raw_ratio = accel_vol / (std_vol_slope + 1e-8)
        raw_signal = -1 * raw_ratio.fillna(0)
        
        # 6. Chuẩn hóa Rank
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_041(df: pd.DataFrame, ema_window=6, delay_step=2, std_window=20):
        
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        ratio = (df['close'] / (vwap + 1e-8)).ewm(span=ema_window, adjust=False).mean()
        
        mom_ratio = ratio - ratio.shift(delay_step)
        
        std_ratio = ratio.rolling(std_window).std(ddof=0)
        
        
        raw_signal = mom_ratio / (std_ratio + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_signal.fillna(0), std_window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_042(df: pd.DataFrame, rank_window=18):
       
        ret2 = df['close'].pct_change(2).fillna(0)
        ret5 = df['close'].pct_change(5).fillna(0)
        
        # 2. Tính TsRank trong 18 phiên (Vị thế của lợi nhuận hiện tại so với quá khứ)
        rank_ret2 = ret2.rolling(rank_window).rank(pct=True)
        rank_ret5 = ret5.rolling(rank_window).rank(pct=True)
        
        
        raw_signal = (rank_ret2 - rank_ret5).fillna(0)
        
        # 4. Chuẩn hóa về [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_043(df: pd.DataFrame, ema_window=6, delay_step=3, std_window=14):
       
        ema_price = df['close'].ewm(span=ema_window, adjust=False).mean()
        
        velocity_ema = ema_price - ema_price.shift(delay_step)
   
        std_velocity = velocity_ema.rolling(std_window).std(ddof=0)
        
        # 4. Tỷ lệ chuẩn hóa và áp dụng Neg (Đánh đảo chiều)
        raw_ratio = velocity_ema / (std_velocity + 1e-8)
        raw_signal = -1 * raw_ratio.fillna(0)
        
        # 5. CsRank chuẩn hóa về [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_044(df: pd.DataFrame, rank_window=15):
        
        ret2 = df['close'].pct_change(2).fillna(0)
        ret7 = df['close'].pct_change(7).fillna(0)
        
        rank_ret2 = ret2.rolling(rank_window).rank(pct=True)
        rank_ret7 = ret7.rolling(rank_window).rank(pct=True)
        
        raw_signal = (rank_ret2 - rank_ret7).fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_045(df: pd.DataFrame, reg_window=12, delay_step=4, std_window=22):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        price_slope = df['close'].rolling(reg_window).cov(ser_index) / (x_var + 1e-8)
        
        # 2. Tính sự thay đổi (Delta) của Slope sau 4 phiên
        delta_slope = price_slope - price_slope.shift(delay_step)
        
        # 3. Chuẩn hóa rủi ro bằng Std của Slope trong 22 phiên
        # Cửa sổ 22 giúp làm mượt tín hiệu cực tốt so với các bản 15, 18 phiên trước đó.
        std_slope = price_slope.rolling(std_window).std(ddof=0)
        
        # 4. Tính tỷ lệ Risk-Adjusted và Neg (Đánh đảo chiều)
        raw_ratio = delta_slope / (std_slope + 1e-8)
        raw_signal = -1 * raw_ratio.fillna(0)
        
        # 5. CsRank chuẩn hóa
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_046(df: pd.DataFrame, ret_window=1, ema_window=6, delay_step=3, std_window=14):
       
        returns = df['close'].pct_change(ret_window).fillna(0)
        ema_ret = returns.ewm(span=ema_window, adjust=False).mean()
        
        # 2. Tính Vận tốc (Momentum) của EMA Returns sau 3 phiên
        mom_ema_ret = ema_ret - ema_ret.shift(delay_step)
        
        # 3. Chuẩn hóa rủi ro bằng Std của chính EMA Returns (14 phiên)
        std_ema_ret = ema_ret.rolling(std_window).std(ddof=0)
        
        # 4. Tín hiệu thuận xu hướng (do 2 lần Neg triệt tiêu)
        raw_signal = mom_ema_ret / (std_ema_ret + 1e-8)
        
        # 5. CsRank chuẩn hóa
        ranked_signal = O.ts_rank_normalized(raw_signal.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_047(df: pd.DataFrame, w_fast=10, w_slow=30):
       
        hl_spread = df['high'] - df['low']
        vol_hl = hl_spread.rolling(w_fast).std(ddof=0)
        
        # 2. Điều kiện Regime: Check đáy biến động trong chu kỳ w_slow
        is_low_vol = vol_hl <= vol_hl.rolling(w_slow).min()
        
        # 3. Nhánh Momentum (Dùng bước nhảy cố định là 2 để giữ độ nhạy)
        sig_mom = -1 * (df['close'].diff(2).fillna(0))
        sig_mom_ranked = sig_mom.rolling(w_slow).rank(pct=True) * 2 - 1
        
        # 4. Nhánh Regression (Dùng w_fast để tính độ dốc)
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(w_fast))
        sig_reg = df['close'].rolling(w_fast).cov(ser_index) / (x_var + 1e-8)
        sig_reg_ranked = sig_reg.rolling(w_slow).rank(pct=True) * 2 - 1
        
        # 5. Kết hợp và chuẩn hóa cuối cùng
        raw_signal = np.where(is_low_vol, sig_mom_ranked, sig_reg_ranked)
        final_ranked = pd.Series(raw_signal, index=df.index).rolling(w_slow).rank(pct=True)
        
        return -1 * (2 * final_ranked - 1).fillna(0)

    @staticmethod
    def alpha_factor_miner_048(df: pd.DataFrame, w_fast=12, w_slow=20):
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(w_fast))
        price_slope = df['close'].rolling(w_fast).cov(ser_index) / (x_var + 1e-8)
        
        # 2. Tính gia tốc (Acceleration) 
        # Độ trễ được fix theo tỷ lệ 1/3 của w_fast để giảm tham số
        delay_step = max(1, w_fast // 3) 
        accel_slope = price_slope - price_slope.shift(delay_step)
        
        # 3. Chuẩn hóa rủi ro bằng Std của Delta giá (dùng w_slow)
        # Delta($close, 1) chính là biến động nến đơn lẻ
        price_delta = df['close'].diff(1).fillna(0)
        std_vol = price_delta.rolling(w_slow).std(ddof=0)
        
        # 4. Tính tín hiệu và Neg (Đánh đảo chiều)
        raw_ratio = accel_slope / (std_vol + 1e-8)
        raw_signal = -1 * raw_ratio.fillna(0)
        
        # 5. Chuẩn hóa Rank cuối cùng theo w_slow
        ranked_signal = raw_signal.rolling(w_slow).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_049(df: pd.DataFrame, window=5):
        if 'vwap' not in df.columns:
            vwap = (df['high'] + df['low'] + df['close']) / 3
        else:
            vwap = df['vwap']
            
        ema_vwap = vwap.ewm(span=window, adjust=False).mean()
      
        relative_diff = (df['close'] - ema_vwap) / (ema_vwap.abs() + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(relative_diff.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_050(df: pd.DataFrame, window=30):
        vol_std = O.ts_std(df['matchingVolume'], window)
    
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        avg_amt = O.ts_mean(amt, window)
        
        ratio = vol_std / (avg_amt + 1e-8)
        
        z_signal = O.zscore(ratio, window)
        
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_051(df: pd.DataFrame, reg_window=25, delta_step=5):
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
       
        # Để tối ưu tính toán rolling, ta dùng công thức: Slope = Cov(y, x) / Var(x)
        ser_index = pd.Series(np.arange(len(df)), index=df.index)
        x_var = np.var(np.arange(reg_window))
        
        # Dùng O.ts_cov từ class của bạn
        amt_slope = O.ts_cov(amt, ser_index, reg_window) / (x_var + 1e-8)
        
        accel_amt = O.ts_delta(amt_slope, delta_step)
        
      
        z_accel = O.zscore(accel_amt.fillna(0), 20)
        raw_signal = -1 * z_accel.fillna(0)
        
        # 5. Đưa về [-1, 1] qua O.ts_rank_normalized
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_052(df: pd.DataFrame, skew_window=40):
        returns = df['close'].pct_change(1).fillna(0)
        
        skewness = returns.rolling(skew_window).skew().fillna(0)
        
        rank_ret = O.ts_rank_normalized(returns, 20)
        
       
        condition = skewness > 0
        raw_signal = np.where(condition, -1 * rank_ret, rank_ret)
        
        final_raw = -1 * pd.Series(raw_signal, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(final_raw.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_053(df: pd.DataFrame, corr_window=12, lag_step=3):
        df_vwap = O.compute_vwap(df.copy(), window=100)
        vwap = df_vwap['vwap']
        
        delayed_close = O.ts_lag(df['close'], lag_step)
        
        correlation = O.ts_corr(delayed_close, vwap, corr_window)
        
        raw_signal = correlation.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_054(df: pd.DataFrame, window=5):
        low_min = O.ts_min(df['low'], window)
        high_max = O.ts_max(df['high'], window)
        
        price_range = high_max - low_min
        
        relative_position = (df['close'] - low_min) / (price_range + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(relative_position.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_055(df: pd.DataFrame, corr_window=20, delta_step=3):
        vol = df['matchingVolume']
        amt = df['amt'] if 'amt' in df.columns else df['close'] * vol
        pv_corr = O.ts_corr(vol, amt, corr_window)
        
        delta_corr = O.ts_delta(pv_corr, delta_step)
        
        z_signal = O.zscore(delta_corr.fillna(0), 20)
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_056(df: pd.DataFrame, skew_window=40):
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        
        amt_skew = amt.rolling(skew_window).skew().fillna(0)
        
        rank_amt = O.ts_rank_normalized(amt, 20)
        
        condition = amt_skew > 0
        raw_signal = np.where(condition, rank_amt, -1 * rank_amt)
        
        final_raw = -1 * pd.Series(raw_signal, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(final_raw.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_057(df: pd.DataFrame, window=10):
        def calculate_hma(series, n):
            wma_half = O.ts_weighted_mean(series, max(1, int(n/2)))
            wma_full = O.ts_weighted_mean(series, n)
            
            raw_hma = 2 * wma_half - wma_full
            
            sqrt_n = max(1, int(np.sqrt(n)))
            return O.ts_weighted_mean(raw_hma, sqrt_n)

        hma_close = calculate_hma(df['close'], window)
        

        relative_diff = (df['close'] - hma_close) / (hma_close.abs() + 1e-8)
        
        z_signal = O.zscore(relative_diff.fillna(0), 20)
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_058(df: pd.DataFrame, kurt_window=50):
        vol = df['matchingVolume']
        v_kurt = vol.rolling(kurt_window).kurt().fillna(0)
        
        vol_log = O.log(vol) # Nén mạnh các giá trị cực đại
        vol_sqrt = O.power(vol, 0.5) # Nén vừa phải
        
        condition = v_kurt > 3
        raw_vol_transformed = np.where(condition, vol_log, vol_sqrt)
        
        final_raw = -1 * pd.Series(raw_vol_transformed, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(final_raw.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_059(df: pd.DataFrame, window=20):
        body = df['close'] - df['open']
        
        full_range = (df['high'] - df['low']).abs() + 0.001
        
        efficiency_ratio = body / full_range
        
        raw_signal = -1 * efficiency_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_060(df: pd.DataFrame, skew_window=15):
        df_vwap = O.compute_vwap(df.copy(), window=skew_window * 2)
        vwap = df_vwap['vwap']
        
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        
        relative_ratio = (amt - vwap) / (vwap.abs() + 1e-8)
        
        skew_signal = relative_ratio.rolling(skew_window).skew().fillna(0)
        
        ranked_signal = O.ts_rank_normalized(skew_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0) 

    @staticmethod
    def alpha_factor_miner_061(df: pd.DataFrame, window=25):
        hl_range = (df['high'] - df['low']).abs()
        x = np.arange(window)
        x_mean = np.mean(x)
        x_var = np.var(x)
        
        def get_resid(y_slice):
            if len(y_slice) < window: return 0
            y_mean = np.mean(y_slice)
            beta = np.cov(x, y_slice)[0, 1] / (x_var + 1e-8)
            alpha = y_mean - beta * x_mean
            current_y = y_slice[-1]
            current_x = x[-1]
            return current_y - (alpha + beta * current_x)

        resid = hl_range.rolling(window).apply(get_resid, raw=True)
        
        raw_signal = -1 * resid.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_062(df: pd.DataFrame, std_window=10, beta_window=20):
        df_vwap = O.compute_vwap(df.copy(), window=30)
        vwap = df_vwap['vwap']
        
        returns = df['close'].pct_change(1).fillna(0)
        volatility = O.ts_std(returns, std_window).fillna(0)
 
        cov_y_x = O.ts_cov(vwap, volatility, beta_window)
        var_x = O.ts_std(volatility, beta_window) ** 2
        
        beta_vwap_vol = cov_y_x / (var_x + 1e-8)
        
        z_signal = O.zscore(beta_vwap_vol.fillna(0), 20)
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_063(df: pd.DataFrame, hma_window=12, corr_window=18):
        def calculate_hma(series, n):
            wma_half = O.ts_weighted_mean(series, max(1, int(n/2)))
            wma_full = O.ts_weighted_mean(series, n)
            raw_hma = 2 * wma_half - wma_full
            sqrt_n = max(1, int(np.sqrt(n)))
            return O.ts_weighted_mean(raw_hma, sqrt_n)

        vol = df['matchingVolume']
        amt = df['amt'] if 'amt' in df.columns else df['close'] * vol
        
        hma_amt = calculate_hma(amt, hma_window)
        hma_vol = calculate_hma(vol, hma_window)
        
        pv_corr = O.ts_corr(hma_amt, hma_vol, corr_window)
        
       
        raw_signal = -1 * pv_corr.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_064(df: pd.DataFrame, q_window=50, quantile_level=0.75):
        price_quantile = df['close'].rolling(q_window).quantile(quantile_level)
        
        deviation = df['close'] - price_quantile
        
        z_signal = O.zscore(deviation.fillna(0), 20)
        
        raw_signal = z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_065(df: pd.DataFrame, std_window=15, beta_window=25):
        df_vwap = O.compute_vwap(df.copy(), window=40)
        vwap = df_vwap['vwap']
        
        returns = df['close'].pct_change(1).fillna(0)
        volatility = O.ts_std(returns, std_window).fillna(0)
        
        # 3. Tính Beta giữa VWAP (y) và Volatility (x) trong 25 phiên
        cov_y_x = O.ts_cov(vwap, volatility, beta_window)
        var_x = O.ts_std(volatility, beta_window) ** 2
        
        beta_signal = cov_y_x / (var_x + 1e-8)
        
        # 4. Áp dụng Neg và CsZScore (qua O.zscore)
        z_signal = O.zscore(beta_signal.fillna(0), 20)
        raw_signal = -1 * z_signal.fillna(0)
        
        # 5. Đưa về [-1, 1] qua O.ts_rank_normalized
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_066(df: pd.DataFrame, hma_window=14, corr_window=20):
      
        def calculate_hma(series, n):
            wma_half = O.ts_weighted_mean(series, max(1, int(n/2)))
            wma_full = O.ts_weighted_mean(series, n)
            raw_hma = 2 * wma_half - wma_full
            sqrt_n = max(1, int(np.sqrt(n)))
            return O.ts_weighted_mean(raw_hma, sqrt_n)

        # 1. Chuẩn bị dữ liệu
        vol = df['matchingVolume']
        amt = df['amt'] if 'amt' in df.columns else df['close'] * vol
        
        # 2. Làm mượt bằng HMA (window 14)
        hma_amt = calculate_hma(amt, hma_window)
        hma_vol = calculate_hma(vol, hma_window)
        
        # 3. Tính tương quan rolling (window 20)
        pv_corr = O.ts_corr(hma_amt, hma_vol, corr_window)
        
        # 4. Áp dụng Neg và CsRank (qua O.ts_rank_normalized)
        raw_signal = -1 * pv_corr.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_067(df: pd.DataFrame, q_window=5, quantile_level=0.8):
        upper_quantile = df['close'].rolling(q_window).quantile(quantile_level)
        
        deviation = df['close'] - upper_quantile
        
        z_signal = O.zscore(deviation.fillna(0), q_window)
        
        raw_signal = z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, q_window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_068(df: pd.DataFrame, ema_window=22, kurt_window=40):
        vol = df['matchingVolume']
        
        vol_ema = vol.ewm(span=ema_window, adjust=False).mean()
        
        vol_ratio = vol / (vol_ema + 1e-8)
        
        kurt_signal = vol_ratio.rolling(kurt_window).kurt().fillna(0)
        
        ranked_signal = O.ts_rank_normalized(kurt_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_069(df: pd.DataFrame, return_lag=3, cov_window=12):
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
       
        log_return = np.log(df['close'] / df['close'].shift(return_lag)).fillna(0)
        
        cov_amt_return = O.ts_cov(amt, log_return, cov_window)
        
       
        raw_signal = -1 * cov_amt_return.fillna(0)
        
        # 5. Chuẩn hóa về [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_070(df: pd.DataFrame, window=15):
        df_vwap = O.compute_vwap(df.copy(), window=window*2)
        vwap = df_vwap['vwap']
        
        low_min = O.ts_min(df['low'], window)
        high_max = O.ts_max(df['high'], window)
        
        range_width = high_max - low_min
        relative_position = (vwap - low_min) / (range_width + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(relative_position.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_071(df: pd.DataFrame, window=15):
        def get_slope(series, n):
            x = np.arange(n)
            x_mean = np.mean(x)
            x_var = np.var(x)
            
            def calc_slope(y_slice):
                if len(y_slice) < n: return 0
                y_mean = np.mean(y_slice)
                return np.cov(x, y_slice)[0, 1] / (x_var + 1e-8)
            
            return series.rolling(n).apply(calc_slope, raw=True)

        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        
        slope_close = get_slope(df['close'], window)
        slope_amt = get_slope(amt, window)
        
        divergence = slope_close - slope_amt
        
        ranked_signal = O.ts_rank_normalized(divergence.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_072(df: pd.DataFrame, vol_window=8, skew_window=16):
        returns = df['close'].pct_change(1).fillna(0)
        vol = O.ts_std(returns, vol_window)
        
        vol_threshold = vol.rolling(50).quantile(0.75)
        
        skew_returns = returns.rolling(skew_window).skew().fillna(0)
        
        condition = vol > vol_threshold
        raw_signal = np.where(condition, skew_returns, -1 * skew_returns)
        
        final_raw = -1 * pd.Series(raw_signal, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(final_raw.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_073(df: pd.DataFrame, vol_window=10, slope_window=15):
        returns = df['close'].pct_change(1).fillna(0)
        vol_current = returns.rolling(vol_window).std().fillna(0)
        vol_delayed = vol_current.shift(5).fillna(0)
       
        price_diff_2 = df['close'].diff(2).fillna(0)
        reversal_signal = -1 * O.ts_rank_normalized(price_diff_2, 20)
        
        def get_slope(series, n):
            x = np.arange(n)
            x_var = np.var(x)
            def calc(y):
                if len(y) < n: return 0
                return np.cov(x, y)[0, 1] / (x_var + 1e-8)
            return series.rolling(n).apply(calc, raw=True)
            
        momentum_signal = O.ts_rank_normalized(get_slope(df['close'], slope_window).fillna(0), 20)
        
        condition = vol_current > vol_delayed
        raw_signal = np.where(condition, reversal_signal, momentum_signal)
        
        final_series = pd.Series(raw_signal, index=df.index)
        signal = (2 * O.ts_rank_normalized(final_series, 20)) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_074(df: pd.DataFrame, vol_window=12, skew_window=18, slope_window=10):
        returns = df['close'].pct_change(1).fillna(0)
        vol = returns.rolling(vol_window).std().fillna(0)
        
        vol_threshold = vol.rolling(30).quantile(0.8)
        
        skew_signal = -1 * returns.rolling(skew_window).skew().fillna(0)
        
        def get_slope(series, n):
            x = np.arange(n)
            x_var = np.var(x)
            def calc(y):
                if len(y) < n: return 0
                return np.cov(x, y)[0, 1] / (x_var + 1e-8)
            return series.rolling(n).apply(calc, raw=True)
            
        slope_signal = get_slope(df['close'], slope_window).fillna(0)
        
        # 5. Logic IfElse và chuẩn hóa Rank
        condition = vol > vol_threshold
        raw_signal = np.where(condition, skew_signal, slope_signal)
        
        # Đưa về [-1, 1]
        final_series = pd.Series(raw_signal, index=df.index)
        ranked_signal = O.ts_rank_normalized(final_series, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_075(df: pd.DataFrame, short_window=8, long_window=24):
        hl_range = (df['high'] - df['low']).abs()
        
        std_short = hl_range.rolling(short_window).std().fillna(0)
        std_long = hl_range.rolling(long_window).std().fillna(0)
        
        vol_ratio = std_short / (std_long + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(vol_ratio.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_076(df: pd.DataFrame,window = 20, delta=1):
     
        df_vwap = O.compute_vwap(df.copy(), window)
        vwap = df_vwap['vwap']
        
        price_to_vwap_ratio = df['close'] / (vwap + 1e-8)
       
        ratio_delta = price_to_vwap_ratio.diff(delta).fillna(0)
    
        raw_signal = O.ts_rank_normalized(ratio_delta, window)
        
        signal = (2 * raw_signal) - 1
        
        return signal.fillna(0)


    # print(f'signal min: {signal.min()}, signal max: {signal.max()}')

    @staticmethod
    def alpha_factor_miner_077(df: pd.DataFrame, ema_window=30, zscore_window=10):
       
        hl_range = (df['high'] - df['low']).abs()
        
        hl_ema = hl_range.ewm(span=ema_window, adjust=False).mean()
        
      
        compression_ratio = hl_range / (hl_ema + 1e-8)
        
        z_signal = O.zscore(compression_ratio.fillna(0), zscore_window)
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_078(df: pd.DataFrame, window=20):
        
        returns = df['close'].pct_change(1).fillna(0)
        vol = df['matchingVolume']
        amt = df['amt'] if 'amt' in df.columns else df['close'] * vol
       
        cov_amt_ret = O.ts_cov(amt, returns, window)
        var_ret = returns.rolling(window).var().fillna(0)
        
        beta_signal = cov_amt_ret / (var_ret + 1e-8)
        
        
        raw_signal = -1 * beta_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_079(df: pd.DataFrame, window=40):
       
        returns = df['close'].pct_change(1).fillna(0)
       
        skew_val = returns.rolling(window).skew().fillna(0)
        
        std_val = returns.rolling(window).std().fillna(0)
        vol_weight = np.sqrt(std_val)
        
        raw_signal = skew_val * vol_weight
        
       
        final_raw = -1 * raw_signal
        
        ranked_signal = O.ts_rank_normalized(final_raw, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_080(df: pd.DataFrame, window=20):
        y = df['close']
        x = df['matchingVolume']
        def get_residual(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / (np.var(x_slice) + 1e-8)
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

       
        residual = pd.Series([
            get_residual(y.values[i-window:i], x.values[i-window:i]) 
            if i >= window else 0 
            for i in range(len(y))
        ], index=df.index)
        
       
        z_signal = O.zscore(residual.fillna(0), window)
        raw_signal = -1 * z_signal
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_081(df: pd.DataFrame, window=20, delta=3):
        price_diff = df['close'].diff(delta).fillna(0)
        hl_range = (df['high'] - df['low']).abs()
        
        def calculate_kama_fast(series, n):
            vals = series.values
            n_rows = len(vals)
            
            change = np.abs(series.diff(n).values)
            volatility = series.diff().abs().rolling(n).sum().values
            
            er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility > 1e-6)
            
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
            
            kama = np.zeros(n_rows)
            if n_rows >= n:
                kama[n-1] = np.mean(vals[:n]) 
                for i in range(n, n_rows):
                    kama[i] = kama[i-1] + sc[i] * (vals[i] - kama[i-1])
            
            return pd.Series(kama, index=series.index)

        kama_range = calculate_kama_fast(hl_range, window)
        
        breakout_signal = price_diff / (kama_range + 1e-6)
        
        ranked_signal = O.ts_rank_normalized(breakout_signal.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_082(df: pd.DataFrame, q_window=60, rank_window=20):
        returns = df['close'].pct_change(1).fillna(0)
        quantile_75 = returns.rolling(q_window).quantile(0.75).fillna(0)
        
        price_ts_rank = O.ts_rank_normalized(df['close'], rank_window)
        
        condition = quantile_75 > 0
        inner_signal = np.where(condition, price_ts_rank, -1 * price_ts_rank)
        
        raw_signal = -1 * O.ts_rank_normalized(pd.Series(inner_signal, index=df.index), 20)
        
        signal = (2 * raw_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_083(df: pd.DataFrame, vol_q_window=50, vol_m_window=20, slope_window=15):
        vol = df['matchingVolume']
        vol_q80 = vol.rolling(vol_q_window).quantile(0.8).fillna(0)
        vol_mean20 = vol.rolling(vol_m_window).mean().fillna(0)
        
        def get_slope(series, n):
            x = np.arange(n)
            x_var = np.var(x)
            def calc(y):
                if len(y) < n: return 0
                # Beta = Cov(x, y) / Var(x)
                return np.cov(x, y)[0, 1] / (x_var + 1e-6)
            return series.rolling(n).apply(calc, raw=True)
            
        slope = get_slope(df['close'], slope_window).fillna(0)
        
        condition = vol_q80 > vol_mean20
        inner_signal = np.where(condition, slope, -1 * slope)
        
        z_signal = O.zscore(pd.Series(inner_signal, index=df.index), 20)
        raw_signal = -1 * z_signal.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_084(df: pd.DataFrame, amt_std_window=40, range_window=20, ema_window=40):
        
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        amt_std = amt.rolling(amt_std_window).std().fillna(0)
        
        # 2. Tính biên độ cực đại trong chu kỳ (TsMax High - TsMin Low)
        ts_max_h = df['high'].rolling(range_window).max()
        ts_min_l = df['low'].rolling(range_window).min()
        price_range = (ts_max_h - ts_min_l).fillna(0)
        
        smooth_range = price_range.ewm(span=ema_window, adjust=False).mean()
        
      
        ratio = amt_std / (smooth_range + 1e-6)
        
        ranked_signal = O.ts_rank_normalized(ratio.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_085(df: pd.DataFrame, window=25):
        body = df['close'] - df['open']
        full_range = df['high'] - df['low']
        efficiency = body / (full_range + 1e-6)
        
        y = efficiency.values
        x = df['matchingVolume'].values
        
        def get_residual_fast(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            # Beta = Cov(x, y) / Var(x)
            x_var = np.var(x_slice)
            if x_var < 1e-6: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / x_var
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

        residuals = pd.Series([
            get_residual_fast(y[i-window:i], x[i-window:i]) 
            if i >= window else 0 
            for i in range(len(y))
        ], index=df.index)
        
       
        z_signal = O.zscore(residuals.fillna(0), window)
        raw_signal = -1 * z_signal
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)


    @staticmethod
    def alpha_factor_miner_086(df: pd.DataFrame, corr_window=10, rank_window=15, std_window=15):
        returns = df['close'].pct_change(1).fillna(0)
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
      
        rev_amt_corr = returns.rolling(corr_window).corr(amt).fillna(0)
        
        price_ts_rank = O.ts_rank_normalized(df['close'], rank_window)
        vol = returns.rolling(std_window).std().fillna(0)
        
        condition = rev_amt_corr >= 0
        
        adjusted_rank = price_ts_rank / (np.sqrt(vol) + 1e-6)
        
        inner_signal = np.where(condition, price_ts_rank, adjusted_rank)
        
        raw_signal = -1 * O.ts_rank_normalized(pd.Series(inner_signal, index=df.index), 20)
        
        signal = (2 * raw_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_087(df: pd.DataFrame, resid_window=20, ts_rank_window=10):
        y = df['close'].values
        x = df['matchingVolume'].values
        
        def get_residual_fast(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            x_var = np.var(x_slice)
            # Dùng epsilon 1e-6 cho phương sai của Volume
            if x_var < 1e-6: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / x_var
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

        # Tính toán mảng Residual
        residuals = pd.Series([
            get_residual_fast(y[i-resid_window:i], x[i-resid_window:i]) 
            if i >= resid_window else 0 
            for i in range(len(y))
        ], index=df.index)
        
        ts_rank_resid = O.ts_rank_normalized(residuals, ts_rank_window)
        cs_rank_resid = O.ts_rank_normalized(residuals, 20) # Giả lập CsRank bằng rank cửa sổ rộng hơn
        
        # 4. Tính hiệu số (Crossover)
        diff_signal = ts_rank_resid - cs_rank_resid
        
        final_raw = O.ts_rank_normalized(diff_signal.fillna(0), 20)
        signal = (2 * final_raw) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_088(df: pd.DataFrame, q_window=50, resid_window=30):
        price_q60 = df['close'].rolling(q_window).quantile(0.6).fillna(df['close'])
        y = price_q60.values
        x = df['matchingVolume'].values
        
        def get_residual_fast(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            x_var = np.var(x_slice)
            # Tiêu chuẩn an toàn 1e-6
            if x_var < 1e-6: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / x_var
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

        residuals = pd.Series([
            get_residual_fast(y[i-resid_window:i], x[i-resid_window:i]) 
            if i >= resid_window else 0 
            for i in range(len(y))
        ], index=df.index)
        
       
        z_signal = O.zscore(residuals.fillna(0), 20)
        raw_signal = -1 * z_signal
        
        # 4. Đưa về biên độ chuẩn [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_089(df: pd.DataFrame, skew_window=40):
        vol_skew = df['matchingVolume'].rolling(skew_window).skew().fillna(0)
        
        body = df['close'] - df['open']
        full_range = df['high'] - df['low']
        efficiency = body / (full_range + 1e-6)
        
        raw_combined = vol_skew * efficiency
        
        z_signal = O.zscore(raw_combined.fillna(0), skew_window)
        
        ranked_signal = O.ts_rank_normalized(z_signal, skew_window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_090(df: pd.DataFrame, resid_window=15, ts_rank_window=10):
        y = df['close'].values
        x = df['matchingVolume'].values
        
        def get_residual_fast(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            x_var = np.var(x_slice)
            if x_var < 1e-6: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / x_var
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

        residuals = pd.Series([
            get_residual_fast(y[i-resid_window:i], x[i-resid_window:i]) 
            if i >= resid_window else 0 
            for i in range(len(y))
        ], index=df.index)
        
        ts_rank_resid = O.ts_rank_normalized(residuals, ts_rank_window)
        
        cs_rank_resid = O.ts_rank_normalized(residuals, 20)
        
        diff_signal = ts_rank_resid - cs_rank_resid
        
        final_raw = O.ts_rank_normalized(diff_signal.fillna(0), 20)
        signal = (2 * final_raw) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_091(df: pd.DataFrame, beta_window=30, delay=5):
        returns = df['close'].pct_change(1).fillna(0)
        amt = df['amt'] if 'amt' in df.columns else df['close'] * df['matchingVolume']
        
        def get_beta(s1, s2, n):
            # s1 là Amount, s2 là Returns
            def calc_beta(s1_slice, s2_slice):
                if len(s2_slice) < 2: return 0
                var_s2 = np.var(s2_slice)
                if var_s2 < 1e-6: return 0
                return np.cov(s1_slice, s2_slice)[0, 1] / var_s2
            
            return pd.Series([
                calc_beta(s1.values[i-n:i], s2.values[i-n:i]) 
                if i >= n else 0 
                for i in range(len(s1))
            ], index=s1.index)

        amount_beta = get_beta(amt, returns, beta_window)
       
        beta_diff = amount_beta - amount_beta.shift(delay).fillna(0)
        
        ranked_signal = O.ts_rank_normalized(beta_diff.fillna(0), 20)
        signal = -1 * (2 * ranked_signal - 1)
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_093(df: pd.DataFrame, window=20):
        vol = df['matchingVolume']
       
        ema_vol = vol.ewm(span=window, adjust=False).mean()
        sma_vol = vol.rolling(window).mean()
        
        std_vol = vol.rolling(window).std().fillna(0)
        
        vol_diff_ratio = (ema_vol - sma_vol) / (std_vol + 1e-6)
        
        ranked_signal = O.ts_rank_normalized(vol_diff_ratio.fillna(0), 20)
        signal = -1 * (2 * ranked_signal - 1)
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_094(df: pd.DataFrame, resid_window=22, kama_window=20, max_window=20):
        """
        Name: vwap_residual_adaptive_divergence
        Logic: Hồi quy Close theo VWAP, sau đó so sánh sai số với đường KAMA của chính nó.
        """
        # --- HÀM TÍNH KAMA NỘI BỘ (Để tránh sửa class O) ---
        def _calc_kama_internal(series, n):
            vals = series.values
            n_rows = len(vals)
            # Efficiency Ratio (ER)
            change = np.abs(series.diff(n).values)
            volatility = series.diff().abs().rolling(n).sum().values
            er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility > 1e-6)
            # Smoothing Constant (SC)
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
            kama = np.zeros(n_rows)
            if n_rows >= n:
                kama[n-1] = np.mean(vals[:n]) 
                for i in range(n, n_rows):
                    kama[i] = kama[i-1] + sc[i] * (vals[i] - kama[i-1])
            return pd.Series(kama, index=series.index)
        # -----------------------------------------------

        # 1. Tính VWAP (Nếu chưa có)
        vwap = (df['close'] * df['matchingVolume']).cumsum() / (df['matchingVolume'].cumsum() + 1e-6)

        # 2. Tính Residual (Hồi quy Close theo VWAP trong 22 phiên)
        y = df['close'].values
        x = vwap.values
        
        def get_residual_fast(y_slice, x_slice):
            if len(y_slice) < 2: return 0
            x_var = np.var(x_slice)
            if x_var < 1e-6: return 0
            beta = np.cov(x_slice, y_slice)[0, 1] / x_var
            alpha = np.mean(y_slice) - beta * np.mean(x_slice)
            y_hat = alpha + beta * x_slice[-1]
            return y_slice[-1] - y_hat

        # Tạo series Residual
        resid_series = pd.Series([
            get_residual_fast(y[i-resid_window:i], x[i-resid_window:i]) 
            if i >= resid_window else 0 
            for i in range(len(y))
        ], index=df.index)

        # 3. Gọi hàm KAMA nội bộ
        kama_resid = _calc_kama_internal(resid_series, kama_window)
        
        # 4. Tính mẫu số: TsMax của trị tuyệt đối Residual
        max_resid_range = resid_series.abs().rolling(max_window).max().fillna(0)
        
        # 5. Tính tỷ lệ phân kỳ (Divergence Ratio)
        div_ratio = (resid_series - kama_resid) / (max_resid_range + 1e-6)
        
        # 6. Rank và đưa về biên độ [-1, 1]
        ranked_signal = O.ts_rank_normalized(div_ratio.fillna(0), 20)
        signal = -1 * (2 * ranked_signal - 1)
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_095(df: pd.DataFrame, window_stoch=20, window_resid=30):
       
        min_low = O.ts_min(df['close'], window_stoch)
        max_high = O.ts_max(df['close'], window_stoch)
        
        raw_stoch = (df['close'] - min_low) / (max_high - min_low + 1e-8)
       
        volume_effect = O.ts_corr(raw_stoch, df['matchingVolume'], window_resid) * \
                        (O.ts_std(raw_stoch, window_resid) / (O.ts_std(df['matchingVolume'], window_resid) + 1e-8)) * \
                        (df['matchingVolume'] - O.ts_mean(df['matchingVolume'], window_resid))
        
        resid_signal = raw_stoch - volume_effect
        
        z_signal = O.zscore(resid_signal, window_resid)
        
        ranked_signal = O.ts_rank_normalized(z_signal, window_resid)
        final_signal = 1 - (2 * ranked_signal) 
        
        return -final_signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_096(df: pd.DataFrame, window_reg=20, window_rank=10):
        returns = df['close'].pct_change().fillna(0)
        
        t = pd.Series(range(window_reg), index=returns.index[-window_reg:]) # dummy cho window
        # Để đơn giản và hiệu quả trong rolling:
        def get_slope(y):
            if len(y) < window_reg: return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
            
        slope = returns.rolling(window_reg).apply(get_slope, raw=True)
      
        mean_amt = O.ts_mean(df['matchingVolume'], window_reg)
        std_amt = O.ts_std(df['matchingVolume'], window_reg)
        std_close = O.ts_std(df['close'], window_reg)
        corr_close_amt = O.ts_corr(df['close'], df['matchingVolume'], window_reg)
        
        beta = corr_close_amt * (std_close / (std_amt + 1e-8))
        resid = df['close'] - (O.ts_mean(df['close'], window_reg) + beta * (df['matchingVolume'] - mean_amt))

        ts_rank_resid = O.ts_rank_normalized(resid, window_rank)

        condition_signal = np.where(slope > 0, ts_rank_resid, -ts_rank_resid)
        condition_signal = pd.DataFrame(condition_signal, index=df.index, columns=['sig'])['sig']

        final_rank = O.rank(condition_signal.to_frame().T).iloc[0] 
        normalized_output = O.ts_rank_normalized(condition_signal, window_reg)
        
        signal = 1 - (2 * normalized_output) 
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_097(df: pd.DataFrame, window=20):
        amt = df['close'] * df['matchingVolume']
        returns = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume']

        def calculate_beta(target, reference, w):
            cov = O.ts_cov(target, reference, w)
            var_ref = O.power(O.ts_std(reference, w), 2)
            return cov / (var_ref + 1e-8)

        # 2. Tính Beta($amt, $volume, 20)
        beta_amt_vol = calculate_beta(amt, volume, window)

        # 3. Tính Beta($returns, $volume, 20)
        beta_ret_vol = calculate_beta(returns, volume, window)

        # 4. Phép trừ (Sub)
        raw_signal = beta_amt_vol - beta_ret_vol

        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        
        # Scale về [-1, 1]
        signal = (2 * ranked_signal) - 1

        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_098(df: pd.DataFrame, window_corr=15, window_resid=20):
        """
        Neg(Neg(CsZScore(Resid(Corr($close, $amt, 15), $volume, 20))))
        """
        # 1. Chuẩn bị dữ liệu
        amt = df['close'] * df['matchingVolume']
        volume = df['matchingVolume']
        
        # 2. Tính Corr($close, $amt, 15)
        correlation = O.ts_corr(df['close'], amt, window_corr)
        
        # 3. Tính Resid(correlation, $volume, 20)
        # Sử dụng công thức phần dư: resid = y - (y_mean + beta * (x - x_mean))
        mean_y = O.ts_mean(correlation, window_resid)
        mean_x = O.ts_mean(volume, window_resid)
        
        std_y = O.ts_std(correlation, window_resid)
        std_x = O.ts_std(volume, window_resid)
        corr_yx = O.ts_corr(correlation, volume, window_resid)
        
        beta = corr_yx * (std_y / (std_x + 1e-8))
        resid = correlation - (mean_y + beta * (volume - mean_x))
        
        # 4. CsZScore (Sử dụng zscore từ class O)
        z_signal = O.zscore(resid, window_resid)
        
        # 5. Neg(Neg(...)) và Normalize về [-1, 1]
        # Triệt tiêu Neg kép bằng cách giữ nguyên hướng của z_signal
        # Dùng ts_rank_normalized để ép biên độ theo yêu cầu
        ranked_signal = O.ts_rank_normalized(z_signal, window_resid)
        
        # Signal cuối cùng: (2 * rank) - 1
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_099(df: pd.DataFrame, window_resid=20, window_slope=15):
        close = df['close']
        volume = df['matchingVolume']
        returns = close.pct_change().fillna(0)
        
        mean_y = O.ts_mean(close, window_resid)
        mean_x = O.ts_mean(volume, window_resid)
        std_y = O.ts_std(close, window_resid)
        std_x = O.ts_std(volume, window_resid)
        corr_yx = O.ts_corr(close, volume, window_resid)
        
        beta = corr_yx * (std_y / (std_x + 1e-8))
        resid = close - (mean_y + beta * (volume - mean_x))
        
        def get_linreg_slope(series, w):
            t = np.arange(w)
            std_t = np.std(t)
            def slope_func(y):
                if len(y) < w: return 0
                return np.polyfit(t, y, 1)[0]
            return series.rolling(w).apply(slope_func, raw=True)

        # 4. Tính toán hai thành phần Slope
        slope_resid = get_linreg_slope(resid, window_slope)
        slope_returns = get_linreg_slope(returns, window_slope)
        
        # 5. Phép trừ (Sub)
        raw_signal = slope_resid - slope_returns
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window_resid)
        
        # Signal = (2 * Rank) - 1
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_100(df: pd.DataFrame, window_corr=15, window_resid=20):
        """
        Logic: Neg(Neg(CsZScore(Resid(Corr($close, $amt, 15), $volume, 20))))
        Nhóm: Volume / Liquidity Signal
        """
        # 1. Tính Amount ($amt) và xử lý NaN
        close = df['close'].ffill()
        volume = df['matchingVolume'].ffill()
        amount = close * volume

        # 2. Tính Rolling Correlation (Price & Amount)
        # Tương quan giữa giá và giá trị giao dịch
        correlation = close.rolling(window=window_corr).corr(amount)

        # 3. Tính Residual (Hồi quy Correlation theo Volume)
        # Công thức đơn giản cho Residual của Y trên X: Resid = Y - (Corr(X,Y) * Std(Y) / Std(X)) * X
        # Ở đây ta tối ưu bằng Vectorization
        y = correlation
        x = volume
        
        y_mean = y.rolling(window_resid).mean()
        x_mean = x.rolling(window_resid).mean()
        
        # Tính Beta đơn giản: cov(x,y) / var(x)
        covariance = x.rolling(window_resid).cov(y)
        variance_x = x.rolling(window_resid).var()
        
        beta = covariance / variance_x
        alpha = y_mean - beta * x_mean
        
        # Phần dư (Residual)
        residual = y - (alpha + beta * x)

        # 4. Chuẩn hóa Cross-sectional Z-Score (giả lập qua Rolling vì input là 1 DF đơn lẻ)
        # Trong môi trường nhiều mã, đây sẽ là chuẩn hóa theo hàng (axis=1)
        # Ở đây ta dùng Rolling Z-Score để thay thế cho logic chuỗi thời gian intraday
        zscore = (residual - residual.rolling(window_resid).mean()) / residual.rolling(window_resid).std()

        # 5. Final Signal: Trường hợp B (Dynamic Tanh) để ép về dải [-1, 1]
        # Triệt tiêu 2 lần Neg() = Giữ nguyên dấu
        final_signal = np.tanh(zscore).fillna(0)

        return final_signal

    @staticmethod
    def alpha_factor_miner_101(df: pd.DataFrame, window_resid: int = 20, window_slope: int = 15) -> pd.Series:
        """
        Logic: CsRank(Sub(TsLinRegSlope(Resid($close, $close * $volume, 20), 15), TsLinRegSlope($close, 15)))
        Xử lý lỗi: Dùng Vectorized Rolling Coeffs thay vì apply lồng DataFrame
        """
        # 1. Chuẩn bị dữ liệu
        y = df['close'].ffill()
        x = (df['close'] * df['matchingVolume']).ffill()

        # 2. Tính Residual bằng Vectorization (OLS: y = beta*x + alpha + resid)
        # Tính toán các thành phần lăn (rolling)
        rolling_cov = y.rolling(window_resid).cov(x)
        rolling_var_x = x.rolling(window_resid).var()
        
        # Beta = Cov(x,y) / Var(x)
        beta = rolling_cov / rolling_var_x.replace(0, np.nan)
        
        # Alpha = Mean(y) - Beta * Mean(x)
        alpha = y.rolling(window_resid).mean() - beta * x.rolling(window_resid).mean()
        
        # Resid = y - (beta * x + alpha)
        resid = y - (beta * x + alpha)
        resid = resid.ffill()

        # 3. Tính Slope (Độ dốc) bằng Vectorization cho nhanh và chính xác
        def get_vectorized_slope(series, window):
            # Công thức rút gọn của Slope: (n*Sum(xy) - Sum(x)*Sum(y)) / (n*Sum(x^2) - (Sum(x))^2)
            # Tạo mảng thời gian t từ 0 đến window-1
            t = np.arange(window)
            t_mean = np.mean(t)
            t_var = np.var(t) * window
            
            # Tính thủ công qua rolling để tránh lỗi apply phức tạp
            sum_t = np.sum(t)
            rolling_sum_y = series.rolling(window).sum()
            rolling_sum_ty = series.rolling(window).apply(lambda x: np.sum(x * t), raw=True)
            
            slope = (rolling_sum_ty - (sum_t * rolling_sum_y / window)) / t_var
            return slope

        # Tính độ dốc cho cả Resid và Close
        slope_resid = get_vectorized_slope(resid, window_slope)
        slope_close = get_vectorized_slope(y, window_slope)

        # 4. Phép trừ Sub
        raw_alpha = slope_resid - slope_close

        # 5. Chuẩn hóa TRƯỜNG HỢP C (Z-Score/Clip)
        # Sử dụng window_slope * 2 để có dải quan sát phân phối đủ rộng
        norm_window = window_slope * 2
        mean = raw_alpha.rolling(norm_window).mean()
        std = raw_alpha.rolling(norm_window).std()
        
        signal = ((raw_alpha - mean) / std.replace(0, np.nan)).clip(-1, 1)

        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_102(df: pd.DataFrame, window_ret: int = 20, window_amt: int = 25) -> pd.Series:
        close = df['close'].ffill()
        volume = df['matchingVolume'].replace(0, np.nan).ffill()
        returns = close.pct_change().fillna(0)
        amount = (close * volume)

        def get_beta(dep_var, indep_var, window):
            rolling_cov = dep_var.rolling(window).cov(indep_var)
            rolling_var = indep_var.rolling(window).var()
            return rolling_cov / rolling_var.replace(0, np.nan)

        beta_ret_vol = get_beta(returns, volume, window_ret)
        beta_amt_vol = get_beta(amount, volume, window_amt)

        ratio = beta_amt_vol / beta_ret_vol.replace(0, np.nan)
        
    
        raw_alpha = np.where(beta_ret_vol > 0, ratio, -ratio)
        raw_alpha = pd.Series(raw_alpha, index=df.index).ffill()

      
        rank_window = max(window_ret, window_amt) * 2 
        
        alpha_ranked = (raw_alpha.rolling(rank_window).rank(pct=True) * 2) - 1
        signal = -alpha_ranked

        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_103(df: pd.DataFrame, window_stoch: int = 25, window_resid: int = 30) -> pd.Series:
        close = df['close'].ffill()
        low = df['low'].ffill()
        high = df['high'].ffill()
        volume = df['matchingVolume'].replace(0, np.nan).ffill()
        amount = (close * volume).ffill()

        low_min = low.rolling(window_stoch).min()
        high_max = high.rolling(window_stoch).max()
        
        stoch_k = (close - low_min) / (high_max - low_min).replace(0, np.nan)
        
        stoch_vol = stoch_k / volume
        stoch_vol = stoch_vol.ffill()

        def get_rolling_resid(y, x, window):
            rolling_cov = y.rolling(window).cov(x)
            rolling_var_x = x.rolling(window).var()
            beta = rolling_cov / rolling_var_x.replace(0, np.nan)
            alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
            return y - (beta * x + alpha)

        raw_resid = get_rolling_resid(stoch_vol, amount, window_resid)

        norm_window = window_resid
        rolling_mean = raw_resid.rolling(norm_window).mean()
        rolling_std = raw_resid.rolling(norm_window).std()
        
        # Chuyển đổi về dải [-1, 1]
        signal = ((raw_resid - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-1, 1)

        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_105(df: pd.DataFrame, window_corr: int = 20, window_rank: int = 15) -> pd.Series:
        returns = np.log(df['close'] / df['close'].shift(1))
        volume = df['matchingVolume'].astype(float)

        corr_vol_ret = volume.rolling(window=window_corr).corr(returns)
        
        ts_rank_vol = volume.rolling(window=window_rank).rank(pct=True)
     
        signal_raw = np.where(
            corr_vol_ret > 0,
            ts_rank_vol,
            -ts_rank_vol
        )
        
        signal_series = pd.Series(signal_raw, index=df.index)
     
        final_rank = signal_series.rolling(window=window_corr).rank(pct=True)
        alpha_final = -( (final_rank * 2) - 1 )
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_106(df: pd.DataFrame, window_beta: int = 20, window_hma: int = 35) -> pd.Series:
        
        returns = df['close'].pct_change()
        amt = df['matchingVolume'].astype(float)
        
        rolling_cov = returns.rolling(window=window_beta).cov(amt)
        rolling_var_amt = amt.rolling(window=window_beta).var()
        beta = rolling_cov / rolling_var_amt
        
        tr_range = df['high'].rolling(15).max() - df['low'].rolling(15).min()
        
        def calculate_hma(series, period):
            def wma(s, p):
                weights = np.arange(1, p + 1)
                return s.rolling(p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
            half_p = int(period / 2)
            sqrt_p = int(np.sqrt(period))
            raw_hma = 2 * wma(series, half_p) - wma(series, period)
            return wma(raw_hma, sqrt_p)

        hma_range = calculate_hma(tr_range, window_hma)
        
        raw_signal = beta / hma_range.replace(0, np.nan)
        
        signal_normalized = -np.tanh(raw_signal / raw_signal.rolling(window_hma).std())
        
        return -signal_normalized.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_107(df: pd.DataFrame, window_kurt: int = 50, window_range: int = 25) -> pd.Series:
        returns = np.log(df['close'] / df['close'].shift(1))
        
        kurt = returns.rolling(window=window_kurt).kurt()
        
        ts_max = df['high'].rolling(window=window_range).max()
        ts_min = df['low'].rolling(window=window_range).min()
        price_range = ts_max - ts_min
        
        vol_sqrt = np.sqrt(df['matchingVolume'].astype(float))
        
        denominator = price_range * vol_sqrt
        
        raw_signal = -(kurt / denominator.replace(0, np.nan))
        
        rolling_mean = raw_signal.rolling(window=window_kurt).mean()
        rolling_std = raw_signal.rolling(window=window_kurt).std()
        
        z_score = (raw_signal - rolling_mean) / rolling_std
        signal_final = z_score.clip(-1, 1)
        
        return -signal_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_108(df: pd.DataFrame, window_q: int = 20, window_kama: int = 25) -> pd.Series:
        quantile_80 = df['close'].rolling(window=window_q).quantile(0.8)
        delta_quantile = quantile_80.diff(5)
        
        price_range = df['high'] - df['low']
        
        def calculate_kama(series, period):
            change = series.diff(period).abs()
            volatility = series.diff().abs().rolling(window=period).sum()
            er = change / volatility.replace(0, np.nan)
            
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
            
            kama = np.zeros_like(series)
            initial_val = series.fillna(0).iloc[0]
            kama[0] = initial_val
            
           
            for i in range(1, len(series)):
                if np.isnan(sc.iloc[i]):
                    kama[i] = kama[i-1]
                else:
                    kama[i] = kama[i-1] + sc.iloc[i] * (series.iloc[i] - kama[i-1])
            return pd.Series(kama, index=series.index)

        kama_range = calculate_kama(price_range, window_kama)
        
        raw_signal = delta_quantile / kama_range.replace(0, np.nan)
        
       
        rank_signal = raw_signal.rolling(window=window_kama).rank(pct=True)
        alpha_final = (rank_signal * 2) - 1
        
        return alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_109(df: pd.DataFrame, window_decay: int = 5, window_slope: int = 15) -> pd.Series:
        returns = np.log(df['close'] / df['close'].shift(1))
        volume = df['matchingVolume'].astype(float)
        
        # 2. Tính Decay(volume, 10) - Linear Decay
        def linear_decay(series, n):
            weights = np.arange(1, n + 1)
            return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        decay_volume = linear_decay(volume, window_decay)
        
        # 3. Tính TsLinRegSlope(returns, 15)
        def rolling_slope(series, window):
            x = np.arange(window)
            x_mean = np.mean(x)
            
            def get_slope(y):
                # Công thức OLS Slope: Cov(x, y) / Var(x)
                if np.isnan(y).any():
                    return np.nan
                y_mean = np.mean(y)
                slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
                return slope
            
            return series.rolling(window).apply(get_slope, raw=True)
        
        slope_returns = rolling_slope(returns, window_slope)
        
        raw_signal = decay_volume * slope_returns
        
       
        rank_signal = raw_signal.rolling(window=window_slope).rank(pct=True)
        alpha_final = (rank_signal * 2) - 1
        
        return alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_110(df: pd.DataFrame, window_ret: int = 50, window_amt: int = 20) -> pd.Series:
       
        returns = df['close'].pct_change()
        amt = df['matchingVolume'].astype(float)
        
        q70_ret = returns.rolling(window=window_ret).quantile(0.7)
        
        def fast_rolling_slope(series, window):
            n = window
            x = np.arange(n)
            def get_slope(raw_y):
                if np.isnan(raw_y).any(): return np.nan
                return np.polyfit(x, raw_y, 1)[0]
            
            return series.rolling(window).apply(get_slope, raw=True)

        slope_amt = fast_rolling_slope(amt, window_amt)
        
        raw_signal = np.where(
            q70_ret > 0,
            slope_amt,
            -slope_amt
        )
        
        raw_signal_series = pd.Series(raw_signal, index=df.index)
        
        rank_signal = raw_signal_series.rolling(window=window_ret).rank(pct=True)
        alpha_final = (rank_signal * 2) - 1
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_111(df: pd.DataFrame, window_corr: int = 10, window_std: int = 40) -> pd.Series:
        amt = df['matchingVolume'].astype(float)
        high = df['high'].astype(float)
        
        corr_amt_high = amt.rolling(window=window_corr).corr(high)
        
        std_amt = amt.rolling(window=window_std).std()
        ema_vol = amt.ewm(span=window_std, adjust=False).mean()
        
        vol_ratio = std_amt / ema_vol.replace(0, np.nan)
        
        raw_signal = np.where(
            corr_amt_high > 0,
            vol_ratio,
            -vol_ratio
        )
        
        raw_signal_series = pd.Series(raw_signal, index=df.index)
        
        
        rolling_std_signal = raw_signal_series.rolling(window=window_std).std()
        
        alpha_final = -np.tanh(raw_signal_series / rolling_std_signal.replace(0, np.nan))
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_112(df: pd.DataFrame, window_low: int = 5, window_dema: int = 10) -> pd.Series:
        low_min = df['low'].rolling(window=window_low).min()
        price_dist = df['close'] - low_min
        
        price_range = df['high'] - df['low']
        
        def calculate_dema(series, period):
            ema1 = series.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            return 2 * ema1 - ema2
            
        dema_range = calculate_dema(price_range, window_dema)
        
        raw_signal = price_dist / dema_range.replace(0, np.nan)
        
        rolling_mean = raw_signal.rolling(window=window_dema).mean()
        rolling_std = raw_signal.rolling(window=window_dema).std()
        
        z_score = (raw_signal - rolling_mean) / rolling_std.replace(0, np.nan)
        
        alpha_final = z_score.clip(-1, 1)
        
        return alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_113(df: pd.DataFrame, window_ret: int = 5, window_resid: int = 20) -> pd.Series:
        log_ret_5 = np.log(df['close'] / df['close'].shift(window_ret))
        vol_sqrt = np.sqrt(df['matchingVolume'].astype(float))
        amt = df['matchingVolume'].astype(float)
        
        # Biến phụ thuộc y, Biến độc lập x
        y = (log_ret_5 / vol_sqrt.replace(0, np.nan)).fillna(0)
        x = amt.fillna(0)
        
        # 2. Hàm tính Residual tối ưu
        def get_rolling_residual(y_ser, x_ser, window):
            rolling_y_mean = y_ser.rolling(window).mean()
            rolling_x_mean = x_ser.rolling(window).mean()
            
            rolling_cov_xy = y_ser.rolling(window).cov(x_ser)
            rolling_var_x = x_ser.rolling(window).var()
            
            beta = rolling_cov_xy / rolling_var_x.replace(0, np.nan)
            alpha = rolling_y_mean - beta * rolling_x_mean
            
            resid = y_ser - (beta * x_ser + alpha)
            return resid

        residual = get_rolling_residual(y, x, window_resid)
        
        rolling_mean = residual.rolling(window=window_resid).mean()
        rolling_std = residual.rolling(window=window_resid).std()
        
        z_score = (residual - rolling_mean) / rolling_std.replace(0, np.nan)
        
        alpha_final = (-z_score).clip(-1, 1)
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_114(df: pd.DataFrame, window_beta: int = 30, window_skew: int = 25) -> pd.Series:
        returns = df['close'].pct_change()
        amt = df['matchingVolume'].astype(float)
        volume = df['matchingVolume'].astype(float) 
        
        rolling_cov = amt.rolling(window=window_beta).cov(returns)
        rolling_var_ret = returns.rolling(window=window_beta).var()
        beta = rolling_cov / rolling_var_ret.replace(0, np.nan)
        
        skew_vol = volume.rolling(window=window_skew).skew()
        
        raw_signal = beta * skew_vol
        
        rank_signal = raw_signal.rolling(window=window_beta).rank(pct=True)
        alpha_final = (rank_signal * 2) - 1
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_115(df: pd.DataFrame, window_stoch: int = 2, window_resid: int = 20) -> pd.Series:
        
        low_min = df['low'].rolling(window=window_stoch).min()
        high_max = df['high'].rolling(window=window_stoch).max()
        price_range = high_max - low_min
        
        y = (df['close'] - low_min) / price_range.replace(0, np.nan)
        
        x = df['matchingVolume'].astype(float)
        
        rolling_cov_xy = y.rolling(window=window_resid).cov(x)
        rolling_var_x = x.rolling(window=window_resid).var()
        
        beta = rolling_cov_xy / rolling_var_x.replace(0, np.nan)
        alpha = y.rolling(window=window_resid).mean() - beta * x.rolling(window=window_resid).mean()
        
        # Residual: epsilon = y - (beta * x + alpha)
        residual = y - (beta * x + alpha)
        
        rolling_mean_res = residual.rolling(window=window_resid).mean()
        rolling_std_res = residual.rolling(window=window_resid).std()
        
        z_score = (residual - rolling_mean_res) / rolling_std_res.replace(0, np.nan)
        
        alpha_final = (-z_score).clip(-1, 1)
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_v2_001(df, window=20, rank_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vwap = (df['high'] + df['low'] + df['close']) / 3
        returns = close.pct_change()

        std_hl_3 = (high - low).rolling(window=3).std()
        std_returns_10 = returns.rolling(window=10).std()
        condition = (std_hl_3 / std_returns_10) < 0.8

        delta_close_2 = close.diff(2)
        neg_rank_delta = -delta_close_2.rolling(window=rank_window).rank(pct=True)

        close_vwap_ratio = close / vwap
        ts_rank_8 = close_vwap_ratio.rolling(window=8).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        raw_signal = pd.Series(np.where(condition, neg_rank_delta, ts_rank_8), index=df.index)
        raw_signal = raw_signal.ffill()

        signal = (raw_signal.rolling(window=window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_v2_002(df, window=20, scale_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vwap = (df['high'] + df['low'] + df['close']) / 3
        returns = close.pct_change()

        std_hl_3 = (high - low).rolling(window=3).std()
        std_returns_10 = returns.rolling(window=10).std()
        condition = (std_hl_3 / std_returns_10) < 0.8

        delta_close_2 = close.diff(2)
        neg_rank_delta = -delta_close_2.rolling(window=scale_window).rank(pct=True)

        close_vwap_ratio = close / vwap
        ts_rank_8 = close_vwap_ratio.rolling(window=8).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        raw_signal = pd.Series(np.where(condition, neg_rank_delta, ts_rank_8), index=df.index)
        raw_signal = raw_signal.ffill()

        rolling_std = raw_signal.rolling(window=window).std().replace(0, 1)
        signal = np.tanh(raw_signal / rolling_std)
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_v2_003(df, window=20, zscore_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vwap = (df['high'] + df['low'] + df['close']) / 3
        returns = close.pct_change()

        std_hl_3 = (high - low).rolling(window=3).std()
        std_returns_10 = returns.rolling(window=10).std()
        condition = (std_hl_3 / std_returns_10) < 0.8

        delta_close_2 = close.diff(2)
        neg_rank_delta = -delta_close_2.rolling(window=zscore_window).rank(pct=True)

        close_vwap_ratio = close / vwap
        ts_rank_8 = close_vwap_ratio.rolling(window=8).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        raw_signal = pd.Series(np.where(condition, neg_rank_delta, ts_rank_8), index=df.index)
        raw_signal = raw_signal.ffill()

        rolling_mean = raw_signal.rolling(window=window).mean()
        rolling_std = raw_signal.rolling(window=window).std().replace(0, 1)
        signal = ((raw_signal - rolling_mean) / rolling_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_v2_004(df, window=5, threshold=0):
        high = df['high']
        low = df['low']
        close = df['close']
        vwap = (df['high'] + df['low'] + df['close']) / 3
        returns = close.pct_change()

        std_hl_3 = (high - low).rolling(window=3).std()
        std_returns_10 = returns.rolling(window=10).std()
        condition = (std_hl_3 / std_returns_10) < 0.8

        delta_close_2 = close.diff(2)
        neg_rank_delta = -delta_close_2.rolling(window=window).rank(pct=True)

        close_vwap_ratio = close / vwap
        ts_rank_8 = close_vwap_ratio.rolling(window=8).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        raw_signal = pd.Series(np.where(condition, neg_rank_delta, ts_rank_8), index=df.index)
        raw_signal = raw_signal.ffill()

        signal = np.sign(raw_signal - threshold)
        return -signal.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_001(df, window=6, rank_window=20):
    #     vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
    #     ratio = df['close'] / vwap.replace(0, np.nan)
    #     ema_ratio = ratio.ewm(span=window, adjust=False).mean()
    #     raw = (ema_ratio - ema_ratio.shift(2)) / ema_ratio.rolling(12).std()
    #     raw = raw.ffill()
    #     signal = (raw.rolling(rank_window).rank(pct=True) * 2) - 1
    #     return signal.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_002(df, window=6, std_window=20):
    #     vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
    #     ratio = df['close'] / vwap.replace(0, np.nan)
    #     ema_ratio = ratio.ewm(span=window, adjust=False).mean()
    #     raw = (ema_ratio - ema_ratio.shift(2)) / ema_ratio.rolling(12).std()
    #     raw = raw.ffill()
    #     signal = np.tanh(raw / raw.rolling(std_window).std().replace(0, np.nan))
    #     return signal.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_003(df, window=6, z_window=20):
    #     vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
    #     ratio = df['close'] / vwap.replace(0, np.nan)
    #     ema_ratio = ratio.ewm(span=window, adjust=False).mean()
    #     raw = (ema_ratio - ema_ratio.shift(2)) / ema_ratio.rolling(12).std()
    #     raw = raw.ffill()
    #     rolling_mean = raw.rolling(z_window).mean()
    #     rolling_std = raw.rolling(z_window).std().replace(0, np.nan)
    #     signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
    #     return signal.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_004(df, window=6):
    #     vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
    #     ratio = df['close'] / vwap.replace(0, np.nan)
    #     ema_ratio = ratio.ewm(span=window, adjust=False).mean()
    #     raw = (ema_ratio - ema_ratio.shift(2)) / ema_ratio.rolling(12).std()
    #     raw = raw.ffill()
    #     signal = np.sign(raw)
    #     return signal.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_001(df, window_ema=6, window_std=14):
    #     close = df['close']
    #     returns = close.pct_change()
    #     ema_returns = returns.ewm(span=window_ema, adjust=False).mean()
    #     ema_returns_delayed = ema_returns.shift(3)
    #     numerator = ema_returns - ema_returns_delayed
    #     denominator = ema_returns.rolling(window_std).std()
    #     raw = numerator / denominator
    #     raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    #     return (raw.rolling(252).rank(pct=True) * 2) - 1

    # @staticmethod
    # def alpha_factor_miner_v3_002(df, window_ema=6, window_std=14):
    #     close = df['close']
    #     returns = close.pct_change()
    #     ema_returns = returns.ewm(span=window_ema, adjust=False).mean()
    #     ema_returns_delayed = ema_returns.shift(3)
    #     numerator = ema_returns - ema_returns_delayed
    #     denominator = ema_returns.rolling(window_std).std()
    #     raw = numerator / denominator
    #     raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    #     return np.tanh(raw / raw.rolling(252).std())

    # @staticmethod
    # def alpha_factor_miner_v3_003(df, window_ema=6, window_std=14):
    #     close = df['close']
    #     returns = close.pct_change()
    #     ema_returns = returns.ewm(span=window_ema, adjust=False).mean()
    #     ema_returns_delayed = ema_returns.shift(3)
    #     numerator = ema_returns - ema_returns_delayed
    #     denominator = ema_returns.rolling(window_std).std()
    #     raw = numerator / denominator
    #     raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    #     zscore = (raw - raw.rolling(252).mean()) / raw.rolling(252).std()
    #     return zscore.clip(-1, 1)

    # @staticmethod
    # def alpha_factor_miner_v3_004(df, window_ema=6, window_std=14):
    #     close = df['close']
    #     returns = close.pct_change()
    #     ema_returns = returns.ewm(span=window_ema, adjust=False).mean()
    #     ema_returns_delayed = ema_returns.shift(3)
    #     numerator = ema_returns - ema_returns_delayed
    #     denominator = ema_returns.rolling(window_std).std()
    #     raw = numerator / denominator
    #     raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    #     return np.sign(raw)

    # @staticmethod
    # def alpha_factor_miner_v3_001(df, window=12, std_window=22):
    #     close = df['close']
    #     slope = close.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    #     delayed_slope = slope.shift(4)
    #     numerator = slope - delayed_slope
    #     denominator = slope.rolling(std_window).std()
    #     raw = -1 * (numerator / denominator.replace(0, np.nan))
    #     raw_ranked = raw.rolling(252).rank(pct=True)
    #     signal = (raw_ranked * 2) - 1
    #     return signal.ffill().fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_002(df, window=12, std_window=22):
    #     close = df['close']
    #     slope = close.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    #     delayed_slope = slope.shift(4)
    #     numerator = slope - delayed_slope
    #     denominator = slope.rolling(std_window).std()
    #     raw = -1 * (numerator / denominator.replace(0, np.nan))
    #     rolling_std = raw.rolling(63).std().replace(0, np.nan)
    #     signal = np.tanh(raw / rolling_std)
    #     return signal.ffill().fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_003(df, window=12, std_window=22):
    #     close = df['close']
    #     slope = close.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    #     delayed_slope = slope.shift(4)
    #     numerator = slope - delayed_slope
    #     denominator = slope.rolling(std_window).std()
    #     raw = -1 * (numerator / denominator.replace(0, np.nan))
    #     rolling_mean = raw.rolling(63).mean()
    #     rolling_std = raw.rolling(63).std().replace(0, np.nan)
    #     signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
    #     return signal.ffill().fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_004(df, window=12, std_window=22):
    #     close = df['close']
    #     slope = close.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    #     delayed_slope = slope.shift(4)
    #     numerator = slope - delayed_slope
    #     denominator = slope.rolling(std_window).std()
    #     raw = -1 * (numerator / denominator.replace(0, np.nan))
    #     signal = np.sign(raw)
    #     return signal.ffill().fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_001(df, window=15):
    #     close = df['close']
    #     ret_2 = close.pct_change(2)
    #     ret_7 = close.pct_change(7)
    #     ts_rank_2 = ret_2.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     ts_rank_7 = ret_7.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     raw = ts_rank_2 - ts_rank_7
    #     raw = raw.ffill().fillna(0)
    #     normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
    #     return normalized.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_002(df, window=15):
    #     close = df['close']
    #     ret_2 = close.pct_change(2)
    #     ret_7 = close.pct_change(7)
    #     ts_rank_2 = ret_2.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     ts_rank_7 = ret_7.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     raw = ts_rank_2 - ts_rank_7
    #     raw = raw.ffill().fillna(0)
    #     normalized = np.tanh(raw / raw.rolling(window).std().replace(0, 1))
    #     return normalized.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_003(df, window=15):
    #     close = df['close']
    #     ret_2 = close.pct_change(2)
    #     ret_7 = close.pct_change(7)
    #     ts_rank_2 = ret_2.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     ts_rank_7 = ret_7.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     raw = ts_rank_2 - ts_rank_7
    #     raw = raw.ffill().fillna(0)
    #     zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, 1)
    #     normalized = zscore.clip(-1, 1)
    #     return normalized.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_004(df, window=15):
    #     close = df['close']
    #     ret_2 = close.pct_change(2)
    #     ret_7 = close.pct_change(7)
    #     ts_rank_2 = ret_2.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     ts_rank_7 = ret_7.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    #     raw = ts_rank_2 - ts_rank_7
    #     raw = raw.ffill().fillna(0)
    #     normalized = np.sign(raw)
    #     return normalized.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_005(df: pd.DataFrame, window: int = 15, quantile: float = 0.05) -> pd.Series:
    #     """
    #     Momentum reversal signal using winsorized Fisher transform.
    #     Handles heavy tails while preserving distribution structure.
    #     """
    #     close = df['close']
    #     ret_2 = close.pct_change(2)
    #     ret_7 = close.pct_change(7)

    #     ts_rank_2 = ret_2.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    #     ts_rank_7 = ret_7.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    #     raw = ts_rank_2 - ts_rank_7
    #     raw = raw.ffill()

    #     low = raw.rolling(window).quantile(quantile)
    #     high = raw.rolling(window).quantile(1 - quantile)
    #     winsorized = raw.clip(lower=low, upper=high, axis=0)

    #     normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
    #     return normalized.fillna(0)

    # @staticmethod
    # def alpha_factor_miner_v3_005(df, window=12, ema_window=6, quantile=0.05):
    #     vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
    #     ratio = df['close'] / vwap.replace(0, 1e-9)
    #     ema_ratio = ratio.ewm(span=ema_window, adjust=False).mean()
    #     diff = ema_ratio - ema_ratio.shift(2)
    #     std = ema_ratio.rolling(window=window).std()
    #     raw = diff / (std.replace(0, 1e-9))
    #     raw_ffilled = raw.ffill()
    #     low = raw_ffilled.rolling(window=window).quantile(quantile)
    #     high = raw_ffilled.rolling(window=window).quantile(1 - quantile)
    #     winsorized = raw_ffilled.clip(lower=low, upper=high)
    #     normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
    #     signal = normalized.clip(-1, 1)
    #     return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_001(df, window=20, rank_window=50):
        high = df['high']
        low = df['low']
        close = df['close']
        amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
        high_roll_max = high.rolling(window=15, min_periods=1).max()
        low_roll_min = low.rolling(window=15, min_periods=1).min()
        raw = (close - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
        x = raw
        y = amount
        beta = y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var().replace(0, np.nan)
        alpha = y.rolling(window=window, min_periods=1).mean() - beta * x.rolling(window=window, min_periods=1).mean()
        resid = y - (beta * x + alpha)
        raw_signal = -((resid - resid.rolling(window=window, min_periods=1).mean()) / resid.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        raw_signal = raw_signal.ffill().fillna(0)
        normalized = (raw_signal.rolling(window=rank_window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_factor_miner_v3_002(df, window=20, scale_window=50):
        high = df['high']
        low = df['low']
        close = df['close']
        amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
        high_roll_max = high.rolling(window=15, min_periods=1).max()
        low_roll_min = low.rolling(window=15, min_periods=1).min()
        raw = (close - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
        x = raw
        y = amount
        beta = y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var().replace(0, np.nan)
        alpha = y.rolling(window=window, min_periods=1).mean() - beta * x.rolling(window=window, min_periods=1).mean()
        resid = y - (beta * x + alpha)
        raw_signal = -((resid - resid.rolling(window=window, min_periods=1).mean()) / resid.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        raw_signal = raw_signal.ffill().fillna(0)
        normalized = np.tanh(raw_signal / raw_signal.rolling(window=scale_window, min_periods=1).std().replace(0, np.nan))
        return normalized

    @staticmethod
    def alpha_factor_miner_v3_003(df, window=20, z_window=50):
        high = df['high']
        low = df['low']
        close = df['close']
        amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
        high_roll_max = high.rolling(window=15, min_periods=1).max()
        low_roll_min = low.rolling(window=15, min_periods=1).min()
        raw = (close - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
        x = raw
        y = amount
        beta = y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var().replace(0, np.nan)
        alpha = y.rolling(window=window, min_periods=1).mean() - beta * x.rolling(window=window, min_periods=1).mean()
        resid = y - (beta * x + alpha)
        raw_signal = -((resid - resid.rolling(window=window, min_periods=1).mean()) / resid.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        raw_signal = raw_signal.ffill().fillna(0)
        z = (raw_signal - raw_signal.rolling(window=z_window, min_periods=1).mean()) / raw_signal.rolling(window=z_window, min_periods=1).std().replace(0, np.nan)
        normalized = z.clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_factor_miner_v3_004(df, window=20):
        high = df['high']
        low = df['low']
        close = df['close']
        amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
        high_roll_max = high.rolling(window=15, min_periods=1).max()
        low_roll_min = low.rolling(window=15, min_periods=1).min()
        raw = (close - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
        x = raw
        y = amount
        beta = y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var().replace(0, np.nan)
        alpha = y.rolling(window=window, min_periods=1).mean() - beta * x.rolling(window=window, min_periods=1).mean()
        resid = y - (beta * x + alpha)
        raw_signal = -((resid - resid.rolling(window=window, min_periods=1).mean()) / resid.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        raw_signal = raw_signal.ffill().fillna(0)
        normalized = np.sign(raw_signal)
        return normalized

    @staticmethod
    def alpha_factor_miner_v3_005(df, window=20, winsor_window=100, quantile=0.05):
        high = df['high']
        low = df['low']
        close = df['close']
        amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
        high_roll_max = high.rolling(window=15, min_periods=1).max()
        low_roll_min = low.rolling(window=15, min_periods=1).min()
        raw = (close - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
        x = raw
        y = amount
        beta = y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var().replace(0, np.nan)
        alpha = y.rolling(window=window, min_periods=1).mean() - beta * x.rolling(window=window, min_periods=1).mean()
        resid = y - (beta * x + alpha)
        raw_signal = -((resid - resid.rolling(window=window, min_periods=1).mean()) / resid.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        raw_signal = raw_signal.ffill().fillna(0)
        low_bound = raw_signal.rolling(window=winsor_window, min_periods=1).quantile(quantile)
        high_bound = raw_signal.rolling(window=winsor_window, min_periods=1).quantile(1 - quantile)
        winsorized = raw_signal.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_factor_miner_v3_006(df, window=6):
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        raw = np.where(close == delay_close, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        sum_raw = raw_series.rolling(window=window, min_periods=1).sum()
        rank = sum_raw.rolling(window=window*2).rank(pct=True)
        normalized = (rank * 2) - 1
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_007(df, window=6):
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        raw = np.where(close == delay_close, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        sum_raw = raw_series.rolling(window=window, min_periods=1).sum()
        std = sum_raw.rolling(window=window*2).std()
        normalized = np.tanh(sum_raw / (std + 1e-9))
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_008(df, window=6):
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        raw = np.where(close == delay_close, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        sum_raw = raw_series.rolling(window=window, min_periods=1).sum()
        rolling_mean = sum_raw.rolling(window=window*2).mean()
        rolling_std = sum_raw.rolling(window=window*2).std()
        z_score = (sum_raw - rolling_mean) / (rolling_std + 1e-9)
        normalized = np.clip(z_score, -1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_009(df, window=6):
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        raw = np.where(close == delay_close, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        sum_raw = raw_series.rolling(window=window, min_periods=1).sum()
        normalized = np.sign(sum_raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_010(df, window=6, quantile=0.05):
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        raw = np.where(close == delay_close, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        sum_raw = raw_series.rolling(window=window, min_periods=1).sum()
        winsor_window = window * 3
        low_bound = sum_raw.rolling(window=winsor_window).quantile(quantile)
        high_bound = sum_raw.rolling(window=winsor_window).quantile(1 - quantile)
        winsorized = sum_raw.clip(lower=low_bound, upper=high_bound)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)


    @staticmethod
    def alpha_factor_miner_v3_011(df: pd.DataFrame, window: int = 20) -> pd.Series:
        # Tính weighted price
        weighted_price = (df['open'] * 0.85) + (df['high'] * 0.15)

        # Tính delta 4 periods
        delta = weighted_price.diff(4)

        # Lấy sign của delta
        sign_delta = np.sign(delta)

        # Tính rank của sign
        raw = sign_delta.rolling(window).rank(pct=True)

        # Chuẩn hóa về [-1, 1]
        signal = (raw * 2) - 1

        # Xử lý missing data
        signal = signal.ffill().fillna(0)

        return signal

    @staticmethod
    def alpha_factor_miner_v3_012(df: pd.DataFrame, window: int = 20) -> pd.Series:
        # Tính weighted price
        weighted_price = (df['open'] * 0.85) + (df['high'] * 0.15)

        # Tính delta 4 periods
        delta = weighted_price.diff(4)

        # Lấy sign của delta
        sign_delta = np.sign(delta)

        # Tính rank của sign
        rank_signal = sign_delta.rolling(window).rank(pct=True)

        # Chuẩn hóa bằng tanh để giữ magnitude
        raw = rank_signal - 0.5  # Center around 0
        std_dev = raw.rolling(window).std()
        signal = np.tanh(raw / (std_dev + 1e-9))

        # Xử lý missing data
        signal = signal.ffill().fillna(0)

        return signal

    @staticmethod
    def alpha_factor_miner_v3_013(df: pd.DataFrame, window: int = 20) -> pd.Series:
        # Tính weighted price
        weighted_price = (df['open'] * 0.85) + (df['high'] * 0.15)

        # Tính delta 4 periods
        delta = weighted_price.diff(4)

        # Lấy sign của delta
        sign_delta = np.sign(delta)

        # Tính rank của sign
        rank_signal = sign_delta.rolling(window).rank(pct=True)

        # Chuẩn hóa bằng z-score và clip
        raw = rank_signal
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()

        signal = ((raw - rolling_mean) / (rolling_std + 1e-9)).clip(-1, 1)

        # Xử lý missing data
        signal = signal.ffill().fillna(0)

        return signal

    @staticmethod
    def alpha_factor_miner_v3_014(df: pd.DataFrame, window: int = 20) -> pd.Series:
        # Tính weighted price
        weighted_price = (df['open'] * 0.85) + (df['high'] * 0.15)

        # Tính delta 4 periods
        delta = weighted_price.diff(4)

        # Lấy sign của delta
        sign_delta = np.sign(delta)

        # Tính rank của sign
        rank_signal = sign_delta.rolling(window).rank(pct=True)

        # Áp dụng sign cho breakout
        raw = rank_signal - 0.5  # Center around 0
        signal = np.sign(raw)

        # Xử lý missing data
        signal = signal.ffill().fillna(0)

        return signal

    @staticmethod
    def alpha_factor_miner_v3_015(df: pd.DataFrame, window: int = 20, quantile: float = 0.05) -> pd.Series:
        # Tính weighted price
        weighted_price = (df['open'] * 0.85) + (df['high'] * 0.15)

        # Tính delta 4 periods
        delta = weighted_price.diff(4)

        # Lấy sign của delta
        sign_delta = np.sign(delta)

        # Tính rank của sign
        rank_signal = sign_delta.rolling(window).rank(pct=True)

        # Winsorized Fisher Transform
        raw = rank_signal
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)

        winsorized = raw.clip(lower=low, upper=high)

        # Fisher Transform
        normalized = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        signal = np.arctanh(normalized.clip(-0.99, 0.99))

        # Xử lý missing data
        signal = signal.ffill().fillna(0)

        return signal

    @staticmethod
    def alpha_factor_miner_v3_016(df, window=20):
        # Tính log return
        ret = np.log(df['close'] / df['close'].shift(1))
        # Tính tổng rolling 5 ngày
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        raw = sum_open_5 * sum_ret_5
        # Tính delta so với 10 ngày trước
        raw_diff = raw - raw.shift(10)
        # Chuẩn hóa bằng rolling rank (phân phối đồng nhất, loại bỏ outlier)
        normalized = (raw_diff.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_017(df, window=20):
        # Tính log return
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        raw = sum_open_5 * sum_ret_5
        raw_diff = raw - raw.shift(10)
        # Chuẩn hóa bằng dynamic tanh (giữ cường độ tín hiệu)
        normalized = np.tanh(raw_diff / raw_diff.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_018(df, window=20):
        # Tính log return
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        raw = sum_open_5 * sum_ret_5
        raw_diff = raw - raw.shift(10)
        # Chuẩn hóa bằng rolling z-score và clip (phù hợp cho spread/oscillator)
        mean = raw_diff.rolling(window).mean()
        std = raw_diff.rolling(window).std()
        z = (raw_diff - mean) / std
        normalized = z.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_019(df):
        # Tính log return
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        raw = sum_open_5 * sum_ret_5
        raw_diff = raw - raw.shift(10)
        # Chuẩn hóa bằng sign (breakout/trend thuần túy)
        normalized = np.sign(raw_diff)
        return normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_020(df, p1=0.05, p2=20):
        # Tính log return
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        raw = sum_open_5 * sum_ret_5
        raw_diff = raw - raw.shift(10)
        # Winsorized Fisher (xử lý heavy tails, giữ cấu trúc phân phối)
        low = raw_diff.rolling(p2).quantile(p1)
        high = raw_diff.rolling(p2).quantile(1 - p1)
        winsorized = raw_diff.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_021(df, window=6, smoothing_window=9):
        high = df['high']
        low = df['low']
        close = df['close']

        tsmax_high = high.rolling(window=window).max()
        tsmin_low = low.rolling(window=window).min()

        raw = ((tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9)) * 100

        sma_raw = raw.rolling(window=smoothing_window).mean()

        normalized = (sma_raw.rolling(window=smoothing_window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_022(df, window=6, smoothing_window=9):
        high = df['high']
        low = df['low']
        close = df['close']

        tsmax_high = high.rolling(window=window).max()
        tsmin_low = low.rolling(window=window).min()

        raw = ((tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9)) * 100

        sma_raw = raw.rolling(window=smoothing_window).mean()

        normalized = np.tanh(sma_raw / sma_raw.rolling(window=smoothing_window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_023(df, window=6, smoothing_window=9):
        high = df['high']
        low = df['low']
        close = df['close']

        tsmax_high = high.rolling(window=window).max()
        tsmin_low = low.rolling(window=window).min()

        raw = ((tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9)) * 100

        sma_raw = raw.rolling(window=smoothing_window).mean()

        rolling_mean = sma_raw.rolling(window=smoothing_window).mean()
        rolling_std = sma_raw.rolling(window=smoothing_window).std()

        z_score = (sma_raw - rolling_mean) / (rolling_std + 1e-9)
        normalized = z_score.clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_024(df, window=6, smoothing_window=9):
        high = df['high']
        low = df['low']
        close = df['close']

        tsmax_high = high.rolling(window=window).max()
        tsmin_low = low.rolling(window=window).min()

        raw = ((tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9)) * 100

        sma_raw = raw.rolling(window=smoothing_window).mean()

        normalized = np.sign(sma_raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_v3_025(df, window=6, smoothing_window=9, quantile_level=0.05):
        high = df['high']
        low = df['low']
        close = df['close']

        tsmax_high = high.rolling(window=window).max()
        tsmin_low = low.rolling(window=window).min()

        raw = ((tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9)) * 100

        sma_raw = raw.rolling(window=smoothing_window).mean()

        low_bound = sma_raw.rolling(window=smoothing_window).quantile(quantile_level)
        high_bound = sma_raw.rolling(window=smoothing_window).quantile(1 - quantile_level)

        winsorized = sma_raw.clip(lower=low_bound, upper=high_bound)

        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

class Domains:
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
    def get_list_of_alphas(verbosity=1):
        header = f'\x1b[90mDomains.get_list_of_alphas\x1b[0m: '
        dic_alphas = {}
        # noinspection PyTypeChecker
        base_list = list(range(1, 302)) + ['alpha_trend_efficiency', 'alpha_custom_resid']
        alpha_factor_miner = [f"alpha_factor_miner_{str(i).rjust(3, '0')}" for i in range(0, 300)]
        alpha_factor_miner = [f"alpha_factor_miner_v2_{str(i).rjust(3, '0')}" for i in range(0, 300)]
        alpha_factor_miner = [f"alpha_factor_miner_v3_{str(i).rjust(3, '0')}" for i in range(0, 300)]

        custom_c_list = [f"c{str(i).rjust(2, '0')}" for i in range(1, 51)]
        new_alpha_list = [f"alpha_new_{str(name).rjust(3, '0')}" for name in list(range(1, 101))]
        
        for alpha_name in base_list + custom_c_list + new_alpha_list + alpha_factor_miner:
            if type(alpha_name) == int:
                alpha_name = str(alpha_name).rjust(3, '0')
            if alpha_name[:6] != 'alpha_':
                alpha_name = f'alpha_{alpha_name}'
            dic = globals()
            alpha = None

            if alpha_name in dic:
                alpha = dic[alpha_name]
            else:
                try:
                    alpha = Alphas().__getattribute__(alpha_name)
                except Exception as e:
                    if len(str(e)) == 0: print(e, end='')

            if alpha is None:
                if verbosity >= 2:
                    print(
                        header,
                        f'Found 0 alpha with name=\x1b[91m{alpha_name}\x1b[0m')
            else:
                dic_alphas[alpha_name] = alpha
                # print(f'\n\n\x1b[96m{alpha_name}:\x1b[0m\n', alpha(df))

        if verbosity >= 2:
            print(header,
                  f'Found \x1b[93m{len(dic_alphas)}\x1b[0m alpha functions')
        return dic_alphas




