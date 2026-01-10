import pandas as pd
import json
# from redis import StrictRedis
from datetime import datetime as dt
from datetime import timedelta
from time import time
import numpy as np
import pickle
import datetime

class RESOURCES:
    THE_DIC_FREQS_FN = '/home/ubuntu/duy/new_strategy/gen1_2/dic_freqs.pkl'
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
    @staticmethod
    def ts_rank_normalized(df, window=10):
        df2 = df.rolling(window).rank()
        
        return (df2-1) / (window-1)
    @staticmethod
    def ts_weighted_mean(df, window=10):
        # noinspection PyUnresolvedReferences
        from talib import WMA

        wma = WMA(df,timeperiod=window)
        return wma

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
    def ts_scale(df, k=1):
        """
        Wrapper function to estimate scaling.
        :param df: a pandas DataFrame.
        :return: a pandas DataFrame rescaled df such that sum(abs(df)) = 1
        """
        denom = df.abs().sum()
        if denom == 0:
            return df * 0
        return df / denom * k

    @staticmethod
    def ts_delay(df, period=1):
        """
        Wrapper function to estimate delay.
        :param df: a pandas DataFrame.
        :param period: the delay grade.
        :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
        """
        return df.shift(period)

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
    #####
    @staticmethod
    def alpha_MFI(df,buy_threshold ,sell_threshold, over_sold, over_bought, time_period ):
        
        # buy_threshold = [0 -> 1]
        # sell_threshold = [0 -> 1]
        
        # over_sold = [10 -> 100]
        # over_bought = [10 -> 100]

        # time_period = [10 -> 20]
        
        # buy_threshold=0.7
        # sell_threshold=-0.5     
        # over_sold=70
        # over_bought=30
        # time_period=14

        tp  = (df['high'] + df['low'] + df['close']) / 3.0
        rmf = tp * df['matchingVolume']

        d = tp.diff()
        pos_mf = pd.Series(np.where(d > 0, rmf, 0.0), index=df.index)
        neg_mf = pd.Series(np.where(d < 0, rmf, 0.0), index=df.index)

        pos_roll = pos_mf.rolling(time_period, min_periods=time_period).sum()
        neg_roll = neg_mf.rolling(time_period, min_periods=time_period).sum()

        mfr = pos_roll / neg_roll
        mfi = 100.0 - (100.0 / (1.0 + mfr))

        mfi = (mfi
            .mask((neg_roll == 0) & (pos_roll > 0), 100.0)   
            .mask((pos_roll == 0) & (neg_roll > 0), 0.0)     
            .mask((pos_roll == 0) & (neg_roll == 0))         
            )

        df['mfi'] = mfi
        df['bop'] = (df['close'] - df['open'])/ (df['high']-df['low'])

        conditions = [(df['bop']>buy_threshold) & (df['mfi'] < over_sold),
                      (df['bop']<sell_threshold) & (df['mfi']>over_bought)]
        choices = [1,-1]
        df['signal'] = np.select(conditions,choices,0)
        return df['signal']

    #####
    @staticmethod
    def alpha_ursi(df:pd.DataFrame, window = 14, factor = 14):
        src = df['close']
        upper = src.rolling(window).max()
        lower = src.rolling(window).min()

        r = upper - lower
        d = src.diff()

        diff = np.where(upper > upper.shift(1), r, np.where(lower < lower.shift(1), -r, d))
        diff = pd.Series(diff, index=src.index)

        num = diff.ewm(alpha = 1/window, adjust= False).mean()
        den = diff.abs().ewm(alpha = 1/window, adjust= False).mean()
        arsi = (num / den) * 50 + 50

        signal = arsi.ewm(span = factor, adjust= False).mean()
        df['alpha'] = np.where((arsi > signal) & (arsi.shift(1) < signal.shift(1)) & (arsi.between(0,60)) ,1 , 0 )


        return df['alpha']
    
    
    @staticmethod
    def alpha_418(df:pd.DataFrame,window=10):
        df = df.copy()
        df['quantity'] = df['matchingVolume'] * (df['close']-df['open'])
        df['signal'] = O.ts_rank_normalized(df['quantity'],window)
        df['signal'] = df['signal'] * 2 - 1
        return df['signal']
    
    @staticmethod
    def alpha_rti(df:pd.DataFrame,window=60,factor = 0.6):
        df = df.copy()
        df['upper_trend'] = df['close'] + df['close'].rolling(window=window).std()
        df['lower_trend'] = df['close'] - df['close'].rolling(window=window).std()
        df['UpperTrend'] = df['upper_trend'].rolling(window=window).quantile(factor)
        df['LowerTrend'] = df['lower_trend'].rolling(window=window).quantile(1-factor)
        # df['signal'] = ((df['close'] - df['LowerTrend']) / (df['UpperTrend'] - df['LowerTrend']))

        df['signal'] = ((df['close'] - df['open']) / (df['UpperTrend'] - df['LowerTrend']))
        df['signal'] = O.ts_rank_normalized(df['signal'])
        df['signal'] = df['signal'] * 2 - 1
                                            
        return df['signal']
    
    @staticmethod
    def alpha_donchian_channel(df:pd.DataFrame,window=10):
        df = df.copy()
        df['highestHigh'] = df['high'].rolling(window=window).max()
        df['lowestLow'] = df['low'].rolling(window=window).min()
        df['mid'] = (df['highestHigh'] + df['lowestLow']) / 2
        df['signal'] = (df['close']- df['open']) / (df['highestHigh'] - df['lowestLow'])
        return df['signal']
    
    #####
    
    @staticmethod
    def alpha_kema(df: pd.DataFrame, window=10,factor=2):
        df = df.copy()
        df['ema'] = df['close'].ewm(span=window, adjust=False).mean()
        def atr(data: pd.Series, length) -> pd.Series:
            tr = np.maximum(
                data["high"] - data["low"],
                np.maximum(
                    np.abs(data["high"] - data["close"].shift()),
                    np.abs(data["low"] - data["close"].shift()),
                ),
            )

            first_atr = np.mean(tr[1 : length + 1])
            atr = tr.copy()
            atr[: length + 1] = first_atr
            atr = atr.ewm(span=length, min_periods=length + 1, adjust=False).mean()
            return atr
        df['atr'] = atr(df, 14)
        df['upper'] = df['ema'] + factor * df['atr']
        df['lower'] = df['ema'] - factor * df['atr']
        return df['upper'], df['lower']
    @staticmethod
    def alpha_channel_breakout(df: pd.DataFrame,window=10):
        df = df.copy()
        df['highBound'] = df['high'].rolling(window=window).max().shift(1)
        df['lowBound'] = df['low'].rolling(window=window).min().shift(1)
        df['signal'] = np.select(
            [df['high']>=df['highBound'],df['low']<=df['lowBound']],
            [1,-1],
            np.nan
        )
        return df['signal']

    @staticmethod
    def alpha_keltner(df: pd.DataFrame,window=10):
        df = df.copy()
        df['hlc'] = (df['high'] + df['low'] + df['close'])/3
        # df['xPrice'] = df['hlc'].ewm(span=window,adjust=False).mean()
        df['xPrice'] = df['hlc'].rolling(window = window).mean()
        df['xMove'] = (df['high']-df['low']).ewm(span=window,adjust=False).mean()
        df['xUpper'] = df['xPrice'] + df['xMove']
        df['xLower'] = df['xPrice'] - df['xMove']
        df['signal'] = (df['close']-df['open']) / (df['xUpper'] - df['xLower'])
       
        return df['signal']

    @staticmethod
    def alpha_bbb(df: pd.DataFrame, window=20, factor=1):
        df = df.copy()
        df['basis'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['upper'] = df['basis'] + df['std'] * factor
        df['lower'] = df['basis'] - df['std'] * factor
        df['signal'] = (df['close'] - df['open']) / (df['upper'] - df['lower'])
    
        return df['signal']
    
    @staticmethod
    def alpha_questionable(df):
        signal = df['close'] - df['open']
        return signal

    @staticmethod
    def alpha_zscore(df, window=10):
        signal = O.zscore(
            df['matchingVolume']
                *
            df['close'].diff(),
            window=window)
        signal = signal / window * 2
        return signal

    @staticmethod
    def alpha_001(df,window=20):
        returns = df['return']
        dff = df[['close']].copy()
        ffilter = returns < 0
        dff.loc[ffilter, 'close'] = O.ts_std(returns, window)[ffilter]
        base = dff['close']
        x = O.ts_argmax(O.power(base, 2.), 5)
        signal = (O.ts_rank(x) - 0.5)
        normalized_signal = signal / 5 - 1
        # normalized_signal.min(), normalized_signal.max()  # -0.9, 0.9
        return normalized_signal
    
    @staticmethod
    def alpha_new_001(df: pd.DataFrame, window=20, factor=3):
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        # 2. Sức mạnh khối lượng tương đối
        adv = O.ts_mean(df['matchingVolume'], window=window)
        relative_volume = df['matchingVolume'] / (adv + 0.001)
        # 3. Tín hiệu thô = Động lượng * Khối lượng
        raw_signal = intraday_momentum * relative_volume
        # 4. Làm mượt tín hiệu để giảm nhiễu
        signal = O.ts_weighted_mean(raw_signal, window=factor)
        return signal
    
    @staticmethod
    def alpha_002(df,window=6):
        U.ignore_warnings()
        volume = df['matchingVolume']
        close = df['close']
        openn = df['open']
        signal = (-1 * O.ts_corr(O.ts_rank(O.ts_delta(np.log(volume), 2)),
                                 O.ts_rank(((close - openn) / openn)), window))
        return signal

    @staticmethod
    def alpha_new_002(df: pd.DataFrame,window=14):
        returns = df['return'].fillna(0)
        gains = returns.clip(lower=0)
        losses = abs(returns.clip(upper=0))
    
        # 2. Calculate average gains and losses
        avg_gain = O.ts_mean(gains, window=window)
        avg_loss = O.ts_mean(losses, window=window)
    
        # 3. Calculate RSI
        rs = avg_gain / (avg_loss + 0.00001) # Add epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
    
        # 4. Create a mean-reversion signal centered around 50
        signal = -(rsi - 50)
        
        # 5. Scale signal from [-50, 50] to [-1, 1]
        signal = signal / 50
        
        
        return -signal        
    
    @staticmethod
    def alpha_003(df,window=10):
        signal = -1 * \
                 O.ts_corr(
                     O.ts_rank(df['open']),
                     O.ts_rank(df['matchingVolume']),
                     window=window)
        signal = -signal
        return signal

    @staticmethod
    def alpha_new_003(df: pd.DataFrame,window=20):
        price_range = df['high'] - df['low']
        avg_range = O.ts_mean(price_range, window=window)
        range_expansion = price_range / (avg_range + 1e-9)
        
        position_in_range = (df['close'] - df['low']) / (price_range + 1e-9)
        position_signal = (2 * position_in_range) - 1
        
        raw_signal = range_expansion * position_signal
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal


    @staticmethod
    def alpha_004(df,window=9):
        signal =  1 * O.ts_rank(O.ts_rank(df['low']), window)
        signal = signal / 5 - 1
        return signal

    @staticmethod
    def alpha_new_004(df: pd.DataFrame,window=20,factor=60):
        price_increase = O.ts_rank_normalized(O.ts_delta(df['close'], window), window)
        low_volatility = 1 - O.ts_rank_normalized(O.ts_std(df['return'], factor), window)
        raw_signal = price_increase * low_volatility
        
        # Signal is already [0, 1], map to [-1, 1]
        signal = 2 * raw_signal - 1
        

        
        return signal 
    
    @staticmethod
    def alpha_005(df,window=200):
        df['vwap'] = Domains.compute_vwap(df,window)['vwap']
        signal = 1 * \
                 O.ts_rank(
                     (df['open'] - df['vwap'].rolling(10).mean())) * \
                 (-1 * abs(
                     O.ts_rank((df['close'] - df['vwap']))))
        signal = signal / 50 + 1
        return signal*(-1)

    @staticmethod
    def alpha_new_005(df: pd.DataFrame,window=3):

        directional_range = df['close'] - df['open']
        total_range = df['high'] - df['low']
        
        trend_ratio = directional_range / (total_range + 1e-9)
        
        # The signal is naturally in [-1, 1]. We smooth it to reduce noise.
        signal = O.ts_mean(trend_ratio, window)
        
        
        return signal
    
    @staticmethod
    def alpha_006(df, window=10):
        signal = 1 * O.ts_corr(df['open'], df['matchingVolume'], window)
        return signal

    @staticmethod
    def alpha_new_006(df: pd.DataFrame, window=8):
        higher_high = (df['high'] > df['high'].shift(1)).astype(int)
        higher_low = (df['low'] > df['low'].shift(1)).astype(int)
        
        lower_high = (df['high'] < df['high'].shift(1)).astype(int)
        lower_low = (df['low'] < df['low'].shift(1)).astype(int)
        
        uptrend_score = O.ts_sum(higher_high & higher_low, window)
        downtrend_score = O.ts_sum(lower_high & lower_low, window)
        
        # Normalize score to [-1, 1]
        signal = (uptrend_score - downtrend_score) / window
        
        
        return signal
    
    @staticmethod
    def alpha_007(df, halflife=0, window=100):
        c = df['close']
        v = df['matchingVolume']
        adv20 = df.rolling(20)\
                  .matchingVolume\
                  .mean()

        delta7 = O.ts_delta(c, 7).fillna(0)
        signal = O.ts_rank(abs(delta7), 60)
        signal = signal.mul(np.sign(delta7)) \
                       .where(adv20 < v, -1)
        signal = (signal.rolling(window).rank() - 50) / 50
        if halflife != 0:
            signal = signal.ewm(halflife=halflife).mean()
        return signal

    @staticmethod
    def alpha_new_007(df: pd.DataFrame,window=30):
        lwma = O.decay_linear(df['close'], d=window)
        raw_signal = df['close'] - lwma
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_008(df, halflife=0,window=10):
        open5 = df['open'].rolling(5).sum()
        return5 = df['return'].rolling(5).sum()
        x = open5 * return5
        signal = 1 * O.ts_rank(x - x.shift(window))
        if halflife != 0:
            signal = signal.ewm(halflife=halflife).mean()
        return signal


    @staticmethod
    def alpha_new_008(df: pd.DataFrame,window=20,factor=2):
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        upper_band = mean + factor * std
        lower_band = mean - factor * std
        
        buy_breakout = df['close'] - upper_band
        sell_breakout = df['close'] - lower_band
        
        raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_009(df,window=5):
        close_diff = df['close'].diff(1)
        signal = close_diff
        flt_min = 0 >= close_diff.rolling(window).min()
        flt_max = close_diff.rolling(window).max() < 0
        signal.loc[flt_min & flt_max] = close_diff
        signal.loc[flt_min & (~flt_max)] = -close_diff

        lower, upper = signal.quantile(0.05), signal.quantile(0.95)
        normalized_signal = signal / (upper - lower)
        return -normalized_signal

    @staticmethod
    def alpha_new_009(df: pd.DataFrame,window=20,factor=3):
        price_accel = O.ts_delta(df['return'], 1) # 'return' is typically close.diff(1)

        # 2. Relative Volume
        adv = O.ts_mean(df['matchingVolume'], window=window)
        relative_volume = df['matchingVolume'] / (adv + 1e-9)

        # 3. Raw Signal: Acceleration confirmed by volume
        raw_signal = price_accel * relative_volume
        
        # 4. Rank, scale, and smooth
        ranked_signal = O.ts_rank_normalized(raw_signal, window=window)
        scaled_signal = 2 * ranked_signal - 1 # Scale from [0, 1] to [-1, 1]
        signal = O.ts_weighted_mean(scaled_signal, window=factor)
        
        return signal
    
    @staticmethod
    def alpha_010(df,window=10):
        close_diff = df['close'].diff(1)
        signal = close_diff
        # If not all lows are lower_low
        flt_min = 0 >= close_diff.rolling(4).min()
        # AND not all_highs are higher high
        flt_max = close_diff.rolling(4).max() >= 0
        # Then trade couter trend
        # (df['matchingVolume'] - df['matchingVolume'].mean())
        signal.loc[flt_min & flt_max] = -close_diff
        normalized_signal = O.ts_rank(signal, window=window) / 5 - 1
        return - normalized_signal

    @staticmethod
    def alpha_new_010(df: pd.DataFrame,window=10):
        # 1. Find the highest high and lowest low over the window
        highest_high = O.ts_max(df['high'], window=window)
        lowest_low = O.ts_min(df['low'], window=window)

        # 2. Calculate position in range (similar to Stochastic Oscillator)
        price_range = highest_high - lowest_low
        position_in_range = (df['close'] - lowest_low) / (price_range + 1e-9)

        # 3. Scale signal to [-1, 1]
        # A value of 1 means close is at the highest high (strong buy)
        # A value of 0 means close is at the lowest low (strong sell)
        signal = 2 * position_in_range - 1
        
  
        return signal

    @staticmethod
    def alpha_011(df,window=3):
        df['vwap'] = Domains.compute_vwap(df)['vwap']
        signal = (
                     O.ts_rank(O.ts_max((df['vwap'] - df['close']), window))
                     +
                     O.ts_rank(O.ts_min((df['vwap'] - df['close']), window))
                 ) \
                 * \
                 O.ts_rank(df['matchingVolume'].diff(3))
        direction = df['vwap'] - df['close']
        signal *= np.sign(direction)
        signal = -signal
        normalized_signal = signal / 200
        return normalized_signal

    @staticmethod
    def alpha_new_011(df: pd.DataFrame,window=50,factor=3):
        df = O.compute_vwap(df, window=window) 
        
        spread = df['close'] - df['vwap']
        
        # 3. Calculate the momentum of the spread
        spread_momentum = O.ts_delta(spread, period=factor)
        
        # 4. Rank and scale the momentum to get a cross-sectional signal
        ranked_signal = O.ts_rank_normalized(spread_momentum, window=window)
        signal = 2 * ranked_signal - 1 # Scale from [0, 1] to [-1, 1]
        
        
        return signal

    @staticmethod
    def alpha_012(df, window=0):

        signal = np.sign(df['matchingVolume'].diff()) * (-1 * df['close'].diff())
        signal = -signal
        # noinspection PyUnresolvedReferences
        lower, upper = signal.quantile(0.05), signal.quantile(0.95)
        normalized_signal = signal / (upper - lower)
        if window != 0:
            normalized_signal = normalized_signal.ewm(halflife=window).mean()
        return normalized_signal

    @staticmethod
    def alpha_new_012(df: pd.DataFrame,window=100,factor=2.5):
        adv = O.ts_mean(df['matchingVolume'], window=window)
        is_volume_spike = df['matchingVolume'] > (adv * factor)
        
        # Signal is the inverse of the daily return on spike days, otherwise 0
        reversal_signal = -df['return'].where(is_volume_spike, 0)
        
        # Rank to create a standardized signal 
        ranked_signal = O.ts_rank_normalized(reversal_signal, window=window)
        signal = 2 * ranked_signal - 1
        
        
        return -signal

    @staticmethod
    def alpha_013(df,window=5):
        signal = -1 \
                 * \
                 O.ts_rank(
                     O.ts_cov(
                         O.ts_rank(df['close']),
                         O.ts_rank(df['matchingVolume']),
                         window=window)
                 )
        signal = (signal - 5) / 5
        return signal

    @staticmethod
    def alpha_new_013(df: pd.DataFrame, window=5):
        
        """
        Logic: Directional Efficiency. Measures how "efficiently" the price moved
        from open to close, relative to the potential range it could have moved in
        that direction. High efficiency is a sign of strong, one-sided conviction.
        smooth_window = [2,3,4]
        """
        

        up_day = df['close'] > df['open']
        down_day = ~up_day
        
        # Efficiency for up days: (close-open) / (high-open)
        up_efficiency = (df['close'] - df['open']) / (df['high'] - df['open'] + 1e-9)
        
        # Efficiency for down days: (close-open) / (open-low)
        down_efficiency = (df['close'] - df['open']) / (df['open'] - df['low'] + 1e-9)
        
        raw_signal = up_efficiency.where(up_day, 0) + down_efficiency.where(down_day, 0)
        
        # Smooth the signal
        signal = O.ts_mean(raw_signal, window=window)

        return signal

    @staticmethod
    def alpha_014(df,window=10):
        signal = (1 * O.ts_rank(df['return'].diff(3))) \
                 * \
                 O.ts_corr(df['open'], df['matchingVolume'], window)
        signal = signal / 10
        return signal

    @staticmethod
    def alpha_new_014(df: pd.DataFrame,window=10,factor=0.7):
        """
        Logic: Climactic Move Reversal. A climactic move (a day with both very
        high volume and a very wide price range) often signals the end of a trend.
        This alpha bets on a reversal following such a day.
        window = [5,10,...,195,200]
        percentile = [0.65,0.7,...,0.95]
        """  
        is_wide_range = O.ts_rank_normalized(df['high'] - df['low'], window) > factor
        is_high_vol = O.ts_rank_normalized(df['matchingVolume'], window) > factor
        
        # Signal is the negative of the return on climactic days
        raw_signal = -df['return'].where(is_wide_range & is_high_vol, 0)
        
        # Rank and scale
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = 2 * ranked_signal - 1
        
        return -signal
    
    @staticmethod
    def alpha_015(df,window=3):
        signal = 1 \
                 * \
                 O.ts_sum(
                     O.ts_rank(O.ts_corr(O.ts_rank(df['high']),
                                         O.ts_rank(df['matchingVolume']),
                                         window)),
                     window)
        normalized_signal = signal / 15 + 1
        normalized_signal = -normalized_signal
        return normalized_signal

    @staticmethod
    def alpha_new_015(df: pd.DataFrame,window=5,factor=5):
        """
        Logic: Hybrid - ADX Regime Switching. Uses the Average Directional Index (ADX)
        to determine trend strength. If the trend is strong (ADX > threshold), it uses
        a momentum strategy. If the trend is weak, it uses a mean-reversion strategy.
        """
        # Simplified ADX Calculation
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = df['low'].diff().clip(upper=0).abs()
        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        is_trending = adx > factor

        # Momentum and Mean-Reversion signals
        momentum_signal = O.ts_delta(df['close'], 3)
        reversion_signal = -df['return']

        raw_signal = momentum_signal.where(is_trending, reversion_signal)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window=factor)
        signal = 2 * ranked_signal - 1
        
        return signal
    
    @staticmethod
    def alpha_016(df,window=5):
        signal = (
            -1
            *
            O.ts_rank(O.ts_corr(
                O.ts_rank(df['high']),
                O.ts_rank(df['matchingVolume']),
                window))
        )
        HARDCODED_ALPHA_016_CALIBERATION = 0
        signal = signal / 5 + HARDCODED_ALPHA_016_CALIBERATION
        return signal

    @staticmethod
    def alpha_new_016(df: pd.DataFrame,window=20,factor=50):
        """
        Logic: Hybrid - Momentum-Weighted Mean Reversion. A mean-reversion signal
        (from Bollinger Bands) is weighted by long-term momentum. This makes buy
        signals stronger in an uptrend and sell signals stronger in a downtrend.
        """
        # Mean-Reversion Signal (from f22)
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        position_in_band = (df['close'] - (mean - 2*std)) / ((mean + 2*std) - (mean - 2*std) + 1e-9)
        reversion_signal = -(position_in_band - 0.5) * 2

        # Momentum Weight
        momentum_weight = O.ts_rank_normalized(O.ts_delta(df['close'], factor), factor)
        
        signal = reversion_signal * momentum_weight
        
        return -signal
    
    @staticmethod
    def alpha_017(df,window=5):
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()
        direction = np.sign(df['close'].diff(1).diff(1))
        signal = (
            (
                (
                    1
                    *
                    O.ts_rank(O.ts_rank(df['close'], 10))
                )
                *
                O.ts_rank(
                    df['close'].diff(1).diff(1))
            )
            *
            O.ts_rank(O.ts_rank((df['matchingVolume'] / adv20), window)))
        signal = signal / 1000 * direction
        return signal

    @staticmethod
    def alpha_new_017(df: pd.DataFrame,window=20):
        # Momentum Signal: A breakout happened in the last 3 days
        highest_high = O.ts_max(df['high'], window).shift(1)
        breakout_signal = (df['close'] > highest_high).astype(int)
        breakout_recently = O.ts_sum(breakout_signal, 3) > 0

        # Mean-Reversion Signal: Bet against the 2-day trend
        reversion_signal = -O.ts_delta(df['close'], 2)

        raw_signal = reversion_signal.where(breakout_recently, 0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window=window)
        signal = 2 * ranked_signal - 1
        
        return -signal
    
    @staticmethod
    def alpha_018(df,window=10):
        bar_height = df['close'] - df['open']
        signal = \
        (
            -1
            *
            O.ts_rank
            (
                # O.ts_std(abs(bar_height), 5)
                # +
                bar_height,
                window=window
                # +
                # O.ts_corr(df['close'],
                #           df['open'],10)
            )
        )
        normalized_signal = -(signal / 5 + 1) / 0.9 - 1/9
        return normalized_signal

    @staticmethod
    def alpha_new_018(df: pd.DataFrame,window=20,factor=3):
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        bb_width = (4 * std) / (mean + 1e-9)
        
        # Regime: Is volatility expanding?
        is_expanding_vol = O.ts_delta(bb_width, factor) > 0

        # Signal: Bet against the recent trend
        reversion_signal = -O.ts_delta(df['close'], factor)

        raw_signal = reversion_signal.where(is_expanding_vol, 0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window=window)
        signal = 2 * ranked_signal - 1
        
        return -signal
    
    @staticmethod
    def alpha_019(df, window=250):
        delayed_close = df['close'].shift(7)
        signal =                                                        \
            (
                 -np.sign
                 (
                     (df['close'] - delayed_close)
                     +
                     delayed_close
                )
            )                                                           \
            *                                                           \
            (
                1 + O.ts_rank(1 + df['return'].rolling(window).sum())
            )
        min_to_max_range = 9 / 2 #signal.max() - signal.min()
        shift = 1.4435555555600001
        signal = signal / min_to_max_range

        signal += shift
        # signal.hist(bins=100)
        # plt.show()
        # print(signal.min(), signal.max())
        return signal

    @staticmethod
    def alpha_new_019(df: pd.DataFrame, window=10, factor=14):

        momentum_signal = O.ts_delta(df['close'], window)
        ranked_momentum = O.ts_rank_normalized(momentum_signal, window) * 2 - 1

        # Mean-Reversion Brake (Value is high when RSI is low, and low when RSI is high)
        returns = df['return'].fillna(0)
        gains = returns.clip(lower=0).ewm(alpha=1/factor, adjust=False).mean()
        losses = abs(returns.clip(upper=0)).ewm(alpha=1/factor, adjust=False).mean()
        rs = gains / (losses + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        reversion_brake = 1 - (rsi / 100)

        signal = ranked_momentum * reversion_brake
        
        return signal
    
    @staticmethod
    def alpha_020(df,window=10):
        # freq = 30
        # df = glob_obj.dic_freqs[freq]
        low_gap  = (df['open'] - df['low'].shift(1))
        high_gap = (df['open'] - df['high'].shift(1))
        close_gap = df['open'] - df['close'].shift(1)
        signal = -np.sign(low_gap) * \
        (
            (
                (-1 * O.ts_rank(high_gap, window))
                    * O.ts_rank(close_gap, window)
            )
            *
            ((O.ts_rank(low_gap, window) - 5.5) / 4.5 / 100) # best sharpe 2
            # O.ts_rank(low_gap) / 1000 # best sharpe 1.36
        )

        # signal.hist(bins=100)
        # plt.show()
        return signal

    @staticmethod
    def alpha_new_020(df: pd.DataFrame,window=30,factor=2):
        price_change = O.ts_delta(df['close'], factor)
        volume_rank = O.ts_rank_normalized(df['matchingVolume'], window)
        signal = price_change * volume_rank
        signal = O.ts_rank_normalized(signal, window) * 2 - 1
        
        return signal
    
    @staticmethod
    def alpha_021(df,window=20):
        mean2 = df['close'].rolling(2).mean()
        mean8 = df['close'].rolling(8).mean()
        std8 = df['close'].rolling(8).std()

        adv20 = df.rolling(window) \
            .matchingVolume \
            .mean()
        volume = df['matchingVolume']
        relative_vol = volume / adv20
        flt1 = (mean8 + std8) < mean2
        flt2 = mean2 < (mean8 - std8)
        flt3 = 1 <= relative_vol
        signal = df.assign(signal=-1)['signal']
        signal.loc[~flt1 & flt2] = 1
        signal.loc[~flt1 & (~flt2) & flt3] = 1
        signal.loc[~flt1 & (~flt2) & (~flt3)] = -1
        signal = -signal
        # signal = signal.ewm(span=window).mean()
        # signal = signal.rolling(window).mean()
        # signal.hist(bins=100)
        # plt.show()
        return signal

    @staticmethod
    def alpha_new_021(df: pd.DataFrame,window=20):

        # 1. TRIMA Signal
        trima_20 = O.linear_weighted_moving_average(df['close'], window_size=5)
        raw_signal_trima = df['close'] - trima_20

        # 2. Volume Strength Signal
        volume_strength = df['matchingVolume'] / (O.ts_mean(df['matchingVolume'], 5) + 1e-9)

        # 3. Combine Signals
        combined_raw_signal = raw_signal_trima * volume_strength

        # 4. Normalize
        signal = 2 * O.ts_rank_normalized(combined_raw_signal, window=window) - 1

        
        return signal
    
    @staticmethod
    def alpha_new_022(df: pd.DataFrame,window=50):

        # 1. Smoothed BOP Signal
        raw_bop = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        bop_signal = O.ts_mean(raw_bop, 5)

        # 2. Interday Momentum Confirmation
        interday_momentum = O.ts_rank_normalized(df['close'].diff(3), 30)

        # 3. Combine Signals
        combined_raw_signal = bop_signal * interday_momentum

        # 4. Normalize
        signal = 2 * O.ts_rank_normalized(combined_raw_signal, window=window) - 1
        
        return signal
    
    @staticmethod
    def alpha_022(df, window=20):
        """
        great ppc
        Scanner.inspect_top_k:  0 / 15: freq=33 threshold=0.6 n=133 Sharpe=2.39 ppc=1.75
        Scanner.inspect_top_k:  1 / 15: freq=34 threshold=0.7 n=51 Sharpe=1.78 ppc=2.65
        Scanner.inspect_top_k:  2 / 15: freq=33 threshold=0.7 n=73 Sharpe=1.61 ppc=2.00
        Scanner.inspect_top_k:  3 / 15: freq=33 threshold=0.75 n=47 Sharpe=1.51 ppc=2.63
        Scanner.inspect_top_k:  4 / 15: freq=13 threshold=0.3 n=928 Sharpe=1.49 ppc=0.43
        Scanner.inspect_top_k:  5 / 15: freq=34 threshold=0.4 n=251 Sharpe=1.46 ppc=0.93
        Scanner.inspect_top_k:  6 / 15: freq=33 threshold=0.4 n=289 Sharpe=1.43 ppc=0.83
        Scanner.inspect_top_k:  7 / 15: freq=34 threshold=0.6 n=95 Sharpe=1.41 ppc=1.57
        Scanner.inspect_top_k:  8 / 15: freq=32 threshold=0.5 n=225 Sharpe=1.37 ppc=0.74
        Scanner.inspect_top_k:  9 / 15: freq=42 threshold=0.8 n=8 Sharpe=1.36 ppc=2.91
        Scanner.inspect_top_k:  10 / 15: freq=59 threshold=0.7 n=9 Sharpe=1.36 ppc=3.25
        """
        signal = 1 / window * \
                 (
                     O.ts_corr(df['high'],
                               df['matchingVolume'], 5)
                       .diff(5)
                     *
                     O.ts_rank(O.ts_std(df['close'],
                                        window))
                 )
        return -signal

    @staticmethod
    def alpha_023(df, window=20, halflife=0):
        """
        Scanner.inspect_top_k:  0 / 15: freq=49 threshold=0.5 n=151 Sharpe=1.44 ppc=1.17
        Scanner.inspect_top_k:  1 / 15: freq=54 threshold=0.1 n=489 Sharpe=1.43 ppc=0.45
        Scanner.inspect_top_k:  2 / 15: freq=26 threshold=0.4 n=215 Sharpe=1.42 ppc=1.07
        Scanner.inspect_top_k:  3 / 15: freq=51 threshold=0.2 n=359 Sharpe=1.41 ppc=0.61
        Scanner.inspect_top_k:  4 / 15: freq=27 threshold=0.3 n=335 Sharpe=1.38 ppc=0.73
        Scanner.inspect_top_k:  5 / 15: freq=52 threshold=0.2 n=364 Sharpe=1.34 ppc=0.56
        Scanner.inspect_top_k:  6 / 15: freq=28 threshold=0.3 n=296 Sharpe=1.34 ppc=0.88
        Scanner.inspect_top_k:  7 / 15: freq=52 threshold=0.1 n=476 Sharpe=1.33 ppc=0.41
        Scanner.inspect_top_k:  8 / 15: freq=53 threshold=0.2 n=365 Sharpe=1.32 ppc=0.56
        Scanner.inspect_top_k:  9 / 15: freq=26 threshold=0.3 n=305 Sharpe=1.29 ppc=0.76
        Scanner.inspect_top_k:  10 / 15: freq=27 threshold=0.2 n=482 Sharpe=1.28 ppc=0.50

        ================================ Reversed signal ================================
        Scanner.inspect_top_k:  0 / 15: freq=10 threshold=0.85 n=86 Sharpe=1.77 ppc=2.61
        Scanner.inspect_top_k:  1 / 15: freq=14 threshold=0.85 n=84 Sharpe=1.70 ppc=1.95
        Scanner.inspect_top_k:  2 / 15: freq=10 threshold=0.8 n=97 Sharpe=1.58 ppc=2.01
        Scanner.inspect_top_k:  3 / 15: freq=11 threshold=0.85 n=88 Sharpe=1.54 ppc=1.75
        Scanner.inspect_top_k:  4 / 15: freq=11 threshold=0.8 n=95 Sharpe=1.54 ppc=1.65
        Scanner.inspect_top_k:  5 / 15: freq=10 threshold=0.75 n=112 Sharpe=1.53 ppc=1.72
        Scanner.inspect_top_k:  6 / 15: freq=14 threshold=0.75 n=99 Sharpe=1.46 ppc=1.63
        Scanner.inspect_top_k:  7 / 15: freq=14 threshold=0.7 n=109 Sharpe=1.40 ppc=1.46
        Scanner.inspect_top_k:  8 / 15: freq=14 threshold=0.8 n=89 Sharpe=1.39 ppc=1.62
        Scanner.inspect_top_k:  9 / 15: freq=15 threshold=0.85 n=95 Sharpe=1.38 ppc=1.43
        Scanner.inspect_top_k:  10 / 15: freq=13 threshold=0.85 n=93 Sharpe=1.32 ppc=1.31
        """
        signal = df.assign(signal=0)['signal']
        flt = df['high'].rolling(window).mean() < df['high']
        signal[flt] = df.loc[flt, 'high'].diff(2)
        signal /= window
        if halflife != 0:
            signal = signal.ewm(halflife=halflife).mean()
        return signal

    @staticmethod
    def alpha_new_023(df: pd.DataFrame, window=50):
        print(window)
        # 1. OBV Slope Signal
        obv = (np.sign(df['close'].diff()) * df['matchingVolume']).fillna(0).cumsum()
        raw_obv_slope = obv.diff(2)

        # 2. Price Momentum Confirmation
        price_momentum = O.ts_rank_normalized(df['close'].diff(2), window)

        # 3. Combine Signals
        combined_raw_signal = raw_obv_slope * price_momentum

        # 4. Normalize
        signal = 2 * O.ts_rank_normalized(combined_raw_signal, window=window) - 1

        return signal
    
    @staticmethod
    def alpha_024(df, window=100):
        """
        Alpha#24:
        ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || 
        ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) 
        ? (-1 * (close - ts_min(close, 100))) 
        : (-1 * delta(close, 3)))
        """
        close = df['close']
        
        # moving average 100
        sma = O.ts_mean(close, window)

        # delta(sma, 100) / delay(close, 100)
        cond = O.ts_delta(sma, window) / O.ts_delay(close, window)

        # branch 1: -1 * (close - ts_min(close, 100))
        branch1 = -1 * (close - O.ts_min(close, window))

        # branch 2: -1 * delta(close, 3)
        branch2 = -1 * O.ts_delta(close, 3)

        signal = branch1.where(cond <= 0.05, branch2)
        return signal
    
    @staticmethod
    def alpha_030(df,window=20):
        """
        Alpha#30:
        (((1.0 - rank(((sign((close - delay(close, 1))) 
                        + sign((delay(close, 1) - delay(close, 2)))) 
                        + sign((delay(close, 2) - delay(close, 3)))))) 
          * sum(volume, 5)) / sum(volume, 20))
        """
        close = df['close']
        volume = df['matchingVolume']

        # phần biến động giá trong 3 ngày
        price_signal = (
            np.sign(close - O.ts_delay(close, 1)) +
            np.sign(O.ts_delay(close, 1) - O.ts_delay(close, 2)) +
            np.sign(O.ts_delay(close, 2) - O.ts_delay(close, 3))
        )

        # rank theo cross-section
        ranked = O.ts_rank(price_signal)

        # numerator: (1 - rank) * sum(volume, 5)
        numerator = (1.0 - ranked) * O.ts_sum(volume, 5)

        # denominator: sum(volume, window)
        denominator = O.ts_sum(volume, window)

        signal = numerator / denominator
        return signal

    @staticmethod
    def alpha_031(df, window=10):
        """
        Alpha#31:
        ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) 
          + rank((-1 * delta(close, 3)))) 
          + sign(scale(correlation(adv20, low, 12))))

        => dùng window thay cho 10/12
        """
        close = df['close']
        low = df['low']
        adv20 = O.ts_mean(df['matchingVolume'], 20)

        part1 = O.ts_rank(
                    O.ts_rank(
                        O.ts_rank(
                            O.decay_linear(-1 * O.ts_rank(O.ts_rank(O.ts_delta(close, window))),
                                           window)
                        )
                    )
                )

        part2 = O.ts_rank(-1 * O.ts_delta(close, 3))

        part3 = np.sign(
                    O.ts_scale(O.ts_corr(adv20, low, window))
                )

        signal = part1 + part2 + part3
        return signal


    @staticmethod
    def alpha_032(df, window=7):
        """
        Alpha#32:
        (scale(((sum(close, 7) / 7) - close)) 
         + (20 * scale(correlation(vwap, delay(close, 5), 230))))

        => window thay cho 7 và 230
        """
        close = df['close']
        vwap = O.compute_vwap(df)['vwap']

        part1 = O.ts_scale(O.ts_mean(close, window) - close)

        part2 = 20 * O.ts_scale(O.ts_corr(vwap, O.ts_delay(close, 5), window))

        signal = part1 + part2
        return signal


    @staticmethod
    def alpha_033(df,window=10):
        """
        Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        """

        return O.ts_rank(-1 * (1 - (df['open'] / df['close'])),window)


    @staticmethod
    def alpha_034(df, window=10):
        """
        Alpha#34:
        rank((-1 * ((1 - (open / close))^1)))
        """
        open_ = df['open']
        close = df['close']
        signal = O.ts_rank(-1 * (1 - (open_ / close)), window)
        return signal


    @staticmethod
    def alpha_035(df, window=32):
        """
        Alpha#35:
        ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) 
          * (1 - Ts_Rank(returns, 32)))

        => window thay cho 32, còn 16 = window//2
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        returns = close.pct_change()

        part1 = O.ts_rank(volume, window)
        part2 = (1 - O.ts_rank((close + high) - low, window//2))
        part3 = (1 - O.ts_rank(returns, window))

        signal = part1 * part2 * part3
        return signal

    @staticmethod
    def alpha_040(df, window=10):
        """
        Alpha#40:
        ((-1 * rank(stddev(high, window))) * correlation(high, volume, window))
        """
        high = df['high']
        volume = df['matchingVolume']

        # phần 1: -1 * rank(stddev(high, window))
        part1 = -1 * O.ts_rank(O.ts_std(high, window))

        # phần 2: correlation(high, volume, window)
        part2 = O.ts_corr(high, volume, window)

        signal = part1 * part2
        return signal

    
    @staticmethod
    def alpha_038(df,window=10):
        close = df['close']
        openn = df['open']
        signal = ((-1 * O.ts_rank(O.ts_rank(close, window))) * O.ts_rank((close / openn)))
        signal = signal / 50 + 1
        return -signal

    @staticmethod
    def alpha_043(df,window=1):
        """
        ts_rank((volume / adv20), 20)
        *
        ts_rank((-1 * delta(close, 7)), 8)
        """
        close = df['close']
        volume = df['matchingVolume']
        delta = lambda ts, window: ts.diff(window)
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()
        signal = O.ts_rank((volume / adv20), 20) \
                 * \
                 O.ts_rank((-1 * delta(close, 7)), 8)
        signal = signal / 160
        signal = signal * np.sign(delta(close, window))
        return signal

    @staticmethod
    def alpha_044(df,window=5):
        signal = O.ts_corr(
            df['high'],
            O.ts_rank(df['matchingVolume']),
            window)
        signal = - signal
        return signal

    @staticmethod
    def alpha_048(df, window=250):
        """
        Alpha#48:
        (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) 
        * delta(close, 1)) / close), IndClass.subindustry) 
        / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        """

        close = df['close']
        delta_close = O.ts_delta(close, 1)
        delay_close = O.ts_delay(close, 1)
        delta_delay_close = O.ts_delta(delay_close, 1)
        corr = O.ts_corr(delta_close, delta_delay_close, window)
        numerator = (corr * delta_close) / close
        denominator = ((delta_close / delay_close) ** 2).rolling(window).sum()

        signal = numerator / denominator
        return signal
    
    @staticmethod
    def alpha_053(df, window=9):
        close, high, low = df['close'], df['high'], df['low']
        x = ((close - low) - (high - close)) / (close - low)
        return -1 * O.ts_delta(x, window)

    @staticmethod
    def alpha_054(df, window=5):
        close, high, low, open_ = df['close'], df['high'], df['low'], df['open']
        numerator = -1 * ((low - close) * (open_ ** window))
        denominator = (low - high) * (close ** window)
        return numerator / denominator

    @staticmethod
    def alpha_056(df, window=3):
        close = df['close']
        volume = df['matchingVolume']   # thay cap bằng volume

        ret = close.pct_change()

        part1 = O.ts_rank(O.ts_sum(ret, 10) / O.ts_sum(O.ts_sum(ret, 2), window))
        part2 = O.ts_rank(ret * volume)

        return -1 * (part1 * part2)


    @staticmethod
    def alpha_057(df, window=30):
        close = df['close']
        vwap = O.compute_vwap(df, window)['vwap']

        numerator = close - vwap
        denominator = O.decay_linear(O.ts_rank(O.ts_argmax(close, 30)), 2)

        return -1 * (numerator / denominator)


    @staticmethod
    def alpha_058(df, window=30):
        close, vwap = df['close'], O.compute_vwap(df, window)['vwap']
        numerator = close - vwap
        denominator = O.decay_linear(O.ts_rank(O.ts_argmax(close, window)), 2)
        return -1 * (numerator / denominator)

    @staticmethod
    def alpha_059(df, window=200):
        """
        Alpha#59:
        -1 * Ts_Rank(
                decay_linear(
                    correlation(IndNeutralize(vwap, industry), volume, 4),
                16),
            8)
        """
        # VWAP tính từ dữ liệu
        vwap = O.compute_vwap(df, window)['vwap']
        volume = df['matchingVolume']
        corr = O.ts_corr(vwap, volume, 4)
        decayed = O.decay_linear(corr, 16)
        return -1 * O.ts_rank(decayed, 8)


    @staticmethod
    def alpha_060(df, window=10):
        close, high, low, volume = df['close'], df['high'], df['low'], df['matchingVolume']
        part1 = 2 * O.ts_scale(O.ts_rank((( (close - low) - (high - close)) / (high - low)) * volume))
        part2 = O.ts_scale(O.ts_rank(O.ts_argmax(close, window)))
        return -1 * (part1 - part2)

    @staticmethod
    def alpha_063(df, window=180):
        close, open_, volume = df['close'], df['open'], df['matchingVolume']
        vwap = O.compute_vwap(df, window)['vwap']
        adv180 = O.ts_mean(volume, 180)

        part1 = O.ts_rank(O.decay_linear(O.ts_delta(close, 2), 8))

        combo = (vwap * 0.318108) + (open_ * (1 - 0.318108))
        part2 = O.ts_rank(O.decay_linear(O.ts_corr(combo, O.ts_sum(adv180, 37), 13), 12))

        return (part1 - part2) * -1

    @staticmethod
    def alpha_067(df, window=20):
        high, volume = df['high'], df['matchingVolume']
        vwap = O.compute_vwap(df, window)['vwap']
        adv20 = O.ts_mean(volume, 20)

        part1 = O.ts_rank(high - O.ts_min(high, 2))
        part2 = O.ts_rank(O.ts_corr(vwap,adv20, 6))
        return (part1 ** part2) * -1

    @staticmethod
    def alpha_069(df, window=20):
        close, volume = df['close'], df['matchingVolume']
        vwap = O.compute_vwap(df, window)['vwap']
        adv20 = O.ts_mean(volume, 20)

        part1 = O.rank(O.ts_max(O.ts_delta(vwap, 2), 4))

        combo = (close * 0.490655) + (vwap * (1 - 0.490655))
        part2 = O.ts_rank(O.ts_corr(combo, adv20, 5), 9)

        return (part1 ** part2) * -1

    @staticmethod
    def alpha_070(df, window=50):
        close, volume = df['close'], df['matchingVolume']
        vwap = O.compute_vwap(df, window)['vwap']
        adv50 = O.ts_mean(volume, 50)

        part1 = O.rank(O.ts_delta(vwap, 1))
        part2 = O.ts_rank(O.ts_corr(close, adv50, 18), 18)

        return (part1 ** part2) * -1

    @staticmethod
    def alpha_072(df, window=10):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        vwap = O.compute_vwap(df, 40)['vwap']  # 40 để đồng bộ với adv40
        adv40 = O.ts_mean(volume, 40)

        # --- Numerator ---
        hl_mean = (high + low) / 2
        corr1 = O.ts_corr(hl_mean, adv40, int(8.93345))
        decay1 = O.decay_linear(corr1, int(10.1519))
        num = O.ts_rank(decay1)

        # --- Denominator ---
        tsr_vwap = O.ts_rank(vwap, int(3.72469))
        tsr_vol = O.ts_rank(volume, int(18.5188))
        corr2 = O.ts_corr(tsr_vwap, tsr_vol, int(6.86671))
        decay2 = O.decay_linear(corr2, int(2.95011))
        den = O.ts_rank(decay2,window) + 1e-9  # tránh chia 0

        signal = num / den / 5 

        print(signal.min(), signal.max()) 
        return -signal
    
    @staticmethod
    def alpha_084(df: pd.DataFrame, window=7):
    # def alpha_084(dfhalflife=0):
        """
            SignedPower(
                Ts_Rank(
                    (
                        vwap - ts_max(vwap, 15.3217)
                    ), 20.7127
                ),
                delta(close, 4.96796)
            )
        """
        # freq = 30
        # df2 = glob_obj.dic_freqs[freq].copy()
        # logging.warning(f"Len of df2: {len(df2)}")
        # df2 = df.copy(
        close = df['close']
        vwap = Domains.compute_vwap(df)['vwap']
        signal = O.ts_rank(
            O.power(
                O.ts_rank(
                    (
                        vwap - O.ts_max(vwap, 15)
                    ), 21
                ),
                O.ts_delta(close, 5)
            )
        ,21)

        # 2 * ((xi – xmin) / (xmax – xmin)) – 1
        s_min = O.ts_min(signal, window)
        s_max = O.ts_max(signal, window)
        signal = 2 * ((signal - s_min) / (s_max - s_min)) - 1
        return signal

    @staticmethod
    def alpha_086(df: pd.DataFrame, halflife=0, window=20):
        """
        (
            Ts_Rank(
                correlation(
                    close,
                    sum(adv20, 14.7444),
                    6.00049
                ), 20.4195
            )
            <
            rank(
                (
                    (open + close) - (vwap + open)
                )
            )
        ) * -1
        """
        # freq = 30
        # df = glob_obj.dic_freqs[freq]

        openn = df['open']
        close = df['close']
        vwap = Domains.compute_vwap(df)['vwap']
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()

        left = O.ts_rank(
            O.ts_corr(
                close,
                O.ts_sum(adv20, 15),
                6
            ),
            window
        )

        right = O.ts_rank(
            (openn + close) - (vwap + openn),
            window
        )

        df['signal'] = np.where(left < right, left, right)
        df['signal'] = df['signal'] * (close - openn)
        df['signal'] = df['signal'] / df['signal'].abs().rolling(10).max()
        if halflife != 0:
            df['signal'] = df['signal'].ewm(halflife=halflife).mean()
        return df['signal']

    @staticmethod
    def alpha_088(df: pd.DataFrame, halflife=0,window =7):
        """
            min(
                rank(
                    decay_linear(
                        (
                            (rank(open) + rank(low))
                            -
                            (rank(high) + rank(close))
                        ), 8.06882
                    )
                ),
                Ts_Rank(
                    decay_linear(
                        correlation(
                            Ts_Rank(close, 8.44728),
                            Ts_Rank(adv60, 20.6966), 8.01266
                        ), 6.65053
                    ), 2.61957
                )
            )
        """
        # freq = 30
        # df = glob_obj.dic_freqs[freq]

        openn = df['open']
        low = df['low']
        high = df['high']
        close = df['close']
        adv60 = df.rolling(60) \
            .matchingVolume \
            .mean()


        left = O.ts_rank(
            O.decay_linear(
                (
                    (O.ts_rank(openn) + O.ts_rank(low))
                    -
                    (O.ts_rank(high) + O.ts_rank(close))
                ), 8
            )
        )

        right = O.ts_rank(
            O.decay_linear(
                O.ts_corr(
                    O.ts_rank(close, 8),
                    O.ts_rank(adv60, 21), 8
                ), 7
            ), 3
        )

        df['signal'] = np.where(left < right, left, right)
        df['signal'] = df['signal'] * (close - openn)
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        if halflife != 0:
            df['signal'] = df['signal'].ewm(halflife=halflife).mean()

        return df['signal']

    @staticmethod
    def alpha_092(df: pd.DataFrame, halflife=0,window=7):
        """
            Ts_Rank(
                decay_linear(
                    (
                        (
                            (
                                (high + low) / 2
                            ) + close
                        ) < (low + open)
                    ), 14.7221
                ), 18.8683
            ),
            Ts_Rank(
                decay_linear(
                    correlation(
                        rank(low), rank(adv30), 7.58555
                    ), 6.94024
                ), 6.80584
            )
        """
        # freq = 30
        # df = glob_obj.dic_freqs[freq]

        high = df['high']
        low = df['low']
        close = df['close']
        openn = df['open']
        adv30 = df.rolling(30) \
            .matchingVolume \
            .mean()

        left = O.ts_rank(
            O.decay_linear(
                (
                    (
                        (
                            (high + low) / 2
                        ) + close
                    ) < (low + openn)
                ), 15
            ), 19
        )

        right = O.ts_rank(
            O.decay_linear(
                O.ts_corr(
                    O.ts_rank(low), O.ts_rank(adv30), 8
                ), 7
            ), 7
        )

        df['signal'] = np.where(left < right, left, right)
        df['signal'] = df['signal'] * (close - openn)
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        if halflife != 0:
            df['signal'] = df['signal'].ewm(halflife=halflife).mean()

        return df['signal']

    @staticmethod
    def alpha_094(df: pd.DataFrame, window=3):
        vwap = Domains.compute_vwap(df)['vwap']
        adv60 = df.rolling(40)['matchingVolume'].mean()
        fst = O.ts_rank(
            vwap - O.ts_min(vwap, 12),
            10
        )

        scd = O.ts_rank(
            O.ts_corr(
                O.ts_rank(vwap, 20),
                O.ts_rank(adv60, 4),
                18
            ),
            window
        )

        df['signal'] = O.power(fst, scd)
        s_min = O.ts_min(df['signal'], 7)
        s_max = O.ts_max(df['signal'], 7)
        df['signal'] = 2 * ((df['signal'] - s_min) / (s_max - s_min)) - 1

        return df['signal']

    @staticmethod
    def alpha_095(df: pd.DataFrame, halflife=0, window=7):
        """
        (
            rank(
                (
                    open - ts_min(open, 12.4105)
                )
            )
            <
            Ts_Rank(
                (
                    rank(
                        correlation(
                            sum(
                                (
                                    (high + low) / 2
                                ), 19.1351
                            ),
                            sum(adv40, 19.1351),
                            12.8742
                        )
                    )^5
                ), 11.7584
            )
        )
        """
        # freq = 30
        # df = glob_obj.dic_freqs[freq]

        openn = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        adv40 = df.rolling(40) \
            .matchingVolume \
            .mean()

        left = O.ts_rank(
            openn - O.ts_min(openn, 12),
            10
        )

        right = O.ts_rank(
            O.ts_rank(
                O.ts_corr(
                    O.ts_sum(
                        (high + low) / 2,
                        19
                    ),
                    O.ts_sum(adv40, 19),
                    13
                ),
                12
            )
        )

        df['signal'] = np.where(left < right, left, right)

        df['signal'] = df['signal'] * (close - openn)

        # df['signal'].hist()
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        if halflife != 0:
            df['signal'] = df['signal'].ewm(halflife=halflife).mean()

        return df['signal']

    @staticmethod
    def alpha_098(df: pd.DataFrame, window=10):
        # freq = 30
        # df = glob_obj.dic_freqs[freq]
        adv5 = df.rolling(5) \
            .matchingVolume \
            .mean()
        adv15 = df.rolling(15) \
            .matchingVolume \
            .mean()
        vwap = Domains.compute_vwap(df)['vwap']
        openn = df['open']
        left = O.ts_rank(
                    O.decay_linear(
                            O.ts_corr(vwap, O.ts_sum(adv5, 26), 4), 7
                        ),window
                )

        right = O.ts_rank(
                    O.decay_linear(
                        O.ts_rank(O.ts_argmin(O.ts_corr(O.ts_rank(openn),O.ts_rank(adv15), 20), 8), 6),
                        8
                    ),window
                )

        df['signal'] = left - right

        df['signal'] = df['signal'] / df['signal'].abs().rolling(7).mean()
      
        return df['signal']

    @staticmethod
    def alpha_099(df: pd.DataFrame, window=20, halflife=0):
        # freq = 47
        # df = glob_obj.dic_freqs[freq]
        df = df.copy()
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        adv60 = df.rolling(60) \
            .matchingVolume \
            .mean()

        first_cond: pd.Series = O.ts_rank(
            O.ts_corr(
                O.ts_sum((high + low) / 2, 20),
                O.ts_sum(adv60, 20),
                9
            ),
        )

        second_cond = O.ts_rank(
            O.ts_corr(
                low, volume, 6
            )
        )

        # df.loc[first_cond < second_cond, 'signal'] = first_cond
        # df['signal'] = df['signal'].fillna(second_cond)
        df['signal'] = np.where(first_cond < second_cond, first_cond, second_cond)
        # df['signal'] = first_cond.where(first_cond<second_cond,second_cond)
        df['signal'] = df['signal'] * (df['close'] - df['open'])

        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).mean()
        if halflife != 0:
            df['signal'] = df['signal'].ewm(halflife=halflife).mean()
        # df['signal'].hist(bins=100)

        return df['signal']

    @staticmethod
    def alpha_101(df: pd.DataFrame):
        """((close - open) / ((high - low) + .001))"""
        close = df['close']
        openn = df['open']
        high = df['high']
        low = df['low']

        df['signal'] = (close - openn) / ((high - low) + 0.001)
        # df['signal'] = (close - openn) / (high - low) 
        # if window != 0:
        #     df['signal'] = df['signal'].ewm(halflife=window).mean()
        return df['signal']

    @staticmethod
    def alpha_101_volume(df: pd.DataFrame, window=20):
        df = df.copy()
        df['adv20'] = df['matchingVolume'].rolling(window=window).mean()
        signal =((df['close'] - df['open']) / (df['high'] - df['low']) + 0.001) * (df['matchingVolume'] / df['adv20'])
        return signal

    @staticmethod
    def alpha_101_volume_smoothed(df, window=20, factor=3):
        
        # Cái này cũ là alpha_new_001
        """
        Logic: Tính toán động lượng trong ngày (giống Alpha#101) và gia quyền nó
        bằng sức mạnh khối lượng tương đối (khối lượng hiện tại so với trung bình).
        Tín hiệu mạnh khi một động thái giá mạnh được xác nhận bởi khối lượng cao.
        """
        # 1. Động lượng trong ngày
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)

        # 2. Sức mạnh khối lượng tương đối
        adv = O.ts_mean(df['matchingVolume'], window=window)
        relative_volume = df['matchingVolume'] / (adv + 0.001)

        # 3. Tín hiệu thô = Động lượng * Khối lượng
        raw_signal = intraday_momentum * relative_volume

        # 4. Làm mượt tín hiệu để giảm nhiễu
        final_signal = O.ts_weighted_mean(raw_signal, window=factor)

        return final_signal

    @staticmethod
    def alpha_101_trend_confirm(df: pd.DataFrame):
        """
        Biến thể 1: Động lượng xác nhận bởi Xu hướng (Trend Confirmation)
        Ý tưởng: Tín hiệu động lượng trong phiên sẽ đáng tin cậy hơn nếu nó cùng chiều với xu hướng ngắn hạn gần đây.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        trend_sign = O.sign(df['close'] - O.ts_lag(df['close'], 3))
        signal = intraday_momentum * trend_sign
        return signal

    @staticmethod
    def alpha_101_stddev_normalized(df: pd.DataFrame,window=20):
        """
        Biến thể 2: Động lượng điều chỉnh bởi Độ biến động (Volatility-Adjusted Momentum)
        Ý tưởng: Chuẩn hóa động lượng trong phiên bằng độ lệch chuẩn của tỷ suất lợi nhuận để tín hiệu ổn định hơn.
        """
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change()
            
        intraday_price_move = df['close'] - df['open']
        volatility = O.ts_std(df['return'], window)
        signal = intraday_price_move / (volatility + 0.00001)
        return signal

    @staticmethod
    def alpha_101_vwap_vol_rank(df: pd.DataFrame):
        """
        Biến thể 3: Động lượng VWAP gia quyền bởi Xếp hạng Khối lượng
        Ý tưởng: Sử dụng VWAP làm mốc tham chiếu cho động lượng và gia quyền bằng thứ hạng của khối lượng.
        """
        if 'vwap' not in df.columns:
            df = O.compute_vwap(df, window=20)

        vwap_momentum = df['close'] - df['vwap']
        volume_rank = O.ts_rank(df['matchingVolume'], 20)
        signal = vwap_momentum * volume_rank
        return signal

    @staticmethod
    def alpha_101_mean_reversion(df: pd.DataFrame):
        """
        Biến thể 4: Đảo chiều Động lượng trong phiên (Intraday Mean-Reversion)
        Ý tưởng: Đặt cược ngược lại với alpha_101. Nếu một phiên có áp lực mua cực mạnh, phiên tiếp theo có thể sẽ điều chỉnh giảm.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        signal = -1 * O.ts_lag(intraday_momentum, 1)
        return signal

    @staticmethod
    def alpha_101_corr_weighted(df: pd.DataFrame,window=10):
        """
        Biến thể 5: Động lượng gia quyền bởi Tương quan Giá-Khối lượng
        Ý tưởng: Tín hiệu alpha_101 sẽ mạnh hơn nếu giá và khối lượng đang có tương quan dương (cùng tăng hoặc cùng giảm).
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        price_rank = O.ts_rank(df['close'], window)
        volume_rank = O.ts_rank(df['matchingVolume'], window)
        corr_factor = O.ts_corr(price_rank, volume_rank, 10)
        signal = intraday_momentum * corr_factor
        return signal

    @staticmethod
    def alpha_101_trend_strength_weighted(df: pd.DataFrame):
        """
        Biến thể 6: Động lượng gia quyền bởi Sức mạnh Xu hướng
        Ý tưởng: Gia quyền tín hiệu alpha_101 bằng sức mạnh của xu hướng dài hạn, đo bằng khoảng cách từ giá tới đường SMA 50.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        trend_strength = df['close'] - O.ts_mean(df['close'], 50)
        signal = intraday_momentum * trend_strength
        return signal

    @staticmethod
    def alpha_101_positional_combo(df: pd.DataFrame, window=20):
        """
        Biến thể 7: Tín hiệu kết hợp Vị thế trong phiên và Vị thế trong chu kỳ
        Ý tưởng: Kết hợp tín hiệu vị thế trong ngày (alpha_101) với tín hiệu vị thế trong chu kỳ 20 ngày.
        """
        intraday_pos = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        min_low_20 = O.ts_min(df['low'], window)
        max_high_20 = O.ts_max(df['high'], window)
        cycle_pos = (df['close'] - min_low_20) / (max_high_20 - min_low_20 + 0.001)
        signal = intraday_pos + cycle_pos
        return signal

    @staticmethod
    def alpha_101_regime_filter(df: pd.DataFrame,window=60,factor=20):
        """
        Biến thể 8: Tín hiệu Lọc theo Trạng thái Thị trường (Regime Filter)
        Ý tưởng: Alpha tự động chuyển đổi giữa logic theo xu hướng (momentum) và đảo chiều (mean-reversion) dựa trên độ biến động của thị trường.
        """
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change()
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        volatility_rank = O.ts_rank(O.ts_std(df['return'], factor), window)
        # If volatility is in the top half (high vol), trade mean-reversion (-1). Otherwise, trade momentum (+1).
        regime_multiplier = np.where(volatility_rank > 0.5, -1, 1)
        signal = intraday_momentum * regime_multiplier
        return signal

    @staticmethod
    def alpha_101_acceleration(df: pd.DataFrame,window=2):
        """
        Biến thể 9: Gia tốc/Giảm tốc của Động lượng
        Ý tưởng: Đo lường sự thay đổi của động lượng trong phiên, cho biết áp lực mua/bán đang tăng tốc hay giảm tốc.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        signal = O.ts_delta(intraday_momentum, window)
        return signal

    @staticmethod
    def alpha_101_oi_confirm(df: pd.DataFrame,window=1):
        """
        Biến thể 10: Động lượng xác nhận bởi Hợp đồng mở (Open Interest)
        Ý tưởng: Gia quyền tín hiệu động lượng bằng sự thay đổi của Hợp đồng mở (OI), đặc biệt hữu ích cho thị trường phái sinh.
        """
        if 'open_interest' not in df.columns:
            return pd.Series(0, index=df.index)
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        oi_change_sign = O.sign(O.ts_delta(df['open_interest'], window))
        signal = intraday_momentum * oi_change_sign
        return signal

    @staticmethod
    def alpha_101_gap_filtered(df: pd.DataFrame,window=1):
        """
        Biến thể 11: Động lượng Lọc theo Khoảng trống Giá (Gap Filter)
        Ý tưởng: Chỉ giao dịch tín hiệu alpha_101 vào những ngày có khoảng trống giá (gap) lớn lúc mở cửa.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        gap_magnitude = (df['open'] / O.ts_lag(df['close'], window) - 1).abs()
        filter_condition = np.where(gap_magnitude > 0.01, 1, 0)
        signal = intraday_momentum * filter_condition
        return signal

    @staticmethod
    def alpha_101_body_wick_ratio(df: pd.DataFrame):
        """
        Biến thể 12: Tỷ lệ Thân nến trên Bóng nến (Body-to-Wick Ratio)
        Ý tưởng: Đo lường sự quyết đoán của cây nến bằng tỷ lệ giữa kích thước thân nến và tổng kích thước bóng nến.
        """
        body = df['close'] - df['open']
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        total_wicks = upper_wick + lower_wick
        signal = body / (total_wicks + 0.001)
        return signal

    @staticmethod
    def alpha_101_decay_smoothed(df: pd.DataFrame,window=10):
        """
        Biến thể 13: Động lượng được làm mượt theo hàm mũ (Exponentially Smoothed)
        Ý tưởng: Dùng trung bình trượt theo hàm mũ (EMA) để làm mượt tín hiệu alpha_101, đặt trọng số lớn hơn vào các dữ liệu gần đây.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        signal = O.decay_linear(intraday_momentum, window)
        return signal

    @staticmethod
    def alpha_101_plus_alpha_008(df: pd.DataFrame,window=5):
        """
        Biến thể 14: Tín hiệu lai (Hybrid Signal)
        Ý tưởng: Kết hợp alpha_101 (momentum) với một alpha đảo chiều (mean-reversion) để tạo tín hiệu cân bằng hơn.
        """
        alpha_101_signal = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        # Logic của Alpha ID 8: -1 * correlation(high, volume, 5)
        alpha_008_signal = -1 * O.ts_corr(df['high'], df['matchingVolume'], window)
        signal = alpha_101_signal + alpha_008_signal
        return signal

    @staticmethod
    def alpha_101_asymmetric(df: pd.DataFrame):
        """
        Biến thể 15: Động lượng Bất đối xứng (Asymmetric Momentum)
        Ý tưởng: Ưu tiên các tín hiệu mua hơn tín hiệu bán bằng cách gia quyền lớn hơn cho những ngày tăng giá.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        multiplier = np.where(df['close'] > df['open'], 2, 1)
        signal = intraday_momentum * multiplier
        return signal

    @staticmethod
    def alpha_101_zscore(df: pd.DataFrame,window=20):
        """
        Biến thể 16: Động lượng Z-Score
        Ý tưởng: Chuẩn hóa động lượng thô (close - open) bằng Z-score để xem nó bất thường như thế nào so với quá khứ.
        """
        intraday_move = df['close'] - df['open']
        signal = O.zscore(intraday_move, window)
        return signal

    @staticmethod
    def alpha_101_rank_combo(df: pd.DataFrame, window=20):
        """
        Biến thể 17: Tín hiệu kết hợp bằng Xếp hạng (Rank Combination)
        Ý tưởng: Kết hợp thứ hạng của tín hiệu động lượng và tín hiệu khối lượng để giảm ảnh hưởng của các giá trị đột biến.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        adv20 = O.ts_mean(df['matchingVolume'], 20)
        relative_volume = df['matchingVolume'] / (adv20 + 0.001)
        rank_momentum = O.ts_rank(intraday_momentum, window)
        rank_volume = O.ts_rank(relative_volume, window)
        signal = rank_momentum + rank_volume
        return signal

    @staticmethod
    def alpha_101_overnight_confirm(df: pd.DataFrame, window=1):
        """
        Biến thể 18: Động lượng xác nhận bởi Xu hướng Ngoài giờ
        Ý tưởng: So sánh động lượng trong giờ và xu hướng ngoài giờ (overnight). Tín hiệu sẽ mạnh hơn nếu cả hai cùng chiều.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        overnight_gap_sign = O.sign(df['open'] - O.ts_lag(df['close'], window))
        signal = intraday_momentum * overnight_gap_sign
        return signal

    @staticmethod
    def alpha_101_powered(df: pd.DataFrame):
        """
        Biến thể 19: Động lượng Lũy thừa (Powered Momentum)
        Ý tưởng: Áp dụng phép biến đổi lũy thừa để làm nổi bật các giá trị tín hiệu ở hai cực (rất mạnh hoặc rất yếu).
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        signal = O.power(intraday_momentum, 2) * O.sign(intraday_momentum)
        return signal

    @staticmethod
    def alpha_101_day_of_week_filter(df: pd.DataFrame):
        """
        Biến thể 20: Tín hiệu Lọc theo Thứ trong Tuần (Day-of-Week Filter)
        Ý tưởng: Thêm một bộ lọc dựa trên hiệu ứng thống kê của các ngày trong tuần, ví dụ tăng cường tín hiệu vào ngày Thứ Hai.
        """
        intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        # Monday is 0 in dayofweek
        day_of_week_multiplier = np.where(df.index.dayofweek == 0, 1.5, 1)
        signal = intraday_momentum * day_of_week_multiplier
        return signal
    
    @staticmethod
    def alpha_029(df,window=5):
        close = df['close']
        returns = df['return']
        scale = lambda x: x
        delta = lambda ts, window: ts.diff(window)
        delay = lambda ts, window: ts.shift(window)

        signal = (O.ts_min((O.ts_rank(O.ts_rank(scale(np.log((O.ts_min(O.ts_rank(O.ts_rank((-1 * O.ts_rank(delta((close - 1),5))))),2))+ 1)))) * 1),5)+O.ts_rank(delay((-1 * returns), 6), window))
        signal = signal / 6 - 1.2
        signal = signal
        return signal

    @staticmethod
    def alpha_025(df,window=10):
        """
        rank(
            ((((-1 * returns) * adv20) * vwap) * (high - close))
        )
        """


        returns = df['return']
        high = df['high']
        close = df['close']
        adv20 = df.rolling(20)                          \
                  .matchingVolume                       \
                  .mean()
        df = O.compute_vwap(df)
        vwap = df['vwap']
        signal = O.ts_rank((
            ((-1 * returns) * adv20)
            * vwap
            * (high - close)
        ),window)
        signal = signal / 5 - 1
        return signal

    @staticmethod
    def alpha_125(df):
        """
        -rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """
        returns = df['return']
        high = df['high']
        close = df['close']
        adv20 = df.rolling(20)                          \
                  .matchingVolume                       \
                  .mean()
        df = O.compute_vwap(df)
        vwap = df['vwap']
        signal = O.ts_rank((
            ((-1 * returns) * adv20)
            * vwap
            * (high - close)
        ))
        signal = signal / 5 - 1
        signal = -signal
        return signal

    @staticmethod
    def alpha_026(df,window=3):
        """
        (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """
        volume = df['matchingVolume']
        high = df['high']
        signal = \
            -1 * O.ts_max\
            (
                O.ts_corr(O.ts_rank(volume, 5),
                          O.ts_rank(high, 5), 5),
                window
            )
        return signal

    @staticmethod
    def alpha_027(df,window=6):
        """((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)"""
        volume = df['matchingVolume']
        df = O.compute_vwap(df)
        vwap = df['vwap']
        condition = \
        (
            0.5
            <
            O.ts_rank
            (
                (
                    O.ts_corr
                    (
                        O.ts_rank(volume),
                        O.ts_rank(vwap),
                        window
                    )
                    /
                    2.0
                 )
             )
        )

        signal = (condition - 0.5) * 2
        return signal

    @staticmethod
    def alpha_028(df,window=5):
        """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()
        low = df['low']
        high = df['high']
        close = df['close']
        signal = (O.ts_corr(adv20, low, window) + ((high + low) / 2)) - close
        signal = signal / 15
        return -signal

    @staticmethod
    def alpha_036(df,window=200):
        rank = O.ts_rank
        correlation = O.ts_corr
        close = df['close']
        # noinspection PyShadowingBuiltins
        open = df['open']
        volume = df['matchingVolume']
        Ts_Rank = O.ts_rank
        returns = df['return']
        delay = lambda ts, window: ts.shift(window)
        df = O.compute_vwap(df,window)
        vwap = df['vwap']
        adv20 = df.rolling(20)                          \
                  .matchingVolume                       \
                  .mean()
        # noinspection PyShadowingBuiltins
        sum = lambda df, window: df.rolling(window).sum()
        signal = (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close))))+(0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6))))+(0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        signal = signal / 20 - 1.45
        signal = -signal
        return signal

    @staticmethod
    def alpha_037(df,window=200):
        rank = O.ts_rank
        correlation = O.ts_corr
        close = df['close']
        # noinspection PyShadowingBuiltins
        open = df['open']
        delay = lambda ts, window: ts.shift(window)
        signal = (rank(correlation(delay((open - close), 1), close, window)) + rank((open - close)))
        signal = signal / 10 - 1
        return signal

    @staticmethod
    def alpha_039(df,window=20):
        rank = O.ts_rank
        returns = df['return']
        volume = df['matchingVolume']
        adv20 = df.rolling(window) \
            .matchingVolume \
            .mean()
        close = df['close']
        delta = lambda ts, window: ts.diff(window)
        decay_linear = O.decay_linear
        # noinspection PyShadowingBuiltins
        sum = lambda df, window: df.rolling(window).sum()
        signal = ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))

        return -signal

    @staticmethod
    def alpha_041(df,window=200):
        high = df['high']
        low = df['low']
        df = O.compute_vwap(df,window)
        vwap = df['vwap']
        signal =(((high * low).pow(0.5)) - vwap)
        signal =  (signal + 100) / 300
        return -signal

    @staticmethod
    def alpha_042(df,window=200):
        rank = O.ts_rank
        close = df['close']
        df = O.compute_vwap(df,window)
        vwap = df['vwap']
        signal = (rank((vwap - close)) / rank((vwap + close)))
        signal = signal / 5 - 1
        return signal
    @staticmethod
    def alpha_102(df, window_vwap=200, window_vol=20, delta_window=7, rank_window=60, halflife=0):
        c = df['close']
        v = df['matchingVolume']

        # 1. Khối lượng trung bình 20 ngày
        adv20 = v.rolling(window_vol).mean()

        # 2. Biến động giá ngắn hạn
        delta = c.diff(delta_window).fillna(0)
        drift_signal = O.ts_rank(abs(delta), rank_window) * np.sign(delta)

        # 3. VWAP deviation (mean-reversion component)
        df = O.compute_vwap(df, window_vwap)
        vwap = df['vwap']
        vwap_signal = (O.ts_rank(vwap - c, rank_window) /
                    O.ts_rank(vwap + c, rank_window)) - 1

        # 4. Kết hợp hai tín hiệu
        signal = 0.5 * drift_signal + 0.5 * vwap_signal

        # 5. Khuếch đại khi volume cao bất thường
        signal = signal.where(v > adv20, signal * -0.5)

        # 6. Chuẩn hóa
        signal = (signal.rolling(rank_window).rank() - rank_window/2) / (rank_window/2)

        # 7. Smoothing nếu cần
        if halflife > 0:
            signal = signal.ewm(halflife=halflife).mean()

        return signal

    @staticmethod
    def alpha_045(df,window=10):
        def wrapper(func):
            def wrapped_func(*args, **kwargs):
                ts = func(*args, **kwargs)
                flt = ts.abs() > 1
                ts.loc[flt] = np.sign(ts.loc[flt])
                return ts
            return wrapped_func

        rank = O.ts_rank
        delay = lambda df, window: df.shift(window)
        close = df['close']
        correlation = wrapper(O.ts_corr)

        volume = df['matchingVolume']
        # noinspection PyShadowingBuiltins
        sum = lambda df, window: df.rolling(window).sum()
        signal = (-1 * (
            rank((sum(delay(close, 5), 20) / 20),window)
            * correlation(close, volume, 2)
            * rank(correlation(sum(close, 5), sum(close, 20), 2),window)
        ))
        signal = signal / 100
        return signal

    @staticmethod
    def alpha_046(df,window=10):
        delay = lambda df, window: df.shift(window)
        close = df['close']
        signal = pd.DataFrame({'signal': -np.ones_like(close)}, index=df.index)
        flt1 = (0.25 < (((delay(close, window*2) - delay(close, window)) / 10) - ((delay(close, window) - close) / 10)))
        flt2 = ((((delay(close, window*2) - delay(close, window)) / 10) - ((delay(close, window) - close) / 10)) < 0)
        signal.loc[flt1] = -1
        signal.loc[(~flt1) & flt2] = 1
        flt3 = (~flt1) & (~flt2)
        signal.loc[flt3, 'signal'] = (-1 * 1) * (close[flt3] - delay(close[flt3], 1))\
                                                                   .to_numpy()
        signal = signal / 3
        return - signal

    @staticmethod
    def alpha_047(df,window=200):
        delay = lambda df, window: df.shift(window)
        rank = O.ts_rank
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        adv20 = df.rolling(20)                          \
                  .matchingVolume                       \
                  .mean()
        df = O.compute_vwap(df,window)
        vwap = df['vwap']
        signal = ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
        signal = (signal + 5.5) / 5
        return -signal

    @staticmethod
    def alpha_049(df,window=10):
        delay = lambda df, window: df.shift(window)
        close = df['close']
        flt = ((((delay(close, window*2) - delay(close, window)) / 10) - ((delay(close, window) - close) / 10)) < (-1 * 0.1))
        signal = pd.DataFrame(
            {
                'signal': np.ones_like(close)
            },
            index=close.index)
        close = close.loc[~flt]
        signal.loc[~flt, 'signal'] = ((-1 * 1) * (close - delay(close, 1)))
        signal = signal / 30
        signal = -signal
        return signal['signal']

    @staticmethod
    def alpha_050(df,window=5):
        volume = df['matchingVolume']
        rank = O.ts_rank
        df = O.compute_vwap(df)
        vwap = df['vwap']
        correlation = O.ts_corr
        ts_max = O.ts_max
        signal = (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), window)), window))
        signal = (signal + 5.5) * 1.5
        return -signal

    @staticmethod
    def alpha_051(df,window=10):

        close = df['close']
        delay = lambda df, window: df.shift(window)

        flt = ((((delay(close, window*2) - delay(close, window)) / 10) - ((delay(close, window) - close) / 10)) < (-1 *0.05))
        signal = pd.DataFrame(
                     {'signal': 1 * np.ones_like(close)},
                     index=close.index) \
                     ['signal']
        signal.loc[~flt] = ((-1 * 1) * (close - delay(close, 1)))
        signal = signal / 12.5
        signal = -signal
        return signal

    @staticmethod
    def alpha_081(df: pd.DataFrame,window=10):
        df = Domains.compute_vwap(df)
        df['auxi1'] = df['matchingVolume'].rolling(10).mean().rolling(50).sum()
        df['auxi2'] = O.ts_corr(df['vwap'],df['auxi1'],8)
        df['auxi3'] = O.ts_rank(df['auxi2']**4,10)
        # df['auxi3'] = O.ts_rank(O.ts_rank(df['auxi2']**4,10),50)
        df['auxi4'] = np.log(df['auxi3'])
        df['quantity1']= O.ts_rank(df['auxi4'].rolling(15).sum(),window)

        df['auxi5'] = O.ts_corr(O.ts_rank(df['vwap'],10),O.ts_rank(df['matchingVolume'],10),5)
        df['quantity2'] = O.ts_rank(df['auxi5'],window)
        df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1)

        # Alpha#81: ((rank(Log(product(rank((rank(auxi2)^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        return df['signal']

    @staticmethod
    def alpha_151(df: pd.DataFrame):
        df['auxi1'] = (df['close'].shift(20) - df['close'].shift(10))/10
        df['auxi2'] = (df['close'].shift(10)- df['close'])/10
        df['auxi3'] = df['close'].diff(1)
        df['signal'] = np.where(
                                df['auxi1']- df['auxi2'] < -0.05,
                                1,
                                -1*df['auxi3'])
        return df['signal']*(-1)
        #Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    @staticmethod
    def alpha_052(df: pd.DataFrame,window=5):
        df['returns'] = df['close'] / df['close'].shift(1) - 1
        df['quantity1'] = O.ts_min(df['low'],window).shift(window) - O.ts_min(df['low'],window)
        df['auxi1'] = (df['returns'].rolling(240).sum() - df['returns'].rolling(20).sum())/220
        df['quantity2'] = O.ts_rank(df['auxi1'],5)
        df['quantity3'] = O.ts_rank(df['matchingVolume'],5)
        df['signal'] = df['quantity1'] * df['quantity2'] * df['quantity3']
        return df['signal']

    @staticmethod
    def alpha_055(df:pd.DataFrame,window=6):
        df['auxi1'] = df['close'] - O.ts_min(df['low'],12)
        df['auxi2'] = O.ts_max(df['high'],12) - O.ts_min(df['low'],12)
        df['quantity1'] = O.ts_rank(df['auxi1']/df['auxi2'],6)
        df['quantity2'] = O.ts_rank(df['matchingVolume'],6)

        df['signal'] = O.ts_corr(df['quantity1'],df['quantity2'],window)
        return df['signal']

    @staticmethod
    def alpha_061(df:pd.DataFrame,window=18):
        df = Domains.compute_vwap(df)
        df['quantity1'] = O.ts_rank(df['vwap'] - O.ts_min(df['vwap'],16))
        df['quantity2'] = O.ts_rank(O.ts_corr(df['vwap'],df['matchingVolume'].rolling(180).sum(),window))
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1)

        df['signal'] = np.where(df['quantity1'] < df['quantity2'], df['quantity1'], df['quantity2']) * (-1)
        df['signal'] = df['signal'] * (df['close'] - df['open'])

        df['signal'] = np.select([df['signal']>100,df['signal'] < -100],[100,-100],df['signal'])
        df['signal'] = df['signal']/100
        df['signal'] = df['signal'] *7
        df['signal'] = np.select([df['signal']>7,df['signal']<-7],[7,-7],df['signal'])
        df['signal'] = df['signal']/7
        return df['signal']*(-1)

    @staticmethod
    def alpha_062(df:pd.DataFrame,window=10):
        df = df.copy()
        df = Domains.compute_vwap(df)
        df['auxi1'] = O.ts_corr(df['vwap'],df['matchingVolume'].rolling(20).mean().rolling(22).sum(),window)
        df['quantity1'] = O.ts_rank(df['auxi1'])
        df['auxi2'] = O.ts_rank(df['high']+df['low']) + O.ts_rank(df['high'])
        df['auxi3'] = np.where(O.ts_rank(df['open']) < df['auxi2'], 1, -1)
        df['auxi4'] = O.ts_rank(df['open']) + df['auxi3']
        df['quantity2'] = O.ts_rank(df['auxi4'])
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)

        df['signal'] = np.where(
            df['quantity1'] < df['quantity2'],
            df['quantity1'] ,
            df['quantity2']) * (-1)
        df['signal'] = df['signal'] * (df['close'] - df['open'])
        df['signal'] = np.select(
            [df['signal'] > 100,
             df['signal'] < -100],
            [100, -100],
            df['signal'])
        df['signal'] = df['signal'] / 100
        df['signal'] = df['signal'] *7
        df['signal'] = np.select(
            [df['signal'] > 7,
             df['signal'] < -7],
            [7, -7],
            df['signal'])
        df['signal'] = df['signal'] / 7
        return df['signal'] * (-1)

    @staticmethod
    def alpha_262(df: pd.DataFrame):
        df = df.copy()
        df = Domains.compute_vwap(df)
        df['auxi1'] = O.ts_corr(df['vwap'], df['matchingVolume'].rolling(20).mean().rolling(22).sum(), 10)
        df['quantity1'] = O.ts_rank(df['auxi1'])
        df['auxi2'] = O.ts_rank(df['high'] + df['low']) + O.ts_rank(df['high'])
        df['auxi3'] = np.where(O.ts_rank(df['open']) * 2 < df['auxi2'], 1, -1)
        df['auxi4'] = O.ts_rank(df['auxi3'])
        df['quantity2'] = O.ts_rank(df['auxi4'])
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)

        df['signal'] = np.where(df['quantity1'] < df['quantity2'], df['quantity1'], df['quantity2']) * (-1)
        df['signal'] = df['signal'] * (df['close'] - df['open'])
        df['signal'] = df['signal'] / 32

        return df['signal'] * (-1)

    @staticmethod
    def alpha_162(df):
        correlation = O.ts_corr
        rank = O.ts_rank
        openn = df['open']
        high = df['high']
        low = df['low']
        vwap = O.compute_vwap(df)['vwap']
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()
        summ = lambda df, window: df.rolling(window).sum()
        # noinspection PyTypeChecker
        signal = (
            (
                rank(
                    correlation(
                        vwap,
                        summ(adv20, int(22.4101)),
                        int(9.91009)
                    )
                )
                <
                rank(
                    (rank(openn) + rank(openn))
                    <
                    (rank(high + low) + rank(high))
                )
            )
            * -1
        )
        signal = signal * 2 + 1
        # signal.hist()
        # plt.show()
        return signal

    @staticmethod
    def alpha_064(df:pd.DataFrame,window=100):
        df = Domains.compute_vwap(df)
        const = 0.178404
        df['auxi1'] = df['open'] * const + df['low']*(1-const)
        df['auxi2'] = df['auxi1'].rolling(13).sum()
        df['auxi3'] = df['matchingVolume'].rolling(120).mean().rolling(13).sum()
        df['quantity1'] = O.ts_rank(O.ts_corr(df['auxi2'], df['auxi3'],17),10)

        df['auxi4'] = (df['high'] + df['low']) / 2 * const + df['vwap']*(1-const)
        df['auxi5'] = df['auxi4'].diff(4)
        df['quantity2'] = O.ts_rank(df['auxi5'], 10)

        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)
        df['signal'] = np.where(df['quantity1'] < df['quantity2'], df['quantity1'], df['quantity2']) * (-1)
        df['signal'] = df['signal'] * (df['close'] - df['open'])
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        df['signal'] = df['signal'] *5
        df['signal'] = np.select(
            [df['signal']>5,df['signal']<-5],
            [5,-5],
            df['signal'])

        return df['signal'] * (-1)
        #Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),

    @staticmethod
    def alpha_065(df:pd.DataFrame,window=100):
        df = Domains.compute_vwap(df)
        const = 0.00817205
        df['auxi1'] = df['open']* const + df['vwap']*(1-const)
        df['auxi2'] = df['matchingVolume'].rolling(60).mean().rolling(9).sum()
        df['quantity1'] = O.ts_rank(O.ts_corr(df['auxi1'],df['auxi2'],6),10)

        df['auxi3'] = df['open'] - O.ts_min(df['open'],14)
        df['quantity2'] = O.ts_rank(df['auxi3'],10)

        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)
        df['signal'] = np.where(df['quantity1']<df['quantity2'],df['quantity1'],df['quantity2']) * (-1)
        df['signal'] = df['signal']*(df['close']-df['open'])
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()

        return df['signal']*(-1)

    @staticmethod
    def alpha_068(df:pd.DataFrame,window=100):
        const = 0.518371
        df['auxi1'] = O.ts_corr(O.ts_rank(df['high'],10),O.ts_rank(df['matchingVolume'].rolling(15).mean(),10),9)
        df['quantity1'] = O.ts_rank(df['auxi1'],14)

        df['auxi2'] = df['close']*const + df['low']*(1-const)
        df['quantity2'] = O.ts_rank(df['auxi2'].diff(),10)

        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)
        df['signal'] = np.where(df['quantity1']<df['quantity2'],df['quantity1'],df['quantity2']) * (-1)
        df['signal'] = df['signal']*(df['close']-df['open'])
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()

        df['signal'] = df['signal'] *5
        df['signal'] = np.select([df['signal']>5,df['signal']<-5],[5,-5],df['signal'])
        return df['signal']*(-1)

    @staticmethod
    def alpha_074(df:pd.DataFrame,window=100):
    
        df = Domains.compute_vwap(df)
        const = 0.0261661
        df['auxi1'] = df['matchingVolume'].rolling(30).mean().rolling(37).sum()
        df['quantity1'] = O.ts_rank(O.ts_corr(df['close'],df['auxi1'],15),10)
        
        df['auxi2'] = df['high'] * const + df['vwap']*(1-const)
        df['rank1'] =  O.ts_rank(df['auxi2'],10)
        df['rank2'] = O.ts_rank(df['matchingVolume'],10)
        df['quantity3'] = O.ts_corr(
                            df['rank1'], df['rank2'],11)
        df['quantity3'] = df['quantity3'].replace([np.inf,-np.inf,np.nan],0)
        
        df['quantity2'] = O.ts_rank(
                        df['quantity3'],10
        )
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)
        df['signal'] = np.where(df['quantity1']<df['quantity2'],df['quantity1'],df['quantity2']) * (-1)

        df['signal'] = df['signal']*(df['close']-df['open'])
        
        # print(df.tail(20).to_string())
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        
        # print(df.tail(20).to_string())
        
        df['signal'] = df['signal'] *5
        df['signal'] = np.select([df['signal']>5,df['signal']<-5],[5,-5],df['signal'])
        return df['signal']*(-1)

    @staticmethod
    def alpha_075(df:pd.DataFrame,window=100,window_corr_vwap=4,window_corr_volume=12):
        df = Domains.compute_vwap(df)
        df['quantity1'] = O.ts_rank(
            O.ts_corr(df['vwap'],df['matchingVolume'],window_corr_vwap)
            ,10
        )
        df['quantity2'] = O.ts_rank(
            O.ts_corr(
                O.ts_rank(df['low']),O.ts_rank(df['matchingVolume'].rolling(50).mean()),window_corr_volume
            )
            ,10
        )
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)
        df['signal'] = np.where(df['quantity1']<df['quantity2'],df['quantity1'],df['quantity2']) * (-1)
        df['signal'] = df['signal']*(df['close']-df['open'])
        df['signal'] = df['signal'] / df['signal'].abs().rolling(window).max()
        return df['signal']*(-1)

    @staticmethod
    def alpha_077(df,window=10):
        high, low, volume = df['high'], df['low'], df['matchingVolume']
        vwap = O.compute_vwap(df, 40)['vwap']
        adv40 = O.ts_mean(volume, 40)

        part1 = O.ts_rank(O.decay_linear((((high + low) / 2 + high) - (vwap + high)), 20),window)
        part2 = O.ts_rank(O.decay_linear(O.ts_corr((high + low) / 2, adv40, 3), 6),window)

        raw_alpha = np.minimum(part1, part2) / 9
        return raw_alpha
    


    @staticmethod
    def alpha_078(df:pd.DataFrame,window=10):
        df = Domains.compute_vwap(df)
        const = 0.352233

        df['auxi1'] = df['low']*const + df['vwap']*(1-const)
        df['quantity1'] = O.ts_rank(
            O.ts_corr(
                df['auxi1'].rolling(20).sum(),df['matchingVolume'].rolling(40).mean().rolling(20).sum(),7
            )
            ,window
        )
        df['quantity2'] = O.ts_rank(
            O.ts_corr(
                O.ts_rank(df['vwap'],10), O.ts_rank(df['matchingVolume'],10),6
            )
            ,window
        )
        df['signal'] = df['quantity1'] ** df['quantity2']
        return df['signal']

    @staticmethod
    def alpha_083(df:pd.DataFrame,window=10):
        df = Domains.compute_vwap(df)
        df['auxi1'] = df['close'].rolling(5).mean()
        df['auxi2'] = (df['high']-df['low'])/df['auxi1']
        df['quantity1'] = O.ts_rank(df['auxi2'].shift(2),window)
        df['quantity2'] = O.ts_rank(df['matchingVolume'],window)
        df['quantity3'] = df['auxi2'] /(df['vwap']-df['close'])

        df['signal'] = df['quantity1'] * df['quantity2'] / df['quantity3']
        return df['signal']*(-1)

    @staticmethod
    def alpha_149(df:pd.DataFrame):
        df['quantity1'] = (df['close'].shift(20) - 2*df['close'].shift(10) + df['close'])/10
        df['quantity2'] = df['close'].diff(1)
        # df['signal'] = np.where(df['quantity1']<-0.1,
        #                         1,
        #                         df['quantity2']*(-1))
        df['signal'] = np.where(df['quantity1']<-0.1,
                        df['quantity2'],
                        df['quantity2']*(-1))
        df['signal'] = df['signal']/2 + 2/5
        return df['signal']*(-1)

    @staticmethod
    def alpha_150(df:pd.DataFrame):
        df = Domains.compute_vwap(df, 20)
        df['auxi1'] = O.ts_rank(df['matchingVolume'],20)
        df['auxi2'] = O.ts_rank(df['close'],20)

        df['signal'] = \
            O.ts_max(
                O.ts_corr(
                    df['auxi1'],
                    df['auxi2'],
                    20
                )
                ,5
            )
        # df['signal'] = df['signal']*5/3 - 1/10
        return df['signal']

    @staticmethod
    def alpha_136(df:pd.DataFrame,window=200):
        df['returns'] = df['close'] / df['close'].shift(1) - 1
        df = Domains.compute_vwap(df,window)

        df['rank1'] = 2.21*\
            O.ts_rank(
                O.ts_corr(
                    df['close'] - df['open'],
                    df['matchingVolume'].shift(1),
                    15
                ),
                10
            )
        df['rank2'] = 0.7*\
            O.ts_rank(df['open']-df['close'])
        df['rank3'] = 0.73*\
            O.ts_rank(
                O.ts_rank(
                    df['returns'].shift(6)*(-1),
                    6)
                    ,5
            )
        df['rank4'] = O.ts_rank(
            O.ts_corr(
                df['vwap'],df['matchingVolume'].rolling(20).mean(),6
            ).abs()
        ,10)
        df['rank5'] = 0.6*\
            O.ts_rank(
                (df['close'].rolling(200).mean() - df['open'])
                *
                (df['close']-df['open'])
            ,10)
        df['signal'] = df['rank1'] + df['rank2'] + df['rank3'] + df['rank4'] + df['rank5']
        df['signal'] = df['signal']/20 - 1.45
        return df['signal']*(-1)

    @staticmethod
    def alpha_071(df:pd.DataFrame,window=30):
        try:
            df = df.copy()
            df = Domains.compute_vwap(df,window)

            df['auxi1'] = O.ts_corr(
                O.ts_rank(df['close'],3),
                O.ts_rank(df['matchingVolume'].rolling(180).mean(),12),
                18
            )
            df['rank1'] = O.ts_rank(
                O.ts_weighted_mean(df['auxi1'],4),
                16
            )
            df['auxi2'] = O.ts_rank((df['low'] + df['open'] - df['vwap']*2),10)**2
            df['rank2'] = O.ts_rank(
                O.ts_weighted_mean(
                    df['auxi2'],16
                )
                ,4
            )
            df['signal'] = df[['rank1','rank2']].max(axis=1)
            df['signal'] = (df['signal'] - 8.5)/7.5

            df['signal'] = np.where(df['signal']>0,
                                    df['signal']*(df['close']-df['open']),
                                    df['signal']*(df['close']-df['open'])*(-1))
            df['signal'] = df['signal']/5
        except Exception as e:
            U.report_error(e)
        return df['signal']

    @staticmethod
    def alpha_066(df: pd.DataFrame,window=10):
        try:
            df = df.copy()
            df = Domains.compute_vwap(df, 30)
            const = 0.96633
            df['rank1'] = O.ts_rank(
                O.ts_weighted_mean(
                    df['vwap'].diff(4), 7
                )
                , window
            )
            df['auxi1'] = df['low'] * const + df['high'] * (1 - const)
            df['auxi1_5'] = np.where(
                df['open'] == (df['high'] + df['low']) / 2,
                0.01,
                (df['open'] - (df['high'] + df['low']) / 2)
            )
            df['auxi2'] = (df['auxi1'] - df['vwap']) / df['auxi1_5']
            df['rank2'] = O.ts_rank(
                O.ts_weighted_mean(
                    df['auxi2'], 11
                )
                , window
            )
            df['signal'] = (df['rank1'] + df['rank2']) * (-1)
            df['signal'] = (df['signal'] + 14) / 12
            df['signal'] = df['signal'] * (df['close'] - df['open'])
            df['signal'] = df['signal'] / 4
        except Exception as e:
            U.report_error(e)
        return df['signal']

    @staticmethod
    def alpha_073(df:pd.DataFrame,window=20):
        try:
            df = df.copy()
            df = Domains.compute_vwap(df,window)
            const = 0.147155
            df['auxi1'] = df['open']*const + df['low']*(1-const)

            df['rank1'] = O.ts_rank(
                O.linear_weighted_moving_average(
                    df['vwap'].diff(5)
                    ,3
                )
                ,9
            )
            df['rank2'] = O.ts_rank(
                O.linear_weighted_moving_average(
                    df['auxi1'].diff(2) / df['auxi1'] * (-1)
                    ,3
                )
                ,17
            )
            # df['rank1_fake'] = df['rank1'] /
            df['signal'] = df[['rank1','rank2']].max(axis=1)
            df['signal'] = (df['signal'] - 9)/8
            df['signal'] = np.where(df['signal']>0,
                            df['signal']*(df['close']-df['open']),
                            df['signal']*(df['close']-df['open'])*(-1)
                            )
            df['signal'] = df['signal']/4
        except Exception as e:
            U.report_error(e)
        return df['signal']

    @staticmethod
    def alpha_volume_weighted_z_score(df: pd.DataFrame,window=20):
        df = df.copy()
        alpha_raw =  (df['matchingVolume'] * df['close'].diff(1))
        rolling_mean = alpha_raw.rolling(window=window).mean()
        rolling_std = alpha_raw.rolling(window=window).std()
        signal = (alpha_raw - rolling_mean) / rolling_std
        return signal
    
    @staticmethod
    def alpha_c01(df: pd.DataFrame, window=10):
        part1 = Alphas.alpha_101(df, window=window)
        part2 = Alphas.alpha_038(df, window=window)
        combined = O.ts_rank(part1) + O.ts_rank(part2)
        return combined * (-1)

    @staticmethod
    def alpha_c02(df, window=10):
        part1 = Alphas.alpha_002(df, window=window)
        part2 = Alphas.alpha_013(df, window=window)
        combined = O.ts_rank(part1) + O.ts_rank(part2)
        return combined * (-1)
    
    @staticmethod
    def alpha_c05(df,window=10):
        part1 = -1 * O.ts_rank(O.ts_rank(df['low'], window))
        part2 = Alphas.alpha_017(df, window)
        combined = O.ts_rank(part1) + O.ts_rank(part2)
        return combined

    @staticmethod
    def alpha_c06(df, window=10):
        part1 = Alphas.alpha_006(df, window=window)
        part2 = Alphas.alpha_044(df, window=window)
        combined = O.ts_rank(part1) + O.ts_rank(part2)
        return combined
    
    @staticmethod
    def alpha_new_003_v1(df: pd.DataFrame, window=20):
        """
        Variation 1: Volume Confirmation
        Logic: A trend day is more reliable if confirmed by high volume. The original
        signal is weighted by the relative trading volume.
        """
       
        # Original logic
        price_range = df['high'] - df['low']
        avg_range = O.ts_mean(price_range, window=window)
        range_expansion = price_range / (avg_range + 1e-9)
        position_in_range = (df['close'] - df['low']) / (price_range + 1e-9)
        position_signal = (2 * position_in_range) - 1
        original_raw_signal = range_expansion * position_signal

        # Volume weight
        relative_volume = df['matchingVolume'] / (O.ts_mean(df['matchingVolume'], window=window) + 1e-9)
        
        # Weighted signal
        raw_signal = original_raw_signal * relative_volume
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_003_v2(df: pd.DataFrame, window=20):
        """
        Variation 2: Rank-Based Robustness
        Logic: Uses ts_rank on the core components (range and position) to make the
        signal less sensitive to outliers, a common technique for robustness.
        """
        
        price_range = df['high'] - df['low']
        position_in_range = (df['close'] - df['low']) / (price_range + 1e-9)

        # Rank-based components
        range_expansion_rank = O.ts_rank_normalized(price_range, window=window)
        position_rank = O.ts_rank_normalized(position_in_range, window=window)
        
        # Combine ranked signals
        # A high rank in both indicates a strong signal
        raw_signal = range_expansion_rank * (2 * position_rank - 1)
        
        # The signal is already normalized due to ranks, but we can smooth it
        signal = O.ts_weighted_mean(raw_signal, window=3)
        
        return signal

    @staticmethod
    def alpha_new_003_v3(df: pd.DataFrame,window=20):
        """
        Variation 3: Open-Close Efficiency
        Logic: Instead of the close's position in the full day's range, this
        measures the efficiency of the open-to-close move relative to the day's
        total volatility.
        """
        
        price_range = df['high'] - df['low']
        avg_range = O.ts_mean(price_range, window=window)
        range_expansion = price_range / (avg_range + 1e-9)
        
        # Open-Close efficiency signal
        efficiency_signal = (df['close'] - df['open']) / (price_range + 1e-9)
        
        raw_signal = range_expansion * efficiency_signal
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_003_v4(df: pd.DataFrame, window=20):
        """
        Variation 4: Range Acceleration
        Logic: Captures the acceleration of range expansion, not just its size.
        A sudden increase in volatility can be a powerful leading signal.
        """
        
        price_range = df['high'] - df['low']
        
        # Range acceleration component
        range_acceleration = O.ts_delta(price_range, 1) / (O.ts_mean(price_range, window=window) + 1e-9)
        
        # Original position signal
        position_in_range = (df['close'] - df['low']) / (price_range + 1e-9)
        position_signal = (2 * position_in_range) - 1
        
        raw_signal = range_acceleration * position_signal
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_003_v5(df: pd.DataFrame, window=20):
        """
        Variation 5: Exhaustion / Mean-Reversion
        Logic: An extremely wide range with a close near the extremes might signal
        a climactic exhaustion point. This alpha bets on a reversal (mean-reversion)
        following such a strong trend day.
        """
        
        price_range = df['high'] - df['low']
        avg_range = O.ts_mean(price_range, window=window)
        range_expansion = price_range / (avg_range + 1e-9)
        
        position_in_range = (df['close'] - df['low']) / (price_range + 1e-9)
        position_signal = (2 * position_in_range) - 1
        
        # The raw signal is the same as the original alpha
        raw_signal = range_expansion * position_signal
        
        # Bet against the strong move
        reversal_signal = -raw_signal
        
        adaptive_divisor = reversal_signal.abs().rolling(window).max() + 1e-9
        signal = reversal_signal / adaptive_divisor
        
        return signal



    @staticmethod
    def alpha_new_008_v1(df: pd.DataFrame, window=20, factor=2):
        """
        Variation 1: VWAP-based Bands
        Logic: Uses VWAP as the centerline for the bands instead of a simple moving
        average, making the breakout signal sensitive to volume.
        """
        
        vwap = O.compute_vwap(df, window=window)['vwap']
        # Use standard deviation from the VWAP, not the simple mean
        std = df['close'].rolling(window).std()

        upper_band = vwap + factor * std
        lower_band = vwap - factor * std

        buy_breakout = df['close'] - upper_band
        sell_breakout = df['close'] - lower_band
        
        raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_008_v2(df: pd.DataFrame, window=20, factor=2):
        """
        Variation 2: ATR Bands (Keltner Channels)
        Logic: Uses Average True Range (ATR) to define the band width instead of
        standard deviation, which can be more robust to price shocks.
        """
        
        mean = O.ts_mean(df['close'], window)
        
        # Calculate ATR
        tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = O.ts_mean(tr, window)
        
        upper_band = mean + factor * atr
        lower_band = mean - factor * atr
        
        buy_breakout = df['close'] - upper_band
        sell_breakout = df['close'] - lower_band
        
        raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_008_v3(df: pd.DataFrame, window=20, factor=2):
        """
        Variation 3: High/Low Breakout
        Logic: Triggers a signal based on the high or low breaking the bands,
        rather than the close, providing an earlier, more aggressive signal.
        """
        
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        upper_band = mean + factor * std
        lower_band = mean - factor * std

        # Use high for buy breakout, low for sell breakout
        buy_breakout = df['high'] - upper_band
        sell_breakout = df['low'] - lower_band
        
        raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_008_v4(df: pd.DataFrame, window=20, factor=2):
        """
        Variation 4: Volume-Weighted Breakout Magnitude
        Logic: The magnitude of the breakout signal is weighted by relative volume,
        giving more importance to breakouts that occur on high trading activity.
        """
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        upper_band = mean + factor * std
        lower_band = mean - factor * std

        buy_breakout = df['close'] - upper_band
        sell_breakout = df['close'] - lower_band
        
        original_raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)

        # Volume weight
        relative_volume = df['matchingVolume'] / (O.ts_mean(df['matchingVolume'], window=window) + 1e-9)
        
        # Weighted signal
        raw_signal = original_raw_signal * relative_volume
        
        adaptive_divisor = raw_signal.abs().rolling(window).max() + 1e-9
        signal = raw_signal / adaptive_divisor
        
        return signal

    @staticmethod
    def alpha_new_008_v5(df: pd.DataFrame, window=20, factor=2):
        """
        Variation 5: Breakout Fade (Mean Reversion)
        Logic: Bets against the breakout, assuming that a sharp move outside the
        bands is an over-extension that will revert back towards the mean.
        """
        mean = O.ts_mean(df['close'], window)
        std = O.ts_std(df['close'], window)
        upper_band = mean + factor * std
        lower_band = mean - factor * std

        buy_breakout = df['close'] - upper_band
        sell_breakout = df['close'] - lower_band
        
        raw_signal = buy_breakout.where(buy_breakout > 0, 0) + sell_breakout.where(sell_breakout < 0, 0)
        
        # Fade the breakout by taking the negative signal
        reversal_signal = -raw_signal
        
        adaptive_divisor = reversal_signal.abs().rolling(window).max() + 1e-9
        signal = reversal_signal / adaptive_divisor
        
        return signal




    @staticmethod
    def alpha_new_005_up1(df: pd.DataFrame, window=20):
        """
        Upgrade 1: Volume and Trend Context
        Logic: Weights the original decisiveness signal by relative volume and
        the broader trend context to identify higher-quality signals.
        """
        # Original decisiveness signal
        directional_range = df['close'] - df['open']
        total_range = df['high'] - df['low']
        trend_ratio = directional_range / (total_range + 1e-9)
        
        # Volume Weight
        relative_volume = O.ts_rank_normalized(df['matchingVolume'], window)
        
        # Trend Context
        trend_context = O.ts_rank_normalized(O.ts_delta(df['close'], window), window)
        
        # Combined Signal
        raw_signal = trend_ratio * relative_volume * trend_context
        
        # Smooth and scale
        signal = O.ts_weighted_mean(raw_signal, 3)
        signal = signal.clip(-1, 1) # Clip to ensure it's within [-1, 1]
        
        return signal
    
    @staticmethod
    def alpha_118(df,window=10):
        bar_height = df['close'] - df['open']
        signal = \
        (
            -1
            *
            O.ts_rank
            (
                # O.ts_std(abs(bar_height), 5)
                # +
                bar_height,
                window
                # +
                # O.ts_corr(df['close'],
                #           df['open'],10)
            )
        )
        normalized_signal = -(signal / 5 + 1) / 0.9 - 1/9
        return normalized_signal

    @staticmethod
    def alpha_218(df: pd.DataFrame,window=10):
        df = df.copy()
        df['signal'] = O.ts_rank_normalized(df['close']-df['open'],window)
        df['signal'] = df['signal'] * 2 -1
        return df['signal'] 
    
    @staticmethod
    def alpha_262(df: pd.DataFrame,window=10):
        df = df.copy()
        df = O.compute_vwap(df)
        df['auxi1'] = O.ts_corr(df['vwap'], df['matchingVolume'].rolling(20).mean().rolling(22).sum(),10)
        df['quantity1'] = O.ts_rank(df['auxi1'],window)
        df['auxi2'] = O.ts_rank(df['high'] + df['low']) + O.ts_rank(df['high'],window)
        df['auxi3'] = np.where(O.ts_rank(df['open'],window) * 2 < df['auxi2'], 1, -1)
        df['auxi4'] = O.ts_rank(df['auxi3'],window)
        df['quantity2'] = O.ts_rank(df['auxi4'],window)
        # df['signal'] = np.where(df['quantity1']<df['quantity2'],1,-1) * (-1)

        df['signal'] = np.where(df['quantity1'] < df['quantity2'], df['quantity1'], df['quantity2']) * (-1)
        df['signal'] = df['signal'] * (df['close'] - df['open'])
        df['signal'] = df['signal'] / 32

        return df['signal'] * (-1)

    @staticmethod
    def alpha_162(df,window=10):
        correlation = O.ts_corr
        rank = O.ts_rank
        openn = df['open']
        high = df['high']
        low = df['low']
        vwap = O.compute_vwap(df)['vwap']
        adv20 = df.rolling(20) \
            .matchingVolume \
            .mean()
        summ = lambda df, window: df.rolling(window).sum()
        # noinspection PyTypeChecker
        signal = (
            (
                rank(
                    correlation(
                        vwap,
                        summ(adv20, int(22.4101)),
                        int(9.91009)
                    ),window
                )
                <
                rank(
                    (rank(openn) + rank(openn))
                    <
                    (rank(high + low) + rank(high)),window
                )
            )
            * -1
        )
        signal = signal * 2 + 1
        # signal.hist()
        # plt.show()
        return signal

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

        # danh sách alpha mặc định (1-301) + các alpha đặc biệt
        base_list = list(range(1, 302)) + [
            'alpha_zscore', 'alpha_questionable', 'alpha_bbb', 'alpha_keltner',
            'alpha_kema', "alpha_418", "alpha_donchian_channel",
            "alpha_rti", "alpha_ursi","alpha_101_volume","alpha_volume_weighted_z_score",
            "alpha_101_volume_smoothed","alpha_101_trend_confirm","alpha_101_stddev_normalized",
            "alpha_101_vwap_vol_rank","alpha_101_mean_reversion","alpha_101_corr_weighted",
            "alpha_101_trend_strength_weighted","alpha_101_positional_combo","alpha_101_regime_filter",
            "alpha_101_acceleration","alpha_101_oi_confirm","alpha_101_gap_filtered","alpha_101_body_wick_ratio",
            "alpha_101_decay_smoothed","alpha_101_plus_alpha_008","alpha_101_asymmetric","alpha_101_zscore",
            "alpha_101_rank_combo","alpha_101_overnight_confirm","alpha_101_powered","alpha_101_day_of_week_filter",
            "alpha_new_003_v1","alpha_new_003_v2","alpha_new_003_v3","alpha_new_003_v4","alpha_new_003_v5",
            "alpha_new_005_up1","alpha_new_008_v1","alpha_new_008_v2","alpha_new_008_v3","alpha_new_008_v4","alpha_new_008_v5",
        ]

        custom_c_list = [f"c{str(i).rjust(2, '0')}" for i in range(1, 51)]
        new_alpha_list = [f"alpha_new_{str(name).rjust(3, '0')}" for name in list(range(1, 101))]
        
        for alpha_name in base_list + custom_c_list + new_alpha_list:
            if isinstance(alpha_name, int):
                alpha_name = str(alpha_name).rjust(3, '0')
            if not alpha_name.startswith('alpha_'):
                alpha_name = f'alpha_{alpha_name}'

            dic = globals()
            alpha = None

            if alpha_name in dic:
                alpha = dic[alpha_name]
            else:
                try:
                    alpha = Alphas().__getattribute__(alpha_name)
                except Exception as e:
                    if len(str(e)) == 0:
                        print(e, end='')

            if alpha is None:
                if verbosity >= 2:
                    print(header,
                        f'Found 0 alpha with name=\x1b[91m{alpha_name}\x1b[0m')
            else:
                dic_alphas[alpha_name] = alpha

        if verbosity >= 2:
            print(header,
                f'Found \x1b[93m{len(dic_alphas)}\x1b[0m alpha functions')

        return dic_alphas




