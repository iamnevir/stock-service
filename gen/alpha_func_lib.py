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
    ##############################################################################################################################################################################################
    @staticmethod
    def alpha_full_factor_001(df: pd.DataFrame, window=80):
        # 1. Tính Min và Max trong cửa sổ 48 phiên
        # Sử dụng O.ts_min và O.ts_max
        min_close = O.ts_min(df['close'], window)
        max_close = O.ts_max(df['close'], window)
        
        # 2. Tính toán biểu thức lõi (giống công thức Stochastic %K)
        # (close - min) / (max - min + 1e-8)
        # Giá trị này sẽ nằm trong khoảng [0, 1]
        stoch = (df['close'] - min_close) / (max_close - min_close + 1e-8)
        
        # 3. CsRank: Xếp hạng theo chiều ngang (cross-sectional rank)
        # Lưu ý: O.rank của bạn dùng axis=1, tức là nó cần DataFrame 
        # có các cột là các mã cổ phiếu khác nhau.
        ranked_stoch = O.ts_rank_normalized(stoch)
        
        # 4. Neg: Lấy giá trị âm
        signal = 1 - (2 * ranked_stoch)

        return -signal
    @staticmethod
    def alpha_full_factor_004(df: pd.DataFrame, window=50):
        # 1. Tính toán Mean Volume (Sử dụng O.ts_mean)
        mean_vol = O.ts_mean(df['matchingVolume'], window)
        
        # 2. Tính tỷ lệ Khối lượng hiện tại / Khối lượng trung bình
        vol_ratio = df['matchingVolume'] / (mean_vol + 1e-8)
        
        # 3. Tính Time-series Rank của Volume Ratio (Range [0, 1])
        rank_vol = O.ts_rank_normalized(vol_ratio, window)
        
        # 4. Tính Returns (Tỷ suất sinh lời)
        returns = df['close'].pct_change()
        
        # 5. Tính Neg(returns) và Rank nó (Range [0, 1])
        neg_returns = -1 * returns
        rank_neg_ret = O.ts_rank_normalized(neg_returns, window)
        
        # 6. Mul (Nhân hai giá trị Rank lại với nhau)
        raw_signal = rank_vol * rank_neg_ret
        
        signal = (raw_signal * 2) - 1

        
        return -signal
    
    @staticmethod
    def alpha_full_factor_007(df: pd.DataFrame, window=50, factor=10):
        factor = int(factor)
        price_pos = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        rank_price_pos = O.ts_rank_normalized(price_pos, window)
 
        vol_ema = O.decay_linear(df['matchingVolume'], factor)
        

        vol_ema_delta = O.ts_delta(vol_ema, factor)
        rank_vol_delta = O.ts_rank_normalized(vol_ema_delta, window)
        

        signal = rank_vol_delta - rank_price_pos
        
        return -signal
    
    @staticmethod
    def alpha_full_factor_031(df: pd.DataFrame, short_delta=1, trend_delta=3, window_rank=12):
        
        delta_1 = O.ts_delta(df['close'], short_delta)
        delta_3 = O.ts_delta(df['close'], trend_delta)
        
        rank_trend = O.ts_rank_normalized(delta_3, window_rank)
        
        raw_signal = np.where(delta_1 > 0, -1 * rank_trend, rank_trend)
        
        signal = raw_signal * 2 
        signal = np.clip(signal, -1, 1)
        signal = pd.Series(raw_signal, index=df.index)
        
        return -signal

    @staticmethod
    def alpha_full_factor_034(df: pd.DataFrame, window=2, factor=5, window_corr_vwap=15):
        factor = int(factor)
        delta_short = O.ts_delta(df['close'], window)
        delta_long = O.ts_delta(df['close'], factor)
        
        rank_long = O.ts_rank_normalized(delta_long, window_corr_vwap)
        
        raw_signal = np.where(delta_short > 0, -1 * rank_long, rank_long)
        signal = pd.Series(raw_signal, index=df.index)
        
        return -signal 

    @staticmethod
    def alpha_full_factor_046(df: pd.DataFrame, window=5):
        returns = df['close'].pct_change()
        
        short_std = O.ts_std(returns, 12)
        long_std_mean = O.ts_mean(short_std, window)
        price_delta = O.ts_delta(df['close'], 3)
        rev_signal = -1 * price_delta
        
        condition = short_std > long_std_mean
        raw_signal = np.where(condition, rev_signal, 0)
      
        raw_signal_ser = pd.Series(raw_signal, index=df.index)
        rank_final = O.ts_rank_normalized(raw_signal_ser, window)
        
        signal = (rank_final * 2) - 1
        return -signal

    @staticmethod
    def alpha_full_factor_062(df: pd.DataFrame, window=70): 
        
        returns = df["close"].pct_change()
        amt = df["close"] * df["matchingVolume"]
        
        rank_ret = O.ts_rank_normalized(returns, window=window)
        
        amt_std = amt.rolling(window).std()
        
        raw_signal =  (rank_ret / (amt_std + 1e-6))
        
        alpha_val = O.ts_rank_normalized(raw_signal, window=window) * 2 - 1

        return pd.Series(alpha_val, index=df.index).fillna(0)
    
    @staticmethod
    def alpha_full_factor_066(df: pd.DataFrame, window=48, factor=3):
        factor = int(factor)
        volume = df["matchingVolume"]
        amt = df["close"] * volume
        returns = df["close"].pct_change()
        
        vol_kurt = volume.rolling(window).kurt()
        
        amt_delta = amt.diff(factor)
        sig_amt_accel = -1 * O.ts_rank_normalized(amt_delta, window=window)
        
        sig_ret_rev = -1 * (O.ts_rank_normalized(returns, window=window) * 2 - 1)
        
        is_vol_shock = vol_kurt > 3.0
        raw_signal = np.where(is_vol_shock, sig_amt_accel, sig_ret_rev)
        
        alpha_val = pd.Series(raw_signal, index=df.index)
        
        return -alpha_val.fillna(0)

    @staticmethod
    def alpha_full_factor_003(df: pd.DataFrame, window=5, delta_period=1):
        df = O.compute_vwap(df.copy(), window)

        term1 = O.ts_delta(df['matchingVolume'], delta_period)
        term2 = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)

        term1_min = O.ts_min(term1, window).shift(1)
        term1_max = O.ts_max(term1, window).shift(1)
        term1_scaled = 2 * (term1 - term1_min) / (term1_max - term1_min + 1e-8) - 1

        term2_min = O.ts_min(term2, window).shift(1)
        term2_max = O.ts_max(term2, window).shift(1)
        term2_scaled = 2 * (term2 - term2_min) / (term2_max - term2_min + 1e-8) - 1

        alpha = (term1_scaled - term2_scaled) / 2

        return -alpha.clip(-1, 1).fillna(0.0)

    @staticmethod
    def alpha_full_factor_002(df: pd.DataFrame, ma_window=5, norm_window=10):
        ma_val = O.decay_linear(df['close'], ma_window)
        deviation = (df['close'] - ma_val) / (ma_val + 1e-8)

        d_min = O.ts_min(deviation, norm_window)
        d_max = O.ts_max(deviation, norm_window)

        scaled = 2 * (deviation - d_min) / (d_max - d_min + 1e-8) - 1
        signal = -scaled

        return -signal.fillna(0.0)
    @staticmethod
    def alpha_full_factor_072(df: pd.DataFrame, window=50, factor=12):
       
        returns = df["close"].pct_change()
        volume = df["matchingVolume"]
        
        vol_kurt = volume.rolling(window).kurt()
        
        pv_corr = df["close"].rolling(int(factor)).corr(volume)
        sig_corr_rev = -1 * pv_corr.fillna(0)
        
        sig_ret_rev = -1 * (O.ts_rank_normalized(returns, window=window) * 2 - 1)
        is_vol_shock = vol_kurt > 3.0
        raw_signal = np.where(is_vol_shock, sig_corr_rev, sig_ret_rev)
        
        alpha_val = pd.Series(raw_signal, index=df.index)
        
        return -alpha_val.fillna(0)
    
    @staticmethod
    def alpha_full_factor_094(df: pd.DataFrame, window=12, window_corr_vwap=60, window_corr_volume=24):
        returns = df["close"] / df["close"].shift(1) - 1
        
        std_12 = returns.rolling(window).std()
        mean_std_60 = std_12.rolling(window_corr_vwap).mean()
        
        kurt_24 = returns.rolling(window_corr_volume).kurt()

        is_active_but_stable = (std_12 > mean_std_60) & (kurt_24 < 2)

        delta_val = df["close"].diff(3)
        signal_active = -1 * (O.ts_rank_normalized(delta_val, window=24) * 2 - 1)
        signal_normal = -1 * (O.ts_rank_normalized(returns, window=24) * 2 - 1)

        alpha_val = np.where(is_active_but_stable, signal_active, signal_normal)

        return pd.Series(-alpha_val, index=df.index)
    # @staticmethod
    # def alpha_full_factor_080(df: pd.DataFrame, window=24, factor=12):
    #     def get_reg_metrics(series):
    #         y = series.values
    #         x = np.arange(len(y))
    #         if np.any(np.isnan(y)): return 0.0, 0.0, 0.0
    #         slope, intercept = np.polyfit(x, y, 1)
    #         y_pred = slope * x + intercept
    #         ss_res = np.sum((y - y_pred)**2)
    #         ss_tot = np.sum((y - np.mean(y))**2)
    #         r2 = 1 - (ss_res / (ss_tot + 1e-6))
            
    #         resi = y[-1] - y_pred[-1]
    #         return r2, slope, resi

    #     stats = df["close"].rolling(window).apply(lambda x: get_reg_metrics(x)[0], raw=False) # R2
    #     slope_val = df["close"].rolling(window).apply(lambda x: get_reg_metrics(x)[1], raw=False) # Slope 24
        
    #     resi_val = df["close"].rolling(factor).apply(lambda x: get_reg_metrics(x)[2], raw=False)

    #     sig_slope = -1 * (O.ts_rank_normalized(slope_val, window=window) * 2 - 1)

    #     sig_resi = -1 * (O.ts_rank_normalized(resi_val, window=window) * 2 - 1)

    #     alpha_val = np.where(stats > 0.7, sig_slope, sig_resi)

    #     return pd.Series(-alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_080(df, window=24, factor=12):

        def fast_regression(series: pd.Series, window: int):

            y = series.values.astype(float)
            n = len(y)

            x = np.arange(window)
            sum_x = x.sum()
            sum_x2 = (x**2).sum()

            denom = window * sum_x2 - sum_x**2

            y_series = pd.Series(y)

            sum_y = y_series.rolling(window).sum()
            sum_y2 = (y_series**2).rolling(window).sum()

            xy = y_series.rolling(window).apply(
                lambda v: np.dot(v, x), raw=True
            )

            slope = (window * xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / window

            y_last = y_series
            y_pred_last = slope * (window - 1) + intercept

            resi = y_last - y_pred_last

            mean_y = sum_y / window
            ss_tot = sum_y2 - window * mean_y**2

            ss_res = (
                y_series.rolling(window)
                .apply(lambda v: np.sum((v - (slope.loc[v.index] * x + intercept.loc[v.index]))**2), raw=False)
            )

            r2 = 1 - ss_res / (ss_tot + 1e-12)

            return r2, slope, resi
        r2, slope_val, _ = fast_regression(df["close"], window)

        _, _, resi_val = fast_regression(df["close"], factor)

        sig_slope = -1 * (O.ts_rank_normalized(slope_val, window=window) * 2 - 1)

        sig_resi = -1 * (O.ts_rank_normalized(resi_val, window=window) * 2 - 1)

        alpha_val = np.where(r2 > 0.7, sig_slope, sig_resi)

        return pd.Series(-alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_090(df: pd.DataFrame, window=24, factor=12):
        factor = int(factor)
        returns = df["close"].pct_change()

        def get_regression_stats(series):
            y = series.values
            x = np.arange(len(y))
            if np.any(np.isnan(y)): return np.nan, np.nan
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-6))
            return r2, slope

        reg_stats = df["close"].rolling(window).apply(
            lambda x: get_regression_stats(x)[0], raw=False
        )
        
        slope_12 = df["close"].rolling(factor).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0], raw=False
        )
        signal_trend = -1 * (O.ts_rank_normalized(slope_12, window=window) * 2 - 1)

        signal_noise = -1 * (O.ts_rank_normalized(returns, window=factor) * 2 - 1)

        alpha_val = np.where(reg_stats > 0.75, signal_trend, signal_noise)

        return pd.Series(-alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_092(df: pd.DataFrame, window=6, factor=48):
        returns = df["close"] / df["close"].shift(1) - 1
        
        amt = df.get("amt", df["close"] * df["matchingVolume"])
        
        efficiency = returns / (amt + 1e-6)
        
        smoothed_eff = efficiency.ewm(span=window, adjust=False).mean()
       
        rank_val = O.ts_rank_normalized(smoothed_eff, window=factor)
        
        alpha_val = (rank_val * 2 - 1)

        return pd.Series(alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_093(df: pd.DataFrame, window=24, factor=3):
        factor = int(factor)
        returns = df["close"] / df["close"].shift(1) - 1
        amt = df.get("amt", df["close"] * df["matchingVolume"])
        efficiency = returns / (amt + 1e-6)
        
        eff_velocity = efficiency.diff(factor)
        
        rank_val = O.ts_rank_normalized(eff_velocity, window=window)
        
        alpha_val = (rank_val * 2 - 1)

        return pd.Series(alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_095(df: pd.DataFrame, window=48):
        returns = df["close"] / df["close"].shift(1) - 1
        
        skew_val = returns.rolling(window).skew()
        kurt_val = returns.rolling(window).kurt()

        is_extreme_regime = (np.abs(skew_val) > 1.5) | (kurt_val > 4.0)

        def get_residual(series):
            y = series.values
            x = np.arange(len(y))
            if np.any(np.isnan(y)): return np.nan
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * (len(y) - 1) + intercept
            return y[-1] - y_pred

        resi_val = df["close"].rolling(6).apply(get_residual, raw=False)
        signal_extreme = -1 * (O.ts_rank_normalized(resi_val, window=window) * 2 - 1)

        signal_normal = -1 * (O.ts_rank_normalized(returns, window=window) * 2 - 1)

        alpha_val = np.where(is_extreme_regime, signal_extreme, signal_normal)

        return pd.Series(-alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_098(df: pd.DataFrame, window=6, factor=2):
        returns = df["close"] / df["close"].shift(1) - 1
        
        amt = df.get("amt", df["close"] * df["matchingVolume"])
        
        efficiency = returns / (amt + 1e-6)
        
        smoothed_eff = O.ts_weighted_mean(efficiency, window=window)
        
        velocity_eff = O.ts_delta(smoothed_eff, period=int(factor))
        
        rank_val = O.ts_rank_normalized(velocity_eff, window=24)
        
        alpha_val = (rank_val * 2 - 1)

        return pd.Series(alpha_val, index=df.index)
    @staticmethod
    def alpha_full_factor_099(df: pd.DataFrame, fast=6, slow=24):
        returns = df["close"] / df["close"].shift(1) - 1
        
        amt = df.get("amt", df["close"] * df["matchingVolume"])
        
        efficiency = returns / (amt + 1e-6)
        
        ema_fast = efficiency.ewm(span=fast, adjust=False).mean()
        ema_slow = efficiency.ewm(span=slow, adjust=False).mean()
        
        raw_diff = ema_fast - ema_slow
        
        normalized_signal = O.ts_rank_normalized(raw_diff, window=24)
        normalized_signal = (normalized_signal * 2) - 1

        return pd.Series(normalized_signal, index=df.index)
    @staticmethod
    def alpha_full_factor_100(df: pd.DataFrame, window=24, factor=12):
        factor = int(factor)
        def get_residual(series):
            y = series.values
            x = np.arange(len(y))
            if np.any(np.isnan(y)): return np.nan
            # Hồi quy tuyến tính: y = ax + b
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * (len(y) - 1) + intercept
            return y[-1] - y_pred

        resi_1 = df["close"].rolling(window).apply(get_residual, raw=False)

        delta_resi = resi_1.diff(3)

        resi_2 = delta_resi.rolling(factor).apply(get_residual, raw=False)

        final_alpha = O.ts_rank_normalized(resi_2, window=factor)
        final_alpha = (final_alpha * 2) - 1

        return final_alpha
    @staticmethod
    def alpha_full_factor_101(df: pd.DataFrame, window=24, factor=12):
        factor = int(factor)
        returns = df["close"] / df["close"].shift(1) - 1
        skew_val = returns.rolling(window).skew()

        med_ret = returns.rolling(window).median()
        std_ret = returns.rolling(window).std()
        
        robust_zscore = (returns - med_ret) / (std_ret + 1e-6)
        signal_extreme = O.ts_rank_normalized(robust_zscore, window=factor)

        def get_residual(series):
            y = series.values
            x = np.arange(len(y))
            if np.any(np.isnan(y)): return np.nan
            # Hồi quy tuyến tính nhanh
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * (len(y) - 1) + intercept
            return y[-1] - y_pred

        close_resi = df["close"].rolling(factor).apply(get_residual, raw=False)
        signal_stable = O.ts_rank_normalized(close_resi, window=factor)
        raw_signal = np.where(np.abs(skew_val) > 1.0, signal_extreme, signal_stable)
        
        normalized_signal = (raw_signal * 2) - 1

        return pd.Series(normalized_signal, index=df.index)

    @staticmethod
    def alpha_full_factor_105(df: pd.DataFrame, window=50):
        returns = df["close"] / df["close"].shift(1) - 1
        skew_val = returns.rolling(window).skew()

        def get_residual(series):
            y = series.values
            x = np.arange(len(y))
            if np.any(np.isnan(y)): return np.nan
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * (len(y) - 1) + intercept
            return y[-1] - y_pred

        open_resi = df["open"].rolling(window).apply(get_residual, raw=False)
        sign_resi = np.sign(open_resi)

        signed_power_ret = np.sign(returns) * np.power(np.abs(returns), 0.6)
        signal_skew_high = O.ts_rank_normalized(sign_resi, window)
        signal_normal = O.ts_rank_normalized(signed_power_ret, window)
        raw_signal = np.where(skew_val > 0.4, signal_skew_high, signal_normal)
        
        normalized_signal = ((raw_signal * 2) - 1)

        return pd.Series(normalized_signal, index=df.index)


    @staticmethod
    def alpha_full_factor_062_zscore_clipping(df: pd.DataFrame, window=70):
        # 1. Dùng Log Return thay vì Pct Change để giảm nhiễu cực đoan
        log_ret = np.log(df["close"] / df["close"].shift(1))
        
        # 2. Thay vì chia cho amt_std, ta chia cho Volatility của chính nó
        # Đây là logic của Sharpe Ratio: Lợi nhuận / Rủi ro
        rolling_vol = log_ret.rolling(window).std() + 1e-6
        risk_adj_ret = log_ret / rolling_vol
        
        # 3. Làm mượt Amount bằng Decay Linear trước khi tính biến động
        amt = df["close"] * df["matchingVolume"]
        smooth_amt_std = O.decay_linear(amt, window).rolling(window).std()
        
        # 4. Kết hợp: Momentum ổn định / Biến động thanh khoản thấp
        raw_signal = risk_adj_ret / (np.log1p(smooth_amt_std) + 1e-6)
        
        # 5. Quan trọng: Dùng Z-score và Clip để loại bỏ Outliers (kẻ thù của Sharpe)
        z_signal = O.zscore(raw_signal, window=window).clip(-3, 3)
        
        # 6. Ép về [-1, 1]
        return np.tanh(z_signal)
    @staticmethod
    def alpha_full_factor_095_regime_adaptive(df: pd.DataFrame, window=48):
        returns = df["close"].pct_change()
        
        skew_val = returns.rolling(window).skew()
        kurt_val = returns.rolling(window).kurt()

        is_extreme_regime = (np.abs(skew_val) > 1.5) | (kurt_val > 4.0)

        signal_normal = O.zscore(returns, window)

        short_trend = O.ts_mean(df["close"], window=6)
        residuals = (df["close"] - short_trend) / short_trend
        signal_extreme = O.zscore(residuals, window)

        raw_alpha = np.where(is_extreme_regime, signal_extreme, signal_normal)

        alpha_val = np.tanh(raw_alpha / 2.0) 

        return pd.Series(alpha_val, index=df.index).fillna(0)

    @staticmethod
    def alpha_full_factor_066_liq_accel(df: pd.DataFrame, window=48, factor=3):
        factor = int(factor)
        volume = df["matchingVolume"]
        amt = df["close"] * volume
        log_amt = np.log1p(amt)
        returns = df["close"].pct_change()
        vol_kurt = volume.rolling(window).kurt()
        is_vol_shock = vol_kurt > 3.0
        amt_delta = log_amt.diff(factor)
        sig_amt_accel = -1 * O.zscore(amt_delta, window=window)
        
        sig_ret_rev = -1 * np.tanh(O.zscore(returns, window=window) / 2.0)
        
        raw_signal = np.where(is_vol_shock, sig_amt_accel, sig_ret_rev)
        
        alpha_val = -1 * np.tanh(pd.Series(raw_signal, index=df.index).fillna(0))
        
        return alpha_val
    
    @staticmethod
    def alpha_full_factor_090_reg_adaptive(df: pd.DataFrame, window=24, factor=12):
        factor = int(factor)
        returns = df["close"].pct_change()
        log_price = np.log(df["close"])
        
        rolling_time = pd.Series(np.arange(len(df)), index=df.index)
        r_val = log_price.rolling(window).corr(rolling_time)
        r2 = r_val ** 2

        std_y = log_price.rolling(factor).std()
        std_x = np.std(np.arange(factor))
        slope = r_val * (std_y / std_x)

        signal_trend = -1 * np.tanh(O.zscore(slope, window) / 2.0)
        
        signal_noise = -1 * np.tanh(O.zscore(returns, factor) / 2.0)

        raw_alpha = np.where(r2 > 0.75, signal_trend, signal_noise)

        alpha_val = -1 * pd.Series(raw_alpha, index=df.index).fillna(0)

        return alpha_val

    @staticmethod
    def alpha_full_factor_099_eff_macd(df: pd.DataFrame, window=6, factor=12, window_norm=15):
        returns = np.log(df["close"] / df["close"].shift(1))
        
        amt = df.get("amt", df["close"] * df["matchingVolume"])
        log_amt = np.log1p(amt)
        
        efficiency = returns / (log_amt + 1e-6)
        
        ema_f = efficiency.ewm(span=window, adjust=False).mean()
        ema_s = efficiency.ewm(span=factor, adjust=False).mean()
        
        raw_diff = ema_f - ema_s
        
        roll_min = raw_diff.rolling(window_norm).min()
        roll_max = raw_diff.rolling(window_norm).max()
        
        norm_diff = 2 * (raw_diff - roll_min) / (roll_max - roll_min + 1e-8) - 1
        
        return pd.Series(norm_diff, index=df.index).fillna(0)


    @staticmethod
    def alpha_full_factor_046_dynamic_reversion(df: pd.DataFrame, window=50):
        
        candle_range = df['high'] - df['low']
        avg_range = candle_range.rolling(window).mean() + 1e-8
        vol_impact = candle_range / avg_range
        
        price_delta = df['close'].diff(3)
        delta_mean = price_delta.rolling(window).mean()
        delta_std = price_delta.rolling(window).std() + 1e-8
        price_z = (price_delta - delta_mean) / delta_std
        
       
        close_pos = (df['close'] - df['low']) / (candle_range + 1e-8) 
        rejection_force = np.where(price_z > 0, close_pos, 1 - close_pos)
        
        raw_signal = price_z * vol_impact * rejection_force
        
        signal = -np.tanh(raw_signal * 0.5)
        
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_full_factor_007_volume_breakout_trend(df: pd.DataFrame, window=12):
        # Volume Breakout Trend
        returns = df['close'] / df['close'].shift(1) - 1
        ema_vol = df['matchingVolume'].ewm(span=window, adjust=False).mean()
        vol_breakout = df['matchingVolume'] / (ema_vol + 1e-8)
        
        # strong volume -> trend follow
        signal = returns * vol_breakout
        
        # Z-score normalization to [-1, 1]
        z_score = (signal - signal.rolling(window).mean()) / (signal.rolling(window).std() + 1e-8)
        return np.tanh(z_score).fillna(0)
    
    @staticmethod
    def alpha_full_factor_085_rank_vol_efficiency(df: pd.DataFrame, window=12):
        # Time-series rank normalized Amount Efficiency
        returns = df['close'] / df['close'].shift(1) - 1
        amt = df['close'] * df['matchingVolume']
        
        efficiency = returns / (amt.rolling(window).mean() + 1e-8)
        
        signal = O.ts_rank_normalized(efficiency, window=window) * 2 - 1

        return signal.fillna(0)

    @staticmethod
    def alpha_full_factor_b08_signed_power_compress(df: pd.DataFrame, window=24):
        # BREAKTHROUGH: SignedPower Compressed Efficiency
        # SignedPower(Returns / Std(Amount), 0.4) — nén cực trị, khuếch đại tín hiệu nhỏ
        returns = df['close'].pct_change()
        amt = df['matchingVolume'] * df['close']
        amt_std = amt.rolling(window).std()
        eff = returns / (amt_std + 1e-6)
        compressed = np.sign(eff) * np.abs(eff) ** 0.4
        sig = -O.ts_rank_normalized(compressed, window*4)
        return -np.tanh(O.zscore(sig, window=window*4).fillna(0) / 2.0)
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

    @staticmethod
    def alpha_popbo_001(df: pd.DataFrame, window=10):
        """ (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6)) """
        # volume = df.get('matchingVolume', df.get('volume', df['close']*0 + 1))
        volume = df['matchingVolume']
        
        # 1. Delta Log Volume
        log_vol = np.log1p(volume)
        delta_log_vol = log_vol.diff(1)
        rank_vol = delta_log_vol.rolling(window).rank(pct=True)
        
        # 2. Price change percentage
        price_diff = (df['close'] - df['open']) / (df['open'] + 1e-8)
        rank_price = price_diff.rolling(window).rank(pct=True)
        
        # 3. Correlation between ranked volume and ranked price
        correlation = rank_vol.rolling(5).corr(rank_price).fillna(0)
        
        # 4. Final signal 
        signal = correlation
        
        return signal


    @staticmethod
    def alpha_popbo_002(df: pd.DataFrame, window=50):
        """ -1 * delta((((close-low)-(high-close))/(high-low)), 1) """
        # 1. Calculate price location within high-low range
        range_diff = df['high'] - df['low'] + 1e-8
        location = ((df['close'] - df['low']) - (df['high'] - df['close'])) / range_diff
        # 2. Delta of location
        delta_location = location.diff(1)
        # 3. Final raw signal
        raw_signal = -1 * delta_location
        # 4. Normalize to [-1, 1] using ts_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_003(df: pd.DataFrame, window = 1 ,factor=20):
        """ SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1))
        :MAX(HIGH,DELAY(CLOSE,1)))),6) """
        factor = int(factor)
        prev_close = df['close'].shift(1)
        
        # 1. Determine baseline
        min_low_prev = np.minimum(df['low'], prev_close)
        max_high_prev = np.maximum(df['high'], prev_close)
        
        # 2. Conditional differences
        cond_up = df['close'] > prev_close
        cond_down = df['close'] < prev_close
        
        part = pd.Series(0.0, index=df.index)
        part[cond_up] = df['close'] - min_low_prev
        part[cond_down] = df['close'] - max_high_prev
        
        # 3. Rolling Sum
        sum_part = part.rolling(window).sum()
        
        # 4. Normalize
        ranked_signal = sum_part.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_004(df: pd.DataFrame):
        """ ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : ... """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Moving averages and std
        sma8 = df['close'].rolling(8).mean()
        std8 = df['close'].rolling(8).std()
        sma2 = df['close'].rolling(4).mean()
        
        # 2. Volume criteria
        mean_vol20 = volume.rolling(10).mean()
        vol_ratio = volume / (mean_vol20 + 1e-8)
        
        # 3. Conditions
        cond1 = (sma8 + std8) < sma2
        cond2 = (sma8 + std8) > sma2
        cond3 = (sma8 + std8) == sma2
        cond4 = vol_ratio >= 1
        
        # 4. Combine into signal
        signal = pd.Series(0.0, index=df.index)
        signal[cond1] = -1
        signal[cond2] = 1
        signal[cond3] = -1
        signal[cond3 & cond4] = 1

        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_005(df: pd.DataFrame):
        """ -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Time-series rank over 5 days
        tsrank_vol = volume.rolling(5).rank(pct=True)
        tsrank_high = df['high'].rolling(5).rank(pct=True)
        
        # 2. Rolling correlation over 5 days
        corr_5 = tsrank_vol.rolling(5).corr(tsrank_high).fillna(0)
        
        # 3. Rolling Max over 3 days
        max_corr = corr_5.rolling(3).max()
        
        # 4. Final signal (already bounded because corr is [-1, 1])
        signal = -1 * max_corr
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_006(df: pd.DataFrame, window=24):
        """ (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1) """
        # 1. Weighted price
        w_price = (df['open'] * 0.85) + (df['high'] * 0.15)
        
        # 2. Delta
        delta_price = w_price.diff(4)
        
        # 3. Sign
        sign_price = np.sign(delta_price)
        
        # 4. Cross-sectional RANK converted to TS RANK over 'window' days
        ranked_sign = sign_price.rolling(window).rank(pct=True)
        
        # 5. Normalize and flip
        signal = -1 * ((2 * ranked_sign) - 1)
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_007(df: pd.DataFrame, window=30):
        """ ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. VWAP - Close diff
        diff = vwap - df['close']
        max_diff = diff.rolling(3).max()
        min_diff = diff.rolling(3).min()
        
        # 2. Volume Delta
        delta_vol = volume.diff(3)
        
        # 3. Time-Series Ranks instead of Cross-sectional
        rank_max = max_diff.rolling(window).rank(pct=True)
        rank_min = min_diff.rolling(window).rank(pct=True)
        rank_vol = delta_vol.rolling(window).rank(pct=True)
        
        # 4. Combine
        raw_signal = (rank_max + rank_min) * rank_vol
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_008(df: pd.DataFrame,window=1 ,factor=50):
        factor = int(factor)
        """ RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Weighted Price
        w_price = (((df['high'] + df['low']) / 2) * 0.2) + (vwap * 0.8)
        
        # 2. Delta
        delta_p = w_price.diff(window) * -1
        
        # 3. TS Rank (normalize to [-1, 1])
        ranked_delta = delta_p.rolling(factor).rank(pct=True)
        signal = (2 * ranked_delta) - 1
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_009(df: pd.DataFrame,window=3, factor=20):
        factor = int(factor)
        """ SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2) """
        # window 4,5,6,7
        # window_rank 10,100,10
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Current and Previous Midprice
        mid_price = (df['high'] + df['low']) / 2
        prev_mid_price = (df['high'].shift(1) + df['low'].shift(1)) / 2
        
        # 2. Price Change
        change = mid_price - prev_mid_price
        
        # 3. Range to Volume ratio
        range_vol = (df['high'] - df['low']) / (volume + 1e-8)
        
        # 4. Raw metric
        raw_metric = change * range_vol
        
        # 5. SMA (EWM with alpha=2/7)
        sma_val = raw_metric.ewm(alpha=window/7, adjust=False).mean()
        
        # 6. Normalize
        ranked_signal = sma_val.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_010(df: pd.DataFrame, window=40):
        """ (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5)) """
        ret = df['close'].pct_change()
        
        # 1. Rolling STD
        std_ret = ret.rolling(20).std()
        
        # 2. Condition
        cond = ret < 0
        part = pd.Series(0.0, index=df.index)
        part[cond] = std_ret
        part[~cond] = df['close']
        
        # 3. Square and Max
        part_sq = part ** 2
        max_part = part_sq.rolling(5).max()
        
        # 4. Normalize
        ranked_signal = max_part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_011(df: pd.DataFrame, window=60): 
        """ SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Price Location Value
        range_diff = df['high'] - df['low'] + 1e-8
        location = ((df['close'] - df['low']) - (df['high'] - df['close'])) / range_diff
        
        # 2. Volume Weighted Location
        vol_weighted = location * volume
        
        # 3. Sum over 6 days
        sum_val = vol_weighted.rolling(2).sum()
        
        # 4. Normalize
        ranked_signal = sum_val.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_012(df: pd.DataFrame, window=10):
        """ (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP))))) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        # 1. Open minus SMA(VWAP, 10)
        vwap_sma = vwap.rolling(5).mean()
        open_diff = df['open'] - vwap_sma
        rank_open = open_diff.rolling(window).rank(pct=True)
        
        # 2. Abs Close minus VWAP
        abs_diff = (df['close'] - vwap).abs()
        rank_abs = abs_diff.rolling(window).rank(pct=True)
        
        # 3. Combine
        raw_signal = rank_open * (-1 * rank_abs)
        
        # 4. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_013(df: pd.DataFrame, window=15):
        """ (((HIGH * LOW)^0.5) - VWAP) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Geometric Mean of High and Low
        geom_mean = np.sqrt(df['high'] * df['low'])
        
        # 2. Difference from VWAP
        diff = geom_mean - vwap
        
        # 3. Normalize
        ranked_signal = diff.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_014(df: pd.DataFrame, window=100):
        """ CLOSE-DELAY(CLOSE,5) """
        # 1. 5-day Price Change
        change = df['close'] - df['close'].shift(2)
        
        # 2. Normalize
        ranked_signal = change.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_015(df: pd.DataFrame, window=10):
        """ OPEN/DELAY(CLOSE,1)-1 """
        # 1. Overnight Return
        overnight_ret = (df['open'] / df['close'].shift(1)) - 1
        
        # 2. Normalize
        ranked_signal = overnight_ret.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_016(df: pd.DataFrame, window=20):
        """ (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. TS Rank
        rank_vol = volume.rolling(window).rank(pct=True)
        rank_vwap = vwap.rolling(window).rank(pct=True)
        
        # 2. Correlation
        corr_val = rank_vol.rolling(5).corr(rank_vwap).fillna(0)
        
        # 3. TS Rank of Correlation
        rank_corr = corr_val.rolling(window).rank(pct=True)
        
        # 4. TS MAX
        max_val = rank_corr.rolling(5).max()
        
        # 5. Signal
        signal = -1 * max_val
       
        return signal.fillna(0)
        
    @staticmethod
    def alpha_popbo_017(df: pd.DataFrame,window=45, factor=50):
        factor = int(factor)
        # window 35,30,40,45
        # window_rank 10,100,10
        """ RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5) """
        # vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        vwap = (df['high']+df['low']+df['close'])/3
        
        # 1. VWAP minus TSMAX(VWAP, 15)
        max_vwap = vwap.rolling(window).max()
        diff = vwap - max_vwap
        
        # 2. Rank of Diff
        rank_diff = diff.rolling(factor).rank(pct=True)
        
        # 3. Delta Close
        delta_close = df['close'].diff(1)
        
        # 4. Power
        raw_signal = rank_diff ** delta_close
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_018(df: pd.DataFrame,window=1, factor=20):
        factor = int(factor)
        """ CLOSE/DELAY(CLOSE,5) """
        # 1. 5-day Return
        ret5 = df['close'] / df['close'].shift(window)
        
        # 2. Normalize
        ranked_signal = ret5.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_019(df: pd.DataFrame, window=100):
        """ (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)
        :(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE)) """
        delay_close = df['close'].shift(20)
        
        # 1. Conditions
        cond_less = df['close'] < delay_close
        cond_eq = df['close'] == delay_close
        cond_greater = df['close'] > delay_close
        
        # 2. Assign Values
        part = pd.Series(0.0, index=df.index)
        part[cond_less] = (df['close'] - delay_close) / delay_close
        part[cond_eq] = 0
        part[cond_greater] = (df['close'] - delay_close) / df['close']
        
        # 3. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
       
    @staticmethod
    def alpha_popbo_020(df: pd.DataFrame, window=70):
        """ (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100 """
        # 1. 6-day Percentage Return
        ret6 = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
        
        # 2. Normalize
        ranked_signal = ret6.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_021(df: pd.DataFrame, window=20):
        """ REGBETA(MEAN(CLOSE,6),SEQUENCE(6)) """
        # 1. 6-day SMA of close
        mean_close = df['close'].rolling(6).mean()
        
        # 2. Linear regression slope against sequence [1..6]
        seq = np.arange(1, 7)
        def get_slope(y):
            if len(y) < 6 or np.any(np.isnan(y)): return np.nan
            return np.polyfit(seq, y, 1)[0]
            
        slope = mean_close.rolling(6).apply(get_slope, raw=True)
        
        # 3. TS Rank normalization
        ranked_signal = slope.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_022(df: pd.DataFrame,window=5, factor=10):
        factor = int(factor)
        """ SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1) """
        # 1. Mean(Close, 6)
        mean_6 = df['close'].rolling(window).mean()
        
        # 2. Percentage deviation from 6-day mean
        dev = (df['close'] - mean_6) / mean_6
        
        # 3. 3-day change of deviation
        change_dev = dev - dev.shift(1)
        
        # 4. SMA of change (EWM with alpha=1/12)
        sma_val = change_dev.ewm(alpha=1/12, adjust=False).mean()
        
        # 5. TS Rank normalization
        ranked_signal = sma_val.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_023(df: pd.DataFrame,window=15, factor=10):
        factor = int(factor)
        """ SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / ... * 100 """
        # window 3,6,9,12,15
        # window_rank 5,40,5
        prev_close = df['close'].shift(1)
        cond_up = df['close'] > prev_close
        
        std_20 = df['close'].rolling(5).std()
        
        # 1. Up std part
        part_up = pd.Series(0.0, index=df.index)
        part_up[cond_up] = std_20
        
        # 2. Down std part
        part_down = pd.Series(0.0, index=df.index)
        part_down[~cond_up] = std_20
        part_down[cond_up] = 0
        
        # 3. SMA parts (alpha = 1/20)
        sma_up = part_up.ewm(alpha=window/20, adjust=False).mean()
        sma_down = part_down.ewm(alpha=window/20, adjust=False).mean()
        
        # 4. Ratio
        ratio = (sma_up / (sma_up + sma_down + 1e-8)) * 100
        
        # 5. Normalize
        ranked_signal = ratio.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0) 

    @staticmethod
    def alpha_popbo_024(df: pd.DataFrame, window=70):
        """ SMA(CLOSE-DELAY(CLOSE,5),5,1) """
        # 1. 5-day change
        change = df['close'] - df['close'].shift(10)
        
        # 2. SMA (alpha=1/5)
        sma_val = change.ewm(alpha=1/5, adjust=False).mean()
        
        # 3. TS Rank normalization
        ranked_signal = sma_val.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_025(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        ret = df['close'].pct_change()
        
        # 1. Relative Volume
        vol_ratio = volume / (volume.rolling(20).mean() + 1e-8)
        
        # 2. Decaylinear of vol_ratio (span=9 approximation)
        decay_vol = vol_ratio.ewm(span=9).mean()
        rank_decay_vol = decay_vol.rolling(window).rank(pct=True)
        
        # 3. Left term
        delta_close_7 = df['close'].diff(7)
        left_inner = delta_close_7 * (1 - rank_decay_vol)
        rank_left = left_inner.rolling(window).rank(pct=True)
        
        # 4. Right term
        sum_ret_250 = ret.rolling(250).sum()
        rank_right = sum_ret_250.rolling(window).rank(pct=True)
        
        # 5. Combine
        raw_signal = -1 * rank_left * (1 + rank_right)
        
        # 6. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_026(df: pd.DataFrame, window=32):
        """ ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230)))) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left part: mean(close, 7) - close
        mean_7 = df['close'].rolling(5).mean()
        left = mean_7 - df['close']
        
        # 2. Right part: corr(vwap, delay(close, 5), 230)
        delay_close_5 = df['close'].shift(5)
        right = vwap.rolling(230).corr(delay_close_5).fillna(0)
        
        # 3. Raw sum
        raw_signal = left + right
        
        # 4. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
       
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_027(df: pd.DataFrame, window=60):
        """ WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12) """
        # 1. 3-day and 6-day returns
        ret_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
        ret_6 = (df['close'] - df['close'].shift(6)) / df['close'].shift(6) * 100
        
        # 2. Sum of returns
        combined_ret = ret_3 + ret_6
        
        # 3. WMA (decay=0.9, window=12) -> approximation with EWM
        wma_val = combined_ret.ewm(halflife=7).mean()
        
        # 4. Normalize
        ranked_signal = wma_val.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_028(df: pd.DataFrame,window=5, factor=20):
        factor = int(factor)
        """ 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA(...)) """
        #window 5,10,15,20
        #window_rank 10,100,10
        # 1. Stochastic value
        min_low_9 = df['low'].rolling(window).min()
        max_high_9 = df['high'].rolling(window).max()
        stoch = (df['close'] - min_low_9) / (max_high_9 - min_low_9 + 1e-8) * 100
        
        # 2. SMA(stoch, 3) -> alpha=1/3
        sma_stoch = stoch.ewm(alpha=2/3, adjust=False).mean()
        
        # 3. Double SMA
        sma_sma_stoch = sma_stoch.ewm(alpha=2/3, adjust=False).mean()
        
        # 4. KDJ-like composite
        raw_signal = (3 * sma_stoch) - (2 * sma_sma_stoch)
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_029(df: pd.DataFrame, window=1, factor=20):
        factor = int(factor)
        # window 1,2,3,4
        # window_rank = 10,100,10
        """ (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. 6-day return
        ret_6 = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        # 2. Volume-weighted return
        vol_ret = ret_6 * volume
        
        # 3. Normalize
        ranked_signal = vol_ret.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_031(df: pd.DataFrame,window=8, factor=20): 
        factor = int(factor)
        # window 2,4,6,8
        # window_rank = 10,100,10
        """ (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100 """
        # 1. 12-day SMA
        mean_12 = df['close'].rolling(window).mean()
        
        # 2. Percentage deviation
        raw_signal = (df['close'] - mean_12) / mean_12 * 100
        
        # 3. TS Rank normalization
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_032(df: pd.DataFrame, window=10):
        """ (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. TS Rank of High and Volume
        rank_high = df['high'].rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        # 2. 3-day Correlation
        corr_3 = rank_high.rolling(3).corr(rank_vol).fillna(0)
        
        # 3. TS Rank of Correlation (instead of cross-sectional)
        rank_corr = corr_3.rolling(window).rank(pct=True)
        
        # 4. 3-day Sum
        sum_3 = rank_corr.rolling(3).sum()
        
        # 5. Combine & Normalize
        raw_signal = -1 * sum_3
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_033(df: pd.DataFrame, window=60):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        ret = df['close'].pct_change()
        
        # 1. TSMIN(Low, 5) diff
        min_low_5 = df['low'].rolling(5).min()
        term1 = (-1 * min_low_5) + min_low_5.shift(5)
        
        # 2. Return diff rank
        sum_240 = ret.rolling(240).sum()
        sum_20 = ret.rolling(20).sum()
        ret_metric = (sum_240 - sum_20) / 220
        rank_ret_metric = ret_metric.rolling(window).rank(pct=True)
        
        # 3. TS Rank Volume
        tsrank_vol = volume.rolling(5).rank(pct=True)
        
        # 4. Combine
        raw_signal = term1 * rank_ret_metric * tsrank_vol
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_034(df: pd.DataFrame,window =4, factor=20):
        factor = int(factor)
        # window 3,4,5,6,
        # window_rank = 10,100,10
        """ MEAN(CLOSE,12)/CLOSE """
        # 1. 12-day SMA
        mean_12 = df['close'].rolling(window).mean()
        
        # 2. Ratio
        ratio = mean_12 / df['close']
        
        # 3. TS Rank normalization
        ranked_signal = ratio.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_035(df: pd.DataFrame, window=20):
        """ (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. First term
        delta_open = df['open'].diff(1)
        decay_delta = delta_open.ewm(span=15).mean()
        rank_decay_delta = decay_delta.rolling(window).rank(pct=True)
        
        # 2. Second term
        open_comb = (df['open'] * 0.65) + (df['open'] * 0.35)
        corr_17 = volume.rolling(17).corr(open_comb).fillna(0)
        decay_corr = corr_17.ewm(span=7).mean()
        rank_decay_corr = decay_corr.rolling(window).rank(pct=True)
        
        # 3. Min
        min_rank = np.minimum(rank_decay_delta, rank_decay_corr)
        
        # 4. Combine & Normalize (already essentially bounded, but we can standard normalise)
        raw_signal = -1 * min_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_036(df: pd.DataFrame, window=40):
        """ RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. TS Rank
        rank_vol = volume.rolling(window).rank(pct=True)
        rank_vwap = vwap.rolling(window).rank(pct=True)
        
        # 2. Correlation
        corr_6 = rank_vol.rolling(6).corr(rank_vwap).fillna(0)
        
        # 3. Sum over 2 days
        sum_2 = corr_6.rolling(2).sum()
        
        # 4. TS Rank normalize
        ranked_signal = sum_2.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_037(df: pd.DataFrame, window=15):
        """ (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10)))) """
        ret = df['close'].pct_change()
        
        # 1. Calculate internal metric
        sum_open_5 = df['open'].rolling(5).sum()
        sum_ret_5 = ret.rolling(5).sum()
        metric = sum_open_5 * sum_ret_5
        
        # 2. Difference against 10-day delay
        diff = metric - metric.shift(10)
        
        # 3. Combine & Normalize
        raw_signal = -1 * diff
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_038(df: pd.DataFrame, window=24):
        """ (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0) """
        # 1. Check Condition
        sma_high_20 = df['high'].rolling(20).mean()
        cond = sma_high_20 < df['high']
        
        # 2. Calculate values
        part = pd.Series(0.0, index=df.index)
        part[cond] = -1 * df['high'].diff(2)
        
        # 3. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_039(df: pd.DataFrame, window=35):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left term
        delta_close_2 = df['close'].diff(2)
        decay_delta = delta_close_2.ewm(span=8).mean()
        rank_left = decay_delta.rolling(window).rank(pct=True)
        
        # 2. Right term components
        w_price = (vwap * 0.3) + (df['open'] * 0.7)
        mean_vol_180 = volume.rolling(180).mean()
        # Note: SUM(MEAN, 37) is equivalent to moving sum of moving average
        sum_mean_vol = mean_vol_180.rolling(37).sum()
        
        # 3. Correlation and Decay
        corr_14 = w_price.rolling(14).corr(sum_mean_vol).fillna(0)
        decay_corr = corr_14.ewm(span=12).mean()
        rank_right = decay_corr.rolling(window).rank(pct=True)
        
        # 4. Combine & Normalize
        raw_signal = -1 * (rank_left - rank_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_040(df: pd.DataFrame, window=50):
        """ SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100 """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Conditions
        prev_close = df['close'].shift(1)
        cond_up = df['close'] > prev_close
        
        # 2. Up Volume
        vol_up = pd.Series(0.0, index=df.index)
        vol_up[cond_up] = volume
        
        # 3. Down Volume
        vol_down = pd.Series(0.0, index=df.index)
        vol_down[~cond_up] = volume
        
        # 4. Sums and Ratio
        sum_up = vol_up.rolling(26).sum()
        sum_down = vol_down.rolling(26).sum()
        ratio = (sum_up / (sum_down + 1e-8)) * 100
        
        # 5. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_041(df: pd.DataFrame, window=10):
        """ (RANK(MAX(DELTA((VWAP), 3), 5))* -1) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Delta VWAP (3 days)
        delta_vwap = vwap.diff(3)
        
        # 2. Maximum of Delta VWAP (5 days)
        max_delta = delta_vwap.rolling(2).max()
        
        # 3. TS Rank of Max Delta
        rank_max = max_delta.rolling(window).rank(pct=True)
        
        # 4. Final signal and normalization
        raw_signal = -1 * rank_max
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_042(df: pd.DataFrame, window=100):
        """ ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. TS Rank of High STD
        std_high = df['high'].rolling(10).std()
        rank_std = std_high.rolling(window).rank(pct=True)
        
        # 2. Correlation between High and Volume
        corr_10 = df['high'].rolling(10).corr(volume).fillna(0)
        
        # 3. Combine and normalize
        raw_signal = -1 * rank_std * corr_10
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_043(df: pd.DataFrame, window=50):
        """ SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        prev_close = df['close'].shift(1)
        
        # 1. Price conditions
        cond_up = df['close'] > prev_close
        cond_down = df['close'] < prev_close
        
        # 2. Conditional Volume
        part = pd.Series(0.0, index=df.index)
        part[cond_up] = volume
        part[cond_down] = -1 * volume
        
        # 3. Rolling sum
        sum_6 = part.rolling(2).sum()
        
        # 4. Normalize
        ranked_signal = sum_6.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_044(df: pd.DataFrame, window=5):
        """ (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Term 1 (Low and Mean Vol Corr)
        mean_vol = volume.rolling(10).mean()
        corr_7 = df['low'].rolling(7).corr(mean_vol).fillna(0)
        decay_corr = corr_7.ewm(span=6).mean()
        tsrank_1 = decay_corr.rolling(4).rank(pct=True)
        
        # 2. Term 2 (Delta VWAP Decay)
        delta_vwap = vwap.diff(3)
        decay_delta = delta_vwap.ewm(span=10).mean()
        tsrank_2 = decay_delta.rolling(15).rank(pct=True)
        
        # 3. Combine and normalize
        raw_signal = tsrank_1 + tsrank_2
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_045(df: pd.DataFrame, window=5):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        w_price = (df['close'] * 0.6) + (df['open'] * 0.4)
        delta_p = w_price.diff(1)
        rank_left = delta_p.rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol = volume.rolling(100).mean()
        corr_15 = vwap.rolling(15).corr(mean_vol).fillna(0)
        rank_right = corr_15.rolling(window).rank(pct=True)
        
        # 3. Combine and normalize
        raw_signal = rank_left * rank_right
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_046(df: pd.DataFrame, window=60): 
        """ (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE) """
        # 1. Moving Averages
        mean_3 = df['close'].rolling(3).mean()
        mean_6 = df['close'].rolling(6).mean()
        mean_12 = df['close'].rolling(12).mean()
        mean_24 = df['close'].rolling(24).mean()
        
        # 2. Ratio
        ratio = (mean_3 + mean_6 + mean_12 + mean_24) / (4 * df['close'] + 1e-8)
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_047(df: pd.DataFrame,window=7, factor=20):
        factor = int(factor)
        # window 5,6,7,8
        # window_rank 10,100,10
        """ SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1) """
        # 1. Stochastic components
        max_high_6 = df['high'].rolling(3).max()
        min_low_6 = df['low'].rolling(3).min()
        
        # 2. %K Stochastic Formula
        stoch = (max_high_6 - df['close']) / (max_high_6 - min_low_6 + 1e-8) * 100
        
        # 3. SMA (alpha = 1/9)
        sma_stoch = stoch.ewm(alpha=window/9, adjust=False).mean()
        
        # 4. Normalize
        ranked_signal = sma_stoch.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_048(df: pd.DataFrame, window=60):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Signs of daily returns
        sign_1 = np.sign(df['close'].diff(1))
        sign_2 = np.sign(df['close'].shift(1).diff(1))
        sign_3 = np.sign(df['close'].shift(2).diff(1))
        
        # 2. Sum and limit signs
        sum_signs = sign_1 + sign_2 + sign_3
        rank_signs = sum_signs.rolling(window).rank(pct=True)
        
        # 3. Volume factors
        sum_vol_5 = volume.rolling(5).sum()
        sum_vol_20 = volume.rolling(10).sum()
        
        # 4. Combine
        raw_signal = -1 * (rank_signs * sum_vol_5) / (sum_vol_20 + 1e-8)
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_049(df: pd.DataFrame, window=5):
        # 1. Trend Conditions
        cond_up = (df['high'] + df['low']) >= (df['high'].shift(1) + df['low'].shift(1))
        
        # 2. Maximum absolute components
        max_abs_diff = np.maximum(df['high'].diff(1).abs(), df['low'].diff(1).abs())
        
        # 3. Conditional Values
        part1 = pd.Series(0.0, index=df.index)
        part1[~cond_up] = max_abs_diff
        
        part2 = pd.Series(0.0, index=df.index)
        part2[cond_up] = max_abs_diff
        
        # 4. Sums
        sum_part1 = part1.rolling(12).sum()
        sum_part2 = part2.rolling(12).sum()
        
        # 5. Ratio and Normalize
        ratio = sum_part1 / (sum_part1 + sum_part2 + 1e-8)
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_050(df: pd.DataFrame, window=70):
        # 1. Trend Conditions
        cond_up = (df['high'] + df['low']) >= (df['high'].shift(1) + df['low'].shift(1))
        
        # 2. Maximum absolute components
        max_abs_diff = np.maximum(df['high'].diff(1).abs(), df['low'].diff(1).abs())
        
        # 3. Conditional Values
        part1 = pd.Series(0.0, index=df.index)
        part1[~cond_up] = max_abs_diff
        
        part2 = pd.Series(0.0, index=df.index)
        part2[cond_up] = max_abs_diff
        
        # 4. Sums
        sum_part1 = part1.rolling(30).sum()
        sum_part2 = part2.rolling(30).sum()
        
        # 5. Ratio Difference
        ratio_diff = (sum_part1 - sum_part2) / (sum_part1 + sum_part2 + 1e-8)
        
        # 6. Normalize
        ranked_signal = ratio_diff.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)
        
    @staticmethod
    def alpha_popbo_051(df: pd.DataFrame, window=90):
        """ SUM(((HIGH+LOW)<=... """
        cond_down = (df['high'] + df['low']) <= (df['high'].shift(1) + df['low'].shift(1))
        
        max_abs_diff = np.maximum(df['high'].diff(1).abs(), df['low'].diff(1).abs())
        
        part1 = pd.Series(0.0, index=df.index)
        part1[~cond_down] = max_abs_diff
        
        part2 = pd.Series(0.0, index=df.index)
        part2[cond_down] = max_abs_diff
        
        sum_part1 = part1.rolling(30).sum()
        sum_part2 = part2.rolling(30).sum()
        
        ratio = sum_part1 / (sum_part1 + sum_part2 + 1e-8)
        
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_052(df: pd.DataFrame, window=80): 
        """ SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100 """
        vwap = (df['high'] + df['low'] + df['close']) / 3
        delay_vwap = vwap.shift(1)
        
        up_move = np.maximum(0, df['high'] - delay_vwap) 
        down_move = np.maximum(0, delay_vwap - df['low'])
        
        sum_up = up_move.rolling(26).sum()
        sum_down = down_move.rolling(26).sum()
        
        ratio = (sum_up / (sum_down + 1e-8)) * 100
        
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_053(df: pd.DataFrame, window=5):
        """ COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100 """
        cond_up = df['close'] > df['close'].shift(1)
        count_up = cond_up.rolling(12).sum()
        
        ratio = (count_up / 12) * 100
        
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_054(df: pd.DataFrame, window=10):
        """ (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10))) """
        diff_co = df['close'] - df['open']
        abs_diff_co = diff_co.abs()
        
        # 1. 10-day STD of Absolute C-O difference
        std_abs_diff = abs_diff_co.rolling(10).std()
        
        # 2. 10-day Correlation of Close and Open
        corr_10 = df['close'].rolling(10).corr(df['open']).fillna(0)
        
        # 3. Inner expression
        inner_val = std_abs_diff + diff_co + corr_10
        rank_inner = inner_val.rolling(window).rank(pct=True)
        
        # 4. Normalize
        raw_signal = -1 * rank_inner
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    @staticmethod
    def alpha_popbo_055(df: pd.DataFrame, window=30):
        """ SUM(...) * MAX(...) / ... """
        A = (df['high'] - df['close'].shift(1)).abs()
        B = (df['low'] - df['close'].shift(1)).abs()
        C = (df['high'] - df['low'].shift(1)).abs()
        
        cond1 = (A > B) & (A > C)
        cond2 = (B > C) & (B > A)
        cond3 = (C >= A) & (C >= B)
        
        part0 = 16 * (df['close'] + (df['close'] - df['open'])/2 - df['open'].shift(1))
        
        part1 = pd.Series(0.0, index=df.index)
        part1[cond1] = A + B/2 + (df['close'].shift(1) - df['open'].shift(1)).abs() / 4
        part1[cond2] = B + A/2 + (df['close'].shift(1) - df['open'].shift(1)).abs() / 4
        part1[cond3] = C + (df['close'].shift(1) - df['open'].shift(1)).abs() / 4
        
        part2 = np.maximum(A, B)
        
        # 1. Daily term
        daily_metric = (part0 / (part1 + 1e-8)) * part2
        
        # 2. Sum over 20 days
        sum_metric = daily_metric.rolling(20).sum()
        
        # 3. Normalize
        ranked_signal = sum_metric.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_056(df: pd.DataFrame, window=50):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left side rank
        tsmin_open_12 = df['open'].rolling(12).min()
        A = (df['open'] - tsmin_open_12).rolling(window).rank(pct=True)
        
        # 2. Right side inner
        sum_mid_19 = ((df['high'] + df['low']) / 2).rolling(19).sum()
        mean_vol_40 = volume.rolling(40).mean()
        sum_mean_vol_19 = mean_vol_40.rolling(19).sum()
        corr_13 = sum_mid_19.rolling(13).corr(sum_mean_vol_19).fillna(0)
        
        rank_corr = corr_13.rolling(window).rank(pct=True)
        
        # 3. Right side rank
        B = (rank_corr ** 5).rolling(window).rank(pct=True)
        
        # 4. Condition
        cond = A < B
        part = pd.Series(0.0, index=df.index)
        part[cond] = 1.0
        
        # 5. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_057(df: pd.DataFrame, window=80):
        """ SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1) """
        tsmin_low = df['low'].rolling(40).min()
        tsmax_high = df['high'].rolling(40).max()
        
        stoch = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + 1e-8) * 100
        sma_stoch = stoch.ewm(alpha=1/3, adjust=False).mean()
        
        ranked_signal = sma_stoch.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_058(df: pd.DataFrame, window=50):
        """ COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100 """
        cond_up = df['close'] > df['close'].shift(1)
        count_up = cond_up.rolling(20).sum()
        
        ratio = (count_up / 20) * 100
        
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_059(df: pd.DataFrame, window=60):
        prev_close = df['close'].shift(1)
        
        cond_gt = df['close'] > prev_close
        cond_lt = df['close'] < prev_close
        
        part = pd.Series(0.0, index=df.index)
        part[cond_gt] = df['close'] - np.minimum(df['low'], prev_close)
        part[cond_lt] = df['close'] - np.maximum(df['high'], prev_close)
        
        sum_20 = part.rolling(20).sum()
        
        ranked_signal = sum_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_060(df: pd.DataFrame, window=60):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        range_diff = df['high'] - df['low']
        location = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (range_diff + 1e-8)
        
        vol_weighted = location * volume
        sum_20 = vol_weighted.rolling(20).sum()
        
        ranked_signal = sum_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_061(df: pd.DataFrame, window=10):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        delta_vwap = vwap.diff(1)
        decay_delta = delta_vwap.ewm(span=12).mean()
        rank_left = decay_delta.rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol = volume.rolling(80).mean()
        corr_8 = df['low'].rolling(8).corr(mean_vol).fillna(0)
        rank_corr = corr_8.rolling(17).rank(pct=True)
        decay_rank = rank_corr.ewm(span=17).mean()
        rank_right = decay_rank.rolling(window).rank(pct=True)
        
        # 3. Max and Normalize
        max_rank = np.maximum(rank_left, rank_right)
        raw_signal = -1 * max_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    

    @staticmethod
    def alpha_popbo_062(df: pd.DataFrame, window=10):
        """ (-1 * CORR(HIGH, RANK(VOLUME), 5)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Rank of volume
        rank_vol_5 = volume.rolling(5).rank(pct=True)
        
        # 2. Correlation
        corr_5 = df['high'].rolling(5).corr(rank_vol_5).fillna(0)
        
        # 3. Normalize
        raw_signal = -1 * corr_5
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_063(df: pd.DataFrame,window=5, factor=20):
        factor = int(factor)
        # window 2,3,4,5
        # window_rank = 10,100,10
        """ SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100 """
        diff_1 = df['close'].diff(1)
        
        # 1. Up Moves and Absolute Moves
        up_move = np.maximum(0, diff_1)
        abs_move = diff_1.abs()
        
        # 2. SMA (alpha = 1/6)
        sma_up = up_move.ewm(alpha=window/6, adjust=False).mean()
        sma_abs = abs_move.ewm(alpha=window/6, adjust=False).mean()
        
        # 3. RSI-like Ratio
        rsi_like = (sma_up / (sma_abs + 1e-8)) * 100
        
        # 4. Normalize
        ranked_signal = rsi_like.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_064(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        rank_vwap_4 = vwap.rolling(4).rank(pct=True)
        rank_vol_4 = volume.rolling(4).rank(pct=True)
        corr_4_vwap_vol = rank_vwap_4.rolling(4).corr(rank_vol_4).fillna(0)
        decay_left = corr_4_vwap_vol.ewm(span=4).mean()
        rank_left = decay_left.rolling(window).rank(pct=True)
        
        # 2. Right Term
        rank_close_4 = df['close'].rolling(4).rank(pct=True)
        mean_vol_60 = volume.rolling(60).mean()
        rank_mean_vol_60_4 = mean_vol_60.rolling(4).rank(pct=True)
        corr_4_close_vol = rank_close_4.rolling(4).corr(rank_mean_vol_60_4).fillna(0)
        max_corr = corr_4_close_vol.rolling(13).max()
        decay_right = max_corr.ewm(span=14).mean()
        rank_right = decay_right.rolling(window).rank(pct=True)
        
        # 3. Max and Normalize
        max_rank = np.maximum(rank_left, rank_right)
        raw_signal = -1 * max_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_065(df: pd.DataFrame,window=2, factor=20):
        factor = int(factor)
        # window = 2,3,4,5
        # window_rank = 10,100,10
        """ MEAN(CLOSE,6)/CLOSE """
        # 1. 6-day SMA
        mean_6 = df['close'].rolling(window).mean()
        
        # 2. Ratio
        ratio = mean_6 / df['close']
        
        # 3. Normalize
        ranked_signal = ratio.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_066(df: pd.DataFrame, window=50):
        """ (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100 """
        # 1. 6-day SMA
        mean_6 = df['close'].rolling(3).mean()
        
        # 2. Percentage Deviation
        pct_dev = (df['close'] - mean_6) / mean_6 * 100
        
        # 3. Normalize
        ranked_signal = pct_dev.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_067(df: pd.DataFrame,window=5, factor=50):
        factor = int(factor)
        # window 10,13,16,19
        # window_rank 10,100,10
        """ SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100 """
        diff_1 = df['close'].diff(1)
        
        # 1. Up Moves and Absolute Moves
        up_move = np.maximum(0, diff_1)
        abs_move = diff_1.abs()
        
        # 2. SMA (alpha = 1/24)
        sma_up = up_move.ewm(alpha=window/24, adjust=True).mean()
        sma_abs = abs_move.ewm(alpha=window/24, adjust=True).mean()
        
        # 3. RSI-like Ratio
        rsi_like = (sma_up / (sma_abs + 1e-8)) * 100
        
        # 4. Normalize
        ranked_signal = rsi_like.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_068(df: pd.DataFrame, window=80):
        """ SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Midpoint Delta
        midpoint = (df['high'] + df['low']) / 2
        midpoint_delay = midpoint.shift(1)
        delta_mid = midpoint - midpoint_delay
        
        # 2. Range
        range_val = df['high'] - df['low']
        
        # 3. Metric calculation
        metric = (delta_mid * range_val) / (volume + 1e-8)
        
        # 4. SMA (alpha = 2/15)
        sma_15 = metric.ewm(alpha=2/15, adjust=False).mean()
        
        # 5. Normalize
        ranked_signal = sma_15.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    @staticmethod
    def alpha_popbo_069(df: pd.DataFrame, window=60):
        """ DTM and DBM Ratio """
        delay_open = df['open'].shift(1)
        
        # 1. DTM: Buying Pressure
        dtm_cond = df['open'] <= delay_open
        dtm = pd.Series(0.0, index=df.index)
        dtm[~dtm_cond] = np.maximum(df['high'] - df['open'], df['open'] - delay_open)
        
        # 2. DBM: Selling Pressure
        dbm_cond = df['open'] >= delay_open
        dbm = pd.Series(0.0, index=df.index)
        dbm[~dbm_cond] = np.maximum(df['open'] - df['low'], df['open'] - delay_open)
        
        # 3. Sums over 20 days
        sum_dtm = dtm.rolling(20).sum()
        sum_dbm = dbm.rolling(20).sum()
        
        # 4. Conditional Ratio
        cond_gt = sum_dtm > sum_dbm
        cond_lt = sum_dtm < sum_dbm
        
        part = pd.Series(0.0, index=df.index)
        part[cond_gt] = (sum_dtm - sum_dbm) / (sum_dtm + 1e-8)
        part[cond_lt] = (sum_dtm - sum_dbm) / (sum_dbm + 1e-8)
        
        # 5. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_070(df: pd.DataFrame, window=24):
        """ STD(AMOUNT,6) """
        # We calculate Amount as Close * Volume
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        amount = df.get('amount', df['close'] * volume)
        
        # 1. 6-day STD of Amount
        std_6 = amount.rolling(6).std()
        
        # 2. Normalize
        ranked_signal = std_6.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_071(df: pd.DataFrame, window=60):
        """ (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100 """
        mean_24 = df['close'].rolling(24).mean()
        
        # 1. Percentage Deviation
        ratio = (df['close'] - mean_24) / mean_24 * 100
        
        # 2. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_072(df: pd.DataFrame, window=50):
        """ SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1) """
        max_high = df['high'].rolling(6).max()
        min_low = df['low'].rolling(6).min()
        
        # 1. Stochastic value
        stoch = (max_high - df['close']) / (max_high - min_low + 1e-8) * 100
        
        # 2. SMA (alpha = 1/15)
        sma_15 = stoch.ewm(alpha=1/15, adjust=False).mean()
        
        # 3. Normalize
        ranked_signal = sma_15.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_073(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        corr_10 = df['close'].rolling(10).corr(volume).fillna(0)
        decay_16 = corr_10.ewm(span=16).mean()
        decay_4 = decay_16.ewm(span=4).mean()
        tsrank_5 = decay_4.rolling(5).rank(pct=True)
        
        # 2. Right Term
        mean_vol_30 = volume.rolling(30).mean()
        corr_4 = vwap.rolling(4).corr(mean_vol_30).fillna(0)
        decay_3 = corr_4.ewm(span=3).mean()
        rank_right = decay_3.rolling(window).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = -1 * (tsrank_5 - rank_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_074(df: pd.DataFrame, window=10):
        """ (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        w_price = (df['low'] * 0.35) + (vwap * 0.65)
        sum_w_price_20 = w_price.rolling(20).sum()
        
        mean_vol_40 = volume.rolling(40).mean()
        sum_mean_vol_20 = mean_vol_40.rolling(20).sum()
        
        corr_7 = sum_w_price_20.rolling(7).corr(sum_mean_vol_20).fillna(0)
        rank_left = corr_7.rolling(window).rank(pct=True)
        
        # 2. Right Term
        rank_vwap = vwap.rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        corr_6 = rank_vwap.rolling(6).corr(rank_vol).fillna(0)
        rank_right = corr_6.rolling(window).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = rank_left + rank_right
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_075(df: pd.DataFrame, window=10):
        bm_open = df.get('benchmark_open', df['open'])
        bm_close = df.get('benchmark_close', df['close'])
        
        # 1. Conditions
        cond_up = df['close'] > df['open']
        cond_bm_down = bm_close < bm_open
        both_cond = cond_up & cond_bm_down
        
        # 2. Sums over 50 days
        count_both_50 = both_cond.rolling(50).sum()
        count_bm_50 = cond_bm_down.rolling(50).sum()
        
        # 3. Ratio
        ratio = count_both_50 / (count_bm_50 + 1e-8)
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_076(df: pd.DataFrame, window=40):
        """ STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Vol-adjusted absolute return
        ret_abs = (df['close'] / df['close'].shift(1) - 1).abs()
        metric = ret_abs / (volume + 1e-8)
        
        # 2. Rolling stats over 20 days
        std_20 = metric.rolling(20).std()
        mean_20 = metric.rolling(20).mean()
        
        # 3. Ratio
        ratio = std_20 / (mean_20 + 1e-8)
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_077(df: pd.DataFrame, window=60):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        midpoint = (df['high'] + df['low']) / 2
        
        # 1. Left Term
        left_val = (midpoint + df['high']) - (vwap + df['high'])
        decay_left = left_val.ewm(span=20).mean()
        rank_left = decay_left.rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol_40 = volume.rolling(40).mean()
        corr_3 = midpoint.rolling(3).corr(mean_vol_40).fillna(0)
        decay_right = corr_3.ewm(span=6).mean()
        rank_right = decay_right.rolling(window).rank(pct=True)
        
        # 3. Min Function and Normalize
        min_rank = np.minimum(rank_left, rank_right)
        ranked_signal = min_rank.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_078(df: pd.DataFrame,window =2, factor=20):
        factor = int(factor)
        # window 2,3,4,5
        # window_rank 10,100,10
        vwap_approx = (df['high'] + df['low'] + df['close']) / 3
        ma_vwap_12 = vwap_approx.rolling(window).mean()
        
        # 1. Numerator
        numerator = vwap_approx - ma_vwap_12
        
        # 2. Denominator
        mean_dev = (df['close'] - ma_vwap_12).abs().rolling(12).mean()
        denominator = 0.015 * mean_dev
        
        # 3. Ratio and Normalize
        ratio = numerator / (denominator + 1e-8)
        ranked_signal = ratio.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_079(df: pd.DataFrame,window=7, factor=20):
        factor = int(factor)
        # window 4,5,6,7
        # window_rank 10,100,10
        """ SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100 """
        diff_1 = df['close'].diff(1)
        
        # 1. Up Moves and Absolute Moves
        up_move = np.maximum(0, diff_1)
        abs_move = diff_1.abs()
        
        # 2. SMA (alpha = 1/12)
        sma_up = up_move.ewm(alpha=window/12, adjust=False).mean()
        sma_abs = abs_move.ewm(alpha=window/12, adjust=False).mean()
        
        # 3. RSI ratio
        rsi_12 = (sma_up / (sma_abs + 1e-8)) * 100
        
        # 4. Normalize
        ranked_signal = rsi_12.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_080(df: pd.DataFrame, window=80):
        """ (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100 """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. 5-day Percentage Change in Volume
        delay_vol_5 = volume.shift(5)
        ratio = (volume - delay_vol_5) / (delay_vol_5 + 1e-8) * 100
        
        # 2. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_081(df: pd.DataFrame, window=15):
        """ SMA(VOLUME,21,2) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. SMA (alpha = 2/21)
        sma_21 = volume.ewm(alpha=2/21, adjust=False).mean()
        
        # 2. Normalize
        ranked_signal = sma_21.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_082(df: pd.DataFrame, window=60): 
        """ SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1) """
        max_high = df['high'].rolling(6).max()
        min_low = df['low'].rolling(6).min()
        
        # 1. Stochastic value
        stoch = (max_high - df['close']) / (max_high - min_low + 1e-8) * 100
        
        # 2. SMA (alpha = 1/20)
        sma_20 = stoch.ewm(alpha=1/20, adjust=False).mean()
        
        # 3. Normalize
        ranked_signal = sma_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_083(df: pd.DataFrame, window=100):
        """ (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Ranks
        rank_high = df['high'].rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        # 2. Covariance
        cov_5 = rank_high.rolling(5).cov(rank_vol).fillna(0)
        
        # 3. Rank of Covariance
        raw_signal = -1 * cov_5.rolling(window).rank(pct=True)
        
        # 4. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_084(df: pd.DataFrame, window=60): 
        """ SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        prev_close = df['close'].shift(1)
        
        # 1. Conditions
        cond_up = df['close'] > prev_close
        cond_down = df['close'] < prev_close
        
        # 2. Conditional assignments
        part = pd.Series(0.0, index=df.index)
        part[cond_up] = volume
        part[cond_down] = -1 * volume
        
        # 3. Sum over 20 days
        sum_20 = part.rolling(20).sum()
        
        # 4. Normalize
        ranked_signal = sum_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_085(df: pd.DataFrame, window=80):
        """ (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Term (Volume Ratio Rank)
        mean_vol_20 = volume.rolling(20).mean()
        vol_ratio = volume / (mean_vol_20 + 1e-8)
        tsrank_vol_ratio = vol_ratio.rolling(20).rank(pct=True)
        
        # 2. Right Term (Delta Close Rank)
        delta_7 = df['close'].diff(7)
        tsrank_delta = (-1 * delta_7).rolling(8).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = tsrank_vol_ratio * tsrank_delta
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_086(df: pd.DataFrame, window=100):
        """ Conditional return assignment based on 20-day and 10-day delays """
        delay_10 = df['close'].shift(10)
        delay_20 = df['close'].shift(20)
        
        # 1. Complex Metric
        metric = ((delay_20 - delay_10) / 10) - ((delay_10 - df['close']) / 10)
        
        # 2. Conditions
        cond_gt = metric > 0.25
        cond_lt = metric < 0.0
        cond_mid = (~cond_gt) & (~cond_lt)
        
        # 3. Conditional Values
        part = pd.Series(0.0, index=df.index)
        part[cond_gt] = -1.0
        part[cond_lt] = 1.0
        part[cond_mid] = -1 * df['close'].diff(1)
        
        # 4. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_087(df: pd.DataFrame, window=50):
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left term
        delta_4 = vwap.diff(4)
        decay_7_left = delta_4.ewm(span=7).mean()
        rank_left = decay_7_left.rolling(window).rank(pct=True)
        
        # 2. Right term
        # LOW * 0.9 + LOW * 0.1 is just LOW
        num = df['low'] - vwap
        den = df['open'] - (df['high'] + df['low']) / 2
        metric = num / (den + 1e-8)
        
        decay_11_right = metric.ewm(span=11).mean()
        tsrank_7_right = decay_11_right.rolling(7).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = -1 * (rank_left + tsrank_7_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_088(df: pd.DataFrame, window=60):
        """ (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100 """
        delay_20 = df['close'].shift(20)
        
        # 1. 20-day Percentage Change
        ratio = (df['close'] - delay_20) / (delay_20 + 1e-8) * 100
        
        # 2. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_089(df: pd.DataFrame, window=70):
        """ 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2)) """
        # 1. Simple Moving Averages (MACD-like)
        sma_13 = df['close'].ewm(alpha=2/13, adjust=False).mean()
        sma_27 = df['close'].ewm(alpha=2/27, adjust=False).mean()
        
        # 2. Difference
        diff_sma = sma_13 - sma_27
        
        # 3. SMA of Difference (Signal Line)
        sma_diff_10 = diff_sma.ewm(alpha=2/10, adjust=False).mean()
        
        # 4. Final calculation and Normalize
        raw_signal = 2 * (diff_sma - sma_diff_10)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_090(df: pd.DataFrame, window=20):
        """ (RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Ranks
        rank_vwap = vwap.rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        # 2. Correlation
        corr_5 = rank_vwap.rolling(5).corr(rank_vol).fillna(0)
        
        # 3. Combine and Normalize
        raw_signal = -1 * corr_5.rolling(window).rank(pct=True)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_091(df: pd.DataFrame, window=30):
        """ ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Term
        max_close_5 = df['close'].rolling(5).max()
        rank_left = (df['close'] - max_close_5).rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol_40 = volume.rolling(40).mean()
        corr_5 = mean_vol_40.rolling(5).corr(df['low']).fillna(0)
        rank_right = corr_5.rolling(window).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = (rank_left * rank_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
    
    @staticmethod
    def alpha_popbo_092(df: pd.DataFrame, window=50):
        """ (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        w_price = (df['close'] * 0.35) + (vwap * 0.65)
        delta_2 = w_price.diff(2)
        decay_3 = delta_2.ewm(span=3).mean()
        rank_left = decay_3.rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol_180 = volume.rolling(180).mean()
        corr_13 = mean_vol_180.rolling(13).corr(df['close']).fillna(0)
        abs_corr = corr_13.abs()
        decay_5 = abs_corr.ewm(span=5).mean()
        rank_right = decay_5.rolling(15).rank(pct=True)
        
        # 3. Combine and Normalize
        max_rank = np.maximum(rank_left, rank_right)
        raw_signal =  max_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_093(df: pd.DataFrame, window=60):
        """ SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) """
        delay_open = df['open'].shift(1)
        
        # 1. Conditional values
        cond_up = df['open'] >= delay_open
        
        part = pd.Series(0.0, index=df.index)
        part[~cond_up] = np.maximum(df['open'] - df['low'], df['open'] - delay_open)
        
        # 2. Sum over 20 days
        sum_20 = part.rolling(20).sum()
        
        # 3. Normalize
        ranked_signal = sum_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_094(df: pd.DataFrame, window=10):
        """ SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        prev_close = df['close'].shift(1)
        
        # 1. Conditional volume assignments
        cond_up = df['close'] > prev_close
        cond_down = df['close'] < prev_close
        
        part = pd.Series(0.0, index=df.index)
        part[cond_up] = volume
        part[cond_down] = -1 * volume
        
        # 2. Sum over 30 days
        sum_30 = part.rolling(30).sum()
        
        # 3. Normalize
        ranked_signal = sum_30.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_095(df: pd.DataFrame, window=40):
        """ STD(AMOUNT,20) """
        # We assume amount is close * volume if not explicitly provided
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        amount = df.get('amount', df['close'] * volume)
        
        # 1. 20-day STD of Amount
        std_20 = amount.rolling(20).std()
        
        # 2. Normalize
        ranked_signal = std_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_096(df: pd.DataFrame, window=80):
        """ SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) """
        tsmin_low = df['low'].rolling(9).min()
        tsmax_high = df['high'].rolling(9).max()
        
        # 1. Stochastic value
        stoch = (df['close'] - tsmin_low) / (tsmax_high - tsmin_low + 1e-8) * 100
        
        # 2. Double SMA (alpha = 1/3)
        sma_inner = stoch.ewm(alpha=1/3, adjust=False).mean()
        sma_outer = sma_inner.ewm(alpha=1/3, adjust=False).mean()
        
        # 3. Normalize
        ranked_signal = sma_outer.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_097(df: pd.DataFrame, window=80):
        """ STD(VOLUME,10) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. 10-day STD of Volume
        std_10 = volume.rolling(10).std()
        
        # 2. Normalize
        ranked_signal = std_10.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_098(df: pd.DataFrame, window=4, factor=20):
        factor = int(factor)
        mean_100 = df['close'].rolling(factor).mean()
        delta_100 = mean_100.diff(factor)
        delay_100 = df['close'].shift(factor)
        
        # 1. Conditional metric calculation
        metric = delta_100 / (delay_100 + 1e-8)
        cond = metric <= 0.05
        
        tsmin_100 = df['close'].rolling(factor).min()
        delta_3 = df['close'].diff(window)
        
        # 2. Assignments
        part = pd.Series(0.0, index=df.index)
        part[cond] = -1 * (df['close'] - tsmin_100)
        part[~cond] = -1 * delta_3
        
        # 3. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_099(df: pd.DataFrame, window=60):
        """ (-1 * RANK(COV(RANK(CLOSE), RANK(VOLUME), 5))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Ranks
        rank_close = df['close'].rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        # 2. Covariance
        cov_5 = rank_close.rolling(5).cov(rank_vol).fillna(0)
        
        # 3. Combine and Normalize
        raw_signal = -1 * cov_5.rolling(window).rank(pct=True)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_100(df: pd.DataFrame, window=10):
        """ STD(VOLUME,20) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. 20-day STD of Volume
        std_20 = volume.rolling(20).std()
        
        # 2. Normalize
        ranked_signal = std_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_101(df: pd.DataFrame, window=40):
        """ ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Rank
        mean_vol_30 = volume.rolling(30).mean()
        sum_mean_vol_37 = mean_vol_30.rolling(37).sum()
        corr_15 = df['close'].rolling(15).corr(sum_mean_vol_37).fillna(0)
        rank1 = corr_15.rolling(window).rank(pct=True)
        
        # 2. Right Rank
        metric = (df['high'] * 0.1) + (vwap * 0.9)
        rank_metric_11 = metric.rolling(11).rank(pct=True)
        rank_vol_11 = volume.rolling(11).rank(pct=True)
        corr_11 = rank_metric_11.rolling(11).corr(rank_vol_11).fillna(0)
        rank2 = corr_11.rolling(window).rank(pct=True)
        
        # 3. Condition
        cond = rank1 < rank2
        part = pd.Series(0.0, index=df.index)
        part[cond] = 1.0
        
        # 4. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_102(df: pd.DataFrame, window=10):
        """ SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        diff_vol = volume.diff(1)
        
        # 1. Volume Moves
        up_vol = np.maximum(0, diff_vol)
        abs_vol = diff_vol.abs()
        
        # 2. SMA (alpha = 1/6)
        sma_up = up_vol.ewm(alpha=1/6, adjust=False).mean()
        sma_abs = abs_vol.ewm(alpha=1/6, adjust=False).mean()
        
        # 3. Ratio
        ratio = (sma_up / (sma_abs + 1e-8)) * 100
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_103(df: pd.DataFrame, window=40):
        """ ((20-LOWDAY(LOW,20))/20)*100 """
        # 1. Calculate days since 20-day low
        lowday = df['low'].rolling(20).apply(lambda x: 19 - np.argmin(x), raw=True)
        
        # 2. Ratio
        ratio = ((20 - lowday) / 20) * 100
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_104(df: pd.DataFrame, window=10):
        """ (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Delta of Correlation
        corr_5 = df['high'].rolling(5).corr(volume).fillna(0)
        delta_5 = corr_5.diff(5)
        
        # 2. Rank of STD
        std_20 = df['close'].rolling(20).std()
        rank_std = std_20.rolling(window).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = -1 * (delta_5 * rank_std)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_105(df: pd.DataFrame, window=30):
        """ (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Ranks
        rank_open = df['open'].rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        
        # 2. Correlation
        corr_10 = rank_open.rolling(10).corr(rank_vol).fillna(0)
        
        # 3. Combine and Normalize
        raw_signal = -1 * corr_10
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_106(df: pd.DataFrame, window=1, factor = 20):
        factor = int(factor)
        """ CLOSE-DELAY(CLOSE,20) """
        # 1. 20-day Momentum
        delta_20 = df['close'].diff(window)
        
        # 2. Normalize
        ranked_signal = delta_20.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_107(df: pd.DataFrame, window=20):
        """ (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1)))) """
        # 1. Ranks of Gaps
        rank_1 = (df['open'] - df['high'].shift(1)).rolling(window).rank(pct=True)
        rank_2 = (df['open'] - df['close'].shift(1)).rolling(window).rank(pct=True)
        rank_3 = (df['open'] - df['low'].shift(1)).rolling(window).rank(pct=True)
        
        # 2. Combine and Normalize
        raw_signal = -1 * rank_1 * rank_2 * rank_3
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_108(df: pd.DataFrame, window=60):
        """ ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        tsmin_high_2 = df['high'].rolling(2).min()
        rank_left = (df['high'] - tsmin_high_2).rolling(window).rank(pct=True)
        
        # 2. Right Term
        mean_vol_120 = volume.rolling(120).mean()
        corr_6 = vwap.rolling(6).corr(mean_vol_120).fillna(0)
        rank_right = corr_6.rolling(window).rank(pct=True)
        
        # 3. Power and Normalize
        raw_signal = -1 * (rank_left ** rank_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_109(df: pd.DataFrame, window=20):
        """ SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2) """
        range_val = df['high'] - df['low']
        
        # 1. SMAs
        sma_1 = range_val.ewm(alpha=4/10, adjust=False).mean()
        sma_2 = sma_1.ewm(alpha=4/10, adjust=False).mean()
        
        # 2. Ratio
        ratio = sma_1 / (sma_2 + 1e-8)
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_110(df: pd.DataFrame, window=20):
        """ SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 """
        delay_close = df['close'].shift(1)
        
        # 1. Up and Down Ranges
        up_move = np.maximum(0, df['high'] - delay_close)
        down_move = np.maximum(0, delay_close - df['low'])
        
        # 2. 20-day Sums
        sum_up = up_move.rolling(20).sum()
        sum_down = down_move.rolling(20).sum()
        
        # 3. Ratio
        ratio = sum_up / (sum_down + 1e-8) * 100
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_111(df: pd.DataFrame,window=2, factor=20):
        factor = int(factor)
        """ SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Location Value
        range_val = df['high'] - df['low']
        location = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (range_val + 1e-8)
        
        # 2. Metric Series
        metric = volume * location
        
        # 3. SMAs
        sma_11 = metric.ewm(alpha=window/11, adjust=False).mean()
        sma_4 = metric.ewm(alpha=window/4, adjust=False).mean()
        
        # 4. Difference and Normalize
        raw_signal = sma_11 - sma_4
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_112(df: pd.DataFrame, window=40):
        """ (SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM() + SUM())*100 """
        delta_1 = df['close'].diff(1)
        
        # 1. Up and Down absolute moves
        up_move = np.maximum(0, delta_1)
        down_move_abs = np.maximum(0, -delta_1)
        
        # 2. Sums over 12 days
        sum_up_12 = up_move.rolling(window).sum()
        sum_down_12 = down_move_abs.rolling(window).sum()
        
        # 3. Ratio
        numerator = sum_up_12 - sum_down_12
        denominator = sum_up_12 + sum_down_12
        ratio = (numerator / (denominator + 1e-8)) * 100
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_113(df: pd.DataFrame, window=20):
        """ (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2)))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Term Rank
        delay_5 = df['close'].shift(1)
        sum_20_delay_5 = delay_5.rolling(20).sum() / 20
        rank_1 = sum_20_delay_5.rolling(window).rank(pct=True)
        
        # 2. Mid Term
        corr_2_cv = df['close'].rolling(20).corr(volume).fillna(0)
        
        # 3. Right Term Rank
        sum_close_5 = df['close'].rolling(5).sum()
        sum_close_20 = df['close'].rolling(20).sum()
        corr_2_sums = sum_close_5.rolling(2).corr(sum_close_20).fillna(0)
        rank_2 = corr_2_sums.rolling(window).rank(pct=True)
        
        # 4. Combine and Normalize
        raw_signal = -1 * (rank_1 * corr_2_cv * rank_2)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_114(df: pd.DataFrame,window=2, factor=20):
        factor = int(factor)
        """ ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Inner Metric Calculation
        mean_close_5 = df['close'].rolling(factor).mean()
        metric = (df['high'] - df['low']) / (mean_close_5 + 1e-8)
        
        # 2. Numerator Left Rank
        delay_2 = metric.shift(window)
        rank_1 = delay_2.rolling(factor).rank(pct=True)
        
        # 3. Numerator Right Rank
        rank_vol = volume.rolling(factor).rank(pct=True)
        rank_2 = rank_vol.rolling(factor).rank(pct=True)
        
        numerator = rank_1 * rank_2
        
        # 4. Denominator
        denominator = metric / (vwap - df['close'] + 1e-8)
        
        # 5. Combine and Normalize
        raw_signal = numerator / (denominator + 1e-8)
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_115(df: pd.DataFrame, window=30):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Term Base Base Rank
        price_l = (df['high'] * 0.9) + (df['close'] * 0.1)
        mean_vol_30 = volume.rolling(30).mean()
        corr_10 = price_l.rolling(10).corr(mean_vol_30).fillna(0)
        rank_left = corr_10.rolling(window).rank(pct=True)
        
        # 2. Right Term Exponent Rank
        midpoint = (df['high'] + df['low']) / 2
        tsrank_4 = midpoint.rolling(4).rank(pct=True)
        tsrank_vol_10 = volume.rolling(10).rank(pct=True)
        corr_7 = tsrank_4.rolling(7).corr(tsrank_vol_10).fillna(0)
        rank_right = corr_7.rolling(window).rank(pct=True)
        
        # 3. Exponential Combination and Normalize
        raw_signal = rank_left ** rank_right
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_116(df: pd.DataFrame, window=20):
        """ REGBETA(CLOSE,SEQUENCE,20) """
        def rolling_regbeta(y):
            if len(y) < 20: return np.nan
            x = np.arange(20)
            var_x = np.var(x, ddof=1)
            if var_x == 0: return 0
            cov_xy = np.cov(x, y)[0, 1]
            return cov_xy / var_x
            
        beta_20 = df['close'].rolling(20).apply(rolling_regbeta, raw=True)
        
        ranked_signal = beta_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_117(df: pd.DataFrame, window=30):
        """ ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Volume Rank
        tsrank_vol_32 = volume.rolling(32).rank(pct=True)
        
        # 2. Price Metric Rank Inverse
        metric = (df['close'] + df['high']) - df['low']
        tsrank_metric_16 = metric.rolling(16).rank(pct=True)
        
        # 3. Return Rank Inverse
        ret = df['close'].pct_change(1)
        tsrank_ret_32 = ret.rolling(32).rank(pct=True)
        
        # 4. Multiplicative Combination and Normalize
        raw_signal = (tsrank_vol_32 * (1.0 - tsrank_metric_16)) * (1.0 - tsrank_ret_32)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_118(df: pd.DataFrame, window=10):
        """ SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100 """
        ho_diff = df['high'] - df['open']
        ol_diff = df['open'] - df['low']
        
        # 1. 20-day sums
        sum_ho_20 = ho_diff.rolling(window).sum()
        sum_ol_20 = ol_diff.rolling(window).sum()
        
        # 2. Ratio
        ratio = (sum_ho_20 / (sum_ol_20 + 1e-8)) * 100
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_120(df: pd.DataFrame, window=30):
        """ (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE))) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Ranks
        rank_diff = (vwap - df['close']).rolling(window).rank(pct=True)
        rank_sum = (vwap + df['close']).rolling(window).rank(pct=True)
        
        # 2. Ratio
        ratio = rank_diff / (rank_sum + 1e-8)
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_121(df: pd.DataFrame, window=10):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term Base Rank
        tsmin_vwap_12 = vwap.rolling(12).min()
        rank_left = (vwap - tsmin_vwap_12).rolling(window).rank(pct=True)
        
        # 2. Right Term Exponent Rank
        tsrank_vwap_20 = vwap.rolling(20).rank(pct=True)
        mean_vol_60 = volume.rolling(60).mean()
        tsrank_vol_2 = mean_vol_60.rolling(2).rank(pct=True)
        
        corr_18 = tsrank_vwap_20.rolling(18).corr(tsrank_vol_2).fillna(0)
        rank_right = corr_18.rolling(3).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = -1 * (rank_left ** rank_right)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_122(df: pd.DataFrame, window=20):
        """ (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1) """
        log_close = np.log(df['close'].replace(0, np.nan)).fillna(0)
        
        # 1. Triple SMA computation (alpha = 2/13)
        sma_1 = log_close.ewm(alpha=2/13, adjust=False).mean()
        sma_2 = sma_1.ewm(alpha=2/13, adjust=False).mean()
        sma_3 = sma_2.ewm(alpha=2/13, adjust=False).mean()
        
        # 2. Percent change of Triple SMA
        delay_sma = sma_3.shift(1)
        ratio = (sma_3 - delay_sma) / (delay_sma + 1e-8)
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_123(df: pd.DataFrame, window=40):
        """ ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Rank (A)
        midpoint = (df['high'] + df['low']) / 2
        sum_mid_20 = midpoint.rolling(20).sum()
        mean_vol_60 = volume.rolling(60).mean()
        sum_vol_20 = mean_vol_60.rolling(20).sum()
        corr_9 = sum_mid_20.rolling(9).corr(sum_vol_20).fillna(0)
        rank_A = corr_9.rolling(window).rank(pct=True)
        
        # 2. Right Rank (B)
        corr_6 = df['low'].rolling(6).corr(volume).fillna(0)
        rank_B = corr_6.rolling(window).rank(pct=True)
        
        # 3. Condition
        cond = rank_A < rank_B
        part = pd.Series(0.0, index=df.index)
        part[cond] = -1.0
        
        # 4. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_124(df: pd.DataFrame, window=10):
        """ (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2) """
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Numerator
        num = df['close'] - vwap
        
        # 2. Denominator decay rank
        tsmax_30 = df['close'].rolling(30).max()
        rank_max = tsmax_30.rolling(window).rank(pct=True)
        decay_2 = rank_max.ewm(span=2).mean()
        
        # 3. Ratio and Normalize
        ratio = num / (decay_2 + 1e-8)
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_125(df: pd.DataFrame, window=30):
        """ (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term Numerator Rank
        mean_vol_80 = volume.rolling(80).mean()
        corr_17 = vwap.rolling(17).corr(mean_vol_80).fillna(0)
        decay_20 = corr_17.ewm(span=20).mean()
        rank_num = decay_20.rolling(window).rank(pct=True)
        
        # 2. Right Term Denominator Rank
        w_price = (df['close'] * 0.5) + (vwap * 0.5)
        delta_3 = w_price.diff(3)
        decay_16 = delta_3.ewm(span=16).mean()
        rank_den = decay_16.rolling(window).rank(pct=True)
        
        # 3. Ratio and Normalize
        ratio = rank_num / (rank_den + 1e-8)
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_126(df: pd.DataFrame, window=20):
        """ (CLOSE+HIGH+LOW)/3 """
        metric = (df['close'] + df['high'] + df['low']) / 6
        
        ranked_signal = metric.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_127(df: pd.DataFrame, window=20):
        """ (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2) """
        max_12 = df['close'].rolling(window).max()
        
        # 1. Term inside mean
        term = (100 * (df['close'] - max_12) / (max_12 + 1e-8)) ** 2
        
        # 2. Mean and square root (RMS)
        mean_12 = term.rolling(window).mean()
        rms = np.sqrt(mean_12)
        
        # 3. Normalize
        ranked_signal = rms.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_128(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        A = (df['high'] + df['low'] + df['close']) / 3
        delay_A = A.shift(1)
        
        # 1. Conditions
        cond_up = A > delay_A
        cond_down = A < delay_A
        
        # 2. Values
        part_up = pd.Series(0.0, index=df.index)
        part_up[cond_up] = A * volume
        
        part_down = pd.Series(0.0, index=df.index)
        part_down[cond_down] = A * volume
        
        # 3. 14-day Sums
        sum_up = part_up.rolling(window).sum()
        sum_down = part_down.rolling(window).sum()
        
        # 4. RSI-like formula
        rs = sum_up / (sum_down + 1e-8)
        metric = 100 - (100 / (1 + rs))
        
        # 5. Normalize
        ranked_signal = metric.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_129(df: pd.DataFrame, window=40):
        """ SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12) """
        delta_1 = df['close'].diff(1)
        
        cond_down = delta_1 < 0
        
        part = pd.Series(0.0, index=df.index)
        part[cond_down] = delta_1.abs()
        
        sum_12 = part.rolling(window).sum()
        
        ranked_signal = sum_12.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_130(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term Numerator Rank
        midpoint = (df['high'] + df['low']) / 2
        mean_vol_40 = volume.rolling(40).mean()
        corr_9 = midpoint.rolling(9).corr(mean_vol_40).fillna(0)
        decay_10 = corr_9.ewm(span=10).mean()
        rank_num = decay_10.rolling(window).rank(pct=True)
        
        # 2. Right Term Denominator Rank
        rank_vwap = vwap.rolling(window).rank(pct=True)
        rank_vol = volume.rolling(window).rank(pct=True)
        corr_7 = rank_vwap.rolling(7).corr(rank_vol).fillna(0)
        decay_3 = corr_7.ewm(span=3).mean()
        rank_den = decay_3.rolling(window).rank(pct=True)
        
        # 3. Ratio and Normalize
        ratio = rank_num / (rank_den + 1e-8)
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_131(df: pd.DataFrame, window=30):
        """ (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Base Rank
        delta_1 = vwap.diff(1)
        rank_left = delta_1.rolling(window).rank(pct=True)
        
        # 2. Exponent Rank
        mean_vol_50 = volume.rolling(50).mean()
        corr_18 = df['close'].rolling(18).corr(mean_vol_50).fillna(0)
        rank_right = corr_18.rolling(18).rank(pct=True)
        
        # 3. Combine and Normalize
        raw_signal = rank_left ** rank_right
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_132(df: pd.DataFrame, window=5):
        """ MEAN(AMOUNT,20) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        amount = df.get('amount', df['close'] * volume)
        
        mean_20 = amount.rolling(20).mean()
        
        ranked_signal = mean_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_133(df: pd.DataFrame, window=60):
        """ ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100 """
        # HIGHDAY and LOWDAY logic (days ago)
        highday = df['high'].rolling(20).apply(lambda x: 19 - np.argmax(x), raw=True)
        lowday = df['low'].rolling(20).apply(lambda x: 19 - np.argmin(x), raw=True)
        
        left_term = ((20 - highday) / 20) * 100
        right_term = ((20 - lowday) / 20) * 100
        
        raw_signal = left_term - right_term
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_134(df: pd.DataFrame, window=1, factor=20):
        factor = int(factor)
        """ (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        delay_12 = df['close'].shift(window)
        ratio = (df['close'] - delay_12) / (delay_12 + 1e-8)
        
        raw_signal = ratio * volume
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_135(df: pd.DataFrame, window=20):
        """ SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1) """
        delay_20 = df['close'].shift(1)
        ratio = df['close'] / (delay_20 + 1e-8)
        
        delay_1 = ratio.shift(1)
        
        sma_20 = delay_1.ewm(alpha=1/20, adjust=False).mean()
        
        ranked_signal = sma_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_136(df: pd.DataFrame, window=10):
        """ ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        ret = df['close'].pct_change()
        delta_3 = ret.diff(1)
        rank_left = delta_3.rolling(window).rank(pct=True)
        
        corr_10 = df['open'].rolling(10).corr(volume).fillna(0)
        
        raw_signal = -1 * rank_left * corr_10
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_137(df: pd.DataFrame,window=1, factor=10):
        factor = int(factor)
        """ Complex nested ternary expression formula """
        delay_close = df['close'].shift(window)
        delay_open = df['open'].shift(window)
        delay_low = df['low'].shift(window)
        
        # Base terms
        A = (df['high'] - delay_close).abs()
        B = (df['low'] - delay_close).abs()
        C = (df['high'] - delay_low).abs()
        D = (delay_close - delay_open).abs()
        
        # Conditions
        cond1 = (A > B) & (A > C)
        cond2 = (B > C) & (B > A)
        
        # Numerator base
        part0 = 16 * (df['close'] + (df['close'] - df['open'])/2 - delay_open)
        
        # Denominator via conditional logic
        part1 = pd.Series(np.nan, index=df.index)
        part1[cond1] = A + B/2 + D/4
        part1[cond2] = B + A/2 + D/4
        part1[~cond1 & ~cond2] = C + D/4
        part1.replace({0: np.nan}, inplace=True)
        
        # Combine
        max_AB = np.maximum(A, B)
        raw_signal = (part0 / part1) * max_AB
        
        # Normalize
        ranked_signal = raw_signal.rolling(factor).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_138(df: pd.DataFrame, window=10):
        """ ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) 
            - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), 
            TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1) """

        # === INPUT DATA (không dùng get nữa) ===
        volume = df['matchingVolume']
        vwap = (df['high'] + df['low'] + df['close']) / 3

        # === 1. LEFT TERM ===
        w_price = (df['low'] * 0.7) + (vwap * 0.3)
        delta_3 = w_price.diff(3)
        decay_20 = delta_3.ewm(span=20, adjust=False).mean()
        rank_left = decay_20.rolling(window).rank(pct=True)

        # === 2. RIGHT TERM ===
        tsrank_low_8 = df['low'].rolling(8).rank(pct=True)

        mean_vol_60 = volume.rolling(60).mean()
        tsrank_vol_17 = mean_vol_60.rolling(17).rank(pct=True)

        corr_5 = tsrank_low_8.rolling(5).corr(tsrank_vol_17)
        corr_5 = corr_5.fillna(0)

        tsrank_corr_19 = corr_5.rolling(19).rank(pct=True)
        decay_16 = tsrank_corr_19.ewm(span=16, adjust=False).mean()
        tsrank_right = decay_16.rolling(7).rank(pct=True)

        # === 3. COMBINE ===
        raw_signal = -1 * (rank_left - tsrank_right)

        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1

        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_139(df: pd.DataFrame, window=40):
        """ (-1 * CORR(OPEN, VOLUME, 10)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        corr_10 = df['open'].rolling(window).corr(volume).fillna(0)
        
        raw_signal = -1 * corr_10
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_140(df: pd.DataFrame, window=20):
        """ MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3)) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Term Rank
        rank_open = df['open'].rolling(window).rank(pct=True)
        rank_low = df['low'].rolling(window).rank(pct=True)
        rank_high = df['high'].rolling(window).rank(pct=True)
        rank_close = df['close'].rolling(window).rank(pct=True)
        
        term1 = (rank_open + rank_low) - (rank_high + rank_close)
        decay_8 = term1.ewm(span=8).mean()
        rank_left = decay_8.rolling(window).rank(pct=True)
        
        # 2. Right Term Rank
        tsrank_close_8 = df['close'].rolling(8).rank(pct=True)
        mean_vol_60 = volume.rolling(60).mean()
        tsrank_vol_20 = mean_vol_60.rolling(20).rank(pct=True)
        
        corr_8 = tsrank_close_8.rolling(8).corr(tsrank_vol_20).fillna(0)
        decay_7 = corr_8.ewm(span=7).mean()
        tsrank_right = decay_7.rolling(3).rank(pct=True)
        
        # 3. Minimum and Normalize
        min_rank = np.minimum(rank_left, tsrank_right)
        ranked_signal = min_rank.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_141(df: pd.DataFrame, window=20):
        """ (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. High Rank
        rank_high = df['high'].rolling(window).rank(pct=True)
        
        # 2. Volume Rank
        mean_vol_15 = volume.rolling(15).mean()
        rank_vol_15 = mean_vol_15.rolling(window).rank(pct=True)
        
        # 3. Correlation and Normalize
        corr_9 = rank_high.rolling(9).corr(rank_vol_15).fillna(0)
        rank_corr = corr_9.rolling(window).rank(pct=True)
        
        raw_signal = -1 * rank_corr
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_142(df: pd.DataFrame, window=10):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. First Term Rank
        tsrank_close_10 = df['close'].rolling(10).rank(pct=True)
        rank_1 = -1 * tsrank_close_10.rolling(window).rank(pct=True)
        
        # 2. Second Term Rank
        delta_1 = df['close'].diff(1)
        delta_delta = delta_1.diff(1)
        rank_2 = delta_delta.rolling(window).rank(pct=True)
        
        # 3. Third Term Rank
        mean_vol_20 = volume.rolling(20).mean()
        ratio = volume / (mean_vol_20 + 1e-8)
        tsrank_ratio_5 = ratio.rolling(5).rank(pct=True)
        rank_3 = tsrank_ratio_5.rolling(window).rank(pct=True)
        
        # 4. Multiplicative Combination and Normalize
        raw_signal = rank_1 * rank_2 * rank_3
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_143(df: pd.DataFrame, window=20):
        close = df['close'].values
        prev_close = O.ts_lag(df['close'], 1).values
        returns = (close - prev_close) / (prev_close + 1e-8)
        alpha_values = np.ones(len(df)) 
        
        for i in range(1, len(df)):
            if close[i] > prev_close[i]:
                alpha_values[i] = alpha_values[i-1] * (1 + returns[i])
            else:
                alpha_values[i] = alpha_values[i-1]
        
        raw_signal = pd.Series(alpha_values, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_144(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        amount = df.get('amount', df['close'] * volume)
        
        delay_1 = df['close'].shift(1)
        
        # 1. Condition
        cond_down = df['close'] < delay_1
        
        # 2. Term to average
        part = pd.Series(0.0, index=df.index)
        part[cond_down] = ((df['close'] / (delay_1 + 1e-8)) - 1).abs() / (amount + 1e-8)
        
        # 3. Sum and Count (Average)
        sum_20 = part.rolling(window).sum()
        count_20 = cond_down.rolling(window).sum()
        
        ratio = sum_20 / (count_20 + 1e-8)
        
        # 4. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_145(df: pd.DataFrame, window=20):
        """ (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100 """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Volume means
        mean_9 = volume.rolling(9).mean()
        mean_26 = volume.rolling(26).mean()
        mean_12 = volume.rolling(12).mean()
        
        # 2. Ratio
        ratio = (mean_9 - mean_26) / (mean_12 + 1e-8) * 100
        
        # 3. Normalize
        ranked_signal = ratio.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)
    @staticmethod
    def alpha_popbo_146(df: pd.DataFrame, window=5):
        ret = df['close'].pct_change(1).fillna(0)
        
        # 1. Returns vs SMA deviation
        sma_61 = ret.ewm(alpha=2/61, adjust=False).mean()
        diff_ret_sma = ret - sma_61
        
        # 2. Numerator terms
        mean_diff_20 = diff_ret_sma.rolling(20).mean()
        
        # 3. Denominator terms
        denom_inner_squared = (ret - diff_ret_sma) ** 2
        sma_sq_61 = denom_inner_squared.ewm(alpha=2/61, adjust=False).mean()
        
        # 4. Ratio and Normalize
        raw_signal = (mean_diff_20 * diff_ret_sma) / (sma_sq_61 + 1e-8)
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_147(df: pd.DataFrame, window=20):
        """ REGBETA(MEAN(CLOSE,12),SEQUENCE(12)) """
        mean_close_12 = df['close'].rolling(12).mean()
        
        # 1. Rolling Beta
        def rolling_regbeta(y):
            if len(y) < 12: return np.nan
            x = np.arange(12)
            var_x = np.var(x, ddof=1)
            if var_x == 0: return 0
            cov_xy = np.cov(x, y)[0, 1]
            return cov_xy / var_x
            
        beta_12 = mean_close_12.rolling(12).apply(rolling_regbeta, raw=True)
        
        # 2. Normalize
        ranked_signal = beta_12.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_148(df: pd.DataFrame, window=20):
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Left Rank
        mean_vol_60 = volume.rolling(60).mean()
        sum_9 = mean_vol_60.rolling(9).sum()
        corr_6 = df['open'].rolling(6).corr(sum_9).fillna(0)
        rank_left = corr_6.rolling(window).rank(pct=True)
        
        # 2. Right Rank
        tsmin_open_14 = df['open'].rolling(14).min()
        rank_right = (df['open'] - tsmin_open_14).rolling(window).rank(pct=True)
        
        # 3. Condition
        cond = rank_left < rank_right
        part = pd.Series(0.0, index=df.index)
        part[cond] = -1.0
        
        # 4. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_149(df: pd.DataFrame, window=20):
     
        asset_ret = (df['close'] / O.ts_lag(df['close'], 1)) - 1
        
        market_ret = asset_ret.rolling(window).mean()
        
        covariance = asset_ret.rolling(window).cov(market_ret)
        variance = market_ret.rolling(window).var()
        
      
        raw_beta = covariance / (variance + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_beta, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)
    

    @staticmethod
    def alpha_popbo_150(df: pd.DataFrame, window=20):
        """ (CLOSE+HIGH+LOW)/3*VOLUME """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Volume weighted price
        raw_signal = (df['close'] + df['high'] + df['low']) / 3 * volume
        
        # 2. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_151(df: pd.DataFrame, window=1, window_rank=10):
        """ SMA(CLOSE-DELAY(CLOSE,20),20,1) """
        delta_20 = df['close'].diff(window)
        
        # 1. 20-day SMA of momentum
        sma_20 = delta_20.ewm(alpha=window/window_rank, adjust=False).mean()
        
        # 2. Normalize
        ranked_signal = sma_20.rolling(window_rank).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_152(df: pd.DataFrame, window=20):
        ratio = df['close'] / (df['close'].shift(1) + 1e-8)
        delay_ratio = ratio.shift(1)
        
        # 1. First SMA
        sma_ratio = delay_ratio.ewm(alpha=1/9, adjust=False).mean()
        delay_sma = sma_ratio.shift(1)
        
        # 2. MACD-like means
        mean_12 = delay_sma.rolling(12).mean()
        mean_26 = delay_sma.rolling(26).mean()
        diff = mean_12 - mean_26
        
        # 3. Final SMA
        sma_final = diff.ewm(alpha=1/9, adjust=False).mean()
        
        # 4. Normalize
        ranked_signal = sma_final.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_153(df: pd.DataFrame, window=20):
        """ (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 """
        # 1. Multiple moving averages
        mean_3 = df['close'].rolling(3).mean()
        mean_6 = df['close'].rolling(6).mean()
        mean_12 = df['close'].rolling(12).mean()
        mean_24 = df['close'].rolling(24).mean()
        
        # 2. Average of averages
        raw_signal = (mean_3 + mean_6 + mean_12 + mean_24) / 4
        
        # 3. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_154(df: pd.DataFrame, window=20):
        """ (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18))) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term
        min_vwap_16 = vwap.rolling(16).min()
        term_left = vwap - min_vwap_16
        
        # 2. Right Term
        mean_vol_180 = volume.rolling(180).mean()
        term_right = vwap.rolling(18).corr(mean_vol_180).fillna(0)
        
        # 3. Condition
        cond = term_left < term_right
        part = pd.Series(0.0, index=df.index)
        part[cond] = 1.0
        
        # 4. Normalize
        ranked_signal = part.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_155(df: pd.DataFrame, window=20):
        """ SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) """
        volume = df.get('matchingVolume', df.get('volume', df['close']*0+1))
        
        # 1. Volume SMAs
        sma_13 = volume.ewm(alpha=2/13, adjust=False).mean()
        sma_27 = volume.ewm(alpha=2/27, adjust=False).mean()
        diff = sma_13 - sma_27
        
        # 2. SMA of difference
        sma_diff_10 = diff.ewm(alpha=2/10, adjust=False).mean()
        
        # 3. MACD histogram structure
        raw_signal = diff - sma_diff_10
        
        # 4. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_156(df: pd.DataFrame, window=20):
        vwap = df.get('vwap', (df['high']+df['low']+df['close'])/3)
        
        # 1. Left Term Rank
        delta_vwap_5 = vwap.diff(5)
        decay_3_left = delta_vwap_5.ewm(span=3).mean()
        rank_left = decay_3_left.rolling(window).rank(pct=True)
        
        # 2. Right Term Rank
        price_w = (df['open'] * 0.15) + (df['low'] * 0.85)
        delta_price_w_2 = price_w.diff(2)
        metric = -1 * (delta_price_w_2 / (price_w + 1e-8))
        decay_3_right = metric.ewm(span=3).mean()
        rank_right = decay_3_right.rolling(window).rank(pct=True)
        
        # 3. Max and Normalize
        max_rank = np.maximum(rank_left, rank_right)
        raw_signal = -1 * max_rank
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_157(df: pd.DataFrame, window=20):
        delta_5 = (df['close'] - 1).diff(1)
        rank_1 = delta_5.rolling(window).rank(pct=True)
        rank_2 = (-1 * rank_1).rolling(window).rank(pct=True)
        rank_3 = rank_2.rolling(window).rank(pct=True)
        
        tsmin_2 = rank_3.rolling(window).min()
        sum_1 = tsmin_2.rolling(window).sum()
        
        log_val = np.log(np.maximum(sum_1, 1e-5))
        rank_4 = log_val.rolling(window).rank(pct=True)
        rank_5 = rank_4.rolling(window).rank(pct=True)
        
        prod_1 = rank_5.rolling(window).apply(np.prod, raw=True)
        min_5 = prod_1.rolling(window).min()
        
        # Right term
        ret = df['close'].pct_change(1).fillna(0)
        delay_ret_6 = (-1 * ret).shift(1)
        tsrank_5 = delay_ret_6.rolling(window).rank(pct=True)
        
        # Combine and Normalize
        raw_signal = min_5 + tsrank_5
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)
        

    @staticmethod
    def alpha_popbo_158(df: pd.DataFrame, window=15):
        """ ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE """
        sma_15 = df['close'].ewm(alpha=5/window, adjust=False).mean()
        
        # 1. Deviations from SMA
        term1 = df['high'] - sma_15
        term2 = df['low'] - sma_15
        
        # 2. Ratio
        raw_signal = (term1 - term2) / (df['close'] + 1e-8)
        
        # 3. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_159(df: pd.DataFrame, window=10):
        """ Complex accumulation of terms over 6, 12, 24 periods """
        delay_1 = df['close'].shift(1)
        
        lower = np.minimum(df['low'], delay_1)
        upper = np.maximum(df['high'], delay_1)
        tr = upper - lower
        
        # 1. 6-day
        sum_lower_6 = lower.rolling(6).sum()
        sum_tr_6 = tr.rolling(6).sum()
        term_6 = (df['close'] - sum_lower_6) / (sum_tr_6 + 1e-8) * 12 * 24
        
        # 2. 12-day
        sum_lower_12 = lower.rolling(12).sum()
        sum_tr_12 = tr.rolling(12).sum()
        term_12 = (df['close'] - sum_lower_12) / (sum_tr_12 + 1e-8) * 6 * 24
        
        # 3. 24-day
        sum_lower_24 = lower.rolling(24).sum()
        sum_tr_24 = tr.rolling(24).sum()
        term_24 = (df['close'] - sum_lower_24) / (sum_tr_24 + 1e-8) * 6 * 12
        
        # 4. Weighted Combine
        raw_signal = (term_6 + term_12 + term_24) * 100 / (6*12 + 6*24 + 12*24)
        
        # 5. Normalize
        ranked_signal = raw_signal.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_160(df: pd.DataFrame, window=20):
        """ SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) """
        delay_close_1 = df['close'].shift(1)
        
        # 1. Condition
        cond = df['close'] <= delay_close_1
        
        # 2. Conditional Values
        std_20 = df['close'].rolling(20).std()
        part = pd.Series(0.0, index=df.index)
        part[cond] = std_20
        
        # 3. SMA
        sma_20 = part.ewm(alpha=1/20, adjust=False).mean()
        
        # 4. Normalize
        ranked_signal = sma_20.rolling(window).rank(pct=True)
        signal = (2 * ranked_signal) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_161(df: pd.DataFrame, window=20):
        
        h_l = df['high'] - df['low']
        prev_close = O.ts_lag(df['close'], 1)
        
        pc_h = (prev_close - df['high']).abs()
        pc_l = (prev_close - df['low']).abs()
        
        tr = np.maximum(h_l, np.maximum(pc_h, pc_l))
        
        raw_atr = O.ts_mean(tr, 12)
        
        ranked_signal = O.ts_rank_normalized(raw_atr, window)
        
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_162(df: pd.DataFrame,window = 1, window_rank=10):
       
        delta = df['close'].diff(window)
        gain = np.maximum(delta, 0)
        loss = delta.abs()
        
        avg_gain = gain.ewm(alpha=1/window_rank, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window_rank, adjust=False).mean()
        
        rsi = (avg_gain / (avg_loss + 1e-8)) * 100
        
        min_rsi = O.ts_min(rsi, window_rank)
        max_rsi = O.ts_max(rsi, window_rank)
        
        raw_stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-8)
        
        
        ranked_signal = O.ts_rank_normalized(raw_stoch_rsi, window_rank)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_163(df: pd.DataFrame, window=10):
        volume = df.get('volume', df.get('matchingVolume'))
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        
        rev_ret = -1 * df['close'].pct_change(1)
        
        avg_vol = O.ts_mean(volume, 10)
        
        selling_pressure = df['high'] - df['close']
        
      
        raw_signal = rev_ret * avg_vol * vwap * selling_pressure
        
      
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_164(df: pd.DataFrame, window=20):
      
        delta_close = df['close'] - O.ts_lag(df['close'], 1)
        
        part = np.where(delta_close > 0, 1 / (delta_close + 1e-8), 1.0)
        part = pd.Series(part, index=df.index)
        
       
        range_hl = (df['high'] - df['low']).replace(0, np.nan)
        
        min_part_12 = O.ts_min(part, 12)
        core_expr = (part - min_part_12) / range_hl * 100
        
        raw_signal = core_expr.ewm(span=13).mean() 
        
        # 5. Chuẩn hóa về dải -1 đến 1
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_165(df: pd.DataFrame, window=48):
        mean_48 = O.ts_mean(df['close'], window)
        deviation = df['close'] - mean_48
        
        cusum_dev = deviation.rolling(window).sum() 
       
        p1 = cusum_dev.rolling(window).max()
        p2 = cusum_dev.rolling(window).min()
        rs_range = p1 - p2
        
        std_48 = O.ts_std(df['close'], window)
        
        # R/S Ratio
        raw_signal = rs_range / (std_48 + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        # Đảo chiều theo logic gốc của bạn (-1 *)
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_166(df: pd.DataFrame, window=20):
        
        returns = (df['close'] / O.ts_lag(df['close'], 1)) - 1
        
        mean_ret = O.ts_mean(returns, window)
        diff = returns - mean_ret
        p1 = O.ts_sum(diff**3, window)
        
        std_ret = O.ts_std(returns, window)
        p2 = (std_ret**3) * (window - 1) * (window - 2) / window
        
        raw_skew = p1 / (p2 + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_skew, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_167(df: pd.DataFrame, window=40):
        delta_close = df['close'] - O.ts_lag(df['close'], 1)
    
        positive_flow = np.where(delta_close > 0, delta_close, 0.0)
        positive_flow = pd.Series(positive_flow, index=df.index)
        
        raw_sum = O.ts_sum(positive_flow, 12)
        
        ranked_signal = O.ts_rank_normalized(raw_sum, window)
        
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_168(df: pd.DataFrame, window=20):
      
        volume = df.get('volume', df.get('matchingVolume'))
        
        avg_vol_20 = O.ts_mean(volume, 20)
        
        raw_ratio = volume / (avg_vol_20 + 1e-8)
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_169(df: pd.DataFrame, window=20):
        delta_close = df['close'] - O.ts_lag(df['close'], 1)
        
        smooth_delta = delta_close.ewm(alpha=1/9, adjust=False).mean()
        
        delayed_smooth = O.ts_lag(smooth_delta, 1)
        
        fast_ma = O.ts_mean(delayed_smooth, 12)
        slow_ma = O.ts_mean(delayed_smooth, 26)
        macd_line = fast_ma - slow_ma
        
        raw_signal = macd_line.ewm(alpha=1/10, adjust=False).mean()
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_170(df: pd.DataFrame, window=1, factor=20):
        factor = int(factor)
       
        rank_inv_close = O.ts_rank_normalized(1 / (df['close'] + 1e-8), factor)
        
        vol = df.get('volume', df.get('matchingVolume'))
        vol_ratio = vol / (O.ts_mean(vol, 20) + 1e-8)
        
        high_rank_pressure = O.ts_rank_normalized(df['high'] - df['close'], factor)
        avg_high_5 = O.ts_mean(df['high'], window)
        relative_high = (df['high'] * high_rank_pressure) / (avg_high_5 + 1e-8)
        
        left_term = (rank_inv_close * vol_ratio) * relative_high
     
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        vwap_delta = vwap - O.ts_lag(vwap, window)
        right_term = O.ts_rank_normalized(vwap_delta, factor)
        
        raw_signal = left_term - right_term
        
        ranked_signal = O.ts_rank_normalized(raw_signal, factor)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_171(df: pd.DataFrame, window=40):
        
        lower_part = df['low'] - df['close']
        
        upper_part = df['close'] - df['high']
        
        price_ratio = (df['open'] / (df['close'] + 1e-8)) ** 5
        
        
        raw_signal = (-1 * (lower_part * price_ratio)) / (upper_part - 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_172(df: pd.DataFrame, window=20):
        prev_close = O.ts_lag(df['close'], 1)
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - prev_close).abs()
        l_pc = (df['low'] - prev_close).abs()
        tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
        sum_tr_14 = O.ts_sum(tr, 14)
        
        hd = df['high'] - O.ts_lag(df['high'], 1)
        ld = O.ts_lag(df['low'], 1) - df['low']
        
        plus_dm = np.where((hd > 0) & (hd > ld), hd, 0.0)
        minus_dm = np.where((ld > 0) & (ld > hd), ld, 0.0)
        
        plus_di = O.ts_sum(pd.Series(plus_dm, index=df.index), 14) * 100 / (sum_tr_14 + 1e-8)
        minus_di = O.ts_sum(pd.Series(minus_dm, index=df.index), 14) * 100 / (sum_tr_14 + 1e-8)
        
        # 4. Tính DX (Directional Movement Index)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
        
        raw_adx = O.ts_mean(dx, 6)
        
        ranked_signal = O.ts_rank_normalized(raw_adx, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_173(df: pd.DataFrame, window=5):
        def quick_sma(series, n=40):
            return series.ewm(span=n, adjust=False).mean()

        ema1 = quick_sma(df['close'], 40)
        
        ema2 = quick_sma(ema1, 40)
        
        ema3_log = quick_sma(quick_sma(quick_sma(np.log1p(df['close']), 40), 40), 40)
        ema3 = np.exp(ema3_log) - 1
        
        raw_tema_variant = (3 * ema1) - (2 * ema2) + ema3
        
        
        price_diff = df['close'] - raw_tema_variant
        
        ranked_signal = O.ts_rank_normalized(price_diff, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_174(df: pd.DataFrame, window=5):
        prev_close = O.ts_lag(df['close'], 1)
        cond = df['close'] > prev_close
        
        std_20 = O.ts_std(df['close'], 5)
        
        upside_vol = np.where(cond, std_20, 0.0)
        upside_vol_series = pd.Series(upside_vol, index=df.index)
        
        raw_signal = upside_vol_series.ewm(alpha=1/5, adjust=False).mean()
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_175(df: pd.DataFrame, window=40):
        
        prev_close = O.ts_lag(df['close'], 1)
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - prev_close).abs()
        l_pc = (df['low'] - prev_close).abs()
        
        tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
        
        raw_atr_6 = O.ts_mean(pd.Series(tr, index=df.index), 6)
        
        
        ranked_signal = O.ts_rank_normalized(raw_atr_6, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_176(df: pd.DataFrame, window=20):
       
        low_12 = O.ts_min(df['low'], 12)
        high_12 = O.ts_max(df['high'], 12)
        
        stoch_k = (df['close'] - low_12) / (high_12 - low_12 + 1e-8)
        
        rank_stoch = O.ts_rank_normalized(stoch_k, window)
        
        volume = df.get('volume', df.get('matchingVolume'))
        rank_volume = O.ts_rank_normalized(volume, window)
        
        raw_corr = O.ts_corr(rank_stoch, rank_volume, 6)
        
        return raw_corr.fillna(0)

    @staticmethod
    def alpha_popbo_177(df: pd.DataFrame, window=20):
        peak_index = O.ts_argmax(df['high'], window) - 1
        
        days_since_high = (window - 1) - peak_index
        
        aroon_up = ((window - days_since_high) / window) * 100
        
        ranked_signal = O.ts_rank_normalized(aroon_up, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_178(df: pd.DataFrame,window=1, factor=20):
        factor = int(factor)
        returns = (df['close'] - O.ts_lag(df['close'], window)) / (O.ts_lag(df['close'], 1) + 1e-8)
        
        volume = df.get('volume', df.get('matchingVolume'))
        raw_flow = returns * volume
        
        ranked_signal = O.ts_rank_normalized(raw_flow, factor)
        
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    

    @staticmethod
    def alpha_popbo_179(df: pd.DataFrame, window=20):
      
        volume = df.get('volume', df.get('matchingVolume'))
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        corr_vwap_vol = O.ts_corr(vwap, volume, 4)
        rank_corr1 = O.ts_rank_normalized(corr_vwap_vol, window)
        
        
        rank_low = O.ts_rank_normalized(df['low'], window)
        rank_avg_vol_50 = O.ts_rank_normalized(O.ts_mean(volume, 50), window)
        
        corr_low_vol50 = O.ts_corr(rank_low, rank_avg_vol_50, 12)
        rank_corr2 = O.ts_rank_normalized(corr_low_vol50, window)
     
        raw_signal = rank_corr1 * rank_corr2
        
        final_rank = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * final_rank) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_180(df: pd.DataFrame, window=20):
    
        vol = df.get('volume', df.get('matchingVolume'))
        avg_vol_20 = O.ts_mean(vol, 20)
        
        cond = vol > avg_vol_20

        delta_7 = O.ts_delta(df['close'], 7)
        rank_abs_delta = O.ts_rank_normalized(delta_7.abs(), 60)
        high_vol_signal = -1 * rank_abs_delta * O.sign(delta_7)
        
        low_vol_signal = -1 * O.ts_rank_normalized(vol, window)
        
        # --- Kết hợp ---
        raw_signal = np.where(cond, high_vol_signal, low_vol_signal)
        raw_signal = pd.Series(raw_signal, index=df.index)
        
        final_rank = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * final_rank) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_181(df: pd.DataFrame, window=20):
        
        ret_asset = (df['close'] / O.ts_lag(df['close'], 1)) - 1
        
        mean_ret = O.ts_mean(ret_asset, window)
        excess = ret_asset - mean_ret
        
        numerator = O.ts_sum(O.sign(excess) * (excess ** 2), window)
        
        denominator = O.ts_sum(excess ** 3, window)
        
        raw_signal = numerator / (denominator + 1e-8)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_182(df: pd.DataFrame, window=20):
       
        asset_up = df['close'] > df['open']
        asset_down = df['close'] < df['open']
        
        sma_20 = O.ts_mean(df['close'], 20)
        market_up = sma_20 > O.ts_lag(sma_20, 1)
        market_down = sma_20 < O.ts_lag(sma_20, 1)
        
        sync_condition = (asset_up & market_up) | (asset_down & market_down)
        
        sync_count = pd.Series(sync_condition.astype(int), index=df.index).rolling(window).sum()
        
        raw_ratio = sync_count / window
       
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_183(df: pd.DataFrame, window=24):
        
        def get_rs_range(x):
            cumsum = np.cumsum(x - np.mean(x))
            return np.max(cumsum) - np.min(cumsum)

        r_range = df['close'].rolling(window).apply(get_rs_range, raw=True)
        
        s_std = O.ts_std(df['close'], 24)
        
        rs_ratio = r_range / (s_std + 1e-8)
        
       
        ranked_signal = O.ts_rank_normalized(rs_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_184(df: pd.DataFrame, window=20):
        candle_range = df['open'] - df['close']
        lagged_range = O.ts_lag(candle_range, 1)
      
        corr_long_term = O.ts_corr(lagged_range, df['close'], 200)
        rank_corr = O.ts_rank_normalized(corr_long_term, window)
        
        rank_current_range = O.ts_rank_normalized(candle_range, window)
        
        raw_signal = rank_corr + rank_current_range
        
        final_rank = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * final_rank) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_popbo_185(df: pd.DataFrame, window=20):
        
        body_relative_range = 1 - (df['open'] / (df['close'] + 1e-8))
        
        raw_signal = -1 * (body_relative_range ** 2)
        
       
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_186(df: pd.DataFrame, window=20):
        prev_close = O.ts_lag(df['close'], 1)
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - prev_close).abs()
        l_pc = (df['low'] - prev_close).abs()
        tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
        sum_tr_14 = O.ts_sum(tr, 14)
        
        # 2. Tính Directional Movement (HD và LD)
        hd = df['high'] - O.ts_lag(df['high'], 1)
        ld = O.ts_lag(df['low'], 1) - df['low']
        
        # Lọc tín hiệu +DM và -DM (Dùng np.where thay cho copy/loc)
        plus_dm = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0.0), index=df.index)
        
        # 3. Tính DI+ và DI-
        plus_di = O.ts_sum(plus_dm, 14) * 100 / (sum_tr_14 + 1e-8)
        minus_di = O.ts_sum(minus_dm, 14) * 100 / (sum_tr_14 + 1e-8)
        
        # 4. Tính DX và ADX (Mean 6)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
        adx_current = O.ts_mean(dx, 6)
        
        # 5. Lấy trung bình ADX hiện tại và ADX trễ 6 phiên
        adx_lagged = O.ts_lag(adx_current, 6)
        raw_signal = (adx_current + adx_lagged) / 2
        
        # 6. Chuẩn hóa về dải -1 đến 1
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_187(df: pd.DataFrame, window=20):
        
        prev_open = O.ts_lag(df['open'], 1)
        
        cond = df['open'] > prev_open
        
       
        impulse = np.maximum((df['high'] - df['open']), (df['open'] - prev_open))
        
        bull_movement = np.where(cond, impulse, 0.0)
        bull_movement_series = pd.Series(bull_movement, index=df.index)
        
        raw_sum = O.ts_sum(bull_movement_series, 20)
        
       
        ranked_signal = O.ts_rank_normalized(raw_sum, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_188(df: pd.DataFrame, window=20):
      
        hl_range = df['high'] - df['low']
        avg_range = O.ts_mean(hl_range, 11)
        
        raw_deviation = ((hl_range - avg_range) / (avg_range + 1e-8)) * 100
      
        ranked_signal = O.ts_rank_normalized(raw_deviation, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_189(df: pd.DataFrame, window=20):
        
        ma_6 = O.ts_mean(df['close'], 6)
        
        abs_deviation = (df['close'] - ma_6).abs()
        
        raw_mad = O.ts_mean(abs_deviation, 6)
       
        ranked_signal = O.ts_rank_normalized(raw_mad, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_190(df: pd.DataFrame, window=20):
        
        ret = (df['close'] / O.ts_lag(df['close'], 1)) - 1
        
        target_ret = (df['close'] / O.ts_lag(df['close'], 19))**(1/20) - 1
        
        diff = ret - target_ret
        diff_sq = diff**2
        
        upside_mask = ret > target_ret
        upside_sum_sq = pd.Series(np.where(upside_mask, diff_sq, 0.0), index=df.index).rolling(window).sum()
        upside_count = pd.Series(upside_mask.astype(int), index=df.index).rolling(window).sum()
        
        downside_mask = ret < target_ret
        downside_sum_sq = pd.Series(np.where(downside_mask, diff_sq, 0.0), index=df.index).rolling(window).sum()
        downside_count = pd.Series(downside_mask.astype(int), index=df.index).rolling(window).sum()
        
        
        numerator = (upside_count - 1) * downside_sum_sq
        denominator = downside_count * upside_sum_sq
        
        raw_ratio = np.log((numerator + 1e-6) / (denominator + 1e-6))
        raw_ratio = pd.Series(raw_ratio, index=df.index)
        
        ranked_signal = O.ts_rank_normalized(raw_ratio, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_popbo_191(df: pd.DataFrame, window=20):
        
        vol = df.get('volume', df.get('matchingVolume'))
        
        corr_vol_low = O.ts_corr(O.ts_mean(vol, 20), df['low'], 5)
        
        
        mid_price = (df['high'] + df['low']) / 2
        price_bias = (mid_price - df['close']) / (df['close'] + 1e-8)
        
        raw_signal = corr_vol_low + price_bias
        
        ranked_signal = O.ts_rank_normalized(raw_signal, window)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    # FACTOR MINOR
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
    def alpha_factor_miner_010(df: pd.DataFrame, window=5, factor=10):
        factor = int(factor)
       
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        delta_ret = returns_ser.diff(window)
        
        std_ret = returns_ser.rolling(factor).std(ddof=0)
       
        raw_ratio = delta_ret / (std_ret + 1e-8)
        
      
        raw_signal = -1 * raw_ratio.fillna(0)
        
        ranked_signal = O.ts_rank_normalized(raw_signal, 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_011(df: pd.DataFrame, window=5, factor=20):
        factor = int(factor)
        close_vals = df['close'].values
        returns = np.diff(close_vals, prepend=close_vals[0]) / (close_vals + 1e-8)
        returns_ser = pd.Series(returns, index=df.index)
        
        std_10 = returns_ser.rolling(window).std(ddof=0)
        std_25 = returns_ser.rolling(factor).std(ddof=0)
        
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
    def alpha_factor_miner_013(df: pd.DataFrame, window=12, factor=3):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        price_vwap_ratio = df['close'] / (vwap + 1e-8)
        
        
        smoothed_ratio = price_vwap_ratio.ewm(span=window, adjust=False).mean()
        
        
        delta_ratio = smoothed_ratio - smoothed_ratio.shift(factor)
        
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
    def alpha_factor_miner_031(df: pd.DataFrame, window=18, factor=3):
        factor = int(factor)
        def get_last_resid(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            line = slope * (len(y) - 1) + intercept
            return y[-1] - line

        resid = df['close'].rolling(window=window).apply(get_last_resid, raw=True)
        
        
        delta_resid = resid - resid.shift(factor)
        
        
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
    def alpha_factor_miner_034(df: pd.DataFrame, window=16, factor=4):
        factor = int(factor)
        def get_last_resid(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            # Tính giá trị dự báo tại điểm cuối cùng
            line_val = slope * (len(y) - 1) + intercept
            return y[-1] - line_val

        # Sử dụng rolling apply để tính phần dư tại mỗi bước
        resid = df['close'].rolling(window=window).apply(get_last_resid, raw=True)
        
        # 2. Tính sự thay đổi của Residual sau 4 phiên
        delta_resid = resid - resid.shift(factor)
        
        
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
    def alpha_factor_miner_076(df: pd.DataFrame,window = 20, factor=1):
        factor = int(factor)
     
        df_vwap = O.compute_vwap(df.copy(), window)
        vwap = df_vwap['vwap']
        
        price_to_vwap_ratio = df['close'] / (vwap + 1e-8)
       
        ratio_delta = price_to_vwap_ratio.diff(factor).fillna(0)
    
        raw_signal = O.ts_rank_normalized(ratio_delta, window)
        
        signal = (2 * raw_signal) - 1
        
        return signal.fillna(0)

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
    def alpha_factor_miner_081(df: pd.DataFrame, window=20, factor=3):
        factor = int(factor)
        price_diff = df['close'].diff(factor).fillna(0)
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
        
        ts_max_h = df['high'].rolling(range_window).max()
        ts_min_l = df['low'].rolling(range_window).min()
        price_range = (ts_max_h - ts_min_l).fillna(0)
        
        smooth_range = price_range.ewm(span=ema_window, adjust=False).mean()
        
      
        ratio = amt_std / (smooth_range + 1e-6)
        
        ranked_signal = O.ts_rank_normalized(ratio.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return -signal.fillna(0)

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
        
        return signal.fillna(0)

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
        
        return -signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_104(df: pd.DataFrame, window=25):
        """
        ID: 28
        Name: relative_volume_shadow_imbalance
        Formula: ($volume/(TS_MAX($volume, 25) + 1e-8)) * 
                 (($high - $open)/$open - ($open - $low)/$open)
        Logic: Trọng số hóa sự mất cân bằng bóng nến bằng tỷ lệ Volume hiện tại so với đỉnh 25 phiên.
        """
        # 1. Tính toán Relative Volume Intensity
        # Giá trị nằm trong khoảng [0, 1]. 1 nghĩa là Volume đang ở đỉnh 25 phiên.
        max_vol = df['matchingVolume'].rolling(window).max()
        vol_intensity = df['matchingVolume'] / (max_vol + 1e-8)
        
        # 2. Tính toán Shadow Imbalance (Bất đối xứng bóng nến)
        # (Bóng trên / Open) - (Bóng dưới / Open)
        upper_shadow = (df['high'] - df['open']) / (df['open'] + 1e-8)
        lower_shadow = (df['open'] - df['low']) / (df['open'] + 1e-8)
        imbalance = upper_shadow - lower_shadow
        
        # 3. Kết hợp bằng phép nhân
        # Ý nghĩa: Nếu có sự lệch bóng nến cực lớn nhưng Vol thấp, tín hiệu sẽ bị triệt tiêu.
        # Tín hiệu chỉ bùng nổ khi "Rút chân/Đẩy đỉnh" đi kèm Vol đột biến.
        raw_signal = vol_intensity * imbalance
        
        # 4. Chuẩn hóa Rank [-1, 1]
        ranked_signal = O.ts_rank_normalized(raw_signal.fillna(0), 20)
        signal = (2 * ranked_signal) - 1
        
        return signal.fillna(0)

    @staticmethod
    def alpha_factor_miner_115(df: pd.DataFrame, window: int = 2, factor: int = 20) -> pd.Series:
        factor = int(factor)
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        price_range = high_max - low_min
        
        y = (df['close'] - low_min) / price_range.replace(0, np.nan)
        
        x = df['matchingVolume'].astype(float)
        
        rolling_cov_xy = y.rolling(window=factor).cov(x)
        rolling_var_x = x.rolling(window=factor).var()
        
        beta = rolling_cov_xy / rolling_var_x.replace(0, np.nan)
        alpha = y.rolling(window=factor).mean() - beta * x.rolling(window=factor).mean()
        
        # Residual: epsilon = y - (beta * x + alpha)
        residual = y - (beta * x + alpha)
        
        rolling_mean_res = residual.rolling(window=factor).mean()
        rolling_std_res = residual.rolling(window=factor).std()
        
        z_score = (residual - rolling_mean_res) / rolling_std_res.replace(0, np.nan)
        
        alpha_final = (-z_score).clip(-1, 1)
        
        return -alpha_final.ffill().fillna(0)

    @staticmethod
    def alpha_mining_001_rank(df, window=13, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['close'] - vwap) / vwap.rolling(window).std()
        raw = raw.ffill()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_mining_001_tanh(df, window=13, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['close'] - vwap) / vwap.rolling(window).std()
        raw = raw.ffill()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_mining_001_zscore(df, window=13, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['close'] - vwap) / vwap.rolling(window).std()
        raw = raw.ffill()
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_mining_001_sign(df, window=13, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['close'] - vwap) / vwap.rolling(window).std()
        raw = raw.ffill()
        normalized = np.sign(raw)
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_mining_001_wf(df, window=13, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['close'] - vwap) / vwap.rolling(window).std()
        raw = raw.ffill()
        p1 = 0.05
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_popbo_advance_v2_026_tanh(df, window=20, sub_window=3):
        # Raw calculation
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        close = df['close']
        part1 = (close.rolling(sub_window).sum() / sub_window) - close
        part2 = vwap.rolling(window).corr(close.shift(5))
        raw = part1 + part2
        # Normalization: Dynamic Tanh (Case B)
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-9))
        normalized = normalized.ffill().fillna(0)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_advance_v2_003_rank(df, window=1, rank_window=20):
        # Raw calculation
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        cond_up = close > prev_close
        cond_down = close < prev_close
        cond_eq = close == prev_close
        min_val = pd.concat([low, prev_close], axis=1).min(axis=1)
        max_val = pd.concat([high, prev_close], axis=1).max(axis=1)
        raw = np.where(cond_eq, 0, np.where(cond_up, close - min_val, close - max_val))
        raw_series = pd.Series(raw, index=df.index)
        # Rolling sum
        sum_raw = raw_series.rolling(window, min_periods=1).sum()
        # Normalization: Rolling Rank (Case A)
        signal = (sum_raw.rolling(rank_window, min_periods=1).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_popbo_advance_v2_013_wf(df, window=30):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['high'] * df['low']).pow(0.5) - vwap
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_advance_v2_013_tanh(df, window=10):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['high'] * df['low']).pow(0.5) - vwap
        std_dev = raw.rolling(window).std()
        normalized = np.tanh(raw / std_dev.where(std_dev != 0, 1))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_advance_v2_026_wf(df, window=40, sub_window=3):
        # Raw calculation
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        close = df['close']
        part1 = (close.rolling(sub_window).sum() / sub_window) - close
        part2 = vwap.rolling(window).corr(close.shift(5))
        raw = part1 + part2
        # Normalization: Winsorized Fisher (Case E) - Hardcoded quantile and winsor window
        p1 = 0.05  # Hardcoded quantile threshold
        p2 = 100   # Hardcoded winsorization window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform approximation
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_003_tanh(df, window=1, factor=10):
        factor = int(factor)
        # Logic gốc: Tính tổng lũy kế trong 6 ngày của chênh lệch giá điều chỉnh.
        # Chuẩn hóa B (Dynamic Tanh): Giữ lại cường độ (magnitude) của tín hiệu.
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        cond_up = close > delay_close
        cond_down = close < delay_close
        min_low_delay = pd.concat([low, delay_close], axis=1).min(axis=1)
        max_high_delay = pd.concat([high, delay_close], axis=1).max(axis=1)
        raw = np.where(close == delay_close, 0,
                    np.where(cond_up, close - min_low_delay,
                                close - max_high_delay))
        raw_series = pd.Series(raw, index=df.index)
        raw_sum = raw_series.rolling(window).sum()
        std_dev = raw_sum.rolling(factor).std().replace(0, np.nan)
        normalized = np.tanh(raw_sum / std_dev)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_007_wf(df, window=40, factor=1):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        volume = df.get('matchingVolume', df.get('volume', 1))

        diff = vwap - df['close']

        max_diff = diff.rolling(window=3).max()
        min_diff = diff.rolling(window=3).min()

        rank_max = max_diff.rolling(window=window).rank(pct=True)
        rank_min = min_diff.rolling(window=window).rank(pct=True)

        delta_vol = volume.diff(3)
        rank_delta = delta_vol.rolling(window=factor).rank(pct=True)

        raw = (rank_max + rank_min) * rank_delta

        p1 = 0.05
        p2 = window
        low = raw.rolling(window=p2).quantile(p1)
        high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)

        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_011_tanh(df, window=1, factor=100):
        factor = int(factor)
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan)
        raw = ((close - low) - (high - close)) / hl_range * volume
        raw_sum = raw.rolling(window).sum()
        normalized = np.tanh(raw_sum / raw_sum.rolling(factor).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_013_zscore(df, window=30):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (df['high'] * df['low'])**0.5 - vwap
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-9)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_014_zscore(df, window=5):
        close = df['close']
        raw = close - close.shift(5)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        z_score = (raw - mean) / std.where(std != 0, np.nan)
        normalized = z_score.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_017_tanh(df, window=20, factor=2):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        max_vwap = vwap.rolling(window).max()
        raw = (vwap - max_vwap).rank(pct=True)
        delta_close = df['close'].diff(factor)
        raw = raw * delta_close
        raw = raw.ffill()
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-9))
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_031_tanh(df, window=3, factor=10):
        factor = int(factor)
        close = df['close']
        raw = (close - close.rolling(window).mean()) / (close.rolling(window).mean() + 1e-9) * 100
        normalized = np.tanh(raw / (raw.rolling(factor).std() + 1e-9))
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_popbo_new_034_zscore(df, window=3, factor=70):
        factor = int(factor)
        close = df['close']
        raw = close.rolling(window).mean() / close
        rolling_mean = raw.rolling(factor).mean()
        rolling_std = raw.rolling(factor).std().replace(0, np.nan)
        z_score = (raw - rolling_mean) / rolling_std
        normalized = z_score.clip(-1, 1)
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_popbo_new_047_sign(df, window=5, factor=10):
        factor = int(factor)
        high = df['high']
        low = df['low']
        close = df['close']
        tsmax_high = high.rolling(window).max()
        tsmin_low = low.rolling(window).min()
        raw = (tsmax_high - close) / (tsmax_high - tsmin_low + 1e-9) * 100
        sma = raw.ewm(span=factor, adjust=False).mean()
        diff = sma.diff()
        normalized = pd.Series(np.sign(diff), index=df.index)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_048_wf(df, window=10, factor=40):
        factor = int(factor)
        p1 = 0.05
        p2 = 100
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sign1 = np.sign(close - close.shift(1))
        sign2 = np.sign(close.shift(1) - close.shift(2))
        sign3 = np.sign(close.shift(2) - close.shift(3))
        raw_sum = sign1 + sign2 + sign3
        raw_rank = raw_sum.rolling(window=window, min_periods=1).rank(pct=True)
        raw = -1 * raw_rank * volume.rolling(window=factor, min_periods=1).sum() / (volume.rolling(window=window, min_periods=1).sum() + 1e-9)
        low = raw.rolling(window=p2, min_periods=1).quantile(p1)
        high = raw.rolling(window=p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_054_rank(df, window=40, factor=30):
        factor = int(factor)
        # Raw components
        close_open_diff = df['close'] - df['open']
        abs_close_open = np.abs(close_open_diff)
        std_abs = abs_close_open.rolling(window).std()
        corr_coef = df['close'].rolling(window).corr(df['open'])
        # Raw alpha
        raw = -1 * (std_abs + close_open_diff + corr_coef)
        # Normalization: Rolling Rank (Case A)
        normalized = (raw.rolling(factor).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_059_wf(df, window=1, factor=60):
        factor = int(factor)
        # Raw calculation
        close = df['close']
        low = df['low']
        high = df['high']
        delay_close = close.shift(1)
        cond1 = close == delay_close
        cond2 = close > delay_close
        min_val = pd.concat([low, delay_close], axis=1).min(axis=1)
        max_val = pd.concat([high, delay_close], axis=1).max(axis=1)
        raw = pd.Series(np.where(cond1, 0, np.where(cond2, close - min_val, close - max_val)), index=df.index)
        raw_sum = raw.rolling(window, min_periods=1).sum()
        # Normalization: Winsorized Fisher (Case E) for heavy tails
        p1 = 0.05  # Hardcoded quantile threshold
        low_bound = raw_sum.rolling(factor).quantile(p1)
        high_bound = raw_sum.rolling(factor).quantile(1 - p1)
        winsorized = raw_sum.clip(lower=low_bound, upper=high_bound)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_065_wf(df, window=3, factor=100):
        factor = int(factor)
        close = df['close']
        raw = close.rolling(window).mean() / close
        p1 = 0.05
        low = raw.rolling(factor).quantile(p1)
        high = raw.rolling(factor).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_120_rank(df, window=20):
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw = (vwap - df['close']) / (vwap + df['close'] + 1e-9)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_124_rank(df, window=10, factor=30):
        factor = int(factor)
        close = df['close']
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        raw_diff = close - vwap
        tsmax = close.rolling(window).max()
        rank_tsmax = tsmax.rolling(window).rank(pct=True)
        
        decay_factor = rank_tsmax.rolling(factor).mean()
        decay_factor = decay_factor.replace(0, np.nan)
        raw = raw_diff / decay_factor
        param = max(20, window // 2)  
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        normalized = normalized.ffill().fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_163_tanh(df, window=10):
        # Logic gốc tương tự
        ret = df['close'].pct_change()
        mean_vol = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        high_close = df['high'] - df['close']
        raw = (-1 * ret) * mean_vol * vwap * high_close
        raw = raw.ffill()
        # Dynamic Tanh: giữ cường độ
        param = max(window, 10)
        std = raw.rolling(param).std().replace(0, np.nan)
        normalized = np.tanh(raw / std)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_popbo_new_178_rank(df, window=25):
        # Raw: (close - prev_close) / prev_close * volume
        raw = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan) * df.get('matchingVolume', df.get('volume', 1))
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_popbo_new_191_wf(df, window=30, factor=30):
        factor = int(factor)
        mean_vol = df['matchingVolume'].rolling(window).mean()
        corr_term = mean_vol.rolling(factor).corr(df['low'])
        hl_avg = (df['high'] + df['low']) / 2
        raw = (corr_term + hl_avg) - df['close']
        # Normalization: Winsorized Fisher (Case E)
        p1 = 0.05  # Hardcoded quantile for winsorization
        p2 = max(window, factor)  # Rolling window for quantile calculation
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform approximation
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_001_rank(df, window=20, factor=20):
        factor=int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        vwap_std = vwap.rolling(window).std()
        raw = -(df['close'] - vwap) / vwap_std.replace(0, np.nan)
        raw = raw.ffill()
        normalized = (raw.rolling(factor).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_001_tanh(df, window=20, factor=90):
        factor=int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        vwap_std = vwap.rolling(window).std()
        raw = -(df['close'] - vwap) / vwap_std.replace(0, np.nan)
        raw = raw.ffill()
        rolling_std = raw.rolling(factor).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_001_zscore(df, window=20, factor=30):
        factor=int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        vwap_std = vwap.rolling(window).std()
        raw = -(df['close'] - vwap) / vwap_std.replace(0, np.nan)
        raw = raw.ffill()
        rolling_mean = raw.rolling(factor).mean()
        rolling_std = raw.rolling(factor).std().replace(0, np.nan)
        zscore = (raw - rolling_mean) / rolling_std
        normalized = zscore.clip(-1, 1)
        return -normalized.fillna(0)

    

    @staticmethod
    def alpha_factor_miner_new_001_wf(df, window=20, factor=30):
        factor = int(factor)
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        vwap_std = vwap.rolling(window).std()
        raw = -(df['close'] - vwap) / vwap_std.replace(0, np.nan)
        raw = raw.ffill()
        p1 = 0.05
        low = raw.rolling(factor).quantile(p1)
        high = raw.rolling(factor).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_076_wf(df, window=100, factor=30):
        factor = int(factor)
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (close - low) / (high - low + 1e-9)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        intercept = raw.rolling(window).mean() - slope * days.rolling(window).mean()
        resid = raw - (slope * days + intercept)
        signal = -resid
        p1 = 0.05
        p2 = factor
        low_win = signal.rolling(p2).quantile(p1)
        high_win = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low_win, upper=high_win, axis=0)
        normalized = np.arctanh(((winsorized - low_win) / (high_win - low_win + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_076_tanh(df, window=70, factor=20):
        factor=int(factor)
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (close - low) / (high - low + 1e-9)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        intercept = raw.rolling(window).mean() - slope * days.rolling(window).mean()
        resid = raw - (slope * days + intercept)
        signal = -resid
        normalized = np.tanh(signal / signal.rolling(factor).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_076_zscore(df, window=90, factor=20):
        factor=int(factor)
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (close - low) / (high - low + 1e-9)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        intercept = raw.rolling(window).mean() - slope * days.rolling(window).mean()
        resid = raw - (slope * days + intercept)
        signal = -resid
        normalized = ((signal - signal.rolling(factor).mean()) / signal.rolling(factor).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_factor_miner_new_213_tanh(df, window=60, factor=5):
        factor=int(factor)
        # Trường hợp B: Dynamic Tanh, giữ lại cường độ.
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        amt = df.get('amount', close * volume)
        # TsArgMin xấp xỉ bằng rank
        close_rank_pct = close.rolling(window).rank(pct=True)
        ts_argmin = (1 - close_rank_pct) * (window - 1)
        # TsEntropy dùng sliding window
        amt_arr = amt.values
        shape = amt_arr.shape[0] - window + 1
        if shape <= 0:
            return pd.Series(np.nan, index=df.index)
        amt_windows = np.lib.stride_tricks.sliding_window_view(amt_arr, window)
        eps = 1e-12
        p = amt_windows / (amt_windows.sum(axis=1, keepdims=True) + eps)
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        entropy_series = pd.Series(np.nan, index=df.index)
        entropy_series.iloc[window-1:] = entropy
        entropy_series = entropy_series.ffill()
        # Regression Residual
        y = ts_argmin
        x = entropy_series
        cov_xy = y.rolling(factor).cov(x)
        var_x = x.rolling(factor).var()
        beta = cov_xy / var_x.replace(0, np.nan)
        mean_y = y.rolling(factor).mean()
        mean_x = x.rolling(factor).mean()
        alpha = mean_y - beta * mean_x
        y_pred = beta * x + alpha
        resi = y - y_pred
        raw = -resi
        # Chuẩn hóa B: Dynamic Tanh
        param = window  # dùng window làm tham số rolling std
        normalized = np.tanh(raw / raw.rolling(param).std().replace(0, np.nan))
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_001_rank(df, window=20, factor=0.1):
        factor = float(factor)
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(factor)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Rolling Rank (Case A) - removes noise, uniform distribution
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_tanh(df, window=20, factor=0.1):
        factor = float(factor)
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(factor)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Dynamic Tanh (Case B) - preserves magnitude
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-9))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_wf(df, window=10, factor=0.1):
        factor = float(factor)
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(factor)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Winsorized Fisher (Case E) - heavy tails, preserve distribution
        p1 = 0.05  # Hardcoded winsorization percentile
        p2 = 60    # Hardcoded rolling window for quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(low, high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_004_tanh(df, window=70, factor=1):
        factor=int(factor)
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa B: Dynamic Tanh để giữ lại cường độ (magnitude).
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa B: Dynamic Tanh
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0).clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_004_zscore(df, window=70, factor=1):
        factor =  int(factor)
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa C: Rolling Z-Score/Clip cho các công thức Spread/Oscillator.
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa C: Rolling Z-Score với clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0) * factor


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
            'alpha_zscore', 'alpha_questionable', 'alpha_bbb', 'alpha_keltner', 'alpha_popbo_advance_v2_026_tanh', 'alpha_popbo_advance_v2_003_rank',
            'alpha_kema', "alpha_418", "alpha_donchian_channel", 'alpha_popbo_advance_v2_013_wf', 'alpha_popbo_advance_v2_013_tanh', 'alpha_popbo_advance_v2_026_wf',
            "alpha_rti", "alpha_ursi","alpha_101_volume","alpha_volume_weighted_z_score",
            "alpha_101_volume_smoothed","alpha_101_trend_confirm","alpha_101_stddev_normalized",
            "alpha_101_vwap_vol_rank","alpha_101_mean_reversion","alpha_101_corr_weighted",
            "alpha_101_trend_strength_weighted","alpha_101_positional_combo","alpha_101_regime_filter",
            "alpha_101_acceleration","alpha_101_oi_confirm","alpha_101_gap_filtered","alpha_101_body_wick_ratio",
            "alpha_101_decay_smoothed","alpha_101_plus_alpha_008","alpha_101_asymmetric","alpha_101_zscore",
            "alpha_101_rank_combo","alpha_101_overnight_confirm","alpha_101_powered","alpha_101_day_of_week_filter",
            "alpha_new_003_v1","alpha_new_003_v2","alpha_new_003_v3","alpha_new_003_v4","alpha_new_003_v5",
            "alpha_new_005_up1","alpha_new_008_v1","alpha_new_008_v2","alpha_new_008_v3","alpha_new_008_v4","alpha_new_008_v5",
            'alpha_full_factor_062_zscore_clipping',"alpha_full_factor_095_regime_adaptive", 'alpha_full_factor_066_liq_accel',
            'alpha_full_factor_090_reg_adaptive', 'alpha_full_factor_099_eff_macd', 'alpha_full_factor_046_dynamic_reversion',
            'alpha_full_factor_007_volume_breakout_trend', 'alpha_full_factor_085_rank_vol_efficiency', 'alpha_full_factor_b08_signed_power_compress',
            'alpha_mining_001_rank', 'alpha_mining_001_tanh', 'alpha_mining_001_zscore', 'alpha_mining_001_sign', 'alpha_mining_001_wf',

            'alpha_popbo_new_003_tanh', 'alpha_popbo_new_007_wf', 'alpha_popbo_new_011_tanh', 'alpha_popbo_new_013_zscore', 'alpha_popbo_new_014_zscore',
            'alpha_popbo_new_017_tanh', 'alpha_popbo_new_031_tanh', 'alpha_popbo_new_034_zscore', 'alpha_popbo_new_047_sign', 'alpha_popbo_new_048_wf',
            'alpha_popbo_new_054_rank', 'alpha_popbo_new_059_wf', 'alpha_popbo_new_065_wf', 'alpha_popbo_new_120_rank', 'alpha_popbo_new_124_rank',
            'alpha_popbo_new_163_tanh', 'alpha_popbo_new_178_rank', 'alpha_popbo_new_191_wf',

            'alpha_factor_miner_new_001_rank', 'alpha_factor_miner_new_001_tanh', 'alpha_factor_miner_new_001_zscore', 'alpha_factor_miner_new_001_wf',
            'alpha_factor_miner_new_076_tanh', 'alpha_factor_miner_new_076_zscore', 'alpha_factor_miner_new_076_wf', 'alpha_factor_miner_new_213_tanh',


            'alpha_quanta_001_rank', 'alpha_quanta_001_tanh', 'alpha_quanta_001_wf', 'alpha_quanta_004_tanh', 'alpha_quanta_004_zscore'
        ]

        custom_c_list = [f"c{str(i).rjust(2, '0')}" for i in range(1, 51)]
        new_alpha_list = [f"alpha_new_{str(name).rjust(3, '0')}" for name in list(range(1, 101))]
        alpha_full_factor = [f"alpha_full_factor_{str(i).rjust(3, '0')}" for i in range(1, 110)]
        alpha_popbo = [f"alpha_popbo_{str(i).rjust(3, '0')}" for i in range(1, 199)]
        alpha_factor_miner = [f"alpha_factor_miner_{str(i).rjust(3, '0')}" for i in range(0, 300)]
        
        for alpha_name in base_list + custom_c_list + new_alpha_list + alpha_full_factor + alpha_popbo + alpha_factor_miner:
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




