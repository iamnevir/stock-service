import pandas as pd
import alpha_func_lib
import numpy as np
from matplotlib import pyplot as plt
import json
from itertools import product
import pickle
from random import shuffle
from bokeh.util.browser import view
from plotly import express as px

from random import shuffle

import warnings



class U:
    @staticmethod
    def report_error(e, function_name="unnamed_foo"):
        from datetime import datetime as dt
        from datetime import datetime
        from traceback import print_exc
        print(datetime.now().strftime("%H:%M:%S"), end=" ")
        print(f"GREEN{function_name}()ENDC Có lỗi xảy ra: REDBG{e}ENDC type: REDBG{type(e).__name__}ENDC"
              f"\nArguments:BLUE{e.args}ENDC".replace("REDBG", '\33[41m').
              replace("ENDC", '\033[0m').replace("GREEN", '\33[32m').replace("BLUE", '\33[34m'))
        print_exc()
        print(f"\x1b[95m{dt.now().strftime('%H:%M:%S')} "
              f"\x1b[102m\x1b[30mError ({e}) handled (hopefully), "
              f"continuing as if nothing happened...\x1b30m")


class C:
    L0 = 4.4
    UNREASONABLE_SHARPE = -999
    MAX_L1 = 10
    CUT_MORNING_SESSION = 'cut_morning_session'
    CUT_ALL_BUT_LAST_15M = 'cut_all_but_last_15m'


class LIB_RESOURCES:
    DIR_PLOTED_ALPHAS = '/tmp/plotted_alphas'


class Alpha_Domains2:
    FLAG_SAVE_DF_ALPHA = True

    @staticmethod
    def compute_mdd_vectorized(df_1d, verbosity=1):
        import numpy as np
        def max_so_far(arr):
            max_sofar = np.maximum.accumulate(arr)
            return max_sofar

        if 'cumNetProfit' in df_1d:
            ds = df_1d['cumNetProfit']
        else:
            ds = df_1d['netProfit'].cumsum()
        high_water_mark = max_so_far(ds)
        lst_cdd = high_water_mark - ds
        lst_mdd = max_so_far(lst_cdd)
        lst_cdd_pct = lst_cdd / df_1d['open']
        lst_mdd_pct = max_so_far(lst_cdd_pct)
        cdd, mdd = lst_cdd[-1], lst_mdd[-1]
        cdd_pct, mdd_pct = lst_cdd_pct[-1], lst_mdd_pct[-1]

        if verbosity >= 2:
            print(f"mdd=\x1b[93m{mdd:,.2f}\x1b[0m mdd_pct={100 * mdd_pct:.1f}% "
                  f"cdd=\x1b[93m{cdd:,.2f}\x1b[0m cdd_pct={100 * cdd_pct:.1f}%")

        return mdd, mdd_pct, cdd, cdd_pct

    @staticmethod
    def detect_book_cutting(df):

        ds = df.groupby('time')['turnover'].mean()
        morning_start = '10:15:00'
        afternoon_end = '14:28:00'
        mid_day = '13:00:00'
        ds = ds[(ds.index < afternoon_end) & (ds.index > morning_start)]
        ds = ds[ds > 0]

        if ds.index[0] > mid_day:
            s = ds.index[0]
            cut_book_text = f'{s[3:5]}+'

        elif ds.index[-1] < mid_day:
            s = ds.index[-1]
            cut_book_text = f'{s[3:5]}-'
        else:
            cut_book_text = ''
        if len(cut_book_text) > 0: cut_book_text += ' '
        return cut_book_text

    @staticmethod
    def compute_performance_from_df_1d(df_1d, flag_compute_mdd=True, l0=C.L0,
                                       return_dic=False):
        def compute_mdd_vectorized(df_1d, verbosity=1):
            import numpy as np
            def max_so_far(arr):
                max_sofar = np.maximum.accumulate(arr)
                return max_sofar

            if 'cumNetProfit' in df_1d: ds = df_1d['cumNetProfit']
            else: ds = df_1d['netProfit'].cumsum()
            high_water_mark = max_so_far(ds)
            lst_cdd = high_water_mark - ds
            lst_mdd = max_so_far(lst_cdd)
            lst_cdd_pct = lst_cdd / df_1d['open']
            lst_mdd_pct = max_so_far(lst_cdd_pct)
            cdd, mdd = lst_cdd[-1], lst_mdd[-1]
            cdd_pct, mdd_pct = lst_cdd_pct[-1], lst_mdd_pct[-1]

            if verbosity >= 2:
                print(f"mdd=\x1b[93m{mdd:,.2f}\x1b[0m mdd_pct={100 * mdd_pct:.1f}% "
                      f"cdd=\x1b[93m{cdd:,.2f}\x1b[0m cdd_pct={100 * cdd_pct:.1f}%")

            return mdd, mdd_pct, cdd, cdd_pct

        ds = df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
        num_days = len(ds)
        #####################################################################################
        sharpe = ds.mean() / ds.std() * 250 ** 0.5
        aroe = ds.sum() * 250 / len(ds)
        profit_factor = -ds[ds > 0].sum() / ds[ds < 0].sum()
        last = df_1d['netProfit'].iloc[-1]
        last_day = df_1d.index[-1]
        ppc = df_1d['netProfit'].sum() / df_1d['turnover'].sum()
        tvr = df_1d['turnover'].mean()

        def build_dic():
            dic = {
                'aroe': round(aroe, 4),
                'sharpe': round(sharpe, 2),
                'turnover': round(tvr, 4),
                'ppc': round(ppc, 3),
                'profitFactor': round(profit_factor, 2),
                'last': round(last, 2),
                'mddText': mdd_text.strip(),
                'numDays': num_days
            }
            return dic
        if not flag_compute_mdd:
            mdd_text = ''

            if return_dic: return build_dic()
            return aroe, sharpe, tvr, ppc, profit_factor,\
                   num_days, last, last_day, mdd_text
        else:
            mdd, mdd_pct, cdd, cdd_pct = compute_mdd_vectorized(df_1d)
            rrr = aroe / mdd_pct
            mdd_text = f'mdd={mdd:.1f}/{cdd:.1f}' \
                       f'({100 * mdd_pct * l0:,.2f}% ' \
                       f'{rrr:,.1f}) '
            if return_dic:
                dic = build_dic()
                dic['mdd'] = round(mdd, 2)
                dic['mddPct'] = round(mdd_pct, 4)
                dic['cdd'] = round(cdd, 2)
                dic['cddPct'] = round(cdd_pct, 4)
                return dic

            return aroe, sharpe, tvr, ppc, profit_factor, \
                   num_days, last, last_day, \
                   mdd, mdd_pct, cdd, cdd_pct, mdd_text, rrr

    @staticmethod
    def compute_performance_stats(df, fee=None, flag_compute_mdd=True, alpha_name='',
                                  start=None, end=None, l1=1.0, l0=C.L0, plot=True,
                                  auto_detect_cut_book_text=True, skip_profit=False,
                                  suffix='', in_place=False):
        import numpy as np
        ################################################################
        if not in_place: df = df.copy()
        ds_pos = df['position']


        df['position'] = l1 * ds_pos
        flt = df['position'].abs() > 1
        df.loc[flt, 'position'] = np.sign(df.loc[flt, 'position'])
        assert df['position'].abs().max() <= 1, 'Sai, position không được > 1'
        ################################################################
        df['action'] = df['position'].diff().fillna(0)
        df['turnover'] = df['action'].abs()
        if fee is None:
            assert 'fee' in df, \
                   f'Sai, fee_params = None thì df cần trườn "fee" '
            assert 'turnover' in df, \
                   f'Sai, fee_params = None thì df cần trường "turnover" '
            fee = round(df['fee'].sum() / df['turnover'].sum(), 2)
        else:
            df['fee'] = df['turnover'] * fee

        if not skip_profit:
            df['grossProfit'] = df['position'] * df['priceChange']
            df['netProfit'] = df['grossProfit'] - df['fee']
            df['cumNetProfit'] = df['netProfit'].cumsum()
        df_1d = df \
            .groupby('day') \
            .agg({'netProfit': 'sum',
                  'turnover': 'sum',
                  'open': 'first',
                  'close': 'last'})
        if start is not None:
            df_1d = df_1d[df_1d.index >= start].copy()
        if end is not None:
            df_1d = df_1d[df_1d.index <= end].copy()

        #####################################################################################
        res = Alpha_Domains2.compute_performance_from_df_1d(df_1d, flag_compute_mdd, l0=l0)
        if not flag_compute_mdd:
            aroe, sharpe, tvr, ppc, profit_factor, \
            num_days, last, last_day, mdd_text = res
        else:
            aroe, sharpe, tvr, ppc, profit_factor, \
            num_days, last, last_day, \
            mdd, mdd_pct, cdd, cdd_pct, mdd_text, rrr = res

        new_report = {
            'alphaName': alpha_name,
            'aroe': round(aroe * l0, 4),
            'sharpe': round(sharpe, 2),
            'turnover': round(tvr, 3),
            'ppc': round(ppc, 3),
            'profitFactor': round(profit_factor, 2),
            'numDays': num_days,
            'last': round(last, 2),
            'l1': l1,
            'fee': fee}
        if flag_compute_mdd:
            new_report['mdd'] = round(mdd, 1)
            new_report['mddPct'] = round(mdd_pct, 3) * l0
            new_report['cdd'] = round(cdd, 3)
            new_report['rrr'] = round(rrr / mdd_pct, 2)
        #####################################################################################

        if l1 != 1:
            l1_text = f'L1={l1:.1f} '
        else:
            l1_text = ''
        #####################################################################################
        old_name = alpha_name
        flt3 = df['session'].shift(1) == 'unconditionalATC'
        has_overnight_profit = df[flt3]['netProfit'].abs().sum() > 0
        if has_overnight_profit and len(alpha_name) > 0 and alpha_name[-1] != '*':
            alpha_name += '*'
        if len(alpha_name) > 0: alpha_name = alpha_name + ' '
        if not auto_detect_cut_book_text:
            cut_book_text = ''
        else:
            cut_book_text = Alpha_Domains2.detect_book_cutting(df)
        if type(suffix) != str: suffix = f"L1={suffix}"
        if len(suffix) > 0: suffix = suffix + ' '
        fig_title = f"{alpha_name}{suffix}" \
                    f"{l1_text}" \
                    f"{cut_book_text}" \
                    f"sharpe={sharpe:.2f} " \
                    f"aroe={100 * aroe:.1f}%({l0 * 100 * aroe:,.1f}%) " \
                    f"ppc={ppc:.3f} " \
                    f"tvr={100 * tvr:.1f}% " \
                    f"{mdd_text}" \
                    f"fee={fee:,.2f} " \
                    f"{last_day}({num_days:,}) " \
                    f"last={last:,.2f}"
        if plot:
            label = "\x1b[90mcompute_performance_stats(): \x1b[0m"
            print(f"{label} {fig_title} pf={new_report['profitFactor']}", end="")
            if flag_compute_mdd: print(f" rrr={rrr:,.2f}")
            Alpha_Domains2.plot_daily_performance(df_1d, fig_title)

        #####################################################################################
        if Alpha_Domains2.FLAG_SAVE_DF_ALPHA and (len(alpha_name) > 0):
            label = '\x1b[90mAlpha_Domains2.compute_performance_stats\x1b[0m" '
            print(label, end="")
            fn = f'{LIB_RESOURCES.DIR_PLOTED_ALPHAS}/{old_name}.pickle'
            with open(fn, 'wb') as file:
                import pickle
                pickle.dump(df, file)
            alpha_func_lib.U.execute_cmd(f'ls -lahtr {fn}')
        return fig_title, new_report, df_1d

    @staticmethod
    def plot_daily_performance(df_1d, fig_title='', days_per_tick=10):
        from matplotlib import pyplot as plt
        if len(fig_title) < 120:
            plt.figure(figsize=(12.5, 7))
        else:
            if len(fig_title) < 140:
                plt.figure(figsize=(13.5, 7))
            else:
                plt.figure(figsize=(14.2, 7))
        idx = df_1d.index.map(lambda x: x[2:])
        plt.bar(idx, df_1d['netProfit'].cumsum())
        plt.title(fig_title.strip())
        x_ticks = [day for day, i in zip(idx, reversed(range(len(idx)))) if
                   i % days_per_tick == 0]
        if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
        plt.xticks(x_ticks, rotation=25)
        plt.show()

    @staticmethod
    def plot_price_position_profit(df, report, day=None, show_profit=True):
        from matplotlib import pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        if day is None:
            day = df['day'].iloc[-1]
        cols1 = ['netProfit', 'position', 'close', 'open']
        df_plot = df[df['day'] == day].set_index('time')[cols1]
        df_plot['cumNetProfit'] = df_plot['netProfit'].cumsum()
        mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains2.compute_mdd_vectorized(df_plot)
        idx = df_plot.index.map(lambda x: x[:-3])
        x_ticks = [x for x in ['09:15', '09:30', '10:00', '10:30',
                               '11:00', '11:30', '13:30', '14:00',
                               '14:15', '14:30']
                   if x <= idx[-1]]
        #################################################################################
        ax1.set_title(f"Price (close) day={day}")
        ax1.plot(idx, df_plot['close'], color='orange')
        ax1.set_xticks(x_ticks)
        #################################################################################
        ax2.set_title(f"{report['alphaName']} Position (L1={report['l1']})")
        ax2.bar(idx, df_plot['position'])
        ax2.set_xticks(x_ticks)
        #################################################################################
        if show_profit:
            mmax = df_plot['cumNetProfit'].max()

            lost_points =  mmax - df_plot['cumNetProfit'].iloc[-1]
            ax3.set_title(f"Net profit={df_plot['netProfit'].sum():.1f} "
                          f"max={mmax:.1f} mdd={mdd:,.2f} "
                          f"distance_to_hwk={lost_points:,.2f}")
            ax3.bar(idx, df_plot['netProfit'].cumsum(), color='green')
            ax3.set_xticks(x_ticks)
        #################################################################################
        plt.show()

    @staticmethod
    def compute_delayed_performance(df, start=None, end=None, alpha_name='',
                                    last_time=None, min_delay=0, max_delay=10):
        import pandas as pd
        lst = []
        for delay in range(min_delay, max_delay + 1):
            df = df.copy()
            if last_time is None:
                last_time = df['executionTime'].iloc[-delay - 1]
            df.loc[df['executionTime'] >= last_time, 'position'] = 0
            ds = (df['position'] * df['priceChange']) - df['position'].diff().abs() * (
                        df['fee'].sum() / df['turnover'].sum())
            df['position'] = df['position'].shift(delay)
            ds_delayed = (df['position'] * df['priceChange']) - df['position'].diff().abs() * (
                    df['fee'].sum() / df['turnover'].sum())
            ds_delayed.sum() / ds.sum()

            print(f"delay={delay} ", end="")
            fig_title, report, df_1d = Alpha_Domains2.compute_performance_stats(
                df,
                fee=0.2,
                start=start,
                end=end,
                alpha_name=f"{alpha_name}(+{delay}m)")
            report['delay'] = delay

            lst.append(report)

        return pd.DataFrame(lst)

    @staticmethod
    def compute_profits_with_overnight(bt, plot_on=False):
        df = bt.df_alpha.copy()
        df_on = bt.compute_overnight_profit(plot_on=plot_on).set_index('day')['onProfit']
        dic_on_profit = {k: v for k, v in zip(df_on.index, df_on)}
        flt = df['session'].shift(1) == 'unconditionalATC'
        assert df.loc[flt, 'position'].abs().sum() == df.loc[flt, 'netProfit'].abs().sum() == 0
        df.loc[flt, 'netProfit'] += df.loc[flt, 'day'].map(dic_on_profit)
        df['cumNetProfit'] = df['netProfit'].cumsum()
        return df

    @staticmethod
    def add_morning_010(bt, start=None, end=None, alpha_name='', l1_010=1,
                        plot=False):
        label = '\x1b[90mAlpha_Domains2.add_morning_010\x1b[0m: '
        df_alpha = bt.df_alpha.copy()
        df_alpha.loc[df_alpha['executionTime'] < '13:35:00', 'position'] = 0
        df_alpha = df_alpha.copy()
        fn = '/tmp/plotted_alphas/alpha_010.pickle'
        print(label, end='')
        alpha_func_lib.U.execute_cmd(f'ls -lahtr {fn}')
        print()
        with open(fn, 'rb') as file: df_010 = pickle.load(file)
        assert len(df_alpha) == len(df_010), 'Sai add_morning_010() #1'
        if l1_010 != 1:
            df_010['position'] *= l1_010
            flt = df_010['position'].abs() > 1
            df_010.loc[flt, 'position'] = np.sign(df_010.loc[flt, 'position'])
        df_alpha['position'] += df_010['position']
        assert df_alpha['position'].abs().max() <= 1, 'Sai add_morning_010() #2'

        _ = Alpha_Domains2.compute_performance_stats(
            df_alpha,
            start=start,
            end=end,
            fee=0.2,
            plot=plot,
            in_place=True,
            alpha_name=alpha_name)

        return df_alpha


class Alpha_Funcs:
    @staticmethod
    def alpha_018_busd(df) -> pd.DataFrame:
        if 'netBusd' not in df: df['netBusd'] = df['bu'] - df['sd']
        signal = -1 * alpha_func_lib.O.ts_rank(df['netBusd'] * df['matchingVolume'])  # 31 alphas 3.5 sharpe

        normalized_signal = -(signal / 5 + 1) / 0.9 - 1 / 9
        # noinspection PyTypeChecker
        return normalized_signal

    # l1: 2.837 IS_BUSD: 1.5 combined 1m sharpe=4.21 tvr=178.7% aroe=25.7% fee=0.20
    @staticmethod
    def alpha_zscore_busd(df, window=10):
        if 'netBusd' not in df: df['netBusd'] = df['bu'] - df['sd']
        O = alpha_func_lib.O
        signal = O.zscore(
            O.ts_rank(df['bu'] + df['sd'])
            *
            (df['close'] - df['close']), #
            window=window)
        signal = signal / window
        # signal = signal / 6
        return signal

    @staticmethod
    def alpha_018_ver2(df) -> pd.DataFrame:
        if 'netBusd' not in df: df['netBusd'] = df['bu'] - df['sd']
        # sharpe 3.08
        signal = -1 * alpha_func_lib.O.ts_rank(df['netBusd'] * (df['bu'] + df['sd']))  # 31 alphas 3.5 sharpe

        # combined 1m sharpe=3.39 tvr=384.0% aroe=32.3% fee=0.20 2023_06_01 -> 2023_10_06: 6.54
        # signal = -1 * alpha_func_lib.O.ts_rank((df['close'] - df['open']) * (df['bu'] + df['sd']))  # 31 alphas 3.5 sharpe

        normalized_signal = -(signal / 5 + 1) / 0.9 - 1 / 9
        # noinspection PyTypeChecker
        return normalized_signal

    @staticmethod
    def alpha_bb_breakout(df: pd.DataFrame, window=20, factor=1):
        df = df.copy()
        df['basis'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['upper'] = df['basis'] + df['std'] * factor
        df['lower'] = df['basis'] - df['std'] * factor
        df['signal'] = (df['close'] - df['open']) / (df['upper'] - df['lower'])

        return df['signal']

    @staticmethod
    def alpha_238(df):
        O = alpha_func_lib.O

        df = df.copy()
        close = df['close']
        openn = df['open']
        signal = O.ts_rank(close - openn, 10) + O.ts_rank(close / openn, 10)
        signal = (signal - 11) / 9
        return signal


class Backtest_Params:
    def __init__(self, day_is_start, day_is_end, day_os_start, day_os_end=None,
                 fee_is=0.2, fee_os=0.2, is_book_cutter=None, os_book_cutter=None,
                 div=1):
        if day_os_end is None:
            day_os_end = '2099_01_01'
        self.day_is_start = day_is_start
        self.day_is_end = day_is_end
        self.day_os_start = day_os_start
        self.day_os_end = day_os_end
        self.fee_is = fee_is
        self.fee_os = fee_os
        self.is_book_cutter = is_book_cutter
        self.os_book_cutter = os_book_cutter
        self.div = div


class Domains:
    @staticmethod
    def gen_threshold_list():
        return [(x, y) for x, y in product(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                if x > y]

    @staticmethod
    def gen_third_thursdays(years=None):
        import datetime

        def third_thursdays(year):
            thursdays = []
            for month in range(1, 13):
                current_date = datetime.date(year, month, 1)

                while current_date.weekday() != 3:  # 3 corresponds to Thursday
                    current_date += datetime.timedelta(days=1)

                third_thursday = current_date + datetime.timedelta(days=14)
                thursdays.append(third_thursday)

            return [date.strftime('%Y_%m_%d') for date in thursdays]

        if years is None: years = ['2021', '2022', '2023', '2024']
        lst = []
        for year in years:
            lst += third_thursdays(int(year))

        return lst

    @staticmethod
    def parse_freq(alpha_id):
        return int(alpha_id.split('_')[2].split('m')[0])


class Plotter:
    @staticmethod
    def plot_delayed_value(df_delayed, value='ppc', alpha_name=''):
        df_delayed[value].plot(
            marker='o',
            linestyle='-')

        for index, value in enumerate(df_delayed[value]):
            plt.annotate(
                str(value), (
                    index, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center')
        plt.title(f'Delayed ppc for {alpha_name}')
        plt.show()

    @staticmethod
    def plot_alpha_performance(df_1d, report, prefix=''):
        fig_title = Plotter.compute_title(report, prefix=prefix)
        plt.figure(figsize=(14, 7))

        idx = df_1d.index.map(lambda x: x[2:])
        plt.bar(idx, df_1d['netProfit'].cumsum())
        plt.title(fig_title)
        x_ticks = [day for day, i in zip(idx, reversed(range(len(df_1d.index)))) if
                   i % 10 == 0]

        if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
        plt.xticks(x_ticks, rotation=45)
        plt.show()

    @staticmethod
    def plot_alpha_performance_plotly(df_1d, report, prefix=''):
        fig_title = Plotter.compute_title(report, prefix=prefix)

        # plt.figure(figsize=(13, 7))
        #
        # idx = df_1d.index.map(lambda x: x[2:])
        # plt.bar(idx, df_1d['netProfit'].cumsum())
        # plt.title(fig_title)
        # x_ticks = [day for day, i in zip(idx, reversed(range(len(df_1d.index)))) if
        #            i % 10 == 0]
        #
        # if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
        # plt.xticks(x_ticks, rotation=45)
        # plt.show()
        if 'day' not in df_1d:
            df_1d = df_1d.reset_index()
        if 'cumNetProfit' not in df_1d:
            df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        df_1d['hoverName'] = df_1d['day'] + '     ' + \
                             df_1d['netProfit'].round(2).astype(str)
        fig = px.bar(df_1d.reset_index(),
                     x='day',
                     y='cumNetProfit',
                     hover_name='hoverName')
        fig.update_layout(dict(title=fig_title))
        fig.show(file_name=f'{report["alphaName"]}')

    @staticmethod
    def compute_title(report, use_ratio=False, prefix=''):
        start = report['start'][-8:]
        end = report['end'][-8:]
        aroe = 100 * report['aroe']
        aroe_with_l0 = aroe * C.L0
        if use_ratio:
            ratio = 100 / aroe_with_l0
        else:
            ratio = 1
        if 'l1' in report:
            l1_text = f"L1={round(report['l1'], 2)} "
        else:
            l1_text = ""
        name = report['alphaName'] if 'alphaName' in report \
            else report['name'] if 'name' in report \
            else ''

        if 'inertia' in report:
            inertia_text = f'in={report["inertia"]} '
        else:
            inertia_text = ''
        t = f"{prefix}{name} {report['freq']}m " \
            f"{inertia_text}" \
            f"$={report['sharpe']:.2f} " \
            f"aroe={aroe:.2f}% " \
            f"mdd={report['mdd']:,.1f}" \
            f"({100 * report['mddPct'] * C.L0 * ratio:,.1f}%) " \
            f"tvr={100 * report['tvr']:.1f}% " \
            f"f={report['fee']:.2f} " \
            f"ppc={report['ppc']:.3f} " \
            f"{l1_text}" \
            f"{start}->{end} last={report['lastProfit']:,.2f}"
        return t

    @staticmethod
    def plot_action_distribution(df_pos_1m, prefix=''):
        if len(prefix) > 0: prefix = f'{prefix} '
        df_pos = df_pos_1m.copy()
        df_pos['time'] = df_pos.index.strftime('%H:%M:%S')
        df_pos['day'] = df_pos.index.strftime('%Y_%m_%d')
        df_pos['timeFirst'] = df_pos['timeLast'] = df_pos['time']
        df_pos_5m = df_pos.resample('5min').agg({'timeFirst': 'first', 'timeLast': 'last'})
        df_pos_5m = df_pos_5m[df_pos_5m['timeFirst'].map(lambda x: type(x) == str)].copy()
        del df_pos['timeFirst'], df_pos['timeLast']
        df_pos = df_pos.merge(df_pos_5m[['timeFirst']], left_index=True, right_index=True, how='left')
        df_pos['timeFirst'] = df_pos['timeFirst'].fillna(method='ffill')
        first_pos = df_pos['position'].iloc[0]
        df_pos['action'] = df_pos['position'] \
                           - \
                           df_pos['position'].shift(1, fill_value=first_pos)
        df_pos['turnover'] = df_pos['action'].abs()
        df_pos['turnover'] = df_pos['turnover'] / df_pos['turnover'].sum() * 100
        df_pos2 = df_pos \
            .reset_index() \
            .groupby(['day', 'timeFirst'], as_index=False) \
            .agg({'turnover': 'sum',
                  'groupTime': 'first'}) \
            .set_index('groupTime')
        df_plot = df_pos2.groupby('timeFirst').agg({'turnover': 'sum'})

        df_plot.index = df_plot.index.map(lambda x: x[:5])
        shown_x_ticks = ['09:00', '09:15', '09:30', '10:00', '10:30', '11:00',
                         '11:30', '13:30', '14:00', '14:15', '14:45']
        plt.bar(df_plot.index, df_plot['turnover'])
        plt.title(f'{prefix}Action distribution over Time')
        plt.xticks(shown_x_ticks, rotation=75)
        plt.show()

    @staticmethod
    def plot_frequency_ditribution(df_all_pos, prefix=''):
        def parse_freq(alpha_id):
            return int(alpha_id.split('_')[2].split('m')[0])

        if len(prefix) > 0: prefix = f'{prefix} '
        df_freq_counts = pd.DataFrame(map(parse_freq, df_all_pos.columns))
        df_plot = df_freq_counts[0].value_counts()
        plt.bar(df_plot.index, df_plot.values)
        plt.title(f'{prefix}Frequency distribution')
        plt.show()

    @staticmethod
    def plot_performance_using_plotly(df_1d, fee, prefix='', start=None, end=None):
        if start is not None:
            df_1d = df_1d[df_1d['day'] >= start]
        if end is not None:
            df_1d = df_1d[df_1d['day'] <= end]
        df_1d = df_1d.copy()

        df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        dic = Alpha_Domains.compute_performance_from_df_1d(
            df_1d,
            fee=fee,
            return_dic=True)
        if len(prefix) > 0: prefix = f'{prefix} '
        fig = px.bar(df_1d,
                     x='day', y='cumNetProfit',
                     hover_name='netProfit')
        fig.update_layout(dict(title=prefix + dic['title']))
        Alpha_Domains.compute_performance_from_df_1d(df_1d, fee=fee)
        fig.show()


class Alpha_Domains:
    @staticmethod
    def compute_profit_factor(df):
        flt = df['netProfit'] > 0
        pf = df['netProfit'][flt].sum() / -df['netProfit'][~flt].sum()
        return round(pf, 3)

    @staticmethod
    def create_backtest(freq, alpha_func=None, alpha_name='', df_alpha=None,
                        dic_freqs=None, upper=None, lower=None, fee=0.2):
        if dic_freqs is None: dic_freqs = DIC_FREQS
        if alpha_func is None:
            assert alpha_name is not None and alpha_name in DIC_ALPHAS
            alpha_func = DIC_ALPHAS[alpha_name]
        if df_alpha is None:
            assert freq is not None
            df_alpha = dic_freqs[freq]

        report = {
            'alphaName': alpha_name,
            'freq': freq,
            'fee': fee,
        }
        if upper is not None:
            assert lower is not None
            report['upper'] = upper
            report['lower'] = lower

        return df_alpha, alpha_func, report

    @staticmethod
    def compute_signal(alpha_func, df_alpha):
        df_alpha['position'] = df_alpha['signal'] = alpha_func(df_alpha)

    @staticmethod
    def adjust_positions(df_alpha):
        flt_unexecutable = ~df_alpha['executable']
        df_alpha.loc[flt_unexecutable, 'position'] = np.NaN
        flt_atc = df_alpha['executionTime'] == '14:45:00'
        df_alpha.loc[flt_atc, 'position'] = 0
        df_alpha['position'] = df_alpha['position'].fillna(method='ffill') \
            .fillna(0)

    @staticmethod
    def compute_position_with_inertia(df_alpha, inertia=0.0):
        Alpha_Domains.adjust_positions(df_alpha)
        if inertia == 0: return
        lst = []
        last_pos = 0
        for pos in df_alpha['position']:
            if abs(pos - last_pos) >= inertia:
                last_pos = pos
            lst.append(last_pos)
        df_alpha['position'] = lst
        Alpha_Domains.adjust_positions(df_alpha)

    @staticmethod
    def compute_positions_with_thresholds(df_alpha, upper, lower):
        Alpha_Domains.adjust_positions(df_alpha)
        lst_pos = []
        last_pos = 0
        for pos in df_alpha['position']:
            if abs(pos) >= upper:
                last_pos = np.sign(pos)
            elif (last_pos > 0) and pos < lower:
                last_pos = 0
            elif (last_pos < 0) and pos > -lower:
                last_pos = 0
            lst_pos.append(last_pos)
        df_alpha['position'] = lst_pos
        Alpha_Domains.adjust_positions(df_alpha)

    @staticmethod
    def adjust_on_and_nonexecutable(df_alpha, hold_overnight,
                                    df_atc=None):
        if df_atc is None: df_atc = df_alpha[df_alpha['executionTime'] == '14:45:00']
        if hold_overnight:
            df_alpha.loc[df_atc.index, 'priceChange'] = \
                df_alpha.loc[df_atc.index, 'close'] \
                - \
                df_alpha['open'].shift(-1).loc[df_atc.index]
        else:
            df_alpha.loc[df_alpha['executionTime'] == '14:45:00', 'position'] = 0
        df_alpha.loc[~df_alpha['executable'], 'position'] = np.NaN
        df_alpha['position'] = df_alpha['position'].fillna(method='ffill')

    @staticmethod
    def compute_action_tvr_and_fee(df_alpha, fee):
        df_alpha['action'] = df_alpha['position'].diff(1).fillna(df_alpha['position'].iloc[0])
        df_alpha['turnover'] = df_alpha['action'].abs()
        df_alpha['fee'] = df_alpha['turnover'] * fee

    @staticmethod
    def compute_profits(df_alpha):
        df_alpha['grossProfit'] = df_alpha['position'] * df_alpha['priceChange']
        df_alpha['cumGrossProfit'] = df_alpha['grossProfit'].cumsum()
        df_alpha['netProfit'] = df_alpha['grossProfit'] - df_alpha['fee']
        df_alpha['cumNetProfit'] = df_alpha['netProfit'].cumsum()

    @staticmethod
    def compute_performance(df_alpha, start=None, end=None, report=None, prefix='',
                            verbosity=1, return_df_1d=False, use_plotly=False,
                            plot=False):
        lst_errs = []
        if start is not None:
            df_alpha = df_alpha[df_alpha['day'] >= start]
        if end is not None:
            df_alpha = df_alpha[df_alpha['day'] <= end]

        df_1d = df_alpha \
            .groupby('day') \
            .agg({
            'turnover': 'sum',
            'netProfit': 'sum',
            'open': 'first',
        })
        df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
        try:
            sharpe = df_1d['pctChange'].mean() / df_1d['pctChange'].std() * 250 ** 0.5
        except Exception as e:
            lst_errs.append(f"{e}")
            # U.report_error(e)
            sharpe = C.UNREASONABLE_SHARPE
        roe = df_1d['pctChange'].sum()
        tvr = df_1d['turnover'].mean()
        aroe = roe * 250 / len(df_1d)
        ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)
        # mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains.compute_mdd_unvectorized(df_alpha)
        mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains.compute_mdd_vectorized(df_1d)
        new_report = {
            'sharpe': round(sharpe, 3),
            'aroe': round(aroe, 4),
            'mdd': round(mdd, 3),
            'mddPct': round(mdd_pct, 4),
            'cdd': round(cdd, 3),
            'cddPct': round(cdd_pct, 4),
            'ppc': round(ppc, 4),
            'tvr': round(tvr, 4),
            'start': df_1d.index[0],
            'end': df_1d.index[-1],
            'lastProfit': round(df_1d['netProfit'].iloc[-1], 2)
        }

        if report is not None:
            for key, value in new_report.items():
                report[key] = value
            new_report = report

        if verbosity >= 2:
            print(json.dumps(new_report, indent=4))

        if plot:
            if not use_plotly:
                Plotter.plot_alpha_performance(df_1d, new_report, prefix=prefix)
            else:
                Plotter.plot_alpha_performance_plotly(df_1d, new_report, prefix=prefix)

        if return_df_1d:
            df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
            df_1d = df_1d.reset_index()
            df_1d.index = df_1d['day'].values
            df_1d['netProfit'] = df_1d['netProfit'].round(2)
            return df_1d, new_report
        return new_report

    @staticmethod
    def compute_mdd_vectorized(df_1d, verbosity=1):

        def max_so_far(arr):
            max_sofar = np.maximum.accumulate(arr)
            return max_sofar

        if 'cumNetProfit' in df_1d:
            ds = df_1d['cumNetProfit']
        else:
            ds = df_1d['netProfit'].cumsum()
        high_water_mark = max_so_far(ds)
        lst_cdd = high_water_mark - ds
        lst_mdd = max_so_far(lst_cdd)
        lst_cdd_pct = lst_cdd / df_1d['open']
        lst_mdd_pct = max_so_far(lst_cdd_pct)
        cdd, mdd = lst_cdd[-1], lst_mdd[-1]
        cdd_pct, mdd_pct = lst_cdd_pct[-1], lst_mdd_pct[-1]

        if verbosity >= 2:
            print(f"mdd=\x1b[93m{mdd:,.2f}\x1b[0m mdd_pct={100 * mdd_pct:.1f}% "
                  f"cdd=\x1b[93m{cdd:,.2f}\x1b[0m cdd_pct={100 * cdd_pct:.1f}%")

        return mdd, mdd_pct, cdd, cdd_pct

    @staticmethod
    def print_report(report, prefix='', flag_return=False):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            if 'start' in report:
                start = report['start'][-8:]
            else:
                start = ''

            if 'end' in report:
                end = report['end'][-8:]
            else:
                end = ''
            aroe_with_l0 = 100 * report['aroe'] * C.L0
            ratio = 100 / aroe_with_l0
            if 'upper' in report:
                args = f' \x1b[35m{report["upper"]}/{report["lower"]}\x1b[0m '
            else:
                args = ' '
            if len(prefix) > 0: prefix = f'{prefix} '
            s = f"\r{prefix}" \
                f"\x1b[93m{report['alphaName']}\x1b[0m " \
                f"\x1b[94m{report['freq']}m\x1b[0m" \
                f"{args}" \
                f"sharpe=\x1b[96m{report['sharpe']:.2f}\x1b[0m " \
                f"mdd=\x1b[91m{report['mdd']:,.1f}\x1b[0m " \
                f"({100 * report['mddPct'] * C.L0 * ratio:,.1f}%) " \
                f"tvr=\x1b[33m{100 * report['tvr']:,.1f}%\x1b[0m " \
                f"aroe_with_l0=\x1b[92m" \
                f"{C.L0 * 100 * report['aroe']:.1f}%\x1b[0m " \
                f"fee=\x1b[95m{report['fee']:.2f}\x1b[0m " \
                f"ppc=\x1b[32m{report['ppc']:.3f}\x1b[0m " \
                f"{start}->{end} last=" \
                f"\x1b[90m{report['lastProfit']:,.2f}\x1b[0m"
            if flag_return: return s
            print(s)

    @staticmethod
    def compute_df_executionT(df_alpha):
        from datetime import datetime as dt
        df_execution_time = df_alpha[['executionTime', 'executionT']].copy()
        flt_lunch = ('11:30:00' < df_alpha['executionTime']) & \
                    (df_alpha['executionTime'] < '13:00:00')
        df_execution_time.loc[flt_lunch, 'executionTime'] = '11:31:00'

        flt_atc = ('14:30:00' < df_alpha['executionTime']) & \
                  (df_alpha['executionTime'] < '14:45:00')
        df_execution_time.loc[flt_atc, 'executionTime'] = '14:45:00'
        flt = flt_atc | flt_lunch
        df_execution_time.loc[flt, 'executionT'] = (
                df_alpha.loc[flt, 'day'] + ' ' +
                df_execution_time.loc[flt, 'executionTime']
        ).map(lambda x: dt.strptime(x, '%Y_%m_%d %H:%M:%S'))

        # df_missed = df_alpha[~df_execution_time['executionT'].isin(df_1m['executionT'])]
        # assert len(df_missed) < 10
        return df_execution_time

    @staticmethod
    def collect_position(df_alpha, name=None, DF_1M=None, dic_freqs=None):
        def compute_df_pos_1m(df_alpha, DF_1M=None, dic_freqs=None):
            if dic_freqs is None:
                dic_freqs = DIC_FREQS
            if DF_1M is None: DF_1M = dic_freqs[1]
            df_pos_1m = DF_1M[['executionT']] \
                .reset_index() \
                .merge(df_alpha[['position', 'executionT']],
                       on='executionT',
                       how='left')
            del df_pos_1m['executionT']
            df_pos_1m = df_pos_1m.groupby('groupTime') \
                .last() \
                .fillna(method='ffill') \
                .fillna(0)
            return df_pos_1m

        df_execution_time = Alpha_Domains.compute_df_executionT(df_alpha)
        df_alpha2 = df_alpha.copy()
        df_alpha2[df_execution_time.columns] = df_execution_time
        df_alpha2['position'] = df_alpha['position']
        df_pos_1m = compute_df_pos_1m(
            df_alpha2,
            DF_1M=DF_1M,
            dic_freqs=dic_freqs)
        df_pos_1m = df_pos_1m['position']
        if name is not None: df_pos_1m = df_pos_1m.rename(name)
        df_pos_1m = df_pos_1m.to_frame()

        return df_pos_1m

    @staticmethod
    def apply_l1(df_alpha, l1, verbosity=1):
        if verbosity >= 2:
            label = '\x1b[90mAlpha_Domains.apply_l1\x1b[0m:'
            print(f"{label} applying l1={l1:,.2f}")
        df = df_alpha[['position']].copy()
        # assert df['position'].abs().max() <= 1, \
        #        'Sai, input vào apply_l1 có position > 1 sẵn'
        df['position'] *= l1
        flt = df['position'].abs() > 1
        df.loc[flt, 'position'] = np.sign(df.loc[flt, 'position'])
        df_alpha2 = df_alpha.copy()
        df_alpha2['position'] = df['position']
        return df_alpha2

    @staticmethod
    def auto_compute_l1(df_alpha, fee, name=None, start=None, end=None,
                        TARGET=None, TOLERANCE=0.0002):
        def compute_leveraged_aroe(l1, df_input, fee):
            df_alpha = df_input.copy()
            df_alpha['position'] = Alpha_Domains.apply_l1(df_pos, l1)
            df_alpha['turnover'] = df_alpha['position'].diff().fillna(0).abs()
            gross_profit = df_alpha['position'] * df_alpha['priceChange']
            df_alpha['netProfit'] = gross_profit - df_alpha['turnover'] * fee
            df_1d = df_alpha \
                .groupby('day') \
                .agg({'netProfit': 'sum',
                      'turnover': 'sum',
                      'open': 'first'})
            ds = df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
            # sharpe = ds.mean() / ds.std() * 250**0.5
            aroe = ds.sum() * 250 / len(ds)
            new_l1 = TARGET / aroe * l1
            precision = 1 - abs(aroe / TARGET - 1)
            return aroe, new_l1, precision

        def announce():
            text = f" for \x1b[94m{name}\x1b[0m ".replace('alpha_', '')
            print(f"\r{label}{count} \x1b[92mFOUND\x1b[0m{text}"
                  f"l1=\x1b[93m{best_l1:,.2f}\x1b[0m "
                  f"aroe={100 * best_aroe:.3f}% (vs {100 * TARGET:,.3f})% "
                  f"(\x1b[92m{100 * best_precision:.1f}%\x1b[0m)")

        if TARGET is None:
            TARGET = 1 / C.L0 * 1.0
        if start is not None:
            df_alpha = df_alpha[df_alpha['day'] >= start]
        if end is not None:
            df_alpha = df_alpha[df_alpha['day'] <= start]
        if start is not None and end is not None:
            df_alpha = df_alpha.copy()
        label = "\x1b[90mauto_compute_l1\x1b[0m:"

        l1 = 1
        precision = 0.01
        df_pos = df_alpha[['position']].copy()
        best_l1, best_precision, best_aroe = l1, precision, 0

        for count in range(20):
            print(f"\r{label} attempt #{count}: l1={l1:.2f} "
                  f"precision={precision * 100:.1f}%    ",
                  end="")
            aroe, l1, precision = \
                compute_leveraged_aroe(
                    l1=l1,
                    df_input=df_alpha,
                    fee=fee)

            if best_precision < precision:
                best_precision = precision
                best_l1 = l1
                best_aroe = aroe

            if abs(precision - 1) < TOLERANCE:
                announce()
                return best_l1

        while abs(precision - 1) > TOLERANCE:
            count += 1
            print(f"\r{label} attempt #{count}: l1={l1:.3f} "
                  f"precision={precision * 100:.1f}%    "
                  f"best=(\x1b[92m{best_precision * 100:,.1f}%\x1b[0m "
                  f"{best_l1:.2f}) {best_aroe} {best_aroe * C.L0 * 100:,.1f}%",
                  end="")
            l1 = (np.sign(TARGET - aroe) * np.random.random() * 0.05 + 1) * l1
            aroe, l1, precision = \
                compute_leveraged_aroe(
                    l1=l1,
                    df_input=df_alpha,
                    fee=fee)
            if l1 > C.MAX_L1 or l1 < - 0.3: return 1
            if best_precision < precision:
                best_precision = precision
                best_l1 = l1
                best_aroe = aroe
            if count >= 100:
                return 1

        announce()
        return best_l1

    @staticmethod
    def compute_df_overnight(df_alpha):
        df = df_alpha[['position', 'open', 'close', 'session', 'day']].copy()
        df['nextOpen'] = df['open'].shift(-1, fill_value=df['close'].iloc[-1])
        df['overnightGap'] = df['nextOpen'] - df['close']
        df['prevPosition'] = df['position'].shift(1)
        flt = df['session'] == 'unconditionalATC'
        df_overnight = df[flt].set_index('day')
        df_overnight['grossProfit'] = \
            df_overnight['prevPosition'] \
            * \
            df_overnight['overnightGap']

        return df_overnight

    @staticmethod
    def compute_overnight_performance(df_overnight, start=None, end=None, plot=False,
                                      prefix=None):
        if start is not None:
            df_overnight = df_overnight[df_overnight.index >= start]
        if end is not None:
            df_overnight = df_overnight[df_overnight.index <= end]

        df_overnight = df_overnight.copy()
        ds = df_overnight['pctChange'] = df_overnight['grossProfit'] / df_overnight['close']
        sharpe = ds.mean() / ds.std() * 250 ** 0.5
        aroe = ds.sum() * 250 / len(ds)
        df_overnight['cumGrossProfit'] = df_overnight['grossProfit'].cumsum()
        atc_report = {
            'sharpe': round(sharpe, 3),
            'aroe': round(aroe, 4)
        }

        if plot:
            df_overnight['cumGrossProfit'].plot()
            if len(str(prefix)) > 0: prefix = f'{prefix} '
            fig_title = f"{prefix}sharpe={sharpe:.2f} " \
                        f"aroe={100 * aroe:.1f}% " \
                        f"({100 * aroe * C.L0:.1f}% with L0={C.L0})"
            plt.title(fig_title)
            plt.show()

        return atc_report

    @staticmethod
    def compute_alpha_id(report, in_place=False):
        # if 'name' in report:
        #     name = report['name']
        # elif 'alphaName' in report:
        #     name = report['alphaName']
        # elif 'alpha_name' in report:
        #     name = report['alpha_name'
        name = report['alphaName']
        if 'upper' in report:
            the_id = f"{name}_{report['freq']}m " \
                     f"{report['upper']}_{report['lower']} " \
                     f"(fee={report['fee']:.2f})"
        elif 'inertia' in report:
            the_id = f"{name} {report['freq']}m {report['inertia']}"
        else:
            the_id = f"{name} {report['freq']}m"

        if in_place: report['id'] = the_id

        return the_id

    @staticmethod
    def parse_alpha_id(the_id):
        l = the_id.split('_')
        alpha_name = '_'.join(l[:2])
        try:
            freq = int(l[2].split(' ')[0].replace('m', ''))
        except Exception as e:
            if len(str(e)) == 0: print(f"{e}", end="")
            freq = int(l[3].split(' ')[0].replace('m', ''))
        l = the_id.split(' ')
        upper, lower = map(float, l[1].split('_'))
        fee = float(l[-1].replace('(fee=', '').replace(')', ''))
        dic_init = {
            'alpha_name': alpha_name,
            'freq': freq,
            'upper': upper,
            'lower': lower,
            'fee': fee
        }
        return dic_init

    @staticmethod
    def compute_performance_from_df_1d(df_1d, fee=0.2, num_ticks=8, plot=False,
                                       name='', return_dic=False):
        ds = df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
        sharpe = ds.mean() / ds.std() * 250 ** 0.5
        idx = df_1d['day'].map(lambda x: x[2:]).values
        x_ticks = [day for day, i in zip(idx, reversed(range(len(idx)))) if
                   i % num_ticks == 0]
        if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
        ppc = df_1d['netProfit'].sum() / df_1d['turnover'].sum()
        mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains.compute_mdd_vectorized(df_1d)
        aroe = (df_1d['netProfit'] / df_1d['open']).sum() * 250 / len(df_1d)
        tvr = df_1d['turnover'].mean()

        fig_title = f"{name} " \
                    f"sharpe={sharpe:,.2f} " \
                    f"aroe={100 * aroe:,.1f}% ({C.L0 * 100 * aroe:,.1f}%) " \
                    f"mdd={mdd:,.2f}({100 * mdd_pct * C.L0:,.1f}%) " \
                    f"ppc={ppc:,.3f} tvr={tvr * 100:,.1f}% " \
                    f"fee={fee:,.2f} " \
                    f"last={df_1d['netProfit'].iloc[-1]:.2f} "

        if plot:
            plt.figure(figsize=(12, 6))
            plt.bar(idx, df_1d['netProfit'].cumsum())
            plt.xticks(x_ticks, rotation=25)
            plt.title(fig_title)
            plt.show()
        if return_dic:
            dic = {
                'name': name,
                'sharpe': sharpe,
                'mdd': mdd,
                'cdd': cdd,
                'fee': fee,
                'last': df_1d['netProfit'].iloc[-1],
                'title': fig_title}
            return dic

    @staticmethod
    def compute_position_and_cut_book(bt):
        Alpha_Domains.adjust_positions(bt.df_alpha)
        if 'inertia' in bt.report:
            Alpha_Domains.compute_position_with_inertia(bt.df_alpha, bt.inertia)
        elif 'upper' in bt.report:
            bt.compute_position()
        # noinspection PyCallingNonCallable
        mmax = bt.df_alpha['position'].abs().max()
        if mmax > 1:
            label = "\x1b[90mAlpha_Domains.compute_position_and_cut_book\x1b[0m: "
            print(f"{label}\x1b[94mWarning: absolute position > 1 "
                  f"\x1b[0m(\x1b[93m{bt.name}\x1b[0m)")
        # bt.df_alpha['position'] /=
        flt = bt.df_alpha['position'].abs() >= 1
        bt.df_alpha.loc[flt, 'position'] = np.sign(bt.df_alpha.loc[flt, 'position'])
        assert bt.df_alpha['position'].abs().max() <= 1, 'Sai, abs position lớn hơn 1'
        bt.cb_func(bt.df_alpha)
        Alpha_Domains.adjust_positions(bt.df_alpha)

    @staticmethod
    def combine_alphas_and_compute_performance(configs, fee, l1=1.0, alpha_name='',
                                               start=None, end=None, plot=True):
        label = "\x1b[90mcompute_combined_os_performance\x1b[0m:"
        lst_bts = [Simulator.from_id(config) for config in configs]
        collector = Collector(name='zscore_busd')
        for i, bt in enumerate(lst_bts):
            print(f"\r{label} #{i + 1:,} / {len(lst_bts):,} "
                  f"{bt.report['id']}  ",
                  end="")
            bt.compute_all()
            collector.collect_bt(bt)
            if i + 1 == len(lst_bts): print()


        bt = collector.create_bt(fee=fee)
        bt.report['l1'] = l1
        if l1 != 1: bt.apply_l1(l1=l1)
        bt.name = bt.report['alphaName'] = 'collected'
        num_alphas = len(collector.df_all_pos.columns)
        bt.compute_tvr_and_fee()
        bt.compute_profits()

        # noinspection PyUnusedLocal
        fig_title, os_report, df_1d = Alpha_Domains2.compute_performance_stats(
            df=bt.df_alpha,
            fee=bt.fee,
            in_place=True,
            plot=plot,
            start=start,
            end=end,
            alpha_name=alpha_name,
            suffix=f'({num_alphas:,} alpha) L1={l1}')

        return bt, collector, os_report, df_1d

    @staticmethod
    def add_overnight_performance(bt, plot_on=False):
        label = '\x1b[90madd_overnight_performance\x1b[0m: '
        df = bt.df_alpha.copy()
        df_on = bt.compute_overnight_profit(plot_on=plot_on).set_index('day')['onProfit']

        dic_on_profit = {k: v for k, v in zip(df_on.index, df_on)}
        flt = df['session'].shift(1) == 'unconditionalATC'
        pre_atc_position = df.loc[flt, 'netProfit'].abs().sum()
        pre_atc_profit = df.loc[flt, 'position'].abs().sum()
        assert pre_atc_position == pre_atc_profit == 0, \
               f"{label}\x1b[91mSai, df_alpha đã có sẵn " \
               f"over-night profit hoặc over-night position\x1b[0m"
        df.loc[flt, 'netProfit'] += df.loc[flt, 'day'].map(dic_on_profit)
        df['cumNetProfit'] = df['netProfit'].cumsum()
        bt.name += '*'
        bt.report['alphaName'] += '*'

        return df


class Simulator:
    def __init__(self, alpha_name, fee, freq=1, alpha_func=None, df_alpha=None,
                 dic_freqs=None, upper=None, lower=None, reverse=False,
                 suffix='', inertia=None, use_plotly=False, cb_func=None):
        if dic_freqs is None:
            dic_freqs = DIC_FREQS

        if df_alpha is None:
            assert freq is not None
            df_alpha = dic_freqs[freq].copy()

        if alpha_func is None:
            alpha_func = DIC_ALPHAS[alpha_name]

        self.fee = fee
        self.name = alpha_name
        self.df_alpha = df_alpha
        self.alpha_func = alpha_func
        self.freq = freq
        self.suffix = suffix
        self.reverse_factor = -1 if reverse else 1

        if cb_func is not None:
            if type(cb_func) == str:
                cb_func = Book_Cutter.get_book_cutter(cb_func)
            self.cb_func: type(lambda x: 0) = cb_func

        self.report = {
            'alphaName': self.name,
            'sharpe': np.NaN,
            'freq': self.freq,
            'fee': self.fee,
        }
        if inertia is not None:
            self.report['inertia'] = inertia
        if upper is not None:
            assert lower is not None
            assert inertia is None
            self.upper = upper
            self.lower = lower
            self.report['upper'] = upper
            self.report['lower'] = lower
        elif inertia is not None:
            self.inertia = inertia

        self.use_plotly = use_plotly
        self.report['id'] = self.compute_id()

        """PLACE-HOLDER"""
        self.df_1d = pd.DataFrame()
        self.df_overnight = pd.DataFrame()
        self.df_pos_1m = pd.DataFrame()
        self.df_on_1d = pd.DataFrame()

    def compute_id(self, save_to_report=False, verbosity=1):
        the_id = Alpha_Domains.compute_alpha_id(
            self.report,
            in_place=False)
        if verbosity >= 2:
            print(f'compute_id: {the_id}')
        if save_to_report: self.report['id'] = the_id

        return the_id

    def compute_signal(self):
        signal = self.alpha_func(self.df_alpha)
        if self.reverse_factor == -1:
            signal = -signal
        self.df_alpha['position'] = \
            self.df_alpha['signal'] = \
            signal

    def compute_position(self):
        if 'upper' in self.report:
            assert 'lower' in self.report
            Alpha_Domains.compute_positions_with_thresholds(
                df_alpha=self.df_alpha,
                upper=self.upper,
                lower=self.lower)
        if 'inertia' in self.report:
            Alpha_Domains.compute_position_with_inertia(
                df_alpha=self.df_alpha,
                inertia=self.report['inertia']
            )

    def compute_tvr_and_fee(self):
        Alpha_Domains.compute_action_tvr_and_fee(
            self.df_alpha,
            self.fee)

    def compute_profits(self):
        Alpha_Domains.compute_profits(self.df_alpha)

    def compute_position_and_cut_book(self):
        Alpha_Domains.compute_position_and_cut_book(self)

    def compute_performance(self, start=None, end=None, plot=False,
                            return_df_1d=False, use_plotly=None):
        if use_plotly is None:
            use_plotly = self.use_plotly

        res = Alpha_Domains.compute_performance(
            self.df_alpha,
            report=self.report,
            start=start,
            end=end,
            plot=plot,
            use_plotly=use_plotly,
            return_df_1d=return_df_1d)

        if return_df_1d:
            self.df_1d, self.report = res
        else:
            self.report = res
        return self.report

    def compute_all(self, start=None, end=None, plot=False,
                    skip_computing_position=False):
        if not skip_computing_position:
            self.compute_signal()
            self.compute_position()
        self.compute_tvr_and_fee()
        self.compute_profits()
        report = self.compute_performance(
            start=start,
            end=end,
            plot=plot)

        return report

    def compute_df_pos_1m(self):
        self.df_pos_1m = Alpha_Domains.collect_position(
            df_alpha=self.df_alpha,
            name='position')

        return self.df_pos_1m

    @staticmethod
    def from_report(report, compute_profits=False):
        bt = Simulator(
            alpha_name=report['name'],
            freq=report['freq'],
            dic_freqs=DIC_FREQS,
            upper=report['upper'],
            lower=report['lower'],
            fee=report['fee'])

        if compute_profits:
            bt.compute_signal()
            bt.compute_position()
            bt.compute_tvr_and_fee()
            bt.compute_profits()

        return bt

    @staticmethod
    def from_id(the_id):
        bt = Simulator(**Alpha_Domains.parse_alpha_id(the_id))
        return bt

    @staticmethod
    def dummy_alpha_func(df_alpha):
        if 'signal' in df_alpha:
            return df_alpha['signal']
        else:
            return df_alpha['positions']

    def plot_profit_with_candles(self, day=None):
        def plot_profit(df, ax2):
            idx = df['time'].map(lambda x: x[:5])
            ax2.plot(idx,
                     df['cumNetProfit'].shift(1),
                     label='cumProfit',
                     color='blue')
            ax2.set_xlabel('time')
            ax2.set_ylabel('profit')
            fig_title1 = ''
            ax2.set_title(fig_title1)
            ax2.legend()
            displayed_ticks = ["09:15", "09:30", "10:00", "10:30",
                               "11:00", "11:30", "13:30", "14:00",
                               "14:15", "14:45"]
            ax2.axvline(x=idx[-2], color='green', linestyle='--', label='preATC')
            ax2.axvline(x=idx[-3], color='red', linestyle='--', label='preATC')
            ax2.axvline(x=idx[-4], color='blue', linestyle='--', label='preATC')
            # ax2.axvline(x=idx[-1], color='green', linestyle='--', label='ATC')
            ax2.set_xticks(displayed_ticks)

        def plot_position(df, ax3):
            """AX1: POSITION (BAR_CHART)"""
            df = df.copy()
            df['position'] = df['position'].shift(1).fillna(0)
            idx = df['time'].map(lambda x: x[:5])
            ax3.bar(idx, df['position'])
            ax3.set_xlabel('time')
            ax3.set_ylabel('position')
            fig_title1 = ''
            ax3.set_title(fig_title1)
            displayed_ticks = ["09:15", "09:30", "10:00", "10:30",
                               "11:00", "11:30", "13:30", "14:00",
                               "14:15", "14:45"]
            ax3.axvline(x=idx[-4], color='blue', linestyle='--', label='preATC')
            ax3.axvline(x=idx[-3], color='red', linestyle='--', label='preATC')
            ax3.axvline(x=idx[-2], color='green', linestyle='--', label='preATC')
            ax3.axvline(x=idx[-1], color='green', linestyle='--', label='ATC')
            ax3.set_xticks(displayed_ticks)

        def plot_price_candles(df, ax1):
            df['x'] = range(len(df))
            prices = df.set_index('x')
            width = 0.70
            width2 = 0.35
            up = prices[prices.close >= prices.open]
            down = prices[prices.close < prices.open]
            col1 = 'green'
            col2 = 'red'
            # plot up prices
            ax1.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
            ax1.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
            ax1.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

            # plot down prices
            ax1.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
            ax1.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
            ax1.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)
            ax1.set_title(f'{day} last_price={df["close"][-1]} profit={df["cumNetProfit"].iloc[-1]:,.2f}')
            ax1.axvline(x=df['x'].iloc[-1] - 1, color='green', linestyle='--', label='ATC')
            ax1.axvline(x=df['x'].iloc[-1], color='green', linestyle='--', label='ATC')
            ax1.axvline(x=df['x'].iloc[-1] - 2, color='red', linestyle='--', label='ATC')
            ax1.axvline(x=df['x'].iloc[-1] - 3, color='blue', linestyle='--', label='ATC')
            ax1.axhline(y=df['open'].iloc[-2], color='blue', linestyle='--', label='ATC')
            ax1.axhline(y=df['close'].iloc[-1], color='red', linestyle='--', label='ATC')

        if day is None:
            day = self.df_alpha['day'].iloc[-1]
        if type(day) == int:
            day = self.df_alpha['day'].unique()[day]
        df = self.df_alpha[self.df_alpha['day'] == day]
        df = df.copy()
        df['cumNetProfit'] = df['netProfit'].cumsum()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

        plot_price_candles(df, ax1)
        plot_profit(df, ax2)
        plot_position(df, ax3)

        plt.show()

    def plot_action_distribution(self):
        df_1m = self.compute_df_pos_1m()
        df_1m.columns = ['position']
        Plotter.plot_action_distribution(df_1m, prefix=self.report['id'])

    def plot_combined_intraday_and_overnight_profit(self):
        ds = (self.df_overnight['grossProfit'] / self.df_overnight['close'] + self.df_1d['pctChange'])
        ds.mean() / ds.std() * 250 ** 0.5
        ds.cumsum().plot()
        plt.title(f'{self.name} aroe={ds.sum() * 100:.1f}%')
        plt.show()

    def apply_l1(self, l1=1.0):
        self.report['l1'] = l1
        self.df_alpha = Alpha_Domains.apply_l1(self.df_alpha, l1)

    def compute_overnight_profit(self, start=None, plot_on=False, name=''):
        if name is None or len(name) == 0: name = self.name
        df_alpha = self.df_alpha.copy()
        df_alpha['prevPos'] = df_alpha['position'].shift(1)
        df_alpha['nextOpen'] = df_alpha['open'].shift(-1)
        flt = df_alpha['session'] == 'unconditionalATC'
        df_with_on = df_alpha.loc[flt, ['day', 'prevPos', 'close', 'nextOpen']].copy()

        df_with_on['onPriceChange'] = (df_with_on['nextOpen'] - df_with_on['close']) \
            .fillna(0)
        df_with_on['onProfit'] = df_with_on['prevPos'] * df_with_on['onPriceChange']
        dh_days = Domains.gen_third_thursdays()
        df_with_on.loc[df_with_on['day'].isin(dh_days), 'onProfit'] = 0
        df_with_on['onProfit'] = df_with_on['onProfit'].shift(1).fillna(0)
        report = self.compute_performance(return_df_1d=True)
        df_on_1d = self.df_1d.merge(df_with_on, on='day', how='left')
        self.df_on_1d = df_on_1d

        if start is not None:
            df_on_1d = df_on_1d[df_on_1d['day'] >= start].copy()

        if plot_on:
            ds = df_on_1d['onPctChange'] = df_on_1d['onProfit'] / df_on_1d['open']
            sharpe_on = ds.mean() / ds.std() * 250 ** 0.5
            aroe = ds.sum() * 250 / len(ds)
            plt.figure(figsize=(10, 7))
            num_ticks = 15
            idx = df_on_1d['day'].map(lambda x: x[2:]).values
            x_ticks = [day for day, i in zip(idx, reversed(range(len(idx)))) if
                       i % num_ticks == 0]
            if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
            plt.plot(idx, df_on_1d['onProfit'].cumsum())
            plt.xticks(x_ticks, rotation=25)
            plt.title(f'{name} '
                      f'sharpe(over-night)={sharpe_on:,.2f} '
                      f'aroe={100 * aroe:.1f}%')
            plt.show()

        label = f'\x1b[90mcompute_overnight_profit\x1b[0m:'
        print(f"{label} {report}")

        return df_on_1d

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def create_bt_with_overnight(self):
        bt = self.clone()
        idx = bt.df_alpha \
            .reset_index() \
            .groupby('day') \
            .agg({'groupTime': 'first'}) \
            ['groupTime']
        bt.df_alpha.loc[idx, 'netProfit'] = bt.df_on_1d['onProfit'].to_numpy()
        bt.df_alpha['cumNetProfit'] = bt.df_alpha['netProfit'].cumsum()
        bt.report['alphaName'] += "*"
        return bt

    def plot_ppc_and_sharpe_decay(self, BT_PARAMS):
        lst_reports = []
        for delay in range(16):
            df_alpha = self.df_alpha.copy()
            df_alpha['newPosition'] = df_alpha \
                .groupby(['day', 'time']) \
                .last() \
                .groupby('day') \
                ['position'] \
                .shift(delay) \
                .fillna(0) \
                .to_numpy()
            df_alpha.loc[df_alpha['session'] == 'unconditionalATC', 'newPosition'] = 0
            df_alpha['position'] = df_alpha['newPosition']
            Alpha_Domains.compute_action_tvr_and_fee(
                df_alpha,
                fee=BT_PARAMS.fee_os)
            Alpha_Domains.compute_profits(df_alpha)
            report = Alpha_Domains.compute_performance(
                df_alpha,
                start=BT_PARAMS.day_os_start)
            report['delay'] = delay
            lst_reports.append(report)

        df_reports = pd.DataFrame(lst_reports)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

        ax1.bar(df_reports['delay'], df_reports['sharpe'])
        ax1.set_title('Sharpe decay rate')

        for i, v in enumerate(df_reports['sharpe']):
            ax1.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', fontsize=8)

        ax2.bar(df_reports['delay'], df_reports['ppc'], color='green')
        ax2.set_title('ppc decay rate')

        for i, v in enumerate(df_reports['ppc']):
            ax2.text(i, v + 0.01, "{:.3f}".format(v), ha='center', va='bottom', fontsize=8)
        ax1.set_xlabel("delay (minute(s))")
        ax2.set_xlabel("delay (minute(s))")
        plt.tight_layout()
        plt.show()

    def compute_df_1d(self, start=None, end=None):
        df_alpha = self.df_alpha
        if start is not None:
            df_alpha = df_alpha[df_alpha['day'] >= start]
        if end is not None:
            df_alpha = df_alpha[df_alpha['day'] <= end]

        self.df_1d = df_alpha \
            .groupby('day') \
            .agg({
                'grossProfit': 'sum',
                'netProfit': 'sum',
                'open': 'first',
                'turnover': 'sum'
            })
        self.df_1d['pctChange'] = self.df_1d['netProfit'] / self.df_1d['open']
        self.df_1d['cumNetProfit'] = self.df_1d['netProfit'].cumsum()
        self.df_1d = self.df_1d.reset_index()
        self.df_1d.index = self.df_1d['day']
        self.df_1d['netProfit'] = self.df_1d['netProfit'].round(2)
        # self.df_1d['hoverName'] = self.df_1d['day'] + \
        #                           ' ' + \
        #                           self.df_1d['netProfit'].astype(str)
        return self.df_1d

    def assert_all(self):
        label = f'\x1b[90mSimulator.assert_all()\x1b[0m: ' \
                f'({self.report["id"]}): '
        print(f"\r{label}\x1b[96mAsserting that all absolute "
              "position < 1\x1b[0m: ",
              end="")
        assert self.df_alpha['position'].abs().max() <= 1, \
            '\x1b[91mFailed [x]\n Sai, vẫn có giá trị > 1 \x1b[0m'
        print("\x1b[92m Passed [√]\x1b[0m")
        ###################################################################
        print(f"\r{label}\x1b[96mAsserting that cumNetProfit "
              "is computed correcttly\x1b[0m: ",
              end="")
        assert \
            (
                    (
                            self.df_alpha['position'] * self.df_alpha['priceChange']
                            -
                            self.df_alpha['position'].diff().abs() * self.fee
                    ).cumsum()
                    -
                    self.df_alpha['netProfit'].cumsum()
            ).abs().sum() \
            <= 1e-8, '\x1b[91mFailed [x]\n Sai \x1b[0m'
        print("\x1b[92m Passed [√]\x1b[0m")

    @staticmethod
    def create_equalizer(plot=False):
        fee = 0.2
        alpha_name, freq, inertia = "alpha_010", 4, 0.2
        # alpha_name, freq, inertia = "alpha_003", 9, 0.4,# Sharpe 4.17!
        # alpha_name, freq, inertia, = "alpha_099", 4, 0.1, # shapre 3
        # alpha_name, freq, inertia, = "alpha_099", 4, 0.8,  # shapre 2.88 long term
        # alpha_name, freq, inertia, = "alpha_084", 7, 0.7

        # bt = Simulator(alpha_name="alpha_068", freq=5, fee=0.2, inertia=2,
        # bt = Simulator(alpha_name="alpha_084", freq=7, fee=0.2, inertia=0.7,
        #                use_plotly=True, cb_func=C.CUT_ALL_BUT_LAST_15M)

        cb_func = C.CUT_ALL_BUT_LAST_15M

        bt = Simulator(alpha_name=alpha_name, freq=freq, fee=fee, inertia=inertia, cb_func=cb_func)
        bt.compute_signal()
        bt.compute_position()
        bt.compute_position_and_cut_book()
        bt.compute_tvr_and_fee()
        bt.compute_profits()
        bt.name = bt.report['alphaName'] = 'equalizer (alpha_010)'
        bt.compute_performance(return_df_1d=True)

        if plot:
            bt.compute_performance(plot=True)

        bt_eq = bt

        return bt_eq


class Scanner:
    def __init__(self, lst_alphas, is_threshold, bt_params,
                 min_freq=10, use_round=False, verbosity=1,
                 collector=None, shuffle=True, max_freq=None):
        self.use_round = use_round
        self.bt_params: Backtest_Params = bt_params
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.lst_alphas = lst_alphas
        self.lst_freqs = list(range(min_freq, max_freq + 1))
        self.lst_thresholds = Domains.gen_threshold_list()
        self.collector = collector

        self.lst_params = list(product(
            self.lst_alphas,
            self.lst_freqs,
            self.lst_thresholds))
        if shuffle:
            from random import shuffle as shuffle_list
            shuffle_list(self.lst_params)
        self.label = '\x1b[90mScanner()\x1b[0m:'
        self.is_threshold = is_threshold

        """SCAN LOOP VARIABLES"""
        self.loc = 0
        self.passed_count = 0
        self.is_running = True
        # noinspection PyTypeChecker
        self.bt: Simulator = None
        self.sharpe_is = -999
        self.max_err = -1
        # noinspection PyTypeChecker
        self.dp: type({}) = None
        self.verbosity = verbosity

        """PLACE-HOLDER"""
        self.dic_report_is = {}
        self.dic_report_os = {}
        self.dic_dfs_pos_1m = {}

    def run_main_loop(self, run_in_thread=False):
        def helper():
            while self.is_running:
                self.process_is()
                self.proceess_os()
                self.next()

        self.is_running = True
        if run_in_thread:
            alpha_func_lib.U.threading_func_wrapper(helper)
        else:
            helper()

    def next(self):
        self.loc += 1
        if self.loc == len(self.lst_params):
            self.is_running = False

    def announce(self):
        cond = self.loc == 0 or \
               self.loc >= len(self.lst_params) - 1 \
               or self.loc % 10 == 0
        if cond:
            pass_rate = 100 * len(self.dic_report_os) / len(self.lst_params)
            print(f"\r{self.label} "
                  f"{self.loc + 1:,} / {len(self.lst_params):,} "
                  f"\x1b[93m{self.bt.report['id']}\x1b[0m "
                  f"is_sharpe={self.bt.report['sharpe']:.4f} "
                  f"passed=\x1b[94m{len(self.dic_report_os):,}\x1b[0m "
                  f"(\x1b[92m{pass_rate:.1f}%\x1b[0m > "
                  f"\x1b[95m{self.is_threshold}\x1b[0m)",
                  end="")

    def process_is(self):
        alpha_name, freq, (upper, lower) = self.lst_params[self.loc]

        bt = self.bt = Simulator(
            alpha_name=alpha_name,
            freq=freq,
            upper=upper,
            lower=lower,
            fee=self.bt_params.fee_is)
        bt.compute_signal()
        bt.compute_position()
        if self.bt_params.is_book_cutter is not None:
            # noinspection PyCallingNonCallable
            self.bt_params.is_book_cutter(
                bt.df_alpha,
                end=self.bt_params.day_is_end)
        bt.compute_tvr_and_fee()
        bt.compute_profits()

        bt.report = report_is = bt.compute_performance(
            end=self.bt_params.day_is_end
        )
        self.dic_report_is[report_is['id']] = report_is
        self.announce()
        if self.use_round:
            sharpe_is = round(report_is['sharpe'], 2)
        else:
            sharpe_is = report_is['sharpe']

        self.sharpe_is = sharpe_is
        self.bt = bt

    def proceess_os(self):
        def maybe_announce_os():
            if self.verbosity >= 2:
                Alpha_Domains.print_report(
                    report_os,
                    prefix=f"{self.loc + 1:,} "
                           f"/ "
                           f"{len(self.lst_params):,} "
                           f"({self.passed_count:,} passed)")

        if self.sharpe_is >= self.is_threshold:
            bt = self.bt

            self.passed_count += 1
            bt.report['id'] = bt.report['id'].split('(')[0] + \
                              f'(fee={self.bt_params.fee_os:.2f})'
            if self.bt_params.os_book_cutter is not None:
                # noinspection PyCallingNonCallable
                self.bt_params.os_book_cutter(
                    bt.df_alpha,
                    end=self.bt_params.day_is_end)
            bt.compute_tvr_and_fee()
            bt.compute_profits()

            report_os = bt.compute_performance(
                start=self.bt_params.day_os_start,
                return_df_1d=True,
                plot=False)

            self.dic_report_os[report_os['id']] = report_os
            self.dic_dfs_pos_1m[bt.report['id']] = bt.compute_df_pos_1m()

            maybe_announce_os()
            if self.collector is not None:
                self.collector.collect_df_alpha(
                    bt.df_alpha,
                    name=bt.report['id'])
                self.collector.collect_df_1d(
                    df_1d=bt.df_1d,
                    report=report_os)


class Collector:
    def __init__(self, name, df_1m=None):
        if df_1m is None:
            df_1m = DF_1M.copy()

        self.lst_all_pos = []
        self.dic_df_1d = {}
        self.df_1m = df_1m
        self.name = name
        # noinspection PyTypeChecker
        self.bt: Simulator = None

        """PLACE-HOLDER"""
        self.df_all_pos = pd.DataFrame()
        self.df_pos_1m = pd.DataFrame()

    def collect_bt(self, bt):
        df_pos_1m = Alpha_Domains.collect_position(
            df_alpha=bt.df_alpha,
            name=bt.name,
            DF_1M=self.df_1m,
        )
        self.lst_all_pos.append(df_pos_1m)

    def collect_df_alpha(self, df_alpha, name):
        df_pos_1m = Alpha_Domains.collect_position(
            df_alpha=df_alpha,
            name=name,
            DF_1M=self.df_1m,
        )

        self.lst_all_pos.append(df_pos_1m)

    def collect_df_1d(self, df_1d, report):
        self.dic_df_1d[report['id']] = df_1d

    def create_bt(self, fee, prefix=""):
        if len(prefix) > 0: prefix = f"{prefix} "
        bt = Simulator(
            alpha_name=f"{prefix}({len(self.lst_all_pos)} alphas)",
            alpha_func=Simulator.dummy_alpha_func,
            fee=fee,
            freq=1, )
        bt.report['l1'] = 0
        self.maybe_compute_df_pos_1m()
        bt.df_alpha['position'] = self.df_pos_1m
        self.bt = bt

        return bt

    def combine_positions(self):
        df_all_pos = pd.concat(self.lst_all_pos, axis=1)
        df_pos = df_all_pos.sum(axis=1) / len(df_all_pos.columns)
        return df_all_pos, df_pos

    def maybe_compute_df_pos_1m(self):
        if len(self.df_pos_1m) == 0:
            self.df_all_pos, self.df_pos_1m = self.combine_positions()

    def plot_heatmap(self, title='Correlation'):
        def rename_duplicated_columns(df):
            seen_names = set()
            new_columns = []
            for col in df.columns:
                new_col = col
                i = 1
                while new_col in seen_names:
                    new_col = f"{col}{i}"
                    i += 1
                seen_names.add(new_col)
                new_columns.append(new_col)
            return new_columns

        df = self.df_all_pos.copy()
        df.columns = df.columns.map(lambda x: x.split('(')[0].split('_')[0])
        df.columns = rename_duplicated_columns(df)
        sorted_col = df.corr().mean().sort_values().index.to_list()
        df = df[sorted_col]
        correlation_matrix = df.corr()
        plt.figure(figsize=(8, 6))

        # Add column names to the axes
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5,
            vmin=0, vmax=1)

        # Center-align tick labels
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='center')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')
        plt.title(title)
        plt.show()


class HARDCODED:
    @staticmethod
    def get_hardcoded_alpha_095_54_configs():
        lst = [
            'alpha_095_42m 0.5_0.2 (fee=0.20)',
            'alpha_095_41m 1_0.2 (fee=0.20)',
            'alpha_095_41m 0.4_0.2 (fee=0.20)',
            'alpha_095_57m 0.4_0.1 (fee=0.20)',
            'alpha_095_47m 0.7_0.1 (fee=0.20)',
            'alpha_095_41m 0.7_0.2 (fee=0.20)',
            'alpha_095_50m 1_0.1 (fee=0.20)',
            'alpha_095_49m 1_0.9 (fee=0.20)',
            'alpha_095_49m 1_0.2 (fee=0.20)',
            'alpha_095_48m 0.9_0.4 (fee=0.20)',
            'alpha_095_49m 1_0.8 (fee=0.20)',
            'alpha_095_49m 1_0.1 (fee=0.20)',
            'alpha_095_49m 1_0.3 (fee=0.20)',
            'alpha_095_41m 0.7_0.6 (fee=0.20)',
            'alpha_095_48m 0.7_0.4 (fee=0.20)',
            'alpha_095_49m 1_0.5 (fee=0.20)',
            'alpha_095_49m 0.9_0.5 (fee=0.20)',
            'alpha_095_49m 0.9_0.3 (fee=0.20)',
            'alpha_095_41m 0.6_0.4 (fee=0.20)',
            'alpha_095_48m 0.7_0.2 (fee=0.20)',
            'alpha_095_48m 0.7_0.5 (fee=0.20)',
            'alpha_095_49m 1_0.6 (fee=0.20)',
            'alpha_095_49m 0.9_0.2 (fee=0.20)',
            'alpha_095_41m 0.7_0.3 (fee=0.20)',
            'alpha_095_57m 0.5_0.2 (fee=0.20)',
            'alpha_095_41m 0.6_0.3 (fee=0.20)',
            'alpha_095_41m 0.6_0.2 (fee=0.20)',
            'alpha_095_49m 1_0.4 (fee=0.20)',
            'alpha_095_49m 0.9_0.1 (fee=0.20)',
            'alpha_095_57m 0.4_0.2 (fee=0.20)',
            'alpha_095_49m 1_0.7 (fee=0.20)',
            'alpha_095_41m 0.7_0.1 (fee=0.20)',
            'alpha_095_41m 0.5_0.4 (fee=0.20)',
            'alpha_095_49m 0.9_0.7 (fee=0.20)',
            'alpha_095_49m 0.9_0.6 (fee=0.20)',
            'alpha_095_49m 0.9_0.4 (fee=0.20)',
            'alpha_095_48m 0.7_0.3 (fee=0.20)',
            'alpha_095_42m 0.5_0.1 (fee=0.20)',
            'alpha_095_41m 0.7_0.5 (fee=0.20)',
            'alpha_095_48m 1_0.2 (fee=0.20)',
            'alpha_095_41m 0.5_0.1 (fee=0.20)',
            'alpha_095_41m 1_0.3 (fee=0.20)',
            'alpha_095_41m 0.7_0.4 (fee=0.20)',
            'alpha_095_47m 0.6_0.2 (fee=0.20)',
            'alpha_095_41m 0.5_0.3 (fee=0.20)',
            'alpha_095_41m 0.6_0.1 (fee=0.20)',
            'alpha_095_41m 0.5_0.2 (fee=0.20)',
            'alpha_095_47m 0.5_0.2 (fee=0.20)',
            'alpha_095_41m 0.6_0.5 (fee=0.20)',
            'alpha_095_48m 0.8_0.2 (fee=0.20)',
            'alpha_095_48m 0.9_0.2 (fee=0.20)',
            'alpha_095_49m 0.9_0.8 (fee=0.20)',
            'alpha_095_48m 0.9_0.3 (fee=0.20)',
            'alpha_095_47m 0.7_0.2 (fee=0.20)']

        return lst

    @staticmethod
    def get_hardcoded_alpha_086_6_configs():
        lst = [
            'alpha_086_29m 0.6_0.1 (fee=0.20)',
            'alpha_086_29m 0.6_0.2 (fee=0.20)',
            'alpha_086_29m 1_0.1 (fee=0.20)',
            'alpha_086_29m 0.9_0.1 (fee=0.20)',
            'alpha_086_29m 0.5_0.1 (fee=0.20)',
            'alpha_086_29m 0.8_0.1 (fee=0.20)']

        return lst

    @staticmethod
    def get_hardcoded_alpha_bbb_64_configs():
        return ['alpha_bbb_44m 0.6_0.1 (fee=0.20)',
                'alpha_bbb_57m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_60m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_47m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_51m 0.5_0.4 (fee=0.20)',
                'alpha_bbb_38m 0.5_0.4 (fee=0.20)',
                'alpha_bbb_23m 0.5_0.4 (fee=0.20)',
                'alpha_bbb_23m 0.6_0.2 (fee=0.20)',
                'alpha_bbb_38m 0.5_0.2 (fee=0.20)',
                'alpha_bbb_51m 0.5_0.3 (fee=0.20)',
                'alpha_bbb_51m 0.5_0.2 (fee=0.20)',
                'alpha_bbb_44m 0.5_0.3 (fee=0.20)',
                'alpha_bbb_47m 0.2_0.1 (fee=0.20)',
                'alpha_bbb_60m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_47m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_57m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_47m 0.5_0.4 (fee=0.20)',
                'alpha_bbb_44m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_50m 0.4_0.3 (fee=0.20)',
                'alpha_bbb_23m 0.6_0.3 (fee=0.20)',
                'alpha_bbb_47m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_58m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_50m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_44m 0.7_0.1 (fee=0.20)',
                'alpha_bbb_41m 0.2_0.1 (fee=0.20)',
                'alpha_bbb_47m 0.5_0.3 (fee=0.20)',
                'alpha_bbb_47m 0.3_0.2 (fee=0.20)',
                'alpha_bbb_44m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_51m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_23m 0.6_0.4 (fee=0.20)',
                'alpha_bbb_59m 0.2_0.1 (fee=0.20)',
                'alpha_bbb_44m 0.5_0.4 (fee=0.20)',
                'alpha_bbb_33m 0.3_0.2 (fee=0.20)',
                'alpha_bbb_60m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_44m 0.4_0.3 (fee=0.20)',
                'alpha_bbb_37m 0.2_0.1 (fee=0.20)',
                'alpha_bbb_47m 0.6_0.2 (fee=0.20)',
                'alpha_bbb_51m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_50m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_59m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_23m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_60m 0.4_0.3 (fee=0.20)',
                'alpha_bbb_47m 0.6_0.1 (fee=0.20)',
                'alpha_bbb_33m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_44m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_51m 0.4_0.3 (fee=0.20)',
                'alpha_bbb_38m 0.5_0.1 (fee=0.20)',
                'alpha_bbb_47m 0.5_0.2 (fee=0.20)',
                'alpha_bbb_23m 0.6_0.5 (fee=0.20)',
                'alpha_bbb_58m 0.4_0.1 (fee=0.20)',
                'alpha_bbb_50m 0.3_0.2 (fee=0.20)',
                'alpha_bbb_53m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_23m 0.5_0.2 (fee=0.20)',
                'alpha_bbb_53m 0.3_0.2 (fee=0.20)',
                'alpha_bbb_51m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_23m 0.5_0.3 (fee=0.20)',
                'alpha_bbb_59m 0.3_0.2 (fee=0.20)',
                'alpha_bbb_23m 0.6_0.1 (fee=0.20)',
                'alpha_bbb_50m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_47m 0.4_0.2 (fee=0.20)',
                'alpha_bbb_38m 0.5_0.3 (fee=0.20)',
                'alpha_bbb_44m 0.5_0.2 (fee=0.20)',
                'alpha_bbb_56m 0.3_0.1 (fee=0.20)',
                'alpha_bbb_23m 0.4_0.1 (fee=0.20)']

    @staticmethod
    def get_hardcoded_zscore_busd_29_configs():
        return ['zscore_busd_46m 0.2_0.1 (fee=0.20)',
                'zscore_busd_30m 0.3_0.2 (fee=0.20)',
                'zscore_busd_23m 0.6_0.4 (fee=0.20)',
                'zscore_busd_40m 0.3_0.2 (fee=0.20)',
                'zscore_busd_53m 0.4_0.2 (fee=0.20)',
                'zscore_busd_47m 0.3_0.2 (fee=0.20)',
                'zscore_busd_41m 0.3_0.1 (fee=0.20)',
                'zscore_busd_47m 0.3_0.1 (fee=0.20)',
                'zscore_busd_60m 0.3_0.1 (fee=0.20)',
                'zscore_busd_37m 0.3_0.2 (fee=0.20)',
                'zscore_busd_40m 0.4_0.2 (fee=0.20)',
                'zscore_busd_60m 0.2_0.1 (fee=0.20)',
                'zscore_busd_41m 0.2_0.1 (fee=0.20)',
                'zscore_busd_23m 0.7_0.1 (fee=0.20)',
                'zscore_busd_41m 0.3_0.2 (fee=0.20)',
                'zscore_busd_47m 0.2_0.1 (fee=0.20)',
                'zscore_busd_60m 0.3_0.2 (fee=0.20)',
                'zscore_busd_46m 0.3_0.2 (fee=0.20)',
                'zscore_busd_36m 0.3_0.1 (fee=0.20)',
                'zscore_busd_40m 0.4_0.3 (fee=0.20)',
                'zscore_busd_37m 0.2_0.1 (fee=0.20)',
                'zscore_busd_59m 0.2_0.1 (fee=0.20)',
                'zscore_busd_53m 0.4_0.3 (fee=0.20)',
                'zscore_busd_23m 0.6_0.5 (fee=0.20)',
                'zscore_busd_53m 0.4_0.1 (fee=0.20)',
                'zscore_busd_40m 0.4_0.1 (fee=0.20)',
                'zscore_busd_46m 0.3_0.1 (fee=0.20)',
                'zscore_busd_37m 0.3_0.1 (fee=0.20)',
                'zscore_busd_23m 0.7_0.2 (fee=0.20)']

    @staticmethod
    def get_hardcoded_alpha_065_23_configs():
        return ['alpha_065_49m 0.2_0.1 (fee=0.20)',
                'alpha_065_45m 0.3_0.2 (fee=0.20)',
                'alpha_065_54m 0.2_0.1 (fee=0.20)',
                'alpha_065_25m 0.3_0.2 (fee=0.20)',
                'alpha_065_47m 0.3_0.1 (fee=0.20)',
                'alpha_065_47m 0.4_0.1 (fee=0.20)',
                'alpha_065_23m 0.5_0.1 (fee=0.20)',
                'alpha_065_47m 0.4_0.2 (fee=0.20)',
                'alpha_065_50m 0.3_0.2 (fee=0.20)',
                'alpha_065_50m 0.3_0.1 (fee=0.20)',
                'alpha_065_53m 0.5_0.2 (fee=0.20)',
                'alpha_065_46m 0.2_0.1 (fee=0.20)',
                'alpha_065_40m 0.6_0.2 (fee=0.20)',
                'alpha_065_44m 0.3_0.1 (fee=0.20)',
                'alpha_065_44m 0.3_0.2 (fee=0.20)',
                'alpha_065_40m 0.6_0.1 (fee=0.20)',
                'alpha_065_45m 0.3_0.1 (fee=0.20)',
                'alpha_065_23m 0.5_0.4 (fee=0.20)',
                'alpha_065_23m 0.5_0.2 (fee=0.20)',
                'alpha_065_23m 0.5_0.3 (fee=0.20)',
                'alpha_065_23m 0.2_0.1 (fee=0.20)',
                'alpha_065_47m 0.4_0.3 (fee=0.20)',
                'alpha_065_47m 0.3_0.2 (fee=0.20)']

    @staticmethod
    def get_hardcoded_alpha_018_195_configs():
        return ['alpha_018_59m 0.9_0.3 (fee=0.20)',
                'alpha_018_41m 0.7_0.1 (fee=0.20)',
                'alpha_018_41m 0.7_0.4 (fee=0.20)',
                'alpha_018_41m 1_0.8 (fee=0.20)',
                'alpha_018_59m 1_0.4 (fee=0.20)',
                'alpha_018_37m 0.7_0.5 (fee=0.20)',
                'alpha_018_37m 1_0.5 (fee=0.20)',
                'alpha_018_49m 0.6_0.1 (fee=0.20)',
                'alpha_018_49m 0.9_0.1 (fee=0.20)',
                'alpha_018_49m 0.8_0.5 (fee=0.20)',
                'alpha_018_41m 0.6_0.5 (fee=0.20)',
                'alpha_018_47m 1_0.7 (fee=0.20)',
                'alpha_018_58m 0.7_0.2 (fee=0.20)',
                'alpha_018_46m 0.5_0.1 (fee=0.20)',
                'alpha_018_37m 0.8_0.7 (fee=0.20)',
                'alpha_018_59m 0.7_0.5 (fee=0.20)',
                'alpha_018_41m 0.8_0.4 (fee=0.20)',
                'alpha_018_41m 0.4_0.2 (fee=0.20)',
                'alpha_018_50m 0.9_0.4 (fee=0.20)',
                'alpha_018_41m 0.8_0.2 (fee=0.20)',
                'alpha_018_42m 1_0.5 (fee=0.20)',
                'alpha_018_50m 0.8_0.1 (fee=0.20)',
                'alpha_018_42m 0.7_0.5 (fee=0.20)',
                'alpha_018_41m 0.9_0.6 (fee=0.20)',
                'alpha_018_59m 0.7_0.4 (fee=0.20)',
                'alpha_018_47m 1_0.2 (fee=0.20)',
                'alpha_018_47m 0.8_0.4 (fee=0.20)',
                'alpha_018_46m 1_0.1 (fee=0.20)',
                'alpha_018_46m 1_0.2 (fee=0.20)',
                'alpha_018_47m 1_0.4 (fee=0.20)',
                'alpha_018_41m 0.7_0.3 (fee=0.20)',
                'alpha_018_47m 0.9_0.6 (fee=0.20)',
                'alpha_018_41m 1_0.5 (fee=0.20)',
                'alpha_018_60m 0.9_0.8 (fee=0.20)',
                'alpha_018_41m 0.5_0.3 (fee=0.20)',
                'alpha_018_50m 0.8_0.3 (fee=0.20)',
                'alpha_018_50m 0.8_0.4 (fee=0.20)',
                'alpha_018_46m 0.6_0.2 (fee=0.20)',
                'alpha_018_42m 0.8_0.2 (fee=0.20)',
                'alpha_018_47m 0.8_0.6 (fee=0.20)',
                'alpha_018_49m 0.7_0.5 (fee=0.20)',
                'alpha_018_47m 1_0.8 (fee=0.20)',
                'alpha_018_59m 0.9_0.4 (fee=0.20)',
                'alpha_018_42m 0.9_0.5 (fee=0.20)',
                'alpha_018_37m 0.6_0.5 (fee=0.20)',
                'alpha_018_37m 0.8_0.6 (fee=0.20)',
                'alpha_018_49m 0.6_0.3 (fee=0.20)',
                'alpha_018_46m 0.8_0.1 (fee=0.20)',
                'alpha_018_41m 0.4_0.3 (fee=0.20)',
                'alpha_018_41m 0.8_0.6 (fee=0.20)',
                'alpha_018_42m 0.9_0.4 (fee=0.20)',
                'alpha_018_37m 0.9_0.7 (fee=0.20)',
                'alpha_018_42m 0.8_0.4 (fee=0.20)',
                'alpha_018_46m 0.9_0.2 (fee=0.20)',
                'alpha_018_47m 0.8_0.5 (fee=0.20)',
                'alpha_018_47m 0.9_0.7 (fee=0.20)',
                'alpha_018_37m 0.9_0.4 (fee=0.20)',
                'alpha_018_47m 0.8_0.2 (fee=0.20)',
                'alpha_018_50m 0.8_0.6 (fee=0.20)',
                'alpha_018_58m 0.7_0.4 (fee=0.20)',
                'alpha_018_49m 1_0.1 (fee=0.20)',
                'alpha_018_50m 1_0.6 (fee=0.20)',
                'alpha_018_59m 0.8_0.2 (fee=0.20)',
                'alpha_018_41m 1_0.6 (fee=0.20)',
                'alpha_018_50m 0.9_0.5 (fee=0.20)',
                'alpha_018_46m 0.4_0.1 (fee=0.20)',
                'alpha_018_41m 1_0.9 (fee=0.20)',
                'alpha_018_42m 0.8_0.1 (fee=0.20)',
                'alpha_018_37m 0.9_0.5 (fee=0.20)',
                'alpha_018_47m 0.9_0.1 (fee=0.20)',
                'alpha_018_37m 1_0.6 (fee=0.20)',
                'alpha_018_58m 0.6_0.5 (fee=0.20)',
                'alpha_018_59m 0.9_0.5 (fee=0.20)',
                'alpha_018_42m 0.8_0.5 (fee=0.20)',
                'alpha_018_49m 0.6_0.2 (fee=0.20)',
                'alpha_018_59m 0.6_0.4 (fee=0.20)',
                'alpha_018_37m 0.7_0.4 (fee=0.20)',
                'alpha_018_60m 0.9_0.5 (fee=0.20)',
                'alpha_018_46m 0.6_0.3 (fee=0.20)',
                'alpha_018_37m 0.7_0.6 (fee=0.20)',
                'alpha_018_47m 0.8_0.7 (fee=0.20)',
                'alpha_018_50m 0.8_0.2 (fee=0.20)',
                'alpha_018_59m 0.6_0.5 (fee=0.20)',
                'alpha_018_59m 0.6_0.2 (fee=0.20)',
                'alpha_018_50m 0.9_0.3 (fee=0.20)',
                'alpha_018_59m 0.9_0.2 (fee=0.20)',
                'alpha_018_41m 1_0.1 (fee=0.20)',
                'alpha_018_59m 0.8_0.3 (fee=0.20)',
                'alpha_018_60m 0.8_0.5 (fee=0.20)',
                'alpha_018_58m 0.7_0.1 (fee=0.20)',
                'alpha_018_50m 1_0.1 (fee=0.20)',
                'alpha_018_50m 1_0.4 (fee=0.20)',
                'alpha_018_42m 0.8_0.3 (fee=0.20)',
                'alpha_018_41m 0.7_0.2 (fee=0.20)',
                'alpha_018_50m 0.8_0.7 (fee=0.20)',
                'alpha_018_41m 1_0.2 (fee=0.20)',
                'alpha_018_42m 0.6_0.4 (fee=0.20)',
                'alpha_018_50m 1_0.2 (fee=0.20)',
                'alpha_018_41m 0.4_0.1 (fee=0.20)',
                'alpha_018_50m 0.8_0.5 (fee=0.20)',
                'alpha_018_42m 0.9_0.1 (fee=0.20)',
                'alpha_018_60m 1_0.4 (fee=0.20)',
                'alpha_018_47m 0.8_0.1 (fee=0.20)',
                'alpha_018_59m 0.7_0.3 (fee=0.20)',
                'alpha_018_49m 0.8_0.2 (fee=0.20)',
                'alpha_018_59m 0.6_0.3 (fee=0.20)',
                'alpha_018_49m 0.6_0.4 (fee=0.20)',
                'alpha_018_47m 1_0.6 (fee=0.20)',
                'alpha_018_60m 1_0.8 (fee=0.20)',
                'alpha_018_37m 0.9_0.6 (fee=0.20)',
                'alpha_018_58m 0.6_0.1 (fee=0.20)',
                'alpha_018_47m 0.9_0.4 (fee=0.20)',
                'alpha_018_42m 0.9_0.2 (fee=0.20)',
                'alpha_018_58m 0.7_0.5 (fee=0.20)',
                'alpha_018_42m 1_0.2 (fee=0.20)',
                'alpha_018_41m 0.6_0.1 (fee=0.20)',
                'alpha_018_46m 0.9_0.1 (fee=0.20)',
                'alpha_018_37m 1_0.4 (fee=0.20)',
                'alpha_018_37m 0.7_0.1 (fee=0.20)',
                'alpha_018_41m 0.8_0.7 (fee=0.20)',
                'alpha_018_41m 0.9_0.5 (fee=0.20)',
                'alpha_018_49m 0.7_0.3 (fee=0.20)',
                'alpha_018_59m 0.8_0.5 (fee=0.20)',
                'alpha_018_42m 0.7_0.4 (fee=0.20)',
                'alpha_018_47m 1_0.9 (fee=0.20)',
                'alpha_018_60m 1_0.9 (fee=0.20)',
                'alpha_018_46m 0.6_0.1 (fee=0.20)',
                'alpha_018_49m 0.6_0.5 (fee=0.20)',
                'alpha_018_41m 1_0.3 (fee=0.20)',
                'alpha_018_47m 0.9_0.5 (fee=0.20)',
                'alpha_018_42m 1_0.1 (fee=0.20)',
                'alpha_018_47m 0.8_0.3 (fee=0.20)',
                'alpha_018_49m 0.7_0.4 (fee=0.20)',
                'alpha_018_60m 1_0.5 (fee=0.20)',
                'alpha_018_37m 1_0.7 (fee=0.20)',
                'alpha_018_50m 1_0.3 (fee=0.20)',
                'alpha_018_46m 0.7_0.3 (fee=0.20)',
                'alpha_018_59m 0.7_0.2 (fee=0.20)',
                'alpha_018_46m 0.7_0.2 (fee=0.20)',
                'alpha_018_58m 0.6_0.2 (fee=0.20)',
                'alpha_018_41m 0.8_0.5 (fee=0.20)',
                'alpha_018_41m 0.6_0.3 (fee=0.20)',
                'alpha_018_41m 0.8_0.1 (fee=0.20)',
                'alpha_018_59m 1_0.5 (fee=0.20)',
                'alpha_018_41m 0.8_0.3 (fee=0.20)',
                'alpha_018_59m 0.7_0.1 (fee=0.20)',
                'alpha_018_46m 0.9_0.3 (fee=0.20)',
                'alpha_018_41m 0.5_0.4 (fee=0.20)',
                'alpha_018_58m 0.6_0.4 (fee=0.20)',
                'alpha_018_60m 0.9_0.4 (fee=0.20)',
                'alpha_018_47m 1_0.1 (fee=0.20)',
                'alpha_018_58m 0.6_0.3 (fee=0.20)',
                'alpha_018_59m 0.8_0.4 (fee=0.20)',
                'alpha_018_50m 0.9_0.7 (fee=0.20)',
                'alpha_018_41m 0.9_0.2 (fee=0.20)',
                'alpha_018_42m 0.9_0.3 (fee=0.20)',
                'alpha_018_42m 1_0.4 (fee=0.20)',
                'alpha_018_41m 0.9_0.7 (fee=0.20)',
                'alpha_018_50m 1_0.5 (fee=0.20)',
                'alpha_018_41m 1_0.7 (fee=0.20)',
                'alpha_018_50m 0.9_0.1 (fee=0.20)',
                'alpha_018_41m 0.9_0.4 (fee=0.20)',
                'alpha_018_46m 1_0.3 (fee=0.20)',
                'alpha_018_46m 0.7_0.1 (fee=0.20)',
                'alpha_018_47m 1_0.5 (fee=0.20)',
                'alpha_018_47m 0.9_0.8 (fee=0.20)',
                'alpha_018_49m 0.8_0.1 (fee=0.20)',
                'alpha_018_50m 1_0.7 (fee=0.20)',
                'alpha_018_49m 0.7_0.2 (fee=0.20)',
                'alpha_018_49m 0.8_0.4 (fee=0.20)',
                'alpha_018_37m 0.8_0.5 (fee=0.20)',
                'alpha_018_42m 0.6_0.5 (fee=0.20)',
                'alpha_018_50m 0.9_0.6 (fee=0.20)',
                'alpha_018_47m 1_0.3 (fee=0.20)',
                'alpha_018_59m 1_0.2 (fee=0.20)',
                'alpha_018_47m 0.9_0.3 (fee=0.20)',
                'alpha_018_41m 0.5_0.2 (fee=0.20)',
                'alpha_018_42m 1_0.3 (fee=0.20)',
                'alpha_018_58m 0.7_0.3 (fee=0.20)',
                'alpha_018_41m 1_0.4 (fee=0.20)',
                'alpha_018_50m 0.9_0.2 (fee=0.20)',
                'alpha_018_59m 0.6_0.1 (fee=0.20)',
                'alpha_018_41m 0.6_0.2 (fee=0.20)',
                'alpha_018_47m 0.9_0.2 (fee=0.20)',
                'alpha_018_49m 0.8_0.3 (fee=0.20)',
                'alpha_018_49m 0.7_0.1 (fee=0.20)',
                'alpha_018_41m 0.5_0.1 (fee=0.20)',
                'alpha_018_37m 0.8_0.4 (fee=0.20)',
                'alpha_018_41m 0.9_0.3 (fee=0.20)',
                'alpha_018_60m 0.8_0.4 (fee=0.20)',
                'alpha_018_41m 0.9_0.1 (fee=0.20)',
                'alpha_018_37m 0.6_0.4 (fee=0.20)',
                'alpha_018_41m 0.6_0.4 (fee=0.20)',
                'alpha_018_41m 0.9_0.8 (fee=0.20)',
                'alpha_018_59m 1_0.3 (fee=0.20)']

    @staticmethod
    def get_hardcoded_alpha_zscore_150_configs():
        return ['alpha_zscore_25m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.4 (fee=0.10)',
                'alpha_zscore_60m 0.8_0.6 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.2 (fee=0.10)',
                'alpha_zscore_25m 0.2_0.1 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.4 (fee=0.10)',
                'alpha_zscore_41m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_49m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_38m 0.7_0.5 (fee=0.10)',
                'alpha_zscore_38m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_50m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_41m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_41m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_30m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.6 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.5 (fee=0.10)',
                'alpha_zscore_38m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.4 (fee=0.10)',
                'alpha_zscore_41m 0.4_0.1 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.6 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.2 (fee=0.10)',
                'alpha_zscore_41m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_32m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_49m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_47m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_42m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_60m 0.9_0.5 (fee=0.10)',
                'alpha_zscore_49m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_28m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_49m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_32m 0.7_0.1 (fee=0.10)',
                'alpha_zscore_41m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_41m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_41m 0.7_0.6 (fee=0.10)',
                'alpha_zscore_32m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_50m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_51m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_23m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_42m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.7 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.5 (fee=0.10)',
                'alpha_zscore_28m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_25m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_38m 0.7_0.6 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.3 (fee=0.10)',
                'alpha_zscore_41m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_30m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_32m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_51m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_34m 0.8_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.1 (fee=0.10)',
                'alpha_zscore_60m 0.8_0.5 (fee=0.10)',
                'alpha_zscore_42m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_35m 0.4_0.1 (fee=0.10)',
                'alpha_zscore_49m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_42m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_42m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_34m 0.7_0.1 (fee=0.10)',
                'alpha_zscore_25m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_23m 0.7_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.3 (fee=0.10)',
                'alpha_zscore_38m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_30m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.3 (fee=0.10)',
                'alpha_zscore_50m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_51m 0.8_0.1 (fee=0.10)',
                'alpha_zscore_38m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_34m 0.8_0.2 (fee=0.10)',
                'alpha_zscore_50m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_32m 0.7_0.3 (fee=0.10)',
                'alpha_zscore_51m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_47m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_50m 0.4_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.5 (fee=0.10)',
                'alpha_zscore_47m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_28m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_47m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_30m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.6 (fee=0.10)',
                'alpha_zscore_49m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_49m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_60m 0.9_0.8 (fee=0.10)',
                'alpha_zscore_58m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.6 (fee=0.10)',
                'alpha_zscore_51m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_30m 0.4_0.1 (fee=0.10)',
                'alpha_zscore_34m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_25m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_41m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_47m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_41m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_25m 0.4_0.1 (fee=0.10)',
                'alpha_zscore_38m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_50m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_47m 0.3_0.2 (fee=0.10)',
                'alpha_zscore_51m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_23m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_41m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_35m 0.3_0.2 (fee=0.10)',
                'alpha_zscore_35m 0.4_0.3 (fee=0.10)',
                'alpha_zscore_29m 0.2_0.1 (fee=0.10)',
                'alpha_zscore_30m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_42m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_42m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_28m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_60m 0.9_0.7 (fee=0.10)',
                'alpha_zscore_30m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_41m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_30m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.1 (fee=0.10)',
                'alpha_zscore_29m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_51m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.5 (fee=0.10)',
                'alpha_zscore_38m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_47m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_51m 0.5_0.2 (fee=0.10)',
                'alpha_zscore_50m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_25m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.4 (fee=0.10)',
                'alpha_zscore_51m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.8 (fee=0.10)',
                'alpha_zscore_50m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_42m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_35m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_50m 0.6_0.2 (fee=0.10)',
                'alpha_zscore_25m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_47m 0.3_0.1 (fee=0.10)',
                'alpha_zscore_38m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_29m 0.3_0.2 (fee=0.10)',
                'alpha_zscore_42m 0.6_0.1 (fee=0.10)',
                'alpha_zscore_60m 0.9_0.6 (fee=0.10)',
                'alpha_zscore_32m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_51m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.3 (fee=0.10)',
                'alpha_zscore_50m 0.5_0.4 (fee=0.10)',
                'alpha_zscore_49m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_53m 0.8_0.1 (fee=0.10)',
                'alpha_zscore_51m 0.6_0.5 (fee=0.10)',
                'alpha_zscore_53m 0.7_0.2 (fee=0.10)',
                'alpha_zscore_28m 0.6_0.4 (fee=0.10)',
                'alpha_zscore_35m 0.4_0.2 (fee=0.10)',
                'alpha_zscore_50m 0.5_0.3 (fee=0.10)',
                'alpha_zscore_38m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_50m 0.6_0.3 (fee=0.10)',
                'alpha_zscore_53m 0.9_0.7 (fee=0.10)',
                'alpha_zscore_58m 0.8_0.2 (fee=0.10)',
                'alpha_zscore_23m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_38m 0.5_0.1 (fee=0.10)',
                'alpha_zscore_60m 0.8_0.7 (fee=0.10)']


class Book_Cutter:
    @staticmethod
    def cut_morning_session(df_alpha, start=None, end=None, return_flt=False):
        idx = True
        if start is not None:
            idx = idx & (df_alpha['day'] >= start)
        if end is not None:
            idx = idx & (df_alpha['day'] <= end)
        flt = df_alpha['executionTime'] <= '13:35:00'
        if return_flt: return flt
        df_alpha.loc[idx & flt, 'position'] = 0

    @staticmethod
    def cut_afternoon_session(df_alpha, start=None, end=None, return_flt=False):
        idx = True
        if start is not None:
            idx = idx & (df_alpha['day'] >= start)
        if end is not None:
            idx = idx & (df_alpha['day'] <= end)
        flt = df_alpha['executionTime'] > '11:28:00'
        if return_flt: return flt
        df_alpha.loc[idx & flt, 'position'] = 0

    @staticmethod
    def cut_all_but_last_55m(df_alpha, start=None, end=None, return_flt=False):
        idx = True
        if start is not None:
            idx = idx & (df_alpha['day'] >= start)
        if end is not None:
            idx = idx & (df_alpha['day'] <= end)
        flt = df_alpha['executionTime'] > '13:35:00'
        if return_flt: return flt
        df_alpha.loc[idx & flt, 'position'] = 0

    @staticmethod
    def cut_all_but_last_15m(df_alpha, start=None, end=None, return_flt=False):
        idx = True
        if start is not None:
            idx = idx & (df_alpha['day'] >= start)
        if end is not None:
            idx = idx & (df_alpha['day'] <= end)
        flt = df_alpha['time'] < "14:15:00"
        if return_flt: return flt
        df_alpha.loc[idx & flt, 'position'] = 0

    @staticmethod
    def cut_all_but_last_30m(df_alpha, start=None, end=None, return_flt=False):
        idx = True
        if start is not None:
            idx = idx & (df_alpha['day'] >= start)
        if end is not None:
            idx = idx & (df_alpha['day'] <= end)
        flt = df_alpha['time'] < "14:00:00"
        if return_flt: return flt
        df_alpha.loc[idx & flt, 'position'] = 0

    @staticmethod
    def get_book_cutter(func_name):
        # noinspection PyGlobalUndefined
        global dic_book_cutters
        if 'dic_book_cutters' not in globals():
            dic_book_cutters = {
                C.CUT_ALL_BUT_LAST_15M: Book_Cutter.cut_all_but_last_15m,
                C.CUT_MORNING_SESSION: Book_Cutter.cut_morning_session,
            }
        return dic_book_cutters[func_name]


#####################################################################################
def init(bt_params):
    # noinspection PyGlobalUndefined
    global DIC_FREQS, DIC_ALPHAS

    pd.options.display.max_rows = 50
    pd.options.display.max_columns = 50
    pd.options.display.width = 350
    pd.options.display.float_format = '{:.2f}'.format

    if len(DIC_FREQS) == 0:
        with open('/tmp/dic_freqs_with_busd.pickle', 'rb') as file:
            dic = pickle.load(file)
        for key in dic: DIC_FREQS[key] = dic[key]
        for freq in DIC_FREQS:
            df = DIC_FREQS[freq]
            DIC_FREQS[freq] = df[(df['day'] >= bt_params.day_is_start) &
                                 (df['day'] <= bt_params.day_os_end)]

        dic = alpha_func_lib.Domains.get_list_of_alphas()
        for alpha_name, alpha_func in dic.items():
            DIC_ALPHAS[alpha_name] = alpha_func

            DIC_ALPHAS['zscore_busd'] = Alpha_Funcs.alpha_zscore_busd
            DIC_ALPHAS['alpha_018_busd'] = Alpha_Funcs.alpha_018_busd
            DIC_ALPHAS['alpha_bbb'] = Alpha_Funcs.alpha_bb_breakout
            DIC_ALPHAS['alpha_238'] = Alpha_Funcs.alpha_238

    # noinspection PyGlobalUndefined
    global DIC_TIME_TO_INDEX, DIC_INDEX_TO_TIME, DF_1M
    if 'DIC_TIME_TO_INDEX' not in globals():
        try:
            with open('/tmp/time_to_index.pickle', 'rb') as file:
                DIC_TIME_TO_INDEX, DIC_INDEX_TO_TIME = pickle.load(file)
        except Exception as e:
            print(f"Error when loading DIC_TIME_TO_INDEX from ssd: {e}")
            DIC_TIME_TO_INDEX, DIC_INDEX_TO_TIME = \
                alpha_busd_lib.Adapters.load_time_to_index_from_db()

        DF_1M = DIC_FREQS[1].copy()
    init_plot()
    return DIC_ALPHAS, DIC_FREQS, DIC_TIME_TO_INDEX, DIC_INDEX_TO_TIME, DF_1M

#####################################################################################

def main():
    def get_zscore_busd_37_hardcoded_configs():
        configs = """
            zscore_busd_31m 0.4_0.1 (fee=0.20)
            zscore_busd_31m 0.3_0.1 (fee=0.20)
            zscore_busd_47m 0.3_0.2 (fee=0.20)
            zscore_busd_31m 0.4_0.2 (fee=0.20)
            zscore_busd_53m 0.5_0.4 (fee=0.20)
            zscore_busd_41m 0.3_0.1 (fee=0.20)
            zscore_busd_53m 0.4_0.2 (fee=0.20)
            zscore_busd_38m 0.3_0.2 (fee=0.20)
            zscore_busd_53m 0.4_0.3 (fee=0.20)
            zscore_busd_40m 0.2_0.1 (fee=0.20)
            zscore_busd_25m 0.3_0.1 (fee=0.20)
            zscore_busd_47m 0.2_0.1 (fee=0.20)
            zscore_busd_53m 0.4_0.1 (fee=0.20)
            zscore_busd_47m 0.3_0.1 (fee=0.20)
            zscore_busd_53m 0.3_0.1 (fee=0.20)
            zscore_busd_35m 0.3_0.1 (fee=0.20)
            zscore_busd_30m 0.3_0.1 (fee=0.20)
            zscore_busd_46m 0.2_0.1 (fee=0.20)
            zscore_busd_25m 0.2_0.1 (fee=0.20)
            zscore_busd_46m 0.3_0.1 (fee=0.20)
            zscore_busd_25m 0.4_0.1 (fee=0.20)
            zscore_busd_53m 0.5_0.1 (fee=0.20)
            zscore_busd_54m 0.2_0.1 (fee=0.20)
            zscore_busd_31m 0.5_0.2 (fee=0.20)
            zscore_busd_53m 0.5_0.2 (fee=0.20)
            zscore_busd_31m 0.5_0.1 (fee=0.20)
            zscore_busd_30m 0.3_0.2 (fee=0.20)
            zscore_busd_53m 0.3_0.2 (fee=0.20)
            zscore_busd_35m 0.2_0.1 (fee=0.20)
            zscore_busd_41m 0.3_0.2 (fee=0.20)
            zscore_busd_41m 0.2_0.1 (fee=0.20)
            zscore_busd_53m 0.5_0.3 (fee=0.20)
            zscore_busd_32m 0.2_0.1 (fee=0.20)
            zscore_busd_31m 0.4_0.3 (fee=0.20)
            zscore_busd_32m 0.3_0.1 (fee=0.20)
            zscore_busd_31m 0.5_0.3 (fee=0.20)
            zscore_busd_52m 0.2_0.1 (fee=0.20)""".split('\n')
        configs = [x.strip() for x in configs if len(x.strip()) > 0]

        return configs

    BT_PARAMS = Backtest_Params(
        day_is_start='2022_10_03',
        day_is_end='2023_06_01',
        day_os_start='2023_06_01',
        is_book_cutter=Book_Cutter.cut_morning_session,
        os_book_cutter=Book_Cutter.cut_morning_session,
        div=0,
        fee_is=0.2, fee_os=0.2)
    DIC_ALPHAS, DIC_FREQS, _, _, DF_1M = init(BT_PARAMS)

    bt, collector, os_report, df_1d = Alpha_Domains.combine_alphas_and_compute_performance(
        configs=get_zscore_busd_37_hardcoded_configs(),
        fee=BT_PARAMS.fee_os, l1=1.7,
        alpha_name='zscore_busd',
        plot=False,
        start=BT_PARAMS.day_os_start, end=None)
    bt.compute_overnight_profit(plot_on=True, start=BT_PARAMS.day_os_start)
    bt.df_alpha = Alpha_Domains.add_overnight_performance(bt, plot_on=False)

    # pf=4.6, sharpe=5.49 tvr=120% mdd=4.9%
    _, os_report_on, df_1d_on = Alpha_Domains2.compute_performance_stats(
        bt.df_alpha,
        skip_profit=True,
        start=BT_PARAMS.day_os_start,
        alpha_name='zscore_busd_37')

DIC_ALPHAS, DIC_FREQS = {}, {}

if __name__ == '__main__': main()