import pandas as pd
import numpy as np
class Simulator:
    def __init__(self, alpha_name, freq=1, upper=None, lower=None, fee=0.1, df_alpha=None,DIC_ALPHAS=None, df_1m = None, cutTime = None ):

        self.freq = freq
        self.upper = upper
        self.lower = lower
        self.fee = fee
        
        # df_alpha = df_alpha[(df_alpha['day'] >= '2024_01_01') & (df_alpha['day'] <= '2024_12_31')].copy()
        df_alpha = df_alpha.copy()
        self.df_alpha = df_alpha
        self.df_1m = df_1m
        self.cutTime = cutTime
        
        
        if DIC_ALPHAS:
            self.alpha_func = DIC_ALPHAS[alpha_name]
    
        self.report = {
            'alphaName': alpha_name,
            'sharpe': np.nan,
            'freq': self.freq,
            'fee': self.fee,
            'upper' : self.upper,
            'lower' : self.lower
        }
        self.report['id'] = Alpha_Domains.compute_alpha_id(self.report,in_place=False)
        self.name = self.report['id']
        self.id = self.report['id']

        

    def compute_signal(self):
                
        Alpha_Domains.compute_signal(alpha_func=self.alpha_func, df_alpha=self.df_alpha)
        
       


    def compute_position(self):
        
        Alpha_Domains.compute_positions_with_thresholds(
            df_alpha=self.df_alpha,
            upper=self.upper,
            lower=self.lower)
        
        if self.cutTime:
            self.change_to_df1m()
        
        
    def change_to_df1m(self):       
        df_1M = self.df_1m.reset_index().merge(self.df_alpha[['position', 'executionT']],
                       on='executionT',
                       how='left').set_index('groupTime')
        df_1M['position'] = df_1M['position'].ffill().fillna(0)
        
        df_1M.loc[df_1M['executionTime'] >= self.cutTime, 'position'] = 0.0
        self.df_alpha = df_1M
        
      

    def compute_tvr_and_fee(self):
        Alpha_Domains.compute_action_tvr_and_fee(
            self.df_alpha,
            self.fee)

    def compute_profits(self):
        Alpha_Domains.compute_profits(self.df_alpha)

    def compute_performance(self, start=None, end=None):
 
        self.df_1d, report = Alpha_Domains.compute_performance(
            self.df_alpha,
            start=start,
            end=end)
        self.report.update(report)

  

class Alpha_Domains:
    
    @staticmethod
    def compute_alpha_id(report, in_place=False):
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
    def compute_signal(alpha_func, df_alpha):
        df_alpha['position'] = df_alpha['signal'] = alpha_func(df_alpha)


    @staticmethod
    def adjust_positions(df_alpha):
        flt_unexecutable = ~df_alpha['executable']
        df_alpha.loc[flt_unexecutable, 'position'] = np.nan
        flt_atc = df_alpha['executionTime'] == '14:45:00'
        df_alpha.loc[flt_atc, 'position'] = 0
        # df_alpha['position'] = df_alpha['position'].fillna(method='ffill') \
        #     .fillna(0)
        df_alpha['position'] = df_alpha['position'].ffill().fillna(0)

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
        
        # uptrend_condition = (df_alpha['ha_close'] > df_alpha['ha_open'])
        # downtrend_condition = (df_alpha['ha_close'] < df_alpha['ha_open'])
        # conditions = [uptrend_condition, downtrend_condition]
        # choices = [1, -1]  # 1 cho Up, -1 cho Down
        # df_alpha['HA_Direction'] = np.select(
        #     conditions, 
        #     choices, 
        #     default=0
        # )
        # N = 3
        # df_alpha['is_strong_uptrend'] = df_alpha['HA_Direction'].rolling(N).apply(lambda x: (x == 1).all(), raw=True)
        # df_alpha['is_strong_downtrend'] = df_alpha['HA_Direction'].rolling(N).apply(lambda x: (x == -1).all(), raw=True)
           
        # conditions = [
        #     (df_alpha['position'] == 1) & (df_alpha['is_strong_uptrend'] == 1),
        #     (df_alpha['position'] == -1) & (df_alpha['is_strong_downtrend'] == 1)
        # ]
        # choices = [1, -1]
        # df_alpha['position'] = np.select(conditions, choices, default=0)
        Alpha_Domains.adjust_positions(df_alpha)

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
    def safe_sharpe(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return x.mean() / x.std() * np.sqrt(250)
        
    @staticmethod
    def compute_performance(df_alpha, start=None, end=None):
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
            
            # sharpe = df_1d['pctChange'].mean() / df_1d['pctChange'].std() * 250 ** 0.5
            sharpe = Alpha_Domains.safe_sharpe(df_1d["netProfit"])
        except Exception as e:
            lst_errs.append(f"{e}")
            # U.report_error(e)
            sharpe = -999
        roe = df_1d['pctChange'].sum()
        tvr = df_1d['turnover'].mean()
        aroe = roe * 250 / len(df_1d)
        ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)

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
            'netProfit' : round(df_1d['netProfit'].sum(), 2),
            'lastProfit': round(df_1d['netProfit'].iloc[-1], 2)
        }

        df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        df_1d = df_1d.reset_index()
        df_1d.index = df_1d['day'].values
        df_1d['netProfit'] = df_1d['netProfit'].round(2)
        return df_1d, new_report
    
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
        # cdd, mdd = lst_cdd[-1], lst_mdd[-1]
        cdd, mdd = lst_cdd.iloc[-1], lst_mdd.iloc[-1]
        # cdd_pct, mdd_pct = lst_cdd_pct[-1], lst_mdd_pct[-1]
        cdd_pct, mdd_pct = lst_cdd_pct.iloc[-1], lst_mdd_pct.iloc[-1]

        if verbosity >= 2:
            print(f"mdd=\x1b[93m{mdd:,.2f}\x1b[0m mdd_pct={100 * mdd_pct:.1f}% "
                  f"cdd=\x1b[93m{cdd:,.2f}\x1b[0m cdd_pct={100 * cdd_pct:.1f}%")

        return mdd, mdd_pct, cdd, cdd_pct



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

        return df_execution_time

    @staticmethod
    def collect_position(df_alpha, name=None, DF_1M=None):
        def compute_df_pos_1m(df_alpha, DF_1M=None):

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
            )
        df_pos_1m = df_pos_1m['position']
        if name is not None: df_pos_1m = df_pos_1m.rename(name)
        df_pos_1m = df_pos_1m.to_frame()

        return df_pos_1m
    

    # @staticmethod
    # def apply_l1(df_alpha, l1, verbosity=1):
    #     if verbosity >= 2:
    #         label = '\x1b[90mAlpha_Domains.apply_l1\x1b[0m:'
    #         print(f"{label} applying l1={l1:,.2f}")
    #     df = df_alpha[['position']].copy()
    #     # assert df['position'].abs().max() <= 1, \
    #     #        'Sai, input vào apply_l1 có position > 1 sẵn'
    #     df['position'] *= l1
    #     flt = df['position'].abs() > 1
    #     df.loc[flt, 'position'] = np.sign(df.loc[flt, 'position'])
    #     df_alpha2 = df_alpha.copy()
    #     df_alpha2['position'] = df['position']
    #     return df_alpha2

    # @staticmethod
    # def auto_compute_l1(df_alpha, fee, name=None, start=None, end=None,
    #                     TARGET=None, TOLERANCE=0.0002):
    #     def compute_leveraged_aroe(l1, df_input, fee):
    #         df_alpha = df_input.copy()
    #         df_alpha['position'] = Alpha_Domains.apply_l1(df_pos, l1)
    #         df_alpha['turnover'] = df_alpha['position'].diff().fillna(0).abs()
    #         gross_profit = df_alpha['position'] * df_alpha['priceChange']
    #         df_alpha['netProfit'] = gross_profit - df_alpha['turnover'] * fee
    #         df_1d = df_alpha \
    #             .groupby('day') \
    #             .agg({'netProfit': 'sum',
    #                   'turnover': 'sum',
    #                   'open': 'first'})
    #         ds = df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
    #         # sharpe = ds.mean() / ds.std() * 250**0.5
    #         aroe = ds.sum() * 250 / len(ds)
    #         new_l1 = TARGET / aroe * l1
    #         precision = 1 - abs(aroe / TARGET - 1)
    #         return aroe, new_l1, precision

    #     def announce():
    #         text = f" for \x1b[94m{name}\x1b[0m ".replace('alpha_', '')
    #         print(f"\r{label}{count} \x1b[92mFOUND\x1b[0m{text}"
    #               f"l1=\x1b[93m{best_l1:,.2f}\x1b[0m "
    #               f"aroe={100 * best_aroe:.3f}% (vs {100 * TARGET:,.3f})% "
    #               f"(\x1b[92m{100 * best_precision:.1f}%\x1b[0m)")

    #     if TARGET is None:
    #         TARGET = 1 / C.L0 * 1.0
    #     if start is not None:
    #         df_alpha = df_alpha[df_alpha['day'] >= start]
    #     if end is not None:
    #         df_alpha = df_alpha[df_alpha['day'] <= start]
    #     if start is not None and end is not None:
    #         df_alpha = df_alpha.copy()
    #     label = "\x1b[90mauto_compute_l1\x1b[0m:"

    #     l1 = 1
    #     precision = 0.01
    #     df_pos = df_alpha[['position']].copy()
    #     best_l1, best_precision, best_aroe = l1, precision, 0

    #     for count in range(20):
    #         print(f"\r{label} attempt #{count}: l1={l1:.2f} "
    #               f"precision={precision * 100:.1f}%    ",
    #               end="")
    #         aroe, l1, precision = \
    #             compute_leveraged_aroe(
    #                 l1=l1,
    #                 df_input=df_alpha,
    #                 fee=fee)

    #         if best_precision < precision:
    #             best_precision = precision
    #             best_l1 = l1
    #             best_aroe = aroe

    #         if abs(precision - 1) < TOLERANCE:
    #             announce()
    #             return best_l1

    #     while abs(precision - 1) > TOLERANCE:
    #         count += 1
    #         print(f"\r{label} attempt #{count}: l1={l1:.3f} "
    #               f"precision={precision * 100:.1f}%    "
    #               f"best=(\x1b[92m{best_precision * 100:,.1f}%\x1b[0m "
    #               f"{best_l1:.2f}) {best_aroe} {best_aroe * C.L0 * 100:,.1f}%",
    #               end="")
    #         l1 = (np.sign(TARGET - aroe) * np.random.random() * 0.05 + 1) * l1
    #         aroe, l1, precision = \
    #             compute_leveraged_aroe(
    #                 l1=l1,
    #                 df_input=df_alpha,
    #                 fee=fee)
    #         if l1 > C.MAX_L1 or l1 < - 0.3: return 1
    #         if best_precision < precision:
    #             best_precision = precision
    #             best_l1 = l1
    #             best_aroe = aroe
    #         if count >= 100:
    #             return 1

    #     announce()
    #     return best_l1

    # @staticmethod
    # def compute_df_overnight(df_alpha):
    #     df = df_alpha[['position', 'open', 'close', 'session', 'day']].copy()
    #     df['nextOpen'] = df['open'].shift(-1, fill_value=df['close'].iloc[-1])
    #     df['overnightGap'] = df['nextOpen'] - df['close']
    #     df['prevPosition'] = df['position'].shift(1)
    #     flt = df['session'] == 'unconditionalATC'
    #     df_overnight = df[flt].set_index('day')
    #     df_overnight['grossProfit'] = \
    #         df_overnight['prevPosition'] \
    #         * \
    #         df_overnight['overnightGap']

    #     return df_overnight

    # @staticmethod
    # def compute_overnight_performance(df_overnight, start=None, end=None, plot=False,
    #                                   prefix=None):
    #     if start is not None:
    #         df_overnight = df_overnight[df_overnight.index >= start]
    #     if end is not None:
    #         df_overnight = df_overnight[df_overnight.index <= end]

    #     df_overnight = df_overnight.copy()
    #     ds = df_overnight['pctChange'] = df_overnight['grossProfit'] / df_overnight['close']
    #     sharpe = ds.mean() / ds.std() * 250 ** 0.5
    #     aroe = ds.sum() * 250 / len(ds)
    #     df_overnight['cumGrossProfit'] = df_overnight['grossProfit'].cumsum()
    #     atc_report = {
    #         'sharpe': round(sharpe, 3),
    #         'aroe': round(aroe, 4)
    #     }

    #     if plot:
    #         df_overnight['cumGrossProfit'].plot()
    #         if len(str(prefix)) > 0: prefix = f'{prefix} '
    #         fig_title = f"{prefix}sharpe={sharpe:.2f} " \
    #                     f"aroe={100 * aroe:.1f}% " \
    #                     f"({100 * aroe * C.L0:.1f}% with L0={C.L0})"
    #         plt.title(fig_title)
    #         plt.show()

    #     return atc_report



    # @staticmethod
    # def parse_alpha_id(the_id):
    #     l = the_id.split('_')
    #     alpha_name = '_'.join(l[:2])
    #     try:
    #         freq = int(l[2].split(' ')[0].replace('m', ''))
    #     except Exception as e:
    #         if len(str(e)) == 0: print(f"{e}", end="")
    #         freq = int(l[3].split(' ')[0].replace('m', ''))
    #     l = the_id.split(' ')
    #     upper, lower = map(float, l[1].split('_'))
    #     fee = float(l[-1].replace('(fee=', '').replace(')', ''))
    #     dic_init = {
    #         'alpha_name': alpha_name,
    #         'freq': freq,
    #         'upper': upper,
    #         'lower': lower,
    #         'fee': fee
    #     }
    #     return dic_init

    # @staticmethod
    # def compute_performance_from_df_1d(df_1d, fee=0.2, num_ticks=8, plot=False,
    #                                    name='', return_dic=False):
    #     ds = df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
    #     sharpe = ds.mean() / ds.std() * 250 ** 0.5
    #     idx = df_1d['day'].map(lambda x: x[2:]).values
    #     x_ticks = [day for day, i in zip(idx, reversed(range(len(idx)))) if
    #                i % num_ticks == 0]
    #     if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks
    #     ppc = df_1d['netProfit'].sum() / df_1d['turnover'].sum()
    #     mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains.compute_mdd_vectorized(df_1d)
    #     aroe = (df_1d['netProfit'] / df_1d['open']).sum() * 250 / len(df_1d)
    #     tvr = df_1d['turnover'].mean()

    #     fig_title = f"{name} " \
    #                 f"sharpe={sharpe:,.2f} " \
    #                 f"aroe={100 * aroe:,.1f}% ({C.L0 * 100 * aroe:,.1f}%) " \
    #                 f"mdd={mdd:,.2f}({100 * mdd_pct * C.L0:,.1f}%) " \
    #                 f"ppc={ppc:,.3f} tvr={tvr * 100:,.1f}% " \
    #                 f"fee={fee:,.2f} " \
    #                 f"last={df_1d['netProfit'].iloc[-1]:.2f} "

    #     if plot:
    #         plt.figure(figsize=(12, 6))
    #         plt.bar(idx, df_1d['netProfit'].cumsum())
    #         plt.xticks(x_ticks, rotation=25)
    #         plt.title(fig_title)
    #         plt.show()
    #     if return_dic:
    #         dic = {
    #             'name': name,
    #             'sharpe': sharpe,
    #             'mdd': mdd,
    #             'cdd': cdd,
    #             'fee': fee,
    #             'last': df_1d['netProfit'].iloc[-1],
    #             'title': fig_title}
    #         return dic

    # @staticmethod
    # def compute_position_and_cut_book(bt):
    #     Alpha_Domains.adjust_positions(bt.df_alpha)
    #     if 'inertia' in bt.report:
    #         Alpha_Domains.compute_position_with_inertia(bt.df_alpha, bt.inertia)
    #     elif 'upper' in bt.report:
    #         bt.compute_position()
    #     # noinspection PyCallingNonCallable
    #     mmax = bt.df_alpha['position'].abs().max()
    #     if mmax > 1:
    #         label = "\x1b[90mAlpha_Domains.compute_position_and_cut_book\x1b[0m: "
    #         print(f"{label}\x1b[94mWarning: absolute position > 1 "
    #               f"\x1b[0m(\x1b[93m{bt.name}\x1b[0m)")
    #     # bt.df_alpha['position'] /=
    #     flt = bt.df_alpha['position'].abs() >= 1
    #     bt.df_alpha.loc[flt, 'position'] = np.sign(bt.df_alpha.loc[flt, 'position'])
    #     assert bt.df_alpha['position'].abs().max() <= 1, 'Sai, abs position lớn hơn 1'
    #     bt.cb_func(bt.df_alpha)
    #     Alpha_Domains.adjust_positions(bt.df_alpha)

    # @staticmethod
    # def combine_alphas_and_compute_performance(configs, fee, l1=1.0, alpha_name='',
    #                                            start=None, end=None, plot=True):
    #     label = "\x1b[90mcompute_combined_os_performance\x1b[0m:"
    #     lst_bts = [Simulator.from_id(config) for config in configs]
    #     collector = Collector(name='zscore_busd')
    #     for i, bt in enumerate(lst_bts):
    #         print(f"\r{label} #{i + 1:,} / {len(lst_bts):,} "
    #               f"{bt.report['id']}  ",
    #               end="")
    #         bt.compute_all()
    #         collector.collect_bt(bt)
    #         if i + 1 == len(lst_bts): print()


    #     bt = collector.create_bt(fee=fee)
    #     bt.report['l1'] = l1
    #     if l1 != 1: bt.apply_l1(l1=l1)
    #     bt.name = bt.report['alphaName'] = 'collected'
    #     num_alphas = len(collector.df_all_pos.columns)
    #     bt.compute_tvr_and_fee()
    #     bt.compute_profits()

    #     # noinspection PyUnusedLocal
    #     fig_title, os_report, df_1d = Alpha_Domains2.compute_performance_stats(
    #         df=bt.df_alpha,
    #         fee=bt.fee,
    #         in_place=True,
    #         plot=plot,
    #         start=start,
    #         end=end,
    #         alpha_name=alpha_name,
    #         suffix=f'({num_alphas:,} alpha) L1={l1}')

    #     return bt, collector, os_report, df_1d

    # @staticmethod
    # def add_overnight_performance(bt, plot_on=False):
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


class Collector:
    def __init__(self, name, df_1m=None):
        self.lst_all_pos = []
        self.dic_df_1d = {}
        self.df_1m = df_1m
        self.name = name

        self.bt: Simulator = None

        """PLACE-HOLDER"""
        self.df_all_pos = pd.DataFrame()
        self.df_pos_1m = pd.DataFrame()
        self.l1 = 1
        
    def collect_bt(self, bt):
        df_pos_1m = Alpha_Domains.collect_position(
            df_alpha=bt.df_alpha,
            name=bt.name,
            DF_1M=self.df_1m,
        )
        self.lst_all_pos.append(df_pos_1m)

    def collect_df_1d(self, df_1d, report):
        self.dic_df_1d[report['id']] = df_1d

    def collect_df_alpha(self, df_alpha, name):
        df_pos_1m = Alpha_Domains.collect_position(
            df_alpha=df_alpha,
            name=name,
            DF_1M=self.df_1m,
        )

        self.lst_all_pos.append(df_pos_1m)



    def combine_positions(self):
        df_all_pos = pd.concat(self.lst_all_pos, axis=1)
        df_pos = df_all_pos.sum(axis=1) / len(df_all_pos.columns)
        return df_all_pos, df_pos

    def maybe_compute_df_pos_1m(self):
        if len(self.df_pos_1m) == 0:
            self.df_all_pos, self.df_pos_1m = self.combine_positions()

    def create_bt(self,df_alpha,DIC_ALPHAS, fee):
        bt = Simulator(
            alpha_name=self.name,
            df_alpha=df_alpha,
            fee=fee,
            freq=1)
        self.maybe_compute_df_pos_1m()
        bt.df_alpha['position'] = self.df_pos_1m

        return bt

    def plot_df_1d(self):
        df_1d = pd.concat([df['netProfit'] for df in self.dic_df_1d.values()], axis=1)
        num_alphas = len(self.dic_df_1d)
        ds_1d = df_1d.sum(axis=1) / num_alphas
        idx = df_1d.index.map(lambda x: x[2:])

        x_ticks = [day for day, i in zip(idx, reversed(range(len(idx)))) if
                   i % 15 == 0]
        if x_ticks[0] != idx[0]: x_ticks = [idx[0]] + x_ticks

        fig_title = f"{self.name} ({num_alphas:,} alphas) " \
                    f"daily return {df_1d.index[-1]} " \
                    f"last={ds_1d[-1]:,.2f}"

        plt.figure(figsize=(9, 6))
        plt.bar(idx, ds_1d.cumsum())
        plt.xticks(x_ticks, rotation=25)
        plt.title(fig_title)
        plt.show()

    def apply_l1_and_plot(self, scan_params, plot=True):
        self.bt = bt_combined = self.create_bt(
            fee=scan_params.fee_os)

        bt_combined.df_alpha = Alpha_Domains.apply_l1(
            df_alpha=bt_combined.df_alpha,
            l1=self.l1)

        bt_combined.compute_tvr_and_fee()
        bt_combined.compute_profits()
        bt_combined.report['l1'] = round(self.l1, 2)
        bt_combined.compute_performance(
            start=scan_params.day_os_start,
            plot=plot)

        # noinspection PyUnusedLocal
        def plot_magnitude(bt_combined=bt_combined):
            bt_combined.df_alpha['magnitude'] = bt_combined.df_alpha['position'].abs()
            bt_combined.df_alpha.groupby('time').agg({'turnover': 'mean'}).plot()
            plt.show()

        return bt_combined

    def save_report(self, dic):
        dic[self.bt.report['alphaName'].split('(')[0]] = self.bt.report.copy()

    def compute_fn(self):
        if '(' not in self.bt.report["alphaName"]:
            name = self.bt.report["alphaName"]
            n = 0
        else:
            name = self.bt.report['alphaName'].split('(')[0]
            n = int(self.bt.report['alphaName'].split('(')[1].split(')')[0])
        day0 = self.bt.df_alpha['day'].iloc[0]
        day1 = self.bt.df_alpha['day'].iloc[-1]
        fn = f'/tmp/collector_{name}_{day0}_{day1}_{n}.pickle'

        return fn




