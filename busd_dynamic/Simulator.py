import numpy as np
import pandas as pd
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class MegaBbAccV2():
    def __init__(
            self,
            alpha_name="",
            configs=[],
            fee=150,
            df_alpha=None,
            data_start=None,
            data_end=None,
            
            busd_source="hose500",  # hose500, vn30
     
            position_multiplier=1.0,
            fma_length=200,
            foreign_policy=0,
            replace_filter_value=0,
            moving_average_type="sma",
            filter_col="aggFBusdVn30",
            foreign_multiplier=0,
            foreign_add_column="aggFBusdVn30",
            update_second_only="False",
            
                        
            init_budget = 1,
            booksize_sizing = False 
    ):

        self.alpha_name = alpha_name
        # configs
        self.configs = configs
        self.four_param_configs = ["_".join(x.strip("_alpha_bb_acc_").split("_")[:4]) for x in self.configs]
        # print(self.four_param_configs)
        # exit()
        self.fee = int(fee)
        self.df_alpha = df_alpha
        self.data_start = data_start
        self.data_end = data_end
        self.busd_source = busd_source
        self.n_alphas = len(configs)
    
       # foreign add
        self.foreign_add_column = foreign_add_column
        self.foreign_multiplier = foreign_multiplier
        
        self.mas = pd.DataFrame()
        self.update_second_only = update_second_only
        # multi df
        self.position_df = pd.DataFrame()
        self.turnover_df = pd.DataFrame()
        self.net_profit_df = pd.DataFrame()
        self.turnover_df1d = pd.DataFrame()
        self.net_profit_df1d = pd.DataFrame()
     
        # filter
        self.filter_column = filter_col
        self.replace_filter_value = replace_filter_value
        self.foreign_policy = foreign_policy
        self.fma_length = fma_length
        self.moving_average_type = moving_average_type
        
        # multi scanner
        self.positions = pd.DataFrame()
        ##################################################################################
        """FILLERS"""
        self.df_1d = pd.DataFrame()
        self.report = {}
        # figures
        self.fig_df_1d = None
        self.fig_df_alpha = None
        self.position_multiplier = position_multiplier
        
        
        self.init_budget = init_budget
        self.booksize_sizing = booksize_sizing
        self.hard_dic_budget = pd.DataFrame({'status':  range(101)})
        
        # DELTA = 37.5
        DELTA = 30
        self.hard_dic_budget['cum'] = DELTA
        self.hard_dic_budget['cum'] =  self.hard_dic_budget['status'] *  self.hard_dic_budget['cum']
        self.hard_dic_budget.loc[0, 'cum'] = DELTA
        self.hard_dic_budget['cum'] = self.hard_dic_budget['cum'].cumsum()
        self.hard_dic_budget['action'] = self.hard_dic_budget['status'] + 1

     

    def compute_based_col(self):
        df = self.df_alpha
        if self.data_start:
            df = df[df["day"] >= self.data_start].copy()
        if self.data_end:
            df = df[df["day"] <= self.data_end].copy()
            
        if self.busd_source == "hose500":
            df["based_col"] = df["aggBusd"]
            # df["based_col"] = df["aggBusd"] + df["aggBusdVn30"]
            # df["based_col"] = df["aggBusd"] + df["aggFBusd"]
            # df["based_col"] = df["aggBusd"] + df["aggFBusdVn30"]
            # df["based_col"] = df["aggBusd"] + df["aggFBusd"] + df["aggBusdVn30"] + df["aggFBusdVn30"]
            # df["based_col"] = df["aggBusd"] + df["aggFBusd"] + df["aggBusdVn30"]
            # df["based_col"] = df["aggBusdVn30"] + df["aggFBusd"]
            # df["based_col"] = df["aggBusdVn30"] 
            # df["based_col"] = df["aggBusd"] + df["aggFBusdVn30"] + df["aggBusdVn30"]
            # df["based_col"] = df["aggBusd"] + df["aggFBusd"] + df["aggFBusdVn30"]
            # df["based_col"] = df["aggBusdVn30"] + df["aggFBusd"] + df["aggFBusdVn30"]
            # df["based_col"] = df["aggBusdVn100"]
            # df["based_col"] = df["aggBusdBank"]
            # df["based_col"] = df["aggBusdLeading"]
            # df["based_col"] = df["psAggBusd"]
        else:
            df["based_col"] = df["aggBusdVn30"]
  
        if self.foreign_multiplier != 0:
            df["based_col"] += self.foreign_multiplier * df[self.foreign_add_column]
        if self.update_second_only:
            df = df[df["based_col"].diff(1) != 0].copy()
            df = df[df['time_int'] >= 10091500]
        # priceChange now in this position
        df["priceChange"] = df.groupby("day")["last"].diff(1).shift(-1).fillna(0)
        self.df_alpha = df
        # print(self.df_alpha)

    def kalman_filter_1d(self,series: np.ndarray, R: float, Q: float) -> np.ndarray:
  
        n = len(series)
        if n == 0:
            return np.array([], dtype=float)
        xhat = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        xhat[0] = float(series[0])
        P[0] = 1.0
        for k in range(1, n):
            xhatminus = xhat[k - 1]
            Pminus = P[k - 1] + Q
            K = Pminus / (Pminus + R)
            xhat[k] = xhatminus + K * (float(series[k]) - xhatminus)
            P[k] = (1 - K) * Pminus
  
        return xhat
        

    def compute_mas(self):
        df = self.df_alpha.copy()
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs]
        ma_lengths = set(ma1 + ma2)
        # exit()
        gb = df.groupby("day")
        cols = []
        for day, df_day in gb:
            for ma_length in ma_lengths:
                # df_day['based_col'] = self.kalman_filter_1d(df_day['based_col'].values, R=0.5, Q=1e-7)
                ma = df_day["based_col"].ewm(span=ma_length, adjust=True).mean().values #ema
                df.loc[df_day.index, f"ma_{ma_length}"] = ma
            if self.foreign_policy != 0:
                # support sma only
                filter_vector = np.sign(
                    df_day[self.filter_column] - df_day[self.filter_column].rolling(window=self.fma_length).mean()#sma
                ) * self.foreign_policy
                df.loc[df_day.index, "foreign_filter"] = filter_vector
                cols = ["foreign_filter"]
                self.df_alpha["foreign_filter"] = df["foreign_filter"]
        self.mas = df[cols + [f"ma_{x}" for x in ma_lengths]]
        # print(self.mas)
        
        # import os
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # save_path = os.path.join(current_dir, "mas.pkl")

        # self.mas.to_pickle(save_path)

    def compute_all_position(self):
        df = self.df_alpha.copy()
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs]
        threshold = [int(x.split("_")[2]) for x in self.four_param_configs]
        exit_strength = [int(x.split("_")[3]) for x in self.four_param_configs]
        ma1_matrix = np.array([self.mas[f"ma_{x}"] for x in ma1]).transpose()
        ma2_matrix = np.array([self.mas[f"ma_{x}"] for x in ma2]).transpose()
        # diff & threshold
        diff_matrix = ma2_matrix - ma1_matrix
        # print(diff_matrix)
        threshold_matrix = np.tile(threshold, (len(diff_matrix), 1))
        # position matrix
        position_matrix = np.full(diff_matrix.shape, np.nan)
        position_matrix[diff_matrix > threshold_matrix] = 1
        position_matrix[diff_matrix < -threshold_matrix] = -1
        if np.abs(exit_strength).sum() > 0:
            exit_strength_matrix = np.tile(exit_strength, (len(diff_matrix), 1))
            take_profit_matrix = threshold_matrix * exit_strength_matrix / 100
            position_matrix[np.abs(diff_matrix) < take_profit_matrix] = 0
        # dataframe
        position_df = pd.DataFrame(position_matrix, columns=self.four_param_configs, index=df.index)
        
        # filter
        if self.foreign_policy != 0:
            # keep only 1st signal
            position_df.ffill(axis=0, inplace=True)
            position_df.fillna(0, inplace=True)
            position_df[position_df.diff(1, axis=0) == 0] = np.nan
            filter_matrix = np.sign(np.tile(self.mas["foreign_filter"], (diff_matrix.shape[1], 1))).transpose()
            # apply filter
            filter_df = pd.DataFrame(filter_matrix, index=df.index)
            position_df[position_df.values * filter_df.values == -1 * self.foreign_policy] = self.replace_filter_value
        # lag position
        position_df = position_df.shift(1, fill_value=0)
        # mega position
        # no position after 14:29:30
        position_df[df["time_int"] >= 10142930] = np.nan
        position_df[df["time_int"] >= 10144500] = 0
        position_df[df["time_int"] <= 10091500] = 0
        day_cumcount = df.groupby('day').cumcount().values
        position_df.values[day_cumcount < 1] = 0.0
        position_df = position_df.ffill(axis=0).fillna(0)
        self.position_df = position_df
        self.position_df = position_df
        
    def compute_mega_position(self, **kwargs):
        # print('compute_mega_position called')
        positions = self.position_df.sum(axis=1)
        positions.ffill(inplace=True)
        # print('budget=============:', self.init_budget)

        
        if self.booksize_sizing: # skip
            self.df_alpha["position_init"] = positions.values / self.n_alphas 
            self.sizing_positon(positions)
            
        else:
            self.df_alpha["position"] = ((positions.values / self.n_alphas * self.init_budget).round(6).astype(int))
            self.df_alpha['bookzise'] = self.init_budget
        # print(self.df_alpha)
   
        
        
    def get_budget(self,total_netProfit):
        df_filtered = self.hard_dic_budget[self.hard_dic_budget['cum'] <= total_netProfit]
        if df_filtered.empty:
            return 0
        else:
            return df_filtered['action'].iloc[-1]
        
    def sizing_positon(self,positions):   
        df = self.df_alpha.copy()
        
        lst_day = df['day'].unique()
        
        total_netProfit = 0
        for day in lst_day:
            df_day = df[df['day'] == day].copy()
            booksize = self.init_budget + self.get_budget(total_netProfit)
            df_day['position'] = ((df_day['position_init'] * booksize).round(6).astype(int))
      
            
            df_day['grossProfit'] = df_day['position'] * df_day['priceChange']
            df_day['action'] = df_day['position'] - df_day['position'].shift(1, fill_value=0)
            df_day['turnover'] = df_day['action'].abs()
            df_day['fee'] = df_day['turnover'] * self.fee / 1000
            df_day['netProfit'] = df_day['grossProfit'] - df_day['fee']
            
            intraday_netProfit = df_day['netProfit'].sum()
            
            total_netProfit += intraday_netProfit
            
            self.df_alpha.loc[df_day.index, f"position"] = df_day['position']
            self.df_alpha.loc[df_day.index, f"bookzise"] = booksize
        

 


    def compute_all_sub_alpha_profit(self):
        df = self.position_df.copy()
        gross_profit_matrix = df.values * np.tile(self.df_alpha["priceChange"].values, (df.shape[1], 1)).transpose()
        gross_profit_df = pd.DataFrame(gross_profit_matrix, columns=self.four_param_configs, index=df.index)
        turnover_df = (df - df.shift(1, fill_value=0)).abs()
        net_profit_df = gross_profit_df - turnover_df * self.fee / 1000
        self.turnover_df = turnover_df
        self.net_profit_df = net_profit_df
        # group by day
        turnover_df["day"] = self.df_alpha["day"]
        net_profit_df["day"] = self.df_alpha["day"]
        self.turnover_df1d = turnover_df.groupby("day").sum()
        self.net_profit_df1d = net_profit_df.groupby("day").sum()


    def compute_profit_and_df_1d(self):
        df = self.df_alpha
        # df['priceChange'] = (df['last'].shift(-1) - df['last']).fillna(0)
        df['grossProfit'] = df['position'] * df['priceChange']
        df['action'] = df['position'] - df['position'].shift(1, fill_value=0)
        df['turnover'] = df['action'].abs()
        df['fee'] = df['turnover'] * self.fee / 1000
        df['netProfit'] = df['grossProfit'] - df['fee']
        df['pctChange'] = df['netProfit'] / df['last']
        df_1d = (df.groupby('day')
                 .agg({'grossProfit': 'sum',
                       'turnover': 'sum',
                       'netProfit': 'sum',
                       'pctChange': 'sum',
                       'bookzise' : 'last',
                    #    'totalMatchVolume' : 'last'
                       }))
        # print(df_1d)
        # exit()
        df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        df_1d['cumTurnover'] = df_1d['turnover'].cumsum()
        df_1d[['cumGrossProfit', 'cumTurnover', 'cumNetProfit']] = \
            df_1d[['grossProfit', 'turnover', 'netProfit']].cumsum()
        df_1d[['grossProfit', 'netProfit', 'cumNetProfit', 'cumGrossProfit']] = \
            df_1d[['grossProfit', 'netProfit', 'cumNetProfit', 'cumGrossProfit']].round(2)
        self.df_1d = df_1d
        # print(self.df_1d)
        



    def compute_all(self,):
        self.compute_profit_and_df_1d()
        self.df_alpha['cumTurnover'] = self.df_alpha['turnover'].cumsum()
        self.compute_all_reports()
        text = ' '.join(
            [f"{k} {Utilities.colorize1(v)}" for k, v in
             zip(self.df_1d.index[-4:],
                 self.df_1d['netProfit'].values[-4:])])
        # print(f"{self.configs.__len__()} "
        #       f"fee={self.fee} "
        #       f"tvr={self.report['tvr'] * 100:,.1f}%"
        #       f"\x1b[94m {text}\x1b[0m ")

    def gen_plot_df_1d(self):
        df = self.df_1d
        fig = go.Figure()
        df["hover_text"] = df.index.astype(str) + "<br>" + df["netProfit"].astype(str)
        fig.add_trace(go.Bar(x=df.index.astype(str), y=df['cumNetProfit'], name='netProfit', hovertext=df["hover_text"]))
        title = self.report["title"]
        fig.update_layout(title=title)
        self.fig_df_1d = fig

    def plot_df_1d(self):
        if self.fig_df_1d is None:
            self.gen_plot_df_1d()
        # self.fig_df_1d.show()
        self.fig_df_1d.write_html("/Users/m2/duy/M2_Dashboard/BE/backtest/6.html")

    def plot_df_alpha(self, day):
        # day = 20250210
        df = self.df_alpha.copy()
        df = df[df["day"] == day].copy()
        df["timeX"] = df.index.strftime("%X")
        df["cumNetProfit"] = df["netProfit"].cumsum()
        df["cumTurnover"] = df["turnover"].cumsum()
        
        df_ma = self.mas.copy()
        df_ma['day'] = df_ma.index.strftime('%Y_%m_%d').astype('int')
        df_ma = df_ma[df_ma['day'] == day ]
        
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs][0]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs][0]
        threshold = [int(x.split("_")[2]) for x in self.four_param_configs][0]
        
        df_ma['diff'] = df_ma[f"ma_{ma2}"] - df_ma[f"ma_{ma1}"]
        # print(df_ma)
        # print(df)
        # exit()
        # fig
        n_rows = 4
        fig = make_subplots(rows=n_rows, cols=1, specs=[[{"secondary_y": True}]] * n_rows, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df["timeX"], y=df["last"], name="last"), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df["timeX"], y=df["based_col"], name="based_col="+self.busd_source), row=1, col=1, secondary_y=True)
        
        fig.add_trace(go.Scatter(x=df["timeX"], y=df_ma["diff"], name="diff"), row=2, col=1, secondary_y=False)
        fig.add_trace(
            go.Scatter(
                x=[df["timeX"].min(), df["timeX"].max()],
                y=[threshold, threshold],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=[df["timeX"].min(), df["timeX"].max()],
                y=[-threshold, -threshold],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1, secondary_y=False
        )
 
        fig.add_trace(go.Scatter(x=df["timeX"], y=df["position"], name="position"), row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df["timeX"], y=df["cumNetProfit"], name="cumNetProfit"), row=4, col=1, secondary_y=False)
        fig.update_traces(xaxis='x1')
        fig.update_layout(hovermode='x unified')
        title = f"{day} {self.alpha_name} day_tvr={df['cumTurnover'].values[-1]:.0f} day_pnl={df['cumNetProfit'].values[-1]:.3f}"
        fig.update_layout(title=title)
        self.fig_df_alpha = fig
        fig.write_html("/home/ubuntu/duy/new_strategy/backtest/1.html")
 



    def compute_report(self, df_1d,name, template=None):
        df_1d = df_1d.copy()
        # exit()

        report = {
            'alpha_name': name,
            'sharpe': None,
            "config":self.configs[0],
            # 'aroe': None,
            'tvr': 0,
            'ppc': None,
            # 'cdd': None,
            # 'mdd': None,
            # 'clmr': None,
            # 'title': 'Zero Trades',
            # 'alphaName': self.alpha_name,
            # 'scanTime': pd.Timestamp.now().strftime('%Y_%m_%d %H:%M:%S'),
            # 'osStart': self.os_start,
            'start_day' : df_1d.index.min(),
            'end_day' : df_1d.index.max()
        }
        if 'cdd' not in df_1d:
            df_1d['cdd'] = df_1d['cumNetProfit'].cummax() - df_1d['cumNetProfit']
            df_1d['cdd'] = df_1d['cdd'].round(2)
            df_1d['mdd'] = df_1d['cdd'].cummax()

        tvr = round(df_1d['turnover'].mean(), 3)
        std = df_1d['pctChange'].std()
        if tvr == 0 or std == 0: return report

        # sharpe = round(df_1d['pctChange'].mean() / std * 250**0.5, 3)
        # sharpe = df_1d["netProfit"].mean() / df_1d["netProfit"].std() * 250 ** 0.5
        sharpe = df_1d["netProfit"].mean() / df_1d["netProfit"].std() * len(df_1d) ** 0.5
        ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
        aroe = round(df_1d['pctChange'].mean() * 250 * 4.41, 3)
        cdd = round(df_1d['cdd'].values[-1], 2)
        mdd = round(df_1d['mdd'].values[-1], 2)
        # clmr = round(df_1d['netProfit'].mean() * 250 / mdd, 1)
        scl = round(aroe / tvr * 10, 3)
        title = (
            # f"{self.alpha_name} "
            f"sharpe: {sharpe:.2f} "
            # f"aroe={100*aroe:.1f}% tvr={100*tvr:,.1f}% "
            # f"mdd={mdd:.1f}/{clmr:.1f} "
            # f"scl={scl:.2f} "
            f"ppc={ppc:.3f}"
        )
        report['sharpe'] = float(sharpe)
        # report['aroe'] = aroe
        report['tvr'] = float(tvr)
        report['ppc'] = float(ppc)
        # report['cdd'] = cdd
        # report['mdd'] = mdd 
        # report['clmr'] = clmr
        # report['title'] = title
        report['cumNetProfit'] = float(df_1d['cumNetProfit'].values[-1])
        if template is not None:
            return {**template, **report}
        return report