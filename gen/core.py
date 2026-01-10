from datetime import datetime
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
class Simulator:
    def __init__(self, alpha_name, freq=1, gen_params=None, fee=0.1, df_alpha=None,DIC_ALPHAS=None,params={},df_tick=None,gen=None,stop_loss=0,is_sizing=False,init_sizing=37.5,start=None,end=None,booksize=1,cut_time=None,df_1m=None): 

        self.freq = freq
        self.gen_params = gen_params
        self.fee = fee
        self.stop_loss = stop_loss
        self.gen = gen
        self.start = start
        self.end = end
        self.df_1m = df_1m
        self.cutTime = cut_time
        # df_alpha = df_alpha[(df_alpha['day'] >= start) & (df_alpha['day'] <= end)].copy()
        self.booksize = booksize
        ohlc_cols = ["open","high","low","close"]
        df_alpha.loc[:, ohlc_cols] = df_alpha[ohlc_cols].astype(float)
        self.df_alpha = df_alpha
        self.df_tick = df_tick.copy() if df_tick is not None else None
        
        if DIC_ALPHAS:
            self.alpha_func = DIC_ALPHAS[alpha_name]
        self.params = params
        self.report = {
            'alphaName': alpha_name,
            'sharpe': np.nan,
            'freq': self.freq,
            'fee': self.fee,
            **(self.gen_params if self.gen_params else {})
        }
        self.is_sizing = is_sizing
        self.hard_dic_budget = pd.DataFrame({'status':  range(101)})
        self.hard_dic_budget['cum'] = init_sizing
        self.hard_dic_budget['cum'] =  self.hard_dic_budget['status'] *  self.hard_dic_budget['cum']
        self.hard_dic_budget.loc[0, 'cum'] = init_sizing
        self.hard_dic_budget['cum'] = self.hard_dic_budget['cum'].cumsum()
        self.hard_dic_budget['action'] = self.hard_dic_budget['status'] + 1
        
    def compute_signal(self):
                
        Alpha_Domains.compute_signal(alpha_func=self.alpha_func, df_alpha=self.df_alpha, params=self.params)
        if self.gen == "1_1" and self.gen_params['halflife'] > 0:
            self.df_alpha['signal'] = self.df_alpha['signal'].ewm(halflife=self.gen_params['halflife']).mean()
        self.df_alpha = self.df_alpha[
            (self.df_alpha['day'] >= self.start) &
            (self.df_alpha['day'] <= self.end)
        ]
        

    def compute_position(self):
        if self.gen == "1_1":
            Alpha_Domains.compute_position(
                df_alpha=self.df_alpha,
                threshold=self.gen_params['threshold'],
                )
        elif self.gen == "1_2":
            Alpha_Domains.compute_positions_with_thresholds(
                df_alpha=self.df_alpha,
                upper=self.gen_params['upper'],
                lower=self.gen_params['lower'])
        elif self.gen == "1_3":
            Alpha_Domains.compute_positions_with_signal_score(
                df_alpha=self.df_alpha,
                score_window=self.gen_params['score'],
                entry_score=self.gen_params['entry'],
                exit_score=self.gen_params['exit'])
        elif self.gen == "1_4":
            Alpha_Domains.compute_position_with_velocity(
                df_alpha=self.df_alpha,
                vel_entry=self.gen_params['entry'],
                vel_exit=self.gen_params['exit'],
                smooth_window=self.gen_params['smooth'] )
        
        if self.is_sizing:
            self.df_alpha["position_init"] = self.df_alpha['position'].values 
            self.sizing_positon()
        else:
            self.df_alpha['position'] = self.df_alpha['position'] * self.booksize
            self.df_alpha['booksize'] = self.booksize
    def get_budget(self,total_netProfit):
        df_filtered = self.hard_dic_budget[self.hard_dic_budget['cum'] <= total_netProfit]
        if df_filtered.empty:
            return 0
        else:
            return df_filtered['action'].iloc[-1]
        
    def change_to_df1m(self):
        if self.cutTime:       
            df_1M = self.df_1m.reset_index().merge(self.df_alpha[['position', 'executionT']],
                        on='executionT',
                        how='left').set_index('groupTime')
            df_1M['position'] = df_1M['position'].ffill().fillna(0)
            # print(df_1M)
            df_1M.loc[df_1M['executionTime'] >= self.cutTime, 'position'] = 0.0
            self.df_alpha = df_1M    
            # print(df_1M[df_1M['day'] == "2025_09_05"][['open','executionT',"position"]].tail(10))
        
    def sizing_positon(self):   
        df = self.df_alpha.copy()
        lst_day = df['day'].unique()
        total_netProfit = 0
        for day in lst_day:
            df_day = df[df['day'] == day].copy()
 
            booksize = self.booksize + self.get_budget(total_netProfit)
            df_day['position'] = ((df_day['position_init'] * booksize).round(6).astype(int))
            
            df_day['grossProfit'] = df_day['position'] * df_day['priceChange']
            df_day['action'] = df_day['position'] - df_day['position'].shift(1, fill_value=0)
            df_day['turnover'] = df_day['action'].abs()
            df_day['fee'] = df_day['turnover'] * self.fee 
            df_day['netProfit'] = df_day['grossProfit'] - df_day['fee']
            
            intraday_netProfit = df_day['netProfit'].sum()
            
            total_netProfit += intraday_netProfit
            
            self.df_alpha.loc[df_day.index, f"position"] = df_day['position']
            self.df_alpha.loc[df_day.index, f"booksize"] = booksize
            
    def compute_tvr_and_fee(self):
        Alpha_Domains.compute_action_tvr_and_fee(
            self.df_alpha,
            self.fee)

    def compute_profits(self):
        Alpha_Domains.compute_profits(self.df_alpha,self.stop_loss,self.fee)

    def compute_performance(self, start=None, end=None):
 
        self.df_1d, report = Alpha_Domains.compute_performance(
            self.df_alpha,
            start=start,
            end=end,equity = self.booksize*300)
        self.report.update(report)
    def merge_position_to_ticks(self):
        self.df_alpha = Alpha_Domains.merge_position_to_ticks(self.df_alpha,self.df_tick)
    def compute_df_trade(self):
        self.df_trade = Alpha_Domains.compute_df_trade(self.df_alpha)
    def extract_net_profits(self):
        return Alpha_Domains.extract_net_profits(self.df_alpha)
    
class Alpha_Domains:
    
    @staticmethod
    def compute_signal(alpha_func, df_alpha, params={}):
        
        df_alpha['position'] = df_alpha['signal'] = alpha_func(df_alpha,**params)
       


    def compute_position(df_alpha, threshold):
        
        flt_mediocre = df_alpha['signal'].abs() < threshold
        df_alpha.loc[flt_mediocre, 'signal'] = np.nan
        
        df_alpha['position'] = df_alpha['signal']  
        Alpha_Domains.adjust_positions(df_alpha)
        
        df_alpha['position'] = np.sign(df_alpha['position'])
        
    @staticmethod
    def adjust_positions(df_alpha):
        flt_unexecutable = ~df_alpha['executable']
        df_alpha.loc[flt_unexecutable, 'position'] = np.nan
        flt_atc = df_alpha['executionTime'] == '14:45:00'
        df_alpha.loc[flt_atc, 'position'] = 0
        df_alpha['position'] = df_alpha['position'].ffill().fillna(0)
        
    @staticmethod
    def compute_positions_with_signal_score(df_alpha, score_window=5, entry_score=3, exit_score=1):
        """
        Cộng điểm theo hướng tín hiệu trong cửa sổ thời gian ngắn.
        - Khi tổng điểm >= entry_score -> mở vị thế long
        - Khi tổng điểm <= -entry_score -> mở vị thế short
        - Khi |tổng điểm| <= exit_score -> đóng vị thế
        """
        Alpha_Domains.adjust_positions(df_alpha)
        signed_signal = np.sign(df_alpha['position'].fillna(0))
        rolling_score = signed_signal.rolling(score_window, min_periods=1).sum()

        new_positions = []
        current_pos = 0
        for score in rolling_score:
            if score >= entry_score:
                current_pos = 1
            elif score <= -entry_score:
                current_pos = -1
            elif abs(score) <= exit_score:
                current_pos = 0
            new_positions.append(current_pos)

        df_alpha['position'] = new_positions
        Alpha_Domains.adjust_positions(df_alpha)
    
    @staticmethod
    def compute_position_with_velocity(df_alpha, vel_entry=0.2, vel_exit=0.05, smooth_window=1):
        
        """
        Vào lệnh theo tốc độ thay đổi tín hiệu:
        - Khi velocity > vel_entry -> long
        - Khi velocity < -vel_entry -> short
        - Khi |velocity| < vel_exit -> đóng vị thế
        Có thể làm mượt velocity bằng rolling mean (smooth_window > 1).
        """
        Alpha_Domains.adjust_positions(df_alpha)

        velocity = df_alpha['position'].diff().fillna(0)
        if smooth_window > 1:
            velocity = velocity.rolling(smooth_window, min_periods=1).mean()

        new_positions = []
        current_pos = 0
        for v in velocity:
            if v > vel_entry:
                current_pos = 1
            elif v < -vel_entry:
                current_pos = -1
            elif abs(v) < vel_exit:
                current_pos = 0
            new_positions.append(current_pos)

        df_alpha['position'] = new_positions
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
    def compute_action_tvr_and_fee(df_alpha, fee):
        df_alpha['action'] = df_alpha['position'].diff(1).fillna(df_alpha['position'].iloc[0])
        df_alpha['turnover'] = df_alpha['action'].abs()
        df_alpha['fee'] = df_alpha['turnover'] * fee
        # print(df_alpha['position'].sum())
        
        
    @staticmethod
    def merge_position_to_ticks(df_alpha, df_tick):
        # ===== LỌC tick theo khoảng thời gian của alpha =====
        # df_alpha[df_alpha['day'] == '2024_01_04'].to_csv('/home/ubuntu/nevir/gen1_2/pre_merge.csv')
        df_alpha = df_alpha.set_index('executionT')
        # start, end = df_alpha.index.min(), df_alpha.index.max()
        # df_tick = df_tick.loc[(df_tick.index >= start) & (df_tick.index <= end)]
        
        # ===== MERGE: dùng tick làm timeline chính =====
        df = df_tick[["last","day"]].join(df_alpha[['position','open']], how="left")
        # Tìm chỉ số cuối cùng của mỗi ngày
        is_last_of_day = df['day'].shift(-1) != df['day']
        last_day_indices = df[is_last_of_day].index

        # Gán position cuối ngày = 0
        df.loc[last_day_indices, "position"] = 0

        
        # ===== FILLFORWARD toàn bộ cột từ alpha =====
        for col in ['position','open']:
            if col != "last":   # không ffill last
                df[col] = df[col].ffill().fillna(0)
        
        df["priceChange"] = (
            df.groupby("day")["last"]
            .diff()
            .shift(-1)
            .fillna(0)
        )
        

        # df[df['day'] == 20240104].to_csv('/home/ubuntu/nevir/gen1_2/merged.csv')
        # print(df)
        return df
    
    @staticmethod
    def hard_cut_loss(df, stop_loss=2, fee=0.175):
        if stop_loss is None or stop_loss <= 0:
            return df
        # ====== BƯỚC 2: XÁC ĐỊNH TRADE_ID ======
        df["position_prev"] = df["position"].shift(fill_value=0)
        df["trade_start"] = ((df["position_prev"] == 0) & (df["position"] != 0)) | \
                            (df["position_prev"] * df["position"] < 0)
        df["trade_id"] = df["trade_start"].cumsum()
        df.loc[(df["position"] == 0) & (df["netProfit"] == 0), "trade_id"] = None
        df["trade_id"] = df["trade_id"].ffill()

        # ====== BƯỚC 3: TÍNH LŨY KẾ LÃI/LỖ THEO TRADE ======
        df["trade_cum_profit"] = df.groupby("trade_id")["grossProfit"].cumsum()

        # ====== BƯỚC 4: XÁC ĐỊNH CÁC TRADE BỊ CUT ======
        cut_condition = df["trade_cum_profit"] < -stop_loss
        cut_first_idx = (
            df[cut_condition]
            .sort_index()
            .drop_duplicates("trade_id", keep="first")
        )

        # cut_trade_ids = cut_first_idx["trade_id"].unique()  
        # print("Cut trades:", cut_trade_ids)

        # # Lưu dữ liệu TRƯỚC khi cut loss (dữ liệu gốc)
        # df[df['day'] == '2024_01_04'].to_csv('/home/ubuntu/nevir/gen1_2/trades_origin.csv')

        # ====== BƯỚC 5: CẬP NHẬT SAU CẮT ======
        cut_start_map = dict(zip(cut_first_idx["trade_id"], cut_first_idx.index))
        trade_ids_array = df["trade_id"].values
        index_array = pd.to_datetime(df.index)
        cut_start_idx_array = np.array([
            cut_start_map.get(tid, pd.Timestamp.max) for tid in trade_ids_array
        ])

        cut_mask = pd.Series(index_array >= cut_start_idx_array, index=df.index)
        df.loc[cut_mask.shift(1, fill_value=False), "position"] = 0

        # Tính lại sau khi cut
        df["action"] = df['position'].diff(1).fillna(df['position'].iloc[0])
        df["turnover"] = df["action"].abs()
        df["fee"] = df["turnover"] * fee
        df["grossProfit"] = df["position"] * df["priceChange"]
        df["netProfit"] = df["grossProfit"] - df["fee"]
        # print(df[df['day'] == 20250805])
        # Lưu dữ liệu SAU khi cut loss
        # df[df['day'] == 20250805].to_csv('/home/ubuntu/nevir/gen1_1/trades_cut.csv')
        return df
    
    @staticmethod
    def compute_profits(df_alpha,stop_loss=0,fee=0.175):
        df_alpha['grossProfit'] = df_alpha['position'] * df_alpha['priceChange']
        df_alpha['netProfit'] = df_alpha['grossProfit'] - df_alpha['fee']
        if stop_loss>0:
            df_alpha = Alpha_Domains.hard_cut_loss(df_alpha, stop_loss=stop_loss, fee=fee)
        df_alpha['cumGrossProfit'] = df_alpha['grossProfit'].cumsum()
        df_alpha['cumNetProfit'] = df_alpha['netProfit'].cumsum()

    @staticmethod
    def extract_trades_df(df_alpha):
        df = df_alpha
        # df = df.set_index('executionT').copy()
        df["position_prev"] = df["position"].shift(fill_value=0)
        df["position_next"] = df["position"].shift(-1, fill_value=0)

        # trade_start: khi từ 0 sang có vị thế hoặc đổi chiều
        df["trade_start"] = (((df["position_prev"] == 0) & (df["position"] != 0)) |
                            (df["position_prev"] * df["position"] < 0)).astype(int)

        # trade_end: khi position hiện tại khác position kế tiếp (về 0 hoặc đảo chiều)
        df["trade_end"] = ((df["position"] != 0) & (df["position_next"] != df["position"])).astype(int)


        # Gán trade_id và loại bỏ các đoạn không có vị thế
        df["trade_id"] = df["trade_start"].cumsum()
        df.loc[df["position"] == 0, "trade_id"] = None
        df["trade_id"] = df["trade_id"].ffill()

        trades = []
        # a = df[df['day']==20220125].copy()
        # a.to_csv("/home/ubuntu/nevir/fenix/Backtest/src/api/blueprints/stock/20220125.csv")
        for trade_id, group in df.groupby("trade_id"):
            if pd.isna(trade_id) or group["position"].iloc[0] == 0:
                continue

            entry_row = group[group["trade_start"] == 1].iloc[0]
            entry_time = entry_row.name

            # Lấy dòng kết thúc cuối cùng
            exit_row = group[group["trade_end"] == 1].iloc[-1]
            exit_time = exit_row.name

            # Kiểm tra lý do kết thúc: về 0 hay đảo chiều
            position = exit_row["position"]
            position_next = exit_row["position_next"]

            if position_next == 0:
                # Trade kết thúc do thoát lệnh → giữ dòng exit
                exit_time_next = df.index[df.index.get_loc(exit_time) + 1]
                trade_slice = df.loc[entry_time:exit_time_next]
            else:
                trade_slice = df.loc[entry_time:exit_time]

            gross_profit = trade_slice["grossProfit"].sum()
            fee = trade_slice["fee"].sum()
            net_profit = trade_slice["netProfit"].sum()

            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "gross_profit": gross_profit,
                "fee": fee,
                "net_profit": net_profit,
            })

        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("/home/ubuntu/nevir/gen1_1/trades.csv")
    @staticmethod
    def compute_df_trade(df_alpha):
        """Lưu time_int và turnover của các giao dịch"""
        df_trade = df_alpha[df_alpha["action"] != 0][["action","executionT"]]
        return df_trade
    
    @staticmethod
    def calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
        """
        Tính số ngày làm việc từ start đến end, với chuẩn workdays_per_year (mặc định 250 ngày/năm).

        Args:
            start (int): Ngày bắt đầu dạng yyyymmdd, ví dụ 20240101.
            end (int): Ngày kết thúc dạng yyyymmdd, ví dụ 20250101.
            workdays_per_year (int): Số ngày làm việc mỗi năm (mặc định 250).

        Returns:
            int: Số ngày làm việc theo chuẩn workdays_per_year.
        """
        if type(start) is int:
            start_date = datetime.strptime(str(start), "%Y%m%d")
            end_date = datetime.strptime(str(end), "%Y%m%d")
            
        else:
        # Chuyển int -> datetime
            start_date = datetime.strptime(start, "%Y_%m_%d")
            end_date = datetime.strptime(end, "%Y_%m_%d")

            # Số ngày thực tế giữa 2 mốc
        total_days = (end_date - start_date).days

        # Chuyển đổi sang ngày làm việc
        working_days = total_days * workdays_per_year / 365.0

        return round(working_days)

    @staticmethod
    def compute_performance(df_alpha, start=None, end=None,equity=300):
        
        working_days = Alpha_Domains.calculate_working_days(start,end)
        agg_dict = {
            "open": "sum",
            "turnover": "sum",
            "netProfit": "sum",
        }

        if "booksize" in df_alpha.columns:
            agg_dict["booksize"] = "last"
        df_1d = df_alpha \
            .groupby('day') \
            .agg(agg_dict)
        # print("10 day best loss", df_1d['netProfit'].nsmallest(10))
        # print("10 day best win", df_1d['netProfit'].nlargest(10))
        
        # df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
        mean = df_1d["netProfit"].mean()
        std = df_1d["netProfit"].std()

        if std and not np.isnan(std):
            sharpe = mean / std * working_days  ** 0.5
            # daily_sharpe = mean / std
        else:
            sharpe = np.nan
            # daily_sharpe =  np.nan
        
        # roe = df_1d['pctChange'].sum()
        tvr = df_1d['turnover'].mean()
        # aroe = roe * 250 / len(df_1d)
        ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)
        
        mdd, mdd_pct, cdd, cdd_pct = Alpha_Domains.compute_mdd_vectorized(df_1d,equity)
        # df = df_alpha
        # df["position_prev"] = df["position"].shift(fill_value=0)
        # df["position_next"] = df["position"].shift(-1, fill_value=0)

        # # trade_start: khi từ 0 sang có vị thế hoặc đổi chiều
        # df["trade_start"] = (((df["position_prev"] == 0) & (df["position"] != 0)) |
        #                     (df["position_prev"] * df["position"] < 0)).astype(int)

        # # trade_end: khi position hiện tại khác position kế tiếp (về 0 hoặc đảo chiều)
        # df["trade_end"] = ((df["position"] != 0) & (df["position_next"] != df["position"])).astype(int)


        # # Gán trade_id và loại bỏ các đoạn không có vị thế
        # df["trade_id"] = df["trade_start"].cumsum()
        # df.loc[df["position"] == 0, "trade_id"] = None
        # df["trade_id"] = df["trade_id"].ffill()
        
        # trade_results = df.groupby("trade_id")["netProfit"].sum()
        # num_trades = trade_results.count()
        # winrate = round((trade_results > 0).sum() / num_trades * 100, 2) if num_trades > 0 else None
        # returns = df_1d['netProfit'] / equity
        # winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
        # loss_profits = df_1d[df_1d['netProfit'] < 0]['netProfit']
        # net_profit_pos = winning_profits.sum()
        # net_profit_neg = loss_profits.sum()
        # npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
        # total_profit = winning_profits.sum()
        # shares = winning_profits / total_profit
        # hhi = (shares ** 2).sum()
        # try:
        #     E = (1 + (trade_results[trade_results > 0].mean()/abs(trade_results[trade_results < 0].mean()))) * (winrate/100) - 1
        # except Exception as e:
        #     E = 0
        new_report = {
            'sharpe': round(sharpe, 3),
            # "hhi": round(hhi,3),
            # "psr": round(Alpha_Domains.dsr(returns, daily_sharpe,0),3),
            # "dsr": round(Alpha_Domains.dsr(returns, daily_sharpe),3),
            # 'aroe': round(aroe, 4),
            'mdd': round(mdd, 3),
            'mddPct': round(mdd_pct.iloc[-1], 4),
            # 'cdd': round(cdd, 3),
            # 'cddPct': round(cdd_pct.iloc[-1], 4),
            'ppc': round(ppc, 4),
            # "E":round(E,2),
            'tvr': round(tvr, 4),
            # "npf":float(round(npf, 2)),
            # 'start': df_1d.index[0],
            # 'end': df_1d.index[-1],
            # 'lastProfit': round(df_1d['netProfit'].iloc[-1], 2),
            "netProfit": round(df_1d['netProfit'].sum(), 2),
            "profitPct": round(df_1d['netProfit'].sum(), 2) / equity * 100,
            # "max_loss": round(df_1d['netProfit'].min(), 2),
            # "max_gross": round(df_1d['netProfit'].max(), 2),
            # "winrate":winrate,
            # "num_trades": num_trades,
        }
        
        df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        df_1d = df_1d.reset_index()
        df_1d.index = df_1d['day'].values
        df_1d['netProfit'] = df_1d['netProfit'].round(2)
        df_1d['ccd1'] = cdd_pct
        df_1d['mdd1'] = mdd_pct
        
        # print(df_alpha[df_alpha['day'] == "2025_09_04"][['open','executionT',"position","entryPrice","exitPrice","grossProfit","netProfit"]])
        # Alpha_Domains.extract_trades_df(df_alpha)
        return df_1d, new_report
    @staticmethod
    def dsr(returns,sharpe, sr_benchmark=0.18):
        try:
            def volatility_sharpe(returns):
                sample_size = len(returns)
                skewness = skew(returns)
                _kurtosis = kurtosis(returns, fisher=True, bias=False)
                return np.sqrt((1 - skewness*sharpe + (_kurtosis + 2)/4*sharpe**2) / (sample_size - 1)), sharpe
            
            sigma_sr, sr = volatility_sharpe(returns)
            z = (sr - sr_benchmark) / sigma_sr
            return 0.5 * (1 + erf(z / sqrt(2))) * 100
        except Exception as e:
            return 0
        
    @staticmethod
    def extract_net_profits(df):
        # shift để xác định điểm vào/ra
        pos = df["position"]
        pos_prev = pos.shift(fill_value=0)

        # trade start: từ 0 sang ≠0 hoặc đổi chiều
        trade_start = ((pos_prev == 0) & (pos != 0)) | (pos_prev * pos < 0)

        # trade_id
        trade_id = trade_start.cumsum()
        trade_id = trade_id.where(pos != 0).ffill()

        df = df.copy()
        df["trade_id"] = trade_id

        # Gom theo trade_id → tính tổng netProfit
        net_profits = (
            df.dropna(subset=["trade_id"])
            .groupby("trade_id")["netProfit"]
            .sum()
            .tolist()
        )

        return net_profits
    
    @staticmethod
    def compute_mdd_vectorized(df_1d, equity=300):
        """
        Tính Maximum Drawdown (MDD) có xét đến equity ban đầu.
        - MDD% được tính từ CDD% = (cummax - cumNetProfit) / (equity + cummax)
        """

        if 'cumNetProfit' in df_1d:
            net_profit = df_1d['cumNetProfit']
        else:
            net_profit = df_1d['netProfit'].cumsum()

        cummax = net_profit.cummax()
        cdd = cummax - net_profit
        cdd_pct = (cdd / (equity + cummax) * 100)

        mdd = cdd.cummax().iloc[-1]
        mdd_pct = cdd_pct.cummax()
        cdd_last = cdd.iloc[-1]
        cdd_pct_last = cdd_pct.iloc[-1]

        return mdd, mdd_pct, cdd_last, cdd_pct



