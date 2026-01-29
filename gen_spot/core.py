from datetime import datetime
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
class Simulator:
    def __init__(self, base_name, freq=1, gen_params=None, fee=0.1, df_base=None,DIC_BASES=None,params={},df_tick=None,gen=None,stop_loss=0,is_sizing=False,init_sizing=37.5,start=None,end=None,booksize=1): 

        self.freq = freq
        self.gen_params = gen_params
        self.fee = fee
        self.stop_loss = stop_loss
        self.gen = gen
        self.start = start
        self.end = end
        self.booksize = booksize
        self.df_base = df_base
        self.df_tick = df_tick
        
        self.base_func = DIC_BASES[base_name]
        self.params = params
        self.report = {
            'base_name': base_name,
            'gen':gen,
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
        Base_Domains.compute_signal(base_func=self.base_func, df_base=self.df_base, params=self.params)

        self.df_base = self.df_base[
            (self.df_base['day'] >= self.start) &
            (self.df_base['day'] <= self.end)
        ]
            

    def compute_position(self):
        if self.gen == "1_1":
            Base_Domains.compute_position(
                df_base=self.df_base,
                threshold=self.gen_params['threshold'])
        elif self.gen == "1_2":
            Base_Domains.compute_positions_with_thresholds(
                df_base=self.df_base,
                upper=self.gen_params['upper'],
                lower=self.gen_params['lower'])
        elif self.gen == "1_3":
            Base_Domains.compute_positions_with_signal_score(
                df_base=self.df_base,
                score_window=self.gen_params['score'],
                entry_score=self.gen_params['entry'],
                exit_score=self.gen_params['exit'])
        elif self.gen == "1_4":
            Base_Domains.compute_position_with_velocity(
                df_base=self.df_base,
                vel_entry=self.gen_params['entry'],
                vel_exit=self.gen_params['exit'],
                smooth_window=self.gen_params['smooth'] )
        
        if self.is_sizing:
            self.df_base["position_init"] = self.df_base['position'].values 
            self.sizing_positon()
        else:
            self.df_base['position'] = self.df_base['position'] * self.booksize
            self.df_base['booksize'] = self.booksize
    def get_budget(self,total_netProfit):
        df_filtered = self.hard_dic_budget[self.hard_dic_budget['cum'] <= total_netProfit]
        if df_filtered.empty:
            return 0
        else:
            return df_filtered['action'].iloc[-1]
        
    def sizing_positon(self):   
        df = self.df_base.copy()
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
            
            self.df_base.loc[df_day.index, f"position"] = df_day['position']
            self.df_base.loc[df_day.index, f"booksize"] = booksize
            
    def compute_tvr_and_fee(self):
        Base_Domains.compute_action_tvr_and_fee(
            self.df_base,
            self.fee)

    def compute_profits(self):
        Base_Domains.compute_profits(self.df_base,self.stop_loss,self.fee)

    def compute_performance(self, start=None, end=None):
 
        self.df_1d, report = Base_Domains.compute_performance(
            self.df_base,
            start=start,
            end=end,equity = self.booksize*300)
        self.report.update(report)
    def merge_position_to_ticks(self):
        self.df_base = Base_Domains.merge_position_to_ticks(self.df_base,self.df_tick)
    def compute_df_trade(self):
        self.df_trade = Base_Domains.compute_df_trade(self.df_base)
    def extract_net_profits(self):
        return Base_Domains.extract_net_profits(self.df_base)
    
class Base_Domains:
    @staticmethod
    def compute_signal(base_func, df_base, params={}):
        
        df_base['position'] = df_base['signal'] = base_func(df_base,**params)
        


    def compute_position(df_base, threshold):
        flt_mediocre = df_base['signal'].abs() < threshold
        df_base.loc[flt_mediocre, 'signal'] = np.nan
        
        df_base['position'] = df_base['signal']  
        Base_Domains.adjust_positions(df_base)
        
        df_base['position'] = np.sign(df_base['position'])
    @staticmethod
    def adjust_positions(df_base):
        flt_unexecutable = ~df_base['executable']
        df_base.loc[flt_unexecutable, 'position'] = np.nan
        flt_atc = df_base['executionTime'] == '14:45:00'
        df_base.loc[flt_atc, 'position'] = 0
        df_base['position'] = df_base['position'].ffill().fillna(0)
        
    @staticmethod
    def compute_positions_with_signal_score(df_base, score_window=5, entry_score=3, exit_score=1):
        """
        Cộng điểm theo hướng tín hiệu trong cửa sổ thời gian ngắn.
        - Khi tổng điểm >= entry_score -> mở vị thế long
        - Khi tổng điểm <= -entry_score -> mở vị thế short
        - Khi |tổng điểm| <= exit_score -> đóng vị thế
        """
        Base_Domains.adjust_positions(df_base)
        df_base['signal'] = df_base['signal'].round(6)
        signed_signal = np.sign(df_base['position'].fillna(0))
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

        df_base['position'] = new_positions
        Base_Domains.adjust_positions(df_base)
    
    @staticmethod
    def compute_position_with_velocity(df_base, vel_entry=0.2, vel_exit=0.05, smooth_window=1):
        
        """
        Vào lệnh theo tốc độ thay đổi tín hiệu:
        - Khi velocity > vel_entry -> long
        - Khi velocity < -vel_entry -> short
        - Khi |velocity| < vel_exit -> đóng vị thế
        Có thể làm mượt velocity bằng rolling mean (smooth_window > 1).
        """
        Base_Domains.adjust_positions(df_base)

        velocity = df_base['position'].diff().fillna(0)
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

        df_base['position'] = new_positions
        Base_Domains.adjust_positions(df_base)
            
    @staticmethod
    def compute_positions_with_thresholds(df_base, upper, lower):
        
        Base_Domains.adjust_positions(df_base)
        lst_pos = []
        last_pos = 0
        df_base['signal'] = df_base['signal'].round(6)
        for pos in df_base['signal']:

            if abs(pos) >= upper:
                last_pos = np.sign(pos)
            elif (last_pos > 0) and pos < lower:
                last_pos = 0
            elif (last_pos < 0) and pos > -lower:
                last_pos = 0
            lst_pos.append(last_pos)
        df_base['position'] = lst_pos
        return Base_Domains.adjust_positions(df_base)
        
        
    @staticmethod
    def compute_action_tvr_and_fee(df_base, fee):
        df_base['action'] = df_base['position'].diff(1).fillna(df_base['position'].iloc[0])
        df_base['turnover'] = df_base['action'].abs()
        df_base['fee'] = df_base['turnover'] * fee
        
    @staticmethod
    def merge_position_to_ticks(df_base, df_tick):
        # ===== LỌC tick theo khoảng thời gian của base =====
        # df_base[df_base['day'] == '2024_01_04'].to_csv('/home/ubuntu/nevir/gen1_2/pre_merge.csv')
        df_base = df_base.set_index('executionT')
        # start, end = df_base.index.min(), df_base.index.max()
        # df_tick = df_tick.loc[(df_tick.index >= start) & (df_tick.index <= end)]
        
        # ===== MERGE: dùng tick làm timeline chính =====
        df = df_tick[["last","day"]].join(df_base[['position','open']], how="left")
        # Tìm chỉ số cuối cùng của mỗi ngày
        is_last_of_day = df['day'].shift(-1) != df['day']
        last_day_indices = df[is_last_of_day].index

        # Gán position cuối ngày = 0
        df.loc[last_day_indices, "position"] = 0

        
        # ===== FILLFORWARD toàn bộ cột từ base =====
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
    def compute_profits(df_base,stop_loss=0,fee=0.175):
        df_base['grossProfit'] = df_base['position'] * df_base['priceChange']
        df_base['netProfit'] = df_base['grossProfit'] - df_base['fee']
        if stop_loss>0:
            df_base = Base_Domains.hard_cut_loss(df_base, stop_loss=stop_loss, fee=fee)
        df_base['cumGrossProfit'] = df_base['grossProfit'].cumsum()
        df_base['cumNetProfit'] = df_base['netProfit'].cumsum()

    @staticmethod
    def extract_trades_df(df_base):
        df = df_base
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
    def compute_df_trade(df_base):
        """Lưu time_int và turnover của các giao dịch"""
        df_trade = df_base[df_base["action"] != 0][["action","executionT"]]
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
    def compute_performance(df_base, start=None, end=None,equity=300):
        try:
            lst_errs = []
            
            working_days = Base_Domains.calculate_working_days(start,end)
            agg_dict = {
                # "open": "sum",
                "turnover": "sum",
                "netProfit": "sum",
            }
            
            if "booksize" in df_base.columns:
                agg_dict["booksize"] = "last"
            df_1d = df_base \
                .groupby('day') \
                .agg(agg_dict)
            # print("10 day best loss", df_1d['netProfit'].nsmallest(10))
            # print("10 day best win", df_1d['netProfit'].nlargest(10))
            
            # df_1d['pctChange'] = df_1d['netProfit'] / df_1d['open']
            try:
                # sharpe = df_1d['pctChange'].mean() / df_1d['pctChange'].std() * 250 ** 0.5
                mean = df_1d["netProfit"].mean()
                std = df_1d["netProfit"].std()

                if std and not np.isnan(std):
                    sharpe = mean / std * working_days  ** 0.5
                    daily_sharpe = mean / std
                else:
                    sharpe = np.nan
                    daily_sharpe =  np.nan
            except Exception as e:
                lst_errs.append(f"{e}")
                # U.report_error(e)
                sharpe = -999
            
            # roe = df_1d['pctChange'].sum()
            tvr = df_1d['turnover'].mean()
            # aroe = roe * 250 / len(df_1d)
            ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)
            
            mdd, mdd_pct, cdd, cdd_pct = Base_Domains.compute_mdd_vectorized(df_1d,equity)
            df = df_base
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
            
            trade_results = df.groupby("trade_id")["netProfit"].sum()
            num_trades = trade_results.count()
            winrate = round((trade_results > 0).sum() / num_trades * 100, 2) if num_trades > 0 else None
            returns = df_1d['netProfit'] / equity
            winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
            loss_profits = df_1d[df_1d['netProfit'] < 0]['netProfit']
            net_profit_pos = winning_profits.sum()
            net_profit_neg = loss_profits.sum()
            npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
            total_profit = winning_profits.sum()
            shares = winning_profits / total_profit
            hhi = (shares ** 2).sum()
            try:
                E = (1 + (trade_results[trade_results > 0].mean()/abs(trade_results[trade_results < 0].mean()))) * (winrate/100) - 1
            except Exception as e:
                E = 0
            new_report = {
                'sharpe': round(sharpe, 3),
                "hhi": round(hhi,3),
                "psr": round(Base_Domains.dsr(returns, daily_sharpe,0),3),
                "dsr": round(Base_Domains.dsr(returns, daily_sharpe),3),
                # 'aroe': round(aroe, 4),
                'mdd': round(mdd, 3),
                'mddPct': round(mdd_pct.iloc[-1], 4),
                'cdd': round(cdd, 3),
                'cddPct': round(cdd_pct.iloc[-1], 4),
                'ppc': round(ppc, 4),
                "E":round(E,2),
                'tvr': round(tvr, 4),
                "npf":float(round(npf, 2)),
                'start': df_1d.index[0],
                'end': df_1d.index[-1],
                'lastProfit': round(df_1d['netProfit'].iloc[-1], 2),
                "netProfit": round(df_1d['netProfit'].sum(), 2),
                "profitPct": round(df_1d['netProfit'].sum(), 2) / equity * 100,
                "max_loss": round(df_1d['netProfit'].min(), 2),
                "max_gross": round(df_1d['netProfit'].max(), 2),
                "winrate":winrate,
                "num_trades": num_trades,
            }
            
            df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
            df_1d = df_1d.reset_index()
            df_1d.index = df_1d['day'].values
            df_1d['netProfit'] = df_1d['netProfit'].round(2)
            df_1d['ccd1'] = cdd_pct
            df_1d['mdd1'] = mdd_pct
            # print(df_1d)
            # exit()
            # print(df_base[df_base['day'] == "2022_02_11"][['executionT',"position","entryPrice","exitPrice","grossProfit","netProfit","signal"]])
            # Base_Domains.extract_trades_df(df_base)
            return df_1d, new_report
        except Exception as e:
            print(f"Error in compute_performance: {str(e)}")
            return None, None
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



