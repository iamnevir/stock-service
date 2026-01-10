from math import erf, sqrt
from time import time
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from datetime import datetime

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
    # Chuyển int -> datetime
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")

    # Số ngày thực tế giữa 2 mốc
    total_days = (end_date - start_date).days

    # Chuyển đổi sang ngày làm việc
    working_days = total_days * workdays_per_year / 365.0

    return round(working_days)

class MegaBbAccV2:
    def __init__(
        self,
        alpha_name="",
        configs=[],
        fee=150,
        df_alpha=None,
        data_start=None,
        data_end=None,
        busd_source="hose500",  # Options: "hose500", "vn30"
        position_multiplier=1.0,
        fma_length=200,
        foreign_policy=0,
        replace_filter_value=0,
        moving_average_type="ema",
        filter_col="aggFBusdVn30",
        foreign_multiplier=0,
        foreign_add_column="aggFBusdVn30",
        update_second_only="False",
        delay=1,
        stop_loss = None,
        book_size=1,
        is_sizing=False,
        init_sizing=0
    ):
        self.stop_loss = stop_loss
        self.alpha_name = alpha_name
        self.configs = configs
        self.four_param_configs = [
            "_".join(x.strip("_alpha_bb_acc_").split("_")[:4]) for x in self.configs
        ]
        self.fee = int(fee)
        self.df_alpha = df_alpha
        self.data_start = data_start
        self.data_end = data_end
        self.busd_source = busd_source
        self.book_size = book_size

        self.foreign_add_column = foreign_add_column
        self.foreign_multiplier = foreign_multiplier
        self.update_second_only = update_second_only

        self.position_multiplier = position_multiplier
        self.fma_length = fma_length
        self.moving_average_type = moving_average_type
        self.filter_column = filter_col
        self.replace_filter_value = replace_filter_value
        self.foreign_policy = foreign_policy

        self.mas = pd.DataFrame()
        self.position_df = pd.DataFrame()
        self.turnover_df = pd.DataFrame()
        self.net_profit_df = pd.DataFrame()
        self.turnover_df1d = pd.DataFrame()
        self.net_profit_df1d = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.df_1d = pd.DataFrame()
        self.report = {}
        self.delay = delay
        self.num_cut_trades = 0
        self.n_alphas = len(configs)
        self.is_sizing = is_sizing
        self.hard_dic_budget = pd.DataFrame({'status':  range(101)})
        self.hard_dic_budget['cum'] = init_sizing
        self.hard_dic_budget['cum'] =  self.hard_dic_budget['status'] *  self.hard_dic_budget['cum']
        self.hard_dic_budget.loc[0, 'cum'] = init_sizing
        self.hard_dic_budget['cum'] = self.hard_dic_budget['cum'].cumsum()
        self.hard_dic_budget['action'] = self.hard_dic_budget['status'] + 1
        
    # def compute_based_col(self):
    #     df = self.df_alpha
    #     df = df[(df["day"] >= self.data_start) & (df["day"] <= self.data_end)]
    #     df["based_col"] = 0
    #     for col in self.busd_source.split("_"):
    #         if col == "hose500":
    #             df["based_col"] += df["aggBusd"]
    #         elif col == "fhose500":
    #             df["based_col"] += df["aggFBusd"]
    #         elif col == "vn30":
    #             df["based_col"] += df["aggBusdVn30"]
    #         elif col == "fvn30":
    #             df["based_col"] += df["aggFBusdVn30"]
            
            
    #     df = df[df["based_col"].diff(1) != 0]
    #     df = df[df['time_int'] >= 10091500]
    #     df["priceChange"] = df.groupby("day")["last"].diff(1).shift(-1).fillna(0)
    #     self.df_alpha = df

    def compute_mas(self):
        # df = self.df_alpha.copy()
        df = self.df_alpha
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs]
        ma_lengths = set(ma1 + ma2)

        mode = self.moving_average_type

        def wma_fast(series: pd.Series, length: int) -> pd.Series:
            if length <= 1:
                return series.copy()
            weights = np.arange(1, length + 1)
            divisor = weights.sum()
            wma_values = np.convolve(series.fillna(0).to_numpy(), weights[::-1], mode='valid') / divisor
            result = np.full(series.shape, np.nan)
            result[length - 1:] = wma_values
            return pd.Series(result, index=series.index)

        def calc_ma(x: pd.Series, length: int) -> pd.Series:
            if mode == "ema":
                return x.ewm(span=length, adjust=True).mean()
            elif mode == "dema":
                ema1 = x.ewm(span=length, adjust=True).mean()
                ema2 = ema1.ewm(span=length, adjust=True).mean()
                return 2 * ema1 - ema2
            elif mode == "hma":
                half = int(length / 2)
                sqrt_len = int(np.sqrt(length))
                wma_half = wma_fast(x, half)
                wma_full = wma_fast(x, length)
                diff = 2 * wma_half - wma_full
                return wma_fast(diff, sqrt_len)
            elif mode == "sma":
                return x.rolling(window=length, min_periods=1).mean()
            else:
                raise ValueError(f"Unknown MA mode: {mode}")

        gb = df.groupby("day")
        cols = []

        for day, df_day in gb:
            for ma_length in ma_lengths:
                ma_values = calc_ma(df_day["based_col"], ma_length)
                df.loc[df_day.index, f"ma_{ma_length}"] = ma_values.values

            if self.foreign_policy != 0:
                filter_vector = np.sign(
                    df_day[self.filter_column] - df_day[self.filter_column].rolling(window=self.fma_length).mean()
                ) * self.foreign_policy
                df.loc[df_day.index, "foreign_filter"] = filter_vector
                cols = ["foreign_filter"]
                self.df_alpha["foreign_filter"] = df["foreign_filter"]

        self.mas = df[cols + [f"ma_{x}" for x in ma_lengths]]


    def compute_all_position(self):
        # start_time = time()
        df = self.df_alpha
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs]
        threshold = [int(x.split("_")[2]) for x in self.four_param_configs]
        exit_strength = [float(x.split("_")[3]) for x in self.four_param_configs]

        ma1_matrix = np.array([self.mas[f"ma_{x}"] for x in ma1]).T
        ma2_matrix = np.array([self.mas[f"ma_{x}"] for x in ma2]).T
        
        diff_matrix = ma2_matrix - ma1_matrix
        threshold_matrix = np.tile(threshold, (len(diff_matrix), 1))
        # threshold_matrix = np.abs(ma2_matrix) * (np.tile(threshold, (len(diff_matrix), 1)) / 100.0)
        
        position_matrix = np.full(diff_matrix.shape, np.nan)
        position_matrix[diff_matrix > threshold_matrix] = 1
        position_matrix[diff_matrix < -threshold_matrix] = -1

        if np.abs(exit_strength).sum() > 0:
            exit_matrix = np.tile(exit_strength, (len(diff_matrix), 1))
            take_profit_matrix = threshold_matrix * exit_matrix / 100
            position_matrix[np.abs(diff_matrix) < take_profit_matrix] = 0

        position_df = pd.DataFrame(position_matrix, columns=self.four_param_configs, index=df.index)

        position_df = position_df.shift(self.delay, fill_value=0)

        position_df[df["time_int"] >= 10142930] = np.nan
        position_df[df["time_int"] >= 10144500] = 0
        position_df[df["time_int"] <= 10091500] = 0
        day_cumcount = df.groupby('day').cumcount().values
        position_df.values[day_cumcount < self.delay] = 0.0
        position_df = position_df.ffill(axis=0).fillna(0)
        
        self.position_df = position_df 
        # print(f"time: {time()-start_time}")
        
    def compute_all_position_v3(self):
        # start_time = time()
        df = self.df_alpha
        ma1 = [int(x.split("_")[0]) for x in self.four_param_configs]
        ma2 = [int(x.split("_")[1]) for x in self.four_param_configs]
        threshold = [int(x.split("_")[2]) for x in self.four_param_configs]
        exit_strength = [float(x.split("_")[3]) for x in self.four_param_configs]

        ma1_matrix = np.array([self.mas[f"ma_{x}"] for x in ma1]).T
        ma2_matrix = np.array([self.mas[f"ma_{x}"] for x in ma2]).T
        
        diff_matrix = ma2_matrix - ma1_matrix
        diff_df = pd.DataFrame(diff_matrix)
        
        # Xác định cửa sổ (window) cho dải Bollinger
        window = threshold[0] 
        rolling_ma = diff_df.ewm(span=window, adjust=True).mean()
        rolling_std = diff_df.rolling(window=window).std()
        multiplier = exit_strength[0] 

        upper_threshold_matrix = rolling_ma + (rolling_std * multiplier)
        lower_threshold_matrix = rolling_ma - (rolling_std * multiplier)
        position_matrix = np.full(diff_matrix.shape, np.nan)
        position_matrix[diff_matrix > upper_threshold_matrix.values] = 1
        position_matrix[diff_matrix < lower_threshold_matrix.values] = -1
        # Exit conditions
        # long_exit_mask  = (position_matrix == 1) & (diff_matrix <= rolling_ma.values)
        # short_exit_mask = (position_matrix == -1) & (diff_matrix >= rolling_ma.values)

        # position_matrix[long_exit_mask | short_exit_mask] = 0

        position_df = pd.DataFrame(position_matrix, columns=self.four_param_configs, index=df.index)
        position_df = position_df.shift(self.delay, fill_value=0)

        position_df[df["time_int"] >= 10142930] = np.nan
        position_df[df["time_int"] >= 10144500] = 0
        position_df[df["time_int"] <= 10091500] = 0
        day_cumcount = df.groupby('day').cumcount().values
        position_df.values[day_cumcount < self.delay] = 0.0
        position_df = position_df.ffill(axis=0).fillna(0)
        
        self.position_df = position_df 
        
    def compute_mega_position(self):
        positions = self.position_df.sum(axis=1)
        positions.ffill(inplace=True)
        # self.df_alpha["position"] = positions.values 
        if self.is_sizing:
            self.df_alpha["position_init"] = positions.values / self.n_alphas 
            self.sizing_positon()
        else:
            self.df_alpha["position"] = ((positions.values / self.n_alphas * self.book_size).round(6).astype(int))
            self.df_alpha['booksize'] = self.book_size
        # self.df_alpha["position"] = positions.values / len(self.configs) * self.book_size
        # self.df_alpha["position"] = self.df_alpha["position"].round(6).astype(int)
        
    def get_budget(self,total_netProfit):
        df_filtered = self.hard_dic_budget[self.hard_dic_budget['cum'] <= total_netProfit]
        if df_filtered.empty:
            return 0
        else:
            return df_filtered['action'].iloc[-1]
        
    def sizing_positon(self):   
        df = self.df_alpha.copy()
        
        lst_day = df['day'].unique()
        
        total_netProfit = 0
        for day in lst_day:
            df_day = df[df['day'] == day].copy()
 
            booksize = self.book_size + self.get_budget(total_netProfit)
            df_day['position'] = ((df_day['position_init'] * booksize).round(6).astype(int))
      
            
            df_day['grossProfit'] = df_day['position'] * df_day['priceChange']
            df_day['action'] = df_day['position'] - df_day['position'].shift(1, fill_value=0)
            df_day['turnover'] = df_day['action'].abs()
            df_day['fee'] = df_day['turnover'] * self.fee / 1000
            df_day['netProfit'] = df_day['grossProfit'] - df_day['fee']
            
            intraday_netProfit = df_day['netProfit'].sum()
            
            total_netProfit += intraday_netProfit
            
            self.df_alpha.loc[df_day.index, f"position"] = df_day['position']
            self.df_alpha.loc[df_day.index, f"booksize"] = booksize
            
    def compute_profit_and_df_1d(self):
        df = self.df_alpha

        # ====== BƯỚC 1: TÍNH LẦN ĐẦU ======
        df["grossProfit"] = df["position"] * df["priceChange"]
        df["action"] = df["position"] - df["position"].shift(1, fill_value=0)
        df["turnover"] = df["action"].abs()
        df["fee"] = df["turnover"] * self.fee / 1000
        df["netProfit"] = df["grossProfit"] - df["fee"]
        df["pctChange"] = df["netProfit"] / df["last"]
        if self.stop_loss and self.stop_loss > 0:
            self.hard_cut_loss(df)
        # a = df[df['day']==20250611].copy()
        # a.to_csv("/home/ubuntu/nevir/fenix/Backtest/src/api/blueprints/stock/20250611.csv")
        # ====== BƯỚC 6: GROUP THEO NGÀY ======
        agg_dict = {
            "grossProfit": "sum",
            "turnover": "sum",
            "netProfit": "sum",
            "booksize":"last"
        }

        df_1d = df.groupby("day").agg(agg_dict)
        df_1d[["cumGrossProfit", "cumTurnover", "cumNetProfit"]] = \
            df_1d[["grossProfit", "turnover", "netProfit"]].cumsum()

        cols_to_round = ["grossProfit", "netProfit", "cumNetProfit", "cumGrossProfit","booksize"]

        df_1d[cols_to_round] = df_1d[cols_to_round].round(2)
        self.df_1d = df_1d
        
    def hard_cut_loss(self, df):
        #====== BƯỚC 2: XÁC ĐỊNH TRADE_ID ======
        df["position_prev"] = df["position"].shift(fill_value=0)
        df["trade_start"] = ((df["position_prev"] == 0) & (df["position"] != 0)) | \
                            (df["position_prev"] * df["position"] < 0)
        df["trade_id"] = df["trade_start"].cumsum()
        df.loc[(df["position"] == 0) & (df["netProfit"] == 0), "trade_id"] = None
        df["trade_id"] = df["trade_id"].ffill()

        # ====== BƯỚC 3: TÍNH LŨY KẾ LÃI/LỖ THEO TRADE ======
        df["trade_cum_profit"] = df.groupby("trade_id")["grossProfit"].cumsum()

        # ====== BƯỚC 4: CẮT LỖ TOÀN BỘ TRADE (vectorized, hỗ trợ DatetimeIndex) ======
        

        # a. Lấy các trade bị cut
        cut_condition = (df["time_int"] < 10142930) & (df["trade_cum_profit"] < -self.stop_loss)
        cut_first_idx = (
            df[cut_condition]
            .sort_index()
            .drop_duplicates("trade_id", keep="first")
        )

        # b. Ánh xạ trade_id -> index datetime của tick bị cut đầu tiên
        cut_start_map = dict(zip(cut_first_idx["trade_id"], cut_first_idx.index))

        # c. Tạo mảng ánh xạ: mỗi dòng gán cut_start_index của trade_id tương ứng
        trade_ids_array = df["trade_id"].values
        index_array = pd.to_datetime(df.index)
        cut_start_idx_array = np.array([
            cut_start_map.get(tid, pd.Timestamp.max) for tid in trade_ids_array
        ])

        # d. Tạo mask các dòng cần bị cut (as Series)
        cut_mask = pd.Series(index_array >= cut_start_idx_array, index=df.index)

        # e. Cập nhật position = 0 tại dòng kế tiếp của cut point
        df.loc[cut_mask.shift(1, fill_value=False), "position"] = 0


        self.num_cut_trades = cut_first_idx["trade_id"].nunique()
        
        df.loc[df["time_int"] >= 10142930, 'position'] = np.nan
        df.loc[df["time_int"] >= 10144500, 'position'] = 0
        df.loc[df["time_int"] <= 10091500, 'position'] = 0
        df['position'] = df['position'].ffill().fillna(0)
        
        # ====== BƯỚC 5: TÍNH LẠI SAU CẮT ======
        df["action"] = df["position"] - df["position"].shift(1, fill_value=0)
        df["turnover"] = df["action"].abs()
        df["fee"] = df["turnover"] * self.fee / 1000
        df["grossProfit"] = df["position"] * df["priceChange"]
        df["netProfit"] = df["grossProfit"] - df["fee"]
        df["pctChange"] = df["netProfit"] / df["last"]
        # a = df.copy()
        # a. ("/home/ubuntu/nevir/fenix/Backtest/src/api/blueprints/stock/20250609.csv")
    
    def compute_df_trade(self):
        """Lưu time_int và turnover của các giao dịch"""
        self.df_trade = self.df_alpha[self.df_alpha["action"] != 0][["action"]]
        
    def compute_report(self, df_1d, template=None):
        report = {
            "sharpe": None,
            "tvr": 0,
            "numdays": len(df_1d),
            "start_day": int(df_1d.index[0]),
            "end_day": int(df_1d.index[-1]),
        }
        tvr = round(df_1d["turnover"].mean(), 3)
        std = df_1d["netProfit"].std()
        # print(f"std: {std}, tvr: {tvr}")
        if tvr == 0 or std == 0:
            return report
        config_count = self.book_size
        equity = 300 * config_count
        net_profit = df_1d["cumNetProfit"]
        cummax = net_profit.cummax()
        cdd = cummax - net_profit

        df_1d["cdd"] = cdd.round(2)
        df_1d["cdd1"] = (cdd / (equity + cummax) * 100).round(2)
        df_1d["mdd"] = cdd.cummax()
        df_1d["mdd1"] = df_1d["cdd1"].cummax()
        df_1d["cumNetProfit1"] = net_profit + equity

        mdd_idx = cdd.idxmax()
        start_idx = cummax.loc[:mdd_idx].idxmax()
        recovery_idx = df_1d.loc[mdd_idx:].loc[net_profit >= cummax[mdd_idx]].first_valid_index()
        
        if pd.notna(start_idx):
            recovery_idx = recovery_idx if pd.notna(recovery_idx) else df_1d.index[-1]
            delta = datetime.strptime(str(recovery_idx), "%Y%m%d") - datetime.strptime(str(start_idx), "%Y%m%d")
            mdd_time = delta.days
        else:
            mdd_time = None
        ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
        df = self.df_alpha
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
        # trade_results.to_csv("/home/ubuntu/nevir/fenix/Backtest/src/api/blueprints/stock/trade.csv")
        trade_cum_profit = trade_results.cumsum()
        trade_cum_max = trade_cum_profit.cummax()
        trade_drawdown = trade_cum_max - trade_cum_profit
        mdd2 = ((trade_drawdown / (equity + trade_cum_max)) * 100).max()
        
        num_trades = trade_results.count()
        winrate = round((trade_results > 0).sum() / num_trades * 100, 2) if num_trades > 0 else None
        
        win_profits = df_1d[df_1d["netProfit"] > 0]["netProfit"]
        loss_profits = df_1d[df_1d["netProfit"] < 0]["netProfit"]
        net_profit_pos = win_profits.sum()
        net_profit_neg = loss_profits.sum()
        profit_percent = df_1d["netProfit"].sum() / equity*100
        npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
        working_day = calculate_working_days(self.data_start, self.data_end)
        maxdd = df_1d["mdd"].values[-1]
        df_1d['neg'] = df_1d['netProfit'] < 0

        # Tính số chuỗi liên tiếp bằng cách phân nhóm các đoạn không liên tục
        df_1d['grp'] = (df_1d['neg'] != df_1d['neg'].shift()).cumsum()

        # Lọc các nhóm có netProfit âm, và tìm độ dài lớn nhất
        max_streak = (
            df_1d[df_1d['neg']]
            .groupby('grp')
            .size()
            .max()
        )
        df_1d['posi'] = df_1d['netProfit'] > 0

        # Tính số chuỗi liên tiếp bằng cách phân nhóm các đoạn không liên tục
        df_1d['grp1'] = (df_1d['posi'] != df_1d['posi'].shift()).cumsum()

        # Lọc các nhóm có netProfit âm, và tìm độ dài lớn nhất
        max_gross = (
            df_1d[df_1d['posi']]
            .groupby('grp1')
            .size()
            .max()
        )
        returns = df_1d["netProfit"] / 300
        sharpe = df_1d["netProfit"].mean() / std * np.sqrt(working_day)
        daily_sharpe = df_1d["netProfit"].mean() / std 
        winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
        total_profit = winning_profits.sum()
        shares = winning_profits / total_profit
        hhi = (shares ** 2).sum()
        E = (1 + (trade_results[trade_results > 0].mean()/abs(trade_results[trade_results < 0].mean()))) * (winrate/100) - 1
        # print(trade_results[trade_results > 0].mean(),abs(trade_results[trade_results < 0].mean()),winrate)
        report.update({
            "sharpe": float(round(sharpe, 3)),
            "psr": float(round(self.psr(returns, daily_sharpe),2)),
            "hhi":float(round(hhi, 4)),
            "sortino": float(round(df_1d["netProfit"].mean() / df_1d[df_1d["netProfit"] < 0]["netProfit"].std() * np.sqrt(working_day), 3)),
            "tvr": float(tvr),
            "mdd_point": float(round(df_1d["mdd"].values[-1], 2)),
            "total_net_profit": float(round(df_1d["netProfit"].sum(), 2)),
            "mar": float(round(df_1d["netProfit"].sum() /  maxdd if maxdd !=0 else 1, 2)),
            "mdd_percent": float(round(df_1d["mdd1"].values[-1], 2)),
            "std_netprofit": float(round(std, 2)),
            "mean_netprofit": float(round(df_1d["netProfit"].mean(), 2)),
            "num_trades": int(num_trades),
            "winrate": float(winrate),
            "E":float(round(E,4)),
            # "skew": float(round(skew(self.df_1d["netProfit"], bias=False),3)),
            # "kurt": float(round(kurtosis(self.df_1d["netProfit"], bias=False),3)),
            "npf":float(round(npf, 2)),
            "ppc": float(round(ppc, 3)),
            "profit_percent": float(round(profit_percent, 2)),
            "num_cut_trades": self.num_cut_trades,
            "max_loss":df_1d['netProfit'].min(),
            "max_gross":df_1d['netProfit'].max(),
            "max_loss_day":max_streak,
            "max_win_day":max_gross,
            "mdd_time": mdd_time,
            "mdd_trade": float(round(mdd2, 2)),
        })
        return {**template, **report} if template else report 
    
    

    def psr(self,returns,sharpe):
        try:
            sample_size = len(returns)
            skewness = skew(returns)
            _kurtosis = kurtosis(returns, fisher=True, bias=False)
            sigma_sr = np.sqrt((1 - skewness*sharpe + (_kurtosis + 2)/4*sharpe**2) / (sample_size - 1))
            z = sharpe / sigma_sr
            return 0.5 * (1 + erf(z / sqrt(2))) * 100
        except Exception as e:
            return 0
    
    def extract_trades_df(self):
        df = self.df_alpha
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

        return pd.DataFrame(trades)

    def compute_report_walk(self, df_1d,start,end):
        report = {
            "sharpe": None,
            "tvr": 0,
            "numdays": len(df_1d),
            "start_day": int(df_1d.index[0]),
            "end_day": int(df_1d.index[-1]),
        }
        df_1d[["cumNetProfit"]] = df_1d[["netProfit"]].cumsum()
        if "cdd" not in df_1d:
            cummax = df_1d["cumNetProfit"].cummax()
            df_1d["cdd"] = (cummax - df_1d["cumNetProfit"]).round(2)
            df_1d["cdd1"] = ((300 + cummax - (300 + df_1d["cumNetProfit"])) / (300 + cummax) * 100).round(2)
            df_1d["mdd"] = df_1d["cdd"].cummax()
            df_1d["mdd1"] = df_1d["cdd1"].cummax()
            df_1d["cumNetProfit1"] = df_1d["cumNetProfit"] + 300

        tvr = round(df_1d["turnover"].mean(), 3)
        std = df_1d["netProfit"].std()
        ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
        if tvr == 0 or std == 0:
            return report
        gross_profit_pos = df_1d[df_1d["grossProfit"] > 0]["grossProfit"].sum()
        gross_profit_neg = df_1d[df_1d["grossProfit"] < 0]["grossProfit"].sum()
        net_profit_pos = df_1d[df_1d["netProfit"] > 0]["netProfit"].sum()
        net_profit_neg = df_1d[df_1d["netProfit"] < 0]["netProfit"].sum()
        profit_percent = df_1d["netProfit"].sum() / 300*100
        gpf = gross_profit_pos / abs(gross_profit_neg) if gross_profit_neg != 0 else gross_profit_pos
        npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
        working_day = calculate_working_days(start, end)
        report.update({
            "sharpe": float(round(df_1d["netProfit"].mean() / std * np.sqrt(working_day), 3)),
            "tvr": float(tvr),
            "mdd_point": float(round(df_1d["mdd"].values[-1], 2)),
            "total_net_profit": float(round(df_1d["netProfit"].sum(), 2)),
            "mdd_percent": float(round(df_1d["mdd1"].values[-1], 2)),
            "gpf":float(round(gpf, 2)),
            "npf":float(round(npf, 2)),
            "ppc": float(round(ppc, 3)),
            "profit_percent": float(round(profit_percent, 2)),
            "max_loss":df_1d['netProfit'].min(),
            "max_gross":df_1d['netProfit'].max(),
        })

        return report

        
