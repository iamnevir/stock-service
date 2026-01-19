from datetime import datetime
import multiprocessing
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from gen_spot.core import Simulator as SimulatorGen
def worker_compute_process(index, config, gen, base_name, fee, dic_freqs, df_tick, stop_loss, start, end, DIC_BASES, return_dict):

    # ====== PARSE CONFIG (logic giữ nguyên 100%) ======
    gen_params = {}
    params = {}
    parts = config.split("_")

    def parse_rest(rest, base_name, params):
        if len(rest) >= 1:
            params["window"] = int(rest[0])
        if len(rest) >= 2:
            if base_name == "base_003":
                params["window_rank"] = float(rest[1])
            else:
                params["factor"] = float(rest[1])
        if base_name == "base_005" and len(rest) >= 3:
            params["window_rank"] = int(rest[2])


    if gen == "1_3":
        freq, score, entry, exit, *rest = parts
        freq, score = int(freq), int(score)
        entry, exit = float(entry), float(exit)

        parse_rest(rest, base_name, params)

        gen_params = {
            "score": score,
            "entry": entry,
            "exit": exit
        }

    elif gen == "1_2":
        freq, upper, lower, *rest = parts
        freq = int(freq)
        upper, lower = float(upper), float(lower)

        parse_rest(rest, base_name, params)

        gen_params = {
            "upper": upper,
            "lower": lower
        }

    elif gen == "1_1":
        freq, threshold, *rest = parts
        freq = int(freq)
        threshold = float(threshold)

        parse_rest(rest, base_name, params)

        gen_params = {
            "threshold": threshold
        }


    
    # ====== RUN SIMULATORGEN ======
    bt = SimulatorGen(
        base_name=base_name,
        freq=freq,
        gen_params=gen_params,
        fee=fee,
        df_base=dic_freqs[freq].copy(),
        params=params,
        DIC_BASES=DIC_BASES,
        df_tick=None,
        gen=gen,
        start=start,
        end=end
    )

    bt.compute_signal()
    bt.compute_position()
    df = bt.df_base.copy()
    if 'datetime' not in df.columns:
        df['datetime'] = df['executionT']
    df = df.set_index("datetime")
    df = df[~df.index.duplicated(keep='first')]
    df = df[['position']]
    
    # ====== STOP LOSS ======
    if stop_loss > 0:
        df_ps = df_tick.copy()
        df_pos = pd.merge(df, df_ps, on='datetime', how='outer').sort_index()
        df_pos['position'] = df_pos['position'].ffill().fillna(0)
        df_pos.dropna(inplace=True)

        df_pos["priceChange"] = df_pos.groupby("day")["last"].diff().shift(-1).fillna(0)
        df_pos['grossProfit'] = df_pos['position'] * df_pos['priceChange']
        df_pos['action'] = df_pos['position'] - df_pos['position'].shift(1, fill_value=0)
        df_pos['turnover'] = df_pos['action'].abs()
        df_pos['fee'] = df_pos['turnover'] * fee
        df_pos['netProfit'] = df_pos['grossProfit'] - df_pos['fee']
        df_pos['pctChange'] = df_pos['netProfit'] / df_pos['last']

        # ⚠ self.hard_cut_loss không dùng được ở worker — tự bạn phải xử lý phần này
        # hoặc truyền một hàm standalone

        df2 = df_pos[['position']].rename(columns={'position': index})
    else:
        df_ps = dic_freqs[1]
        if 'datetime' not in df_ps.columns:
            df_ps['datetime'] = df_ps['executionT']
        df_ps = df_ps.set_index("datetime")
        df_pos = pd.merge(df_ps, df, on='datetime', how='left')
        df_pos['position'] = df_pos['position'].ffill().fillna(0)
        df2 = df_pos[['position']].rename(columns={'position': index})

    return_dict[index] = df2

class Simulator:
    def __init__(self, base_name,configs ,fee=0.1, dic_freqs=None,DIC_BASES=None,df_tick=None,start=None,end=None,stop_loss=0,gen='gen1_2',booksize=None,is_sizing=False,init_sizing=37.5):

        self.configs = configs
        self.fee = fee
        self.stop_loss = stop_loss
        self.dic_freqs = dic_freqs
        self.df_tick = df_tick.copy() if df_tick is not None else None
        self.start = start
        self.end = end
        if DIC_BASES:
            self.DIC_BASES = DIC_BASES
            self.base_func = DIC_BASES[base_name]
        self.base_name = base_name
        self.gen = gen
        self.booksize = booksize if booksize is not None else len(configs)
        self.n_bases = len(self.configs)
        self.is_sizing = is_sizing
        self.hard_dic_budget = pd.DataFrame({'status':  range(101)})
        self.hard_dic_budget['cum'] = init_sizing
        self.hard_dic_budget['cum'] =  self.hard_dic_budget['status'] *  self.hard_dic_budget['cum']
        self.hard_dic_budget.loc[0, 'cum'] = init_sizing
        self.hard_dic_budget['cum'] = self.hard_dic_budget['cum'].cumsum()
        self.hard_dic_budget['action'] = self.hard_dic_budget['status'] + 1

    def adjust_positions(self,df_base):
        flt_unexecutable = ~df_base['executable']
        df_base.loc[flt_unexecutable, 'position_init'] = np.nan
        flt_atc = df_base['executionTime'] == '14:45:00'
        df_base.loc[flt_atc, 'position_init'] = 0
        df_base['position_init'] = df_base['position_init'].ffill().fillna(0)
    
    def compute_mega(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []
        for idx, config in enumerate(self.configs):
            p = multiprocessing.Process(
                target=worker_compute_process,
                args=(
                    idx,
                    config,
                    self.gen,
                    self.base_name,
                    self.fee,
                    self.dic_freqs,
                    self.df_tick,
                    self.stop_loss,
                    self.start,
                    self.end,
                    self.DIC_BASES,
                    return_dict
                )
            )
            p.start()
            processes.append(p)
            while len(processes) >= 20:
                # Chờ 1 process hoàn thành rồi xóa khỏi danh sách
                for proc in processes:
                    if not proc.is_alive():
                        proc.join()
                        processes.remove(proc)
                        break

        for p in processes:
            p.join()

        # Ghép kết quả theo thứ tự
        lst_pos = [return_dict[i] for i in sorted(return_dict.keys())]
        
        df_all_pos = pd.concat(lst_pos, axis=1)
        df_all_pos = df_all_pos[~df_all_pos.index.duplicated(keep='first')]
        df_pos_sum = df_all_pos.sum(axis=1).rename('position').to_frame()
        if self.stop_loss > 0:
            df_base = self.df_tick.copy()
            df_base["priceChange"] = (
                df_base.groupby("day")["last"]
                .diff()
                .shift(-1)
                .fillna(0)
            )
            start = int("".join(start.split("_")))
            end = int("".join(end.split("_")))
        else:
            df_base = self.dic_freqs[1].copy()
            start = self.start
            end = self.end
        if self.is_sizing:
            df_base["position_init"] = df_base["executionT"].map(df_pos_sum["position"]) / self.n_bases 
          
            self.adjust_positions(df_base)
            df_base = self.sizing_positon(df_base)
        else:
            df_base["position"] = df_base["executionT"].map(df_pos_sum["position"])
            df_base["position"] = ((df_base['position'] / self.n_bases  * self.booksize).round(6).astype(int))
        df_base['grossProfit'] = df_base['position'] * df_base['priceChange']
        df_base['action'] = df_base['position'] - df_base['position'].shift(1, fill_value=0)
        df_base['turnover'] = df_base['action'].abs()
        df_base['fee'] = df_base['turnover'] * self.fee 
        df_base['netProfit'] = df_base['grossProfit'] - df_base['fee']
        df_base['cumNetProfit'] = df_base['netProfit'].cumsum()
        df_base = df_base[(df_base['day'] >= start) & (df_base['day'] <= end)]
        agg_dict = {
            "grossProfit": "sum",
            "turnover": "sum",
            "netProfit": "sum",
        }
        if "booksize" in df_base.columns:
            agg_dict["booksize"] = "last"
        df_1d = (df_base.groupby('day')
                 .agg(agg_dict))
        df_1d['cumNetProfit'] = df_1d['netProfit'].cumsum()
        df_1d['cumTurnover'] = df_1d['turnover'].cumsum()
        df_1d[['cumGrossProfit', 'cumTurnover', 'cumNetProfit']] = \
            df_1d[['grossProfit', 'turnover', 'netProfit']].cumsum()
        df_1d[['grossProfit', 'netProfit', 'cumNetProfit', 'cumGrossProfit']] = \
            df_1d[['grossProfit', 'netProfit', 'cumNetProfit', 'cumGrossProfit']].round(2)
        self.df_base = df_base
        self.df_1d = df_1d
    
    def get_budget(self,total_netProfit):
        df_filtered = self.hard_dic_budget[self.hard_dic_budget['cum'] <= total_netProfit]
        if df_filtered.empty:
            return 0
        else:
            return df_filtered['action'].iloc[-1]
        
    def sizing_positon(self,df_base):   
        df = df_base.copy()
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
            if day == "2025_01_02":
                print(df_day['position'].sum(),df_day['position_init'].sum(),booksize)
               
            df_base.loc[df_day.index, f"position"] = df_day['position']
            df_base.loc[df_day.index, f"booksize"] = booksize
        return df_base
    
    def hard_cut_loss(self,df):
        if self.stop_loss is None or self.stop_loss <= 0:
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
        cut_condition = df["trade_cum_profit"] < -self.stop_loss
        cut_first_idx = (
            df[cut_condition]
            .sort_index()
            .drop_duplicates("trade_id", keep="first")
        )
        positions = df.index.get_indexer(cut_first_idx.index)
        next_positions = positions + 1
        cut_first_idx = df.iloc[next_positions]
        # ====== BƯỚC 5: CẬP NHẬT SAU CẮT ======
        cut_start_map = dict(zip(cut_first_idx["trade_id"], cut_first_idx.index))
        trade_ids_array = df["trade_id"].values
        index_array = pd.to_datetime(df.index)
        cut_start_idx_array = np.array([
            cut_start_map.get(tid, pd.Timestamp.max) for tid in trade_ids_array
        ])

        cut_mask = pd.Series(index_array >= cut_start_idx_array, index=df.index)
        df.loc[cut_mask, "position"] = 0

        # Tính lại sau khi cut
        # df["action"] = df['position'].diff(1).fillna(df['position'].iloc[0])
        # df["turnover"] = df["action"].abs()
        # df["fee"] = df["turnover"] * fee
        # df["grossProfit"] = df["position"] * df["priceChange"]
        # df["netProfit"] = df["grossProfit"] - df["fee"]

        # Lưu dữ liệu SAU khi cut loss
        # df[df['day'] == '2024_01_04'].to_csv('/home/ubuntu/nevir/gen1_2/trades_cut.csv')
        return df
    
    def compute_performance(self):
 
        self.df_1d, report = Base_Domains.compute_performance(
            self.df_base,
            start=self.start,
            end=self.end,
            equity=300*self.booksize,
            df_1d=self.df_1d
        )
        self.report = report
    
    def extract_net_profits(self):
        return Base_Domains.extract_net_profits(self.df_base)
    def compute_df_trade(self):
        self.df_trade = Base_Domains.compute_df_trade(self.df_base)
class Base_Domains:
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
    def compute_performance(df_base, start=None, end=None,equity =300,df_1d=None):
        lst_errs = []
       
        working_day = Base_Domains.calculate_working_days(start, end)
        try:
            mean = df_1d["netProfit"].mean()
            std = df_1d["netProfit"].std()
            if std and not np.isnan(std):
                sharpe = mean / std * working_day ** 0.5
                daily_sharpe = mean / std
            else:
                sharpe = np.nan
                daily_sharpe = np.nan
        except Exception as e:
            lst_errs.append(f"{e}")
            # U.report_error(e)
            sharpe = -999
        tvr = df_1d['turnover'].mean()
        ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)

        mdd, mdd_pct, cdd, cdd_pct = Base_Domains.compute_mdd_vectorized(df_1d,equity)
        df = df_base
        # print(df[['day','position','grossProfit','fee','netProfit']])
        df["position_prev"] = df["position"].shift(fill_value=0)
        df["position_next"] = df["position"].shift(-1, fill_value=0)

        # trade_start: khi từ 0 sang có vị thế hoặc đổi chiều
        df["trade_start"] = (((df["position_prev"] == 0) & (df["position"] > 0)) |
                            (df["position_prev"] * df["position"] < 0)).astype(int)

        # trade_end: khi position hiện tại khác position kế tiếp (về 0 hoặc đảo chiều)
        df["trade_end"] = (
            (df["position"] != 0) &
            ((df["position_next"] < 0) | (df["position_next"] > 0))
        ).astype(int)



        # Gán trade_id và loại bỏ các đoạn không có vị thế
        df["trade_id"] = df["trade_start"].cumsum()
        df.loc[df["position"] == 0, "trade_id"] = None
        df["trade_id"] = df["trade_id"].ffill()

        trade_results = df.groupby("trade_id")["netProfit"].sum()
        
        num_trades = trade_results.count()
        # print(num_trades)
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
        E = (1 + (trade_results[trade_results > 0].mean()/abs(trade_results[trade_results < 0].mean()))) * (winrate/100) - 1
        new_report = {
            'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 3),
            'mddPct': round(mdd_pct.iloc[-1], 4),
            "hhi": round(hhi,3),
            "psr": round(Base_Domains.dsr(returns, daily_sharpe,0),3),
            # 'cdd': round(cdd, 3),
            # 'cddPct': round(cdd_pct.iloc[-1], 4),
            'ppc': round(ppc, 4),
            "E":round(E,2),
            'tvr': round(tvr, 4),
            # 'start': df_1d.index[0],
            # 'end': df_1d.index[-1],
            "npf":float(round(npf, 2)),
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
        # print(df_base[df_base['day'] == '2025_09_05'][['executionT','position','priceChange','grossProfit','fee','netProfit',"signal"]])
        return df_1d, new_report

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

    def compute_df_trade(df_base):
        """Lưu time_int và turnover của các giao dịch"""
        df_trade = df_base[df_base["action"] != 0][["action","executionT"]]
        return df_trade



    

 




