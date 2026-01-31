from datetime import datetime
from math import erf, sqrt
import sys
from time import time
from itertools import combinations
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

from pymongo import MongoClient, InsertOne
from bson import ObjectId
from busd_auto.MegaBbAccV2 import MegaBbAccV2 as Simulator
from busd_auto.utils import get_mongo_uri, load_data, sanitize_for_bson
from typing import Union, List, Tuple


def calculate_working_days(
    cv: Union[
        str,
        int,
        List[Union[str, int]],
        Tuple[Union[str, int], ...],
        List[List[Union[str, int]]]
    ],
    workdays_per_year: int = 250
) -> int:
    """
    Tính số ngày làm việc từ CV periods.

    Supported formats:
    - "YYYY_MM_DD-YYYY_MM_DD"
    - 20240101-20250101 (int)
    - ["YYYY_MM_DD-YYYY_MM_DD", ...]
    - [20240101, 20250101]
    - [["YYYY_MM_DD", "YYYY_MM_DD"], ...]
    - [[20240101, 20250101], ...]
    """

    def parse_date(d: Union[str, int]) -> datetime:
        if isinstance(d, int):
            return datetime.strptime(str(d), "%Y%m%d")
        elif isinstance(d, str):
            if "_" in d:
                return datetime.strptime(d, "%Y_%m_%d")
            elif d.isdigit():
                return datetime.strptime(d, "%Y%m%d")
        raise ValueError(f"Invalid date format: {d}")

    def parse_period(start, end) -> int:
        s = parse_date(start)
        e = parse_date(end)
        return (e - s).days

    total_days = 0

    # --- CASE 1: single string or int range ---
    if isinstance(cv, str) and "-" in cv:
        start, end = cv.split("-")
        total_days = parse_period(start, end)

    # --- CASE 2: list / tuple ---
    elif isinstance(cv, (list, tuple)):
        for p in cv:

            # [start, end]
            if isinstance(p, (list, tuple)) and len(p) == 2:
                total_days += parse_period(p[0], p[1])

            # "start-end"
            elif isinstance(p, str) and "-" in p:
                start, end = p.split("-")
                total_days += parse_period(start, end)

            else:
                raise ValueError(f"Invalid CV period format: {p}")

    else:
        raise ValueError(f"Unsupported cv type: {type(cv)}")

    working_days = total_days * workdays_per_year / 365.0
    return int(round(working_days))

def psr(returns,sharpe):
        try:
            sample_size = len(returns)
            skewness = skew(returns)
            _kurtosis = kurtosis(returns, fisher=True, bias=False)
            sigma_sr = np.sqrt((1 - skewness*sharpe + (_kurtosis + 2)/4*sharpe**2) / (sample_size - 1))
            z = sharpe / sigma_sr
            return 0.5 * (1 + erf(z / sqrt(2))) * 100
        except Exception as e:
            return 0

def compute_performance_with_df1d(df_1d,cv_dates,equity):
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
        net_profit = df_1d["cumNetProfit"]
        cummax = net_profit.cummax()
        cdd = cummax - net_profit

        df_1d["cdd"] = cdd.round(2)
        df_1d["cdd1"] = (cdd / (equity + cummax) * 100).round(2)
        df_1d["mdd"] = cdd.cummax()
        df_1d["mdd1"] = df_1d["cdd1"].cummax()
        df_1d["cumNetProfit1"] = net_profit + equity

        
        ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
        win_profits = df_1d[df_1d["netProfit"] > 0]["netProfit"]
        loss_profits = df_1d[df_1d["netProfit"] < 0]["netProfit"]
        net_profit_pos = win_profits.sum()
        net_profit_neg = loss_profits.sum()
        profit_percent = df_1d["netProfit"].sum() / equity*100
        npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
        working_day = calculate_working_days(cv_dates)
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
      
        report.update({
            "sharpe": float(round(sharpe, 3)),
            "psr": float(round(psr(returns, daily_sharpe),2)),
            "hhi":float(round(hhi, 4)),
            "sortino": float(round(df_1d["netProfit"].mean() / df_1d[df_1d["netProfit"] < 0]["netProfit"].std() * np.sqrt(working_day), 3)),
            "tvr": float(tvr),
            "mdd_point": float(round(df_1d["mdd"].values[-1], 2)),
            "total_net_profit": float(round(df_1d["netProfit"].sum(), 2)),
            "mar": float(round(df_1d["netProfit"].sum() /  maxdd if maxdd !=0 else 1, 2)),
            "mdd_percent": float(round(df_1d["mdd1"].values[-1], 2)),
            "std_netprofit": float(round(std, 2)),
            "mean_netprofit": float(round(df_1d["netProfit"].mean(), 2)),
            # "skew": float(round(skew(self.df_1d["netProfit"], bias=False),3)),
            # "kurt": float(round(kurtosis(self.df_1d["netProfit"], bias=False),3)),
            "npf":float(round(npf, 2)),
            "ppc": float(round(ppc, 3)),
            "profit_percent": float(round(profit_percent, 2)),
            "max_loss":df_1d['netProfit'].min(),
            "max_gross":df_1d['netProfit'].max(),
            "max_loss_day":max_streak,
            "max_win_day":max_gross,
            "working_days": working_day,
        })
        return report

def precompute_wfa_os(
    source: str,
    wfa_list
):
    PRE = {}

    for i, fa in enumerate(wfa_list):
        os_cfg = fa["os"]
        data_start = os_cfg["start"]
        data_end = os_cfg["end"]
        df_alpha = load_data(source=source, start=data_start, end=data_end)
        strategies = (
            fa.get("correlation", {})
              .get("results", {})
              .get("strategies", [])
        )

        if not strategies or df_alpha is None or len(df_alpha) == 0:
            continue

        bt = Simulator(
            configs=strategies,
            df_alpha=df_alpha,
            data_start=data_start,
            data_end=data_end,
            fee=175,
            delay=1,
            stop_loss=fa['stop_loss'],
            book_size=fa["book_size"],
            is_sizing=fa['is_sizing'],
            init_sizing=fa['init_sizing']
        )
        bt.compute_mas()
        bt.compute_all_position()
        bt.compute_mega_position()
        bt.compute_profit_and_df_1d()
        df = bt.df_1d.copy()
        key = f"os_{i}"
        PRE[key] = {
            "df": df,
            "equity": 300 * fa["book_size"],
            "start": data_start,
            "end": data_end
        }

        print(f"✅ OS {key}: {data_start} → {data_end} | rows={len(df)}")

    return PRE

def get_df_1d_from_cv(cv_keys, PRE):
    dfs = []
    equity = 0
    cv_dates = []

    for k in cv_keys:
        item = PRE[k]
        dfs.append(item["df"])
        equity = item["equity"]

        cv_dates.append([item["start"], item["end"]])

    df = pd.concat(dfs)
    df["cumNetProfit"] = df["netProfit"].cumsum()

    return df, equity, cv_dates

def split_os_to_chunks(os_item):
    def int_to_ts(x):
        return pd.to_datetime(str(x), format="%Y%m%d")

    def ts_to_int(ts):
        return int(ts.strftime("%Y%m%d"))

    start_dt = int_to_ts(os_item["start"]) if isinstance(os_item["start"], int) else pd.to_datetime(os_item["start"])
    end_dt   = int_to_ts(os_item["end"])   if isinstance(os_item["end"], int)   else pd.to_datetime(os_item["end"])

    df = os_item["df"].copy()

    # ---- convert index to Timestamp for processing ----
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")

    df = df.sort_index()

    cut_points = [
        start_dt + pd.DateOffset(months=1),
        start_dt + pd.DateOffset(months=2),
        end_dt
    ]

    chunks = []
    prev = start_dt
    chunk_id = 1

    for cut in cut_points:
        is_last = cut == end_dt

        mask = (
            (df.index >= prev) &
            ((df.index <= cut) if is_last else (df.index < cut))
        )

        df_chunk = df.loc[mask]

        if df_chunk.empty:
            prev = cut
            continue

        # ---- convert index back to int YYYYMMDD ----
        df_chunk = df_chunk.copy()
        df_chunk.index = df_chunk.index.strftime("%Y%m%d").astype(int)

        chunks.append({
            "df": df_chunk,
            "equity": os_item["equity"],
            "start": ts_to_int(prev),
            "end": ts_to_int(cut),
            "chunk_id": chunk_id
        })

        chunk_id += 1
        prev = cut

    return chunks

def run_cpcv_sequential(cv_list, PRE, busd_id):
    reports = []

    for i, cv_keys in enumerate(cv_list, 1):
        df, equity, cv_dates = get_df_1d_from_cv(cv_keys, PRE)
        report = compute_performance_with_df1d(
            cv_dates=cv_dates,        
            equity=equity,
            df_1d=df
        )
        print(equity, report['sharpe'], report['mdd_percent'], report['profit_percent'], report['working_days'])
        reports.append({
            "cv": cv_dates,     # ✅ lưu dạng date
            "busd_id": busd_id,
            **sanitize_for_bson(report)
        })

        if i % 100 == 0:
            print(f"  ▶ processed {i}/{len(cv_list)} CV")

    return reports

def cpcv(busd_id):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    cpcv_collection = busd_db["busd_cpcv"]

    doc = busd_collection.find_one({"_id": ObjectId(busd_id)})
    if not doc:
        print("❌ busd not found")
        return
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "cpcv.status": "running",
        }}
    )
    busd_name = doc["name"]
    source = doc.get("source")

    wfa_list = doc.get("wfa", [])
    if not wfa_list:
        print("❌ No WFA data")
        return
        
    gen_path = 3

    print(f"🔍 CPCV-WFA (SEQUENTIAL)")
    print(f"   busd={busd_name} | source={source}")

    

    start_time = time()
    
    # --- PRECOMPUTE OS ---
    PRE_RAW = precompute_wfa_os(
        source=source,
        wfa_list=wfa_list
    )

    PRE = {}
    for os_key, os_item in PRE_RAW.items():
        chunks = split_os_to_chunks(os_item)
        for j, chunk in enumerate(chunks, 1):
            PRE[f"{os_key}_p{j}"] = chunk

    os_keys = list(PRE.keys())
    print(f"🧩 total chunks = {len(os_keys)}")
    cv_list = list(combinations(os_keys, gen_path))

    print(f"🧩 OS count={len(os_keys)} | CPCV paths={len(cv_list)}")

    # --- RUN CPCV (NO POOL) ---
    reports = run_cpcv_sequential(cv_list, PRE, busd_id)

    print(f"⏱️ CPCV finished in {time() - start_time:.2f}s")

    # --- SAVE MONGO ---
    cpcv_collection.delete_many({"busd_id": busd_id})
    cpcv_collection.create_index("busd_id")
    ops = []
    for r in reports:
        ops.append(InsertOne(r))
        if len(ops) == 500:
            cpcv_collection.bulk_write(ops, ordered=False)
            ops.clear()
    if ops:
        cpcv_collection.bulk_write(ops, ordered=False)

    # --- STATISTICS ---
    profit = np.sort( [r["profit_percent"] for r in reports])
    mdd =  np.sort([r["mdd_percent"] for r in reports])
    sharpe = [r["sharpe"] for r in reports]
    statistics = {
        "profitPct": {
            "mean": round(np.mean(profit), 2),
            "max": round(np.max(profit), 2),
            "min": round(np.min(profit), 2),
            "75pct": round(np.percentile(profit, 25), 2),
            "95pct": round(np.percentile(profit, 5), 2),
            "99pct": round(np.percentile(profit, 1), 2),
        },
        "mddPct": {
            "mean": round(np.mean(mdd), 2),
            "max": round(np.max(mdd), 2),
            "min": round(np.min(mdd), 2),
            "75pct": round(np.percentile(mdd, 75), 2),
            "95pct": round(np.percentile(mdd, 95), 2),
            "99pct": round(np.percentile(mdd, 99), 2),
        },
        "sharpe": {
            "mean": round(np.mean(sharpe), 2),
        }
    }

    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "cpcv.path_count": len(cv_list),
            "cpcv.statistics": statistics,
        }}
    )
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "cpcv.status": "done",
        }}
    )
    print("✅ CPCV-WFA (sequential) saved")


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/wfa_cpcv.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    cpcv(_id)

if __name__ == "__main__":
    main()