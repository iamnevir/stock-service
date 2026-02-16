from datetime import datetime
from math import erf, sqrt
import sys
from time import time
from itertools import combinations
from functools import partial
from multiprocessing import Pool
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

from pymongo import MongoClient, InsertOne
from bson import ObjectId

from auto.utils import get_mongo_uri, load_dic_freqs, sanitize_for_bson
from gen.alpha_func_lib import Domains
from gen.core_mega import Simulator
from typing import Union, List, Tuple


def calculate_working_days(
    cv: Union[
        str,
        List[List[str]],
        Tuple[str, ...],
        List[str]
    ],
    workdays_per_year: int = 250
) -> int:
    """
    Tính số ngày làm việc từ CV periods.

    Supported formats:
    - "YYYY_MM_DD-YYYY_MM_DD"
    - ["YYYY_MM_DD-YYYY_MM_DD", ...]
    - [["YYYY_MM_DD", "YYYY_MM_DD"], ...]   ← WFA CPCV
    """

    def parse_period(start: str, end: str) -> int:
        s = datetime.strptime(start, "%Y_%m_%d")
        e = datetime.strptime(end, "%Y_%m_%d")
        return (e - s).days

    total_days = 0

    # --- CASE 1: single string ---
    if isinstance(cv, str):
        if "-" not in cv:
            raise ValueError(f"Invalid cv string: {cv}")
        start, end = cv.split("-")
        total_days = parse_period(start, end)

    # --- CASE 2: list / tuple ---
    elif isinstance(cv, (list, tuple)):
        for p in cv:

            # ["start", "end"]
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

    return mdd, mdd_pct, cdd_last, cdd_pct
    
def compute_performance_with_df1d(cv,equity =300,df_1d=None):
    lst_errs = []
    
    working_day = calculate_working_days(cv)
    try:
        mean = df_1d["netProfit"].mean()
        std = df_1d["netProfit"].std()
        if std and not np.isnan(std):
            sharpe = mean / std * working_day ** 0.5
            print(working_day, mean, std, sharpe, equity)
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

    mdd, mdd_pct, cdd, cdd_pct = compute_mdd_vectorized(df_1d,equity)
    
    returns = df_1d['netProfit'] / equity
    winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
    loss_profits = df_1d[df_1d['netProfit'] < 0]['netProfit']
    net_profit_pos = winning_profits.sum()
    net_profit_neg = loss_profits.sum()
    npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
    total_profit = winning_profits.sum()
    shares = winning_profits / total_profit
    hhi = (shares ** 2).sum()
    new_report = {
        'sharpe': round(sharpe, 2),
        'mdd': round(mdd, 2),
        'mddPct': round(mdd_pct.iloc[-1], 2),
        "hhi": round(hhi,2),
        "psr": round(dsr(returns, daily_sharpe,0),2),
        'ppc': round(ppc, 2),
        'tvr': round(tvr, 2),
        "npf":float(round(npf, 2)),
        "netProfit": round(df_1d['netProfit'].sum(), 2),
        "profitPct": round(df_1d['netProfit'].sum(), 2) / equity * 100,
        "max_loss": round(df_1d['netProfit'].min(), 2),
        "max_gross": round(df_1d['netProfit'].max(), 2),
    }
    return new_report

def parse_date_safe(x):
    if not isinstance(x, str):
        return x
    for fmt in ("%Y%m%d", "%Y_%m_%d"):
        try:
            return pd.to_datetime(x, format=fmt)
        except ValueError:
            pass
    raise ValueError(f"Unknown date format: {x}")

def split_os_to_chunks(os_item):
    # ---------- helpers ----------
    def parse_date(x):
        if isinstance(x, int):
            return pd.to_datetime(str(x), format="%Y%m%d")
        elif isinstance(x, str):
            if "_" in x:
                return pd.to_datetime(x, format="%Y_%m_%d")
            else:
                return pd.to_datetime(x)
        else:
            return pd.to_datetime(x)  # ⚠️ KHÔNG return x thẳng

    def format_date(ts, mode):
        if mode == "int":
            return int(ts.strftime("%Y%m%d"))
        elif mode == "str":
            return ts.strftime("%Y_%m_%d")

    # ---------- detect mode ----------
    if isinstance(os_item["start"], int):
        mode = "int"
    elif isinstance(os_item["start"], str) and "_" in os_item["start"]:
        mode = "str"
    else:
        mode = "auto"

    # ---------- parse input ----------
    start_dt = parse_date(os_item["start"])
    end_dt   = parse_date(os_item["end"])

    df = os_item["df"].copy()

    # ---------- parse index (PHẢI ra DatetimeIndex) ----------
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index.map(parse_date))

    df = df.sort_index()

    # ---------- cut points ----------
    cut_points = [
        start_dt + pd.DateOffset(months=2),
        start_dt + pd.DateOffset(months=4),
        end_dt
    ]

    # ---------- split ----------
    chunks = []
    prev = start_dt
    chunk_id = 1

    for cut in cut_points:
        is_last = (cut == end_dt)

        mask = (
            (df.index >= prev) &
            ((df.index <= cut) if is_last else (df.index < cut))
        )

        df_chunk = df.loc[mask]

        if df_chunk.empty:
            prev = cut
            continue

        df_chunk = df_chunk.copy()

        # convert index back
        if mode == "int":
            df_chunk.index = df_chunk.index.strftime("%Y%m%d").astype(int)
        else:
            df_chunk.index = df_chunk.index.strftime("%Y_%m_%d")

        chunks.append({
            "df": df_chunk,
            "equity": os_item["equity"],
            "start": format_date(prev, mode),
            "end": format_date(cut, mode),
            "chunk_id": chunk_id
        })

        chunk_id += 1
        prev = cut

    return chunks

def precompute_wfa_os(
    alpha_name,
    gen,
    dic_freqs,
    DIC_ALPHAS,
    df_tick,
    wfa_list,
    source
):
    PRE = {}

    for i, fa in enumerate(wfa_list):
        os_cfg = fa["os"]
        fee = fa.get("fee", 0.175)
        strategies = (
            fa.get("correlation", {})
              .get("results", {})
              .get("strategies", [])
        )

        if not strategies:
            continue

        bt = Simulator(
            alpha_name=alpha_name,
            configs=strategies,
            dic_freqs=dic_freqs,
            DIC_ALPHAS=DIC_ALPHAS,
            df_tick=df_tick,
            start=os_cfg["start"],
            end=os_cfg["end"],
            fee=fee,
            stop_loss=fa["stop_loss"],
            gen=gen,
            booksize=fa["book_size"],
            is_sizing=fa["is_sizing"],
            init_sizing=fa["init_sizing"],
            source=source
        )

        bt.compute_mega()

        df = bt.df_1d.copy()

        key = f"os_{i}"
        PRE[key] = {
            "df": df,
            "equity": 300 * fa["book_size"],
            "start": os_cfg["start"],
            "end": os_cfg["end"]
        }

        print(f"✅ OS {key}: {os_cfg['start']} → {os_cfg['end']} | rows={len(df)}")

    return PRE

def get_df_1d_from_cv(cv_keys, PRE):
    dfs = []
    cv_dates = []
    equity = 0
    for k in cv_keys:
        item = PRE[k]
        dfs.append(item["df"])
        equity = item["equity"]
        cv_dates.append([item["start"], item["end"]])

    df = pd.concat(dfs)
    df["cumNetProfit"] = df["netProfit"].cumsum()

    return df, equity, cv_dates


def run_cpcv_sequential(cv_list, PRE, alpha_id):
    reports = []

    for i, cv_keys in enumerate(cv_list, 1):
        df, equity, cv_dates = get_df_1d_from_cv(cv_keys, PRE)

        report = compute_performance_with_df1d(
            cv=cv_dates,        # ✅ list[list[start, end]]
            equity=equity,
            df_1d=df
        )

        reports.append({
            "cv": cv_dates,     # ✅ lưu dạng date
            "alpha_id": alpha_id,
            **sanitize_for_bson(report)
        })

        if i % 100 == 0:
            print(f"  ▶ processed {i}/{len(cv_list)} CV")

    return reports

def cpcv(alpha_id):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    cpcv_collection = alpha_db["alpha_cpcv"]

    doc = alpha_collection.find_one({"_id": ObjectId(alpha_id)})
    if not doc:
        print("❌ Alpha not found")
        return
    alpha_collection.update_one(
        {"_id": ObjectId(alpha_id)},
        {"$set": {
            "cpcv.status": "running",
        }}
    )
    alpha_name = doc["alpha_name"]
    gen = doc.get("gen", "1_2")
    overnight = doc.get("overnight",False)
    source = doc.get("source",None)
    wfa_list = doc.get("wfa", [])
    if not wfa_list:
        print("❌ No WFA data")
        return
        
    gen_path = 3

    print(f"🔍 CPCV-WFA (SEQUENTIAL)")
    print(f"   alpha={alpha_name} | gen_path={gen_path}")

    dic_freqs = load_dic_freqs(source, overnight)
    DIC_ALPHAS = Domains.get_list_of_alphas()
    df_tick = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")

    start_time = time()

    # --- PRECOMPUTE OS ---
    PRE_RAW = precompute_wfa_os(
        alpha_name=alpha_name,
        gen=gen,
        dic_freqs=dic_freqs,
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=df_tick,
        wfa_list=wfa_list,
        source=source
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
    reports = run_cpcv_sequential(cv_list, PRE, alpha_id)

    print(f"⏱️ CPCV finished in {time() - start_time:.2f}s")

    # --- SAVE MONGO ---
    cpcv_collection.delete_many({"alpha_id": alpha_id})
    cpcv_collection.create_index("alpha_id")
    ops = []
    for r in reports:
        ops.append(InsertOne(r))
        if len(ops) == 500:
            cpcv_collection.bulk_write(ops, ordered=False)
            ops.clear()
    if ops:
        cpcv_collection.bulk_write(ops, ordered=False)

    # --- STATISTICS ---
    profit = np.sort( [r["profitPct"] for r in reports])
    mdd =  np.sort([r["mddPct"] for r in reports])
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

    alpha_collection.update_one(
        {"_id": ObjectId(alpha_id)},
        {"$set": {
            "cpcv.path_count": len(cv_list),
            "cpcv.statistics": statistics,
        }}
    )
    alpha_collection.update_one(
        {"_id": ObjectId(alpha_id)},
        {"$set": {
            "cpcv.status": "done",
        }}
    )
    print("✅ CPCV-WFA (sequential) saved")


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfa_cpcv.py <_id>")
#         sys.exit(1)

#     _id = sys.argv[1]

#     cpcv(_id)

# if __name__ == "__main__":
#     main()