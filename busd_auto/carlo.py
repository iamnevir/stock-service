
import sys
from time import time
from backend.api.blueprints.stock.MegaBbAccV2 import MegaBbAccV2
from busd_auto.utils import load_data
import numpy as np
import pandas as pd

from pymongo import MongoClient
from bson import ObjectId
from busd_auto.utils import get_mongo_uri, sanitize_for_bson

def generate_combinations(profits, sample=10000, day=125):
    indices = np.random.choice(
        len(profits),
        size=(sample, day),
        replace=True,
    )
    return profits[indices]
# def generate_combinations(profits, sample=10000):
#     profits = np.array(profits)
#     return np.array([np.random.permutation(profits) for _ in range(sample)])

def calculate_equity(combinations, cap):
    num_combinations, num_steps = combinations.shape

    equity = np.zeros((num_combinations, num_steps))
    equity[:, 0] = cap + combinations[:, 0]

    for j in range(1, num_steps):
        equity[:, j] = equity[:, j - 1] + combinations[:, j]

    return equity

def calculate_sharpe(mat, day=125):
    """
    returns: sharpe matrix shape (sample,)
    """
    mean = mat.mean(axis=1)
    std = mat.std(axis=1, ddof=1)  # dùng sample std cho đúng Sharpe

    sharpe = mean / std * np.sqrt(day)

    return sharpe

def calculate_drawdown_and_mdd(equity):
    peak_equity = np.maximum.accumulate(equity, axis=1)
    peak_equity = np.nan_to_num(peak_equity, nan=0.0, posinf=0.0, neginf=0.0)
    equity = np.nan_to_num(equity, nan=0.0)
    drawdown = np.where(np.abs(peak_equity) > 1e-10, (peak_equity - equity) / peak_equity, 0)
    mdd = drawdown.max(axis=1) * 100
    return mdd


def calculate_gain(equity, cap, gain):
    final_equity = equity[:, -1]
    gain_count = np.sum(final_equity >= cap * (gain / 100))
    return gain_count


def calculate_ranges(values, ranges, sample):
    return [np.sum(values >= threshold) / sample * 100 for threshold in ranges]


def calculate_ruin_percentages(equity, cap, ruin_ranges):

    # Tạo ra một ma trận với các giá trị ngưỡng ruin
    ruin_thresholds = np.array([cap * (1 - ruin / 100) for ruin in ruin_ranges])

    # Mở rộng chiều của ruin_thresholds để khớp với equity
    ruin_thresholds_expanded = ruin_thresholds[:, np.newaxis, np.newaxis]

    # Kiểm tra tất cả các ngưỡng ruin cùng một lúc
    ruin_matrix = equity < ruin_thresholds_expanded

    # Tính số lượng tổ hợp bị ruin cho từng ngưỡng
    ruin_count = np.any(ruin_matrix, axis=2).sum(axis=1)

    # Tính phần trăm ruin cho từng ngưỡng
    ruin_percentages = (ruin_count / equity.shape[0]) * 100

    return ruin_percentages

def calculate_ruin_median(equity, cap):
    min_equity_per_path = np.min(equity, axis=1)

    ruined_equities = min_equity_per_path[min_equity_per_path < cap]

    ruin_percentages = ((cap - ruined_equities) / cap) * 100

    if len(ruin_percentages) == 0:
        return np.nan

    ruin_median = np.median(ruin_percentages)

    mgr_range = [75,95,99]
    percentiles_mgr = np.percentile(ruin_percentages, mgr_range)
    
    return ruin_median, percentiles_mgr,mgr_range

def precompute_wfa_os(
    source: str,
    wfa_list
):
    
    net_profit_list = []
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

        bt = MegaBbAccV2(
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
        net_profit_list.extend(df["netProfit"].values.tolist())

    return net_profit_list

def calculate_metrics(data,cap=300, day=125, sample=10000):
    profits = np.array(data)
    # profit_percent = profits / 300 
    # risk_init = 0.001
    
    # np.savetxt("/home/ubuntu/nevir/fenix/Backtest/src/api/blueprints/stock/profits.csv", profits, delimiter=",", fmt="%.2f")
    combinations = generate_combinations(profits=profits,day=day)
    
    equity = calculate_equity(combinations, cap)
    sharpe = calculate_sharpe(combinations, day=day)
    mean_sharpe = np.mean(sharpe)
    ruin_ranges = list(range(10, 45, 5)) 
    ruin_percentages = calculate_ruin_percentages(
        equity, cap, ruin_ranges
    )

    mdd = calculate_drawdown_and_mdd(equity)
    
    gains = (equity[:, -1] / cap - 1) * 100
    
    percentile_range = [25, 5, 1]
    percentiles = np.percentile(gains, percentile_range)
    
    mdd_ranges = list(range(10, 45, 5))
    mdd_percentages = calculate_ranges(mdd, mdd_ranges, sample)
    
    range_gain = [100 - i for i in percentile_range]
    mgr,percentiles_mgr, mgr_range = calculate_ruin_median(equity,cap)
    
    mgd_range = [75,95,99]
    percentiles_mgd = np.percentile(mdd, mgd_range)
    
    final =  {
        **{
            f"{threshold}_MDD": round(float(mdd_percentage), 2)
            for threshold, mdd_percentage in zip(mdd_ranges, mdd_percentages)
        },
        **{
            f"{threshold}_Ruin": round(float(ruin_percentage), 2)
            for threshold, ruin_percentage in zip(ruin_ranges, ruin_percentages)
        },
        **{
            f"{threshold}MG": round(float(percentile), 2)
            for threshold, percentile in zip(range_gain, percentiles)
        },
        **{
            f"{threshold}MGR": round(float(percentile), 2)
            for threshold, percentile in zip(mgr_range, percentiles_mgr)
        },
        **{
            f"{threshold}MGD": round(float(percentile), 2)
            for threshold, percentile in zip(mgd_range, percentiles_mgd)
        },
        "MG": round(np.median(gains), 2),
        "MGR": round(mgr, 2),
        "MGD": round(np.median(mdd), 2),
        "mean_sharpe": round(float(mean_sharpe), 2),
        "max_mdd": round(np.max(mdd), 2),
        "max_gain": round(np.max(gains), 2),
        "mean_gain": round(np.mean(gains), 2),
        "mean_mdd": round(np.mean(mdd), 2),
    }
    return sanitize_for_bson(final)

def carlo(busd_id):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]

    doc = busd_collection.find_one({"_id": ObjectId(busd_id)})
    if not doc:
        print("❌ BUSD not found")
        return
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "carlo.status": "running",
        }}
    )
    source = doc.get("source",None)
    wfa_list = doc.get("wfa", [])
    if not wfa_list:
        print("❌ No WFA data")
        return

    print(f"🔍 CARLO-WFA (SEQUENTIAL)")

    start_time = time()
    # --- PRECOMPUTE OS ---
    net_profit_list = precompute_wfa_os(
        source=source,
        wfa_list=wfa_list,
    )
    print(f"⏱️  Time gen: {time() - start_time:.2f} seconds")
    print(f"🧩 Precomputed {len(net_profit_list)} net profit entries")
    # print(net_profit_list[0])
    result = calculate_metrics(net_profit_list,cap=50*300,day=125)
    # print(result)
    print(f"⏱️  Time taken: {time() - start_time:.2f} seconds")
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "carlo.statistics": result,
        }}
    )
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {
            "carlo.status": "done",
        }}
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/carlo.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    carlo(_id)

if __name__ == "__main__":
    main()