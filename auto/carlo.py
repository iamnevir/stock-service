import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient

from bson import ObjectId
from auto.utils import get_mongo_uri, load_dic_freqs, setup_logger
from gen.alpha_func_lib import Domains
from gen.core import Simulator
from gen.core_mega import Simulator as SimulatorMega

def generate_combinations(profits, sample=10000):
    indices = np.random.choice(
        len(profits),
        size=(sample, len(profits)),
        replace=True,
    )
    return profits[indices]
# def generate_combinations(profits, sample=10000):
#     profits = np.array(profits)
#     return np.array([np.random.permutation(profits) for _ in range(sample)])


def generate_combinations_month(profits, sample=10000):
    month_size = len(profits) // 12
    indices = np.random.choice(
        len(profits),
        size=(sample, month_size),
        replace=True,
    )
    return profits[indices]

def generate_combinations_week(profits, sample=10000):
    week_size = len(profits) // 52
    indices = np.random.choice(
        len(profits),
        size=(sample, week_size),
        replace=True,
    )
    return profits[indices]

def calculate_equity(combinations, cap):
    num_combinations, num_steps = combinations.shape

    equity = np.zeros((num_combinations, num_steps))
    equity[:, 0] = cap + combinations[:, 0]

    for j in range(1, num_steps):
        equity[:, j] = equity[:, j - 1] + combinations[:, j]

    return equity
# def calculate_equity(combinations, cap):
#     profit_percent = combinations
#     num_combinations, num_steps = profit_percent.shape

#     equity = np.zeros((num_combinations, num_steps))

#     equity[:, 0] = cap * (1 + profit_percent[:, 0])

#     for j in range(1, num_steps):
#         equity[:, j] = equity[:, j - 1]  + profit_percent[:, j]*300

#     return equity

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

def calculate_metrics(data,cap=300, sample=10000):
    
    profits = np.array(data)
    combinations = generate_combinations(profits)
    
    equity = calculate_equity(combinations, cap)

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
            f"{threshold}_MDD": float(mdd_percentage)
            for threshold, mdd_percentage in zip(mdd_ranges, mdd_percentages)
        },
        **{
            f"{threshold}_Ruin": float(ruin_percentage)
            for threshold, ruin_percentage in zip(ruin_ranges, ruin_percentages)
        },
        **{
            f"{threshold}MG": float(percentile)
            for threshold, percentile in zip(range_gain, percentiles)
        },
        **{
            f"{threshold}MGR": float(percentile)* -1
            for threshold, percentile in zip(mgr_range, percentiles_mgr)
        },
        **{
            f"{threshold}MGD": float(percentile)* -1
            for threshold, percentile in zip(mgd_range, percentiles_mgd)
        },
        "MG": np.median(gains),
        "MGR":mgr*-1,
        "MGD":np.median(mdd)*-1
    }
    if len(data) // 52 > 0:
        
        combinations_m = generate_combinations_month(
             profits
        )
        combinations_w = generate_combinations_week(
             profits
        )
        equity_m = calculate_equity(combinations_m, cap)
        equity_w = calculate_equity(combinations_w, cap)
        gains_m = (equity_m[:, -1] / cap - 1) * 100
        gains_w = (equity_w[:, -1] / cap - 1) * 100
        percentile_range_mw = [1]
        percentiles_m = np.percentile(gains_m, percentile_range_mw)
        percentiles_w = np.percentile(gains_w, percentile_range_mw)
        range_gain = [100 - i for i in percentile_range_mw]
        final.update(
            {
                "MGW": np.median(gains_w, axis=0),
                **{
                    f"{threshold}WGain": float(percentile)
                    for threshold, percentile in zip(range_gain, percentiles_w)
                },
                "MGM": np.median(gains_m, axis=0),
                **{
                    f"{threshold}MGain": float(percentile)
                    for threshold, percentile in zip(range_gain, percentiles_m)
                },
            }
        )

    return final


def carlo_alpha(alpha_name,gen,fee,configs,dic_freqs,DIC_ALPHAS,df_tick,start,end,stop_loss,book_size,is_sizing,init_sizing):
    DIC_ALPHAS = Domains.get_list_of_alphas()
    bt = SimulatorMega(
        alpha_name=alpha_name,
        configs=configs,
        dic_freqs=dic_freqs,
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=df_tick,
        start=start,
        end=end,
        fee=fee,
        stop_loss=stop_loss,
        gen=gen,
        booksize=book_size,
        is_sizing=is_sizing,
        init_sizing=init_sizing
        
    )
    bt.compute_mega()
    bt.compute_performance() 
    profits=bt.extract_net_profits()
    report = calculate_metrics(profits,cap=book_size*300,sample=10000)
    return report

def carlos_alpha(id, one_carlo):
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return
    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen", "1_2")
    wfo = alpha_doc.get("wfo", {})
    periods = wfo.get("period", [])
    fee = wfo.get("fee", 0.175)
    correlation = wfo.get("correlation", {})
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()
    df_tick = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    reports = []
    if one_carlo is not None:
        periods = [item for item in periods if item.get("os","") == one_carlo]
    for period in periods:
        _is = period.get("is", "")
        _os = period.get("os", "")
        start = _os.split("-")[0]
        end = _os.split("-")[1]
        stop_loss = period.get(f"stop_loss", 0)
        book_size = period.get(f"book_size", 1)
        is_sizing = period.get(f"is_sizing",False)
        init_sizing = period.get(f"init_sizing",30)
        configs = next(
            (item.get("strategies", []) for item in correlation.get("results", [])
            if item.get("is") == _is),
            []
        )
        if not configs and len(configs) == 0:
            continue
        report = carlo_alpha(alpha_name,gen,fee,configs,dic_freqs,DIC_ALPHAS,df_tick,start,end,stop_loss,book_size,is_sizing,init_sizing)
        report['period'] = _os
        reports.append(report)
    carlo_list = reports if one_carlo is None else []
    if one_carlo is not None:
        carlo_list = alpha_doc.get("carlo", [])
        new_data = reports[0] if reports else {}

        updated = False
        for item in carlo_list:
            period_value = item["period"]
            if period_value == one_carlo:
                item.update(new_data)
                updated = True
                break

        if not updated:
            carlo_list.append(new_data)
            
    alpha_collection.update_one(
        {"_id": ObjectId(id)},
        {
            "$set": {
                "carlo": carlo_list,
            }
        }
    )