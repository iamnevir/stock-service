import multiprocessing
import time
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from datetime import datetime
from typing import Tuple, Dict, Optional
from multiprocessing import shared_memory
from tqdm import tqdm
from busd_auto.utils import get_mongo_uri
from pymongo import MongoClient, UpdateOne
GLOBAL_DF_MA = None
GLOBAL_DF_RAW = None
GLOBAL_SHM = None
GLOBAL_SHAPE = None


def group_dates_by_week(date_list, collection: Optional[object] = None):
    groups = {}

    for date_str in date_list:
        s_date = str(date_str).strip()
        date_obj = datetime.strptime(s_date, "%Y%m%d")
        year, week_num, _ = date_obj.isocalendar()
        week_key = f"{year}-W{week_num:02d}"

        if week_key not in groups:
            groups[week_key] = {
                "lst_day": [],
                "lst_strategy": [],
                "df_result_1D": pd.DataFrame(),
            }
        groups[week_key]["lst_day"].append(int(date_str))

    sorted_keys = sorted(groups.keys())
    for i, current_key in enumerate(sorted_keys):
        groups[current_key]["next_week"] = sorted_keys[i + 1] if i < len(sorted_keys) - 1 else None

    if collection is not None and sorted_keys:
        ops = []
        for week in sorted_keys:
            lst_day = [int(day) for day in groups[week]["lst_day"]]
            ops.append(
                UpdateOne(
                    {"week": week},
                    {
                        "$setOnInsert": {"week": week},
                        "$addToSet": {"lst_day": {"$each": lst_day}},
                    },
                    upsert=True,
                )
            )

        if ops:
            collection.bulk_write(ops, ordered=False)

    return groups


def get_weeks_for_running(collection):
    query = {
        "lst_day": {"$exists": True, "$ne": []},
    }

    weeks = list(collection.find(query, {"_id": 0, "week": 1, "lst_day": 1, "srcs": 1}).sort("week", 1))
    configs_week = {}
    for item in weeks:
        week = item.get("week")
        lst_day = item.get("lst_day", [])
        if not week or not lst_day:
            continue

        week_days = sorted(int(day) for day in lst_day)
        done_src_modes = set()
        for src_item in item.get("srcs", []):
            src_name = src_item.get("src")
            mode = src_item.get("mode")
            if src_name and mode:
                done_src_modes.add(f"{src_name}__{mode}")

        configs_week[week] = {
            "lst_day": week_days,
            "start_day": week_days[0],
            "end_day": week_days[-1],
            "done_src_modes": done_src_modes,
        }

    return configs_week


def filter_results_by_week(dic_result, top_n=4000, max_tvr_mean=30):
    filtered_results = {}
    total_weeks = len(dic_result)
    print(f"[filter] start: {total_weeks} weeks")

    for week_idx, (week, reports) in enumerate(dic_result.items(), start=1):
        if week_idx == 1 or week_idx % 10 == 0 or week_idx == total_weeks:
            pct_week = (week_idx / total_weeks * 100) if total_weeks else 100
            print(f"[filter] week {week_idx}/{total_weeks} ({pct_week:.2f}%) - {week}")

        if not reports:
            continue

        normalized_reports = []
        for report in reports:
            if isinstance(report, pd.Series):
                normalized_reports.append(report.to_dict())
            elif isinstance(report, dict):
                normalized_reports.append(report)
            elif hasattr(report, "to_dict"):
                normalized_reports.append(report.to_dict())

        if not normalized_reports:
            continue

        df_week = pd.DataFrame(normalized_reports)
        if df_week.empty or "config" not in df_week.columns:
            continue

        # Support both old keys (cumNetProfit/tvr_mean) and current keys (total_net_profit/tvr).
        if "cumNetProfit" in df_week.columns:
            df_week["_profit"] = pd.to_numeric(df_week["cumNetProfit"], errors="coerce")
        else:
            df_week["_profit"] = pd.to_numeric(df_week.get("total_net_profit"), errors="coerce")

        if "tvr_mean" in df_week.columns:
            df_week["_tvr"] = pd.to_numeric(df_week["tvr_mean"], errors="coerce")
        else:
            df_week["_tvr"] = pd.to_numeric(df_week.get("tvr"), errors="coerce")

        df_week = df_week.dropna(subset=["_profit", "_tvr"]) 
        df_week = df_week[df_week["_tvr"] < max_tvr_mean]
        df_week = df_week.sort_values(by="_profit", ascending=False).head(top_n)

        if not df_week.empty:
            configs = df_week["config"].dropna().astype(str).tolist()
            filtered_results[week] = list(dict.fromkeys(configs))

    return filtered_results

def update_week_strategies(collection, filtered_results, source_column: str, mode: str):
    if not filtered_results:
        print(f"[update] no filtered results to write for {source_column} ({mode})")
        return

    print(f"[update] preparing writes for {len(filtered_results)} weeks ({source_column}, {mode})")
    ops = []
    for week, configs in filtered_results.items():
        # Update existing src/mode bucket if present.
        ops.append(
            UpdateOne(
                {
                    "week": week,
                    "srcs": {"$elemMatch": {"src": source_column, "mode": mode}},
                },
                {
                    "$set": {
                        "srcs.$[elem].lst_strategy": configs,
                    },
                    "$unset": {
                        "lst_strategy": "",
                        "strategies": "",
                    },
                },
                array_filters=[{"elem.src": source_column, "elem.mode": mode}],
            )
        )

        # Insert new src/mode bucket if not present.
        ops.append(
            UpdateOne(
                {
                    "week": week,
                    "srcs": {"$not": {"$elemMatch": {"src": source_column, "mode": mode}}},
                },
                {
                    "$push": {
                        "srcs": {
                            "src": source_column,
                            "mode": mode,
                            "lst_strategy": configs,
                        }
                    },
                    "$unset": {
                        "lst_strategy": "",
                        "strategies": "",
                    },
                },
            )
        )

    if ops:
        collection.bulk_write(ops, ordered=False)
        print(f"[update] done: wrote lst_strategy for {len(filtered_results)} weeks ({source_column}, {mode})")

# --- Utils ---
def calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")
    total_days = (end_date - start_date).days
    working_days = total_days * workdays_per_year / 365.0
    return round(working_days)

# --- MA precomputation ---
def precompute_ma(
    df_alpha: pd.DataFrame,
    ma_lengths: set,
    based_col: str = "based_col",
    mode: str = "sma"  # Options: "sma", "ema", "dema", "wma"
) -> pd.DataFrame:
    df = df_alpha.copy()
    gb = df.groupby("day")
    ma_dfs = []

    def wma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.copy()
        weights = np.arange(1, length + 1)
        divisor = weights.sum()
        wma_values = np.convolve(series.fillna(0).to_numpy(), weights[::-1], mode='valid') / divisor
        result = np.full(series.shape, np.nan)
        result[length - 1:] = wma_values
        return pd.Series(result, index=series.index)

    for ma in ma_lengths:
        if mode == "sma":
            ma_series = gb[based_col].transform(lambda x: x.rolling(window=ma, min_periods=1).mean())
        elif mode == "ema":
            ma_series = gb[based_col].transform(lambda x: x.ewm(span=ma, adjust=True).mean())
        elif mode == "dema":
            def calc_dema(x):
                ema1 = x.ewm(span=ma, adjust=True).mean()
                ema2 = ema1.ewm(span=ma, adjust=True).mean()
                return 2 * ema1 - ema2
            ma_series = gb[based_col].transform(calc_dema)
        elif mode == "wma":
            ma_series = gb[based_col].transform(lambda x: wma(x, ma))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'sma', 'ema', 'dema', 'wma'.")
        
        ma_dfs.append(ma_series.rename(f"ma_{ma}"))
        

    ma_all = pd.concat(ma_dfs, axis=1)
    df = pd.concat([df, ma_all], axis=1)

    return df.copy()


# --- Position generation ---
def compute_single_position(df_ma: pd.DataFrame, ma1: int, ma2: int, threshold: int, exit_strength: int = 0, delay: int = 1) -> pd.Series:
    diff = df_ma[f"ma_{ma2}"] - df_ma[f"ma_{ma1}"]
    position = pd.Series(index=diff.index, dtype=np.float32)

    position[diff > threshold] = 1
    position[diff < -threshold] = -1

    if exit_strength != 0:
        take_profit = threshold * exit_strength / 100
        position[np.abs(diff) < take_profit] = 0

    position = position.shift(delay, fill_value=0)
    position[df_ma["time_int"] >= 10142930] = np.nan
    position[df_ma["time_int"] >= 10144500] = 0
    position[df_ma["time_int"] <= 10091500] = 0
    day_vals = df_ma['day'].values
    is_new_day = np.concatenate(([True], day_vals[1:] != day_vals[:-1]))
    new_day_indices = np.where(is_new_day)[0]
    pos_copy = position.values
    n = len(position)

    for start_idx in new_day_indices:
        end_idx = min(start_idx + delay, n)
        pos_copy[start_idx:end_idx] = 0.0
    position = pd.Series(pos_copy, index=position.index)
    position = position.ffill().fillna(0)
    return position

# --- Profit computation ---
def compute_profit(position: pd.Series, df_alpha: pd.DataFrame, fee: int) -> pd.DataFrame:
    df_alpha["position"] = position
    df_alpha["grossProfit"] = df_alpha["position"] * df_alpha["priceChange"]
    df_alpha["action"] = df_alpha["position"].diff().fillna(df_alpha["position"])
    df_alpha["turnover"] = df_alpha["action"].abs()
    df_alpha["fee"] = df_alpha["turnover"] * fee / 1000
    df_alpha["netProfit"] = df_alpha["grossProfit"] - df_alpha["fee"]

    df_1d = (
        df_alpha.groupby("day", sort=False, observed=True)
        .agg(
            turnover=("turnover", "sum"),
            netProfit=("netProfit", "sum")
        )
        .round(2)
    )
    df_1d["cumNetProfit"] = df_1d["netProfit"].cumsum()
    
    return df_1d

    
# --- Reporting ---
def compute_report(df_1d, start, end):
    report = {
        "sharpe": None,
        "tvr": 0,
        "total_net_profit": 0.0,
        "mdd_percent": 0.0,
    }
    equity = 300
    if "cdd" not in df_1d:
        cummax = df_1d["cumNetProfit"].cummax()
        df_1d["cdd"] = (cummax - df_1d["cumNetProfit"]).round(2)
        df_1d["cdd1"] = ((equity+ cummax - (equity + df_1d["cumNetProfit"])) / (equity + cummax) * 100).round(2)
        df_1d["mdd"] = df_1d["cdd"].cummax()
        df_1d["mdd1"] = df_1d["cdd1"].cummax()
        df_1d["cumNetProfit1"] = df_1d["cumNetProfit"] + equity

    tvr = round(df_1d["turnover"].mean(), 3)
    std = df_1d["netProfit"].std()
    
    if tvr == 0 or std == 0:
        return report
    working_day = calculate_working_days(start,end)

    report.update({
        "sharpe": float(round(df_1d["netProfit"].mean() / std * np.sqrt(working_day), 3)),
        "tvr": float(tvr),
        "total_net_profit": float(round(df_1d["netProfit"].sum(), 2)),
        "mdd_percent": float(round(df_1d["mdd1"].values[-1], 2)),
    })

    return report 

# --- Master runner ---
def run_one_config(ma1, ma2, th,es, fee, delay, df_ma, df, start, end) -> Tuple[str, Dict]:
    # start_time = time.time()
    position = compute_single_position(df_ma, ma1, ma2, th, es, delay)
    # print(f"compute_single_position: {time.time()-start_time}")
    # start_time = time.time()
    df_1d = compute_profit(position, df, fee)
    # print(f"compute_profit: {time.time()-start_time}")
    # start_time = time.time()
    report = compute_report(df_1d, start, end)
    # print(f"compute_report: {time.time()-start_time}")
    return report

# --- Entry point ---

# === Worker init / task ===
def init_worker(df_ma_, df_raw_, shm_name_, shape_):
    """Mỗi worker ánh xạ shared memory và dữ liệu đọc-only"""
    global GLOBAL_DF_MA, GLOBAL_DF_RAW, GLOBAL_SHM, GLOBAL_SHAPE
    GLOBAL_DF_MA = df_ma_
    GLOBAL_DF_RAW = df_raw_
    GLOBAL_SHM = shared_memory.SharedMemory(name=shm_name_)
    GLOBAL_SHAPE = shape_



def worker_task(args):
    """Mỗi worker chạy 1 cấu hình và ghi kết quả trực tiếp vào shared memory"""
    idx, ma1, ma2, th, es, fee, delay, config_id, start, end = args

    # === gọi hàm thực tế của bạn ===
    rpt = run_one_config(ma1, ma2, th, es, fee, delay, GLOBAL_DF_MA, GLOBAL_DF_RAW, start, end)
    # ví dụ rpt = {"sharpe": 1.23, "mdd_percent": -5.2, "tvr": 0.31, "ppc": 0.88, "profit_percent": 17.3, "total_net_profit": 12000}

    np_array = np.ndarray(GLOBAL_SHAPE, dtype=np.float64, buffer=GLOBAL_SHM.buf)
    np_array[idx, 0] = rpt.get("sharpe", np.nan)
    np_array[idx, 1] = rpt.get("mdd_percent", np.nan)
    np_array[idx, 2] = rpt.get("tvr", np.nan)
    np_array[idx, 3] = rpt.get("total_net_profit", np.nan)

    
def scan(source_column: str, start_date: int, end_date: int, mode: str, df_base: pd.DataFrame):
    FEE, DELAY = 175, 1
    start_time = time.time()
    
    # --- Filter per requested week window ---
    df = df_base[(df_base["day"] >= start_date) & (df_base["day"] <= end_date)].copy()
    if df.empty:
        print(f"[scan] skip {source_column} {start_date}-{end_date}: empty data")
        return []

    df["based_col"] = 0
    for col in source_column.split("_"):
        if col == "hose500": df["based_col"] += df["aggBusd"]
        elif col == "fhose500": df["based_col"] += df["aggFBusd"]
        elif col == "vn30": df["based_col"] += df["aggBusdVn30"]
        elif col == "fvn30": df["based_col"] += df["aggFBusdVn30"]

    df = df[df["based_col"].diff() != 0]
    df = df[df["time_int"] >= 10091500]
    if df.empty:
        print(f"[scan] skip {source_column} {start_date}-{end_date}: filtered data empty")
        return []

    df["priceChange"] = df.groupby("day")["last"].diff().shift(-1).fillna(0)

    # --- Precompute MA ---
    ma_lengths = {ma for ma1 in range(6, 151) for ma2 in range(5, ma1) for ma in (ma1, ma2)}
    df_ma = precompute_ma(df, ma_lengths, based_col="based_col", mode=mode)

    # --- Build jobs ---
    all_jobs = []
    th_list = list(range(5, 20))
    es_list = list(range(2, 9, 3))
    idx = 0
    for ma1 in range(6, 151):
        for ma2 in range(5, ma1):
            for th in th_list:
                for es in es_list:
                    config_id = f"{ma1}_{ma2}_{th}_{es}"
                    all_jobs.append((idx, ma1, ma2, th, es, FEE, DELAY, config_id, start_date, end_date))
                    idx += 1

    total_jobs = len(all_jobs)
    print(f"Total jobs: {total_jobs}")

    # --- Shared memory cho 6 chỉ tiêu ---
    n_metrics = 4  # sharpe, mdd, tvr, ppc, profit%, net profit
    shm = shared_memory.SharedMemory(create=True, size=total_jobs * n_metrics * 8)
    result_array = np.ndarray((total_jobs, n_metrics), dtype=np.float64, buffer=shm.buf)
    result_array[:] = np.nan

    # --- Run Pool ---
    with multiprocessing.Pool(
        processes=40,
        initializer=init_worker,
        initargs=(df_ma, df, shm.name, result_array.shape)
    ) as pool:
        list(tqdm(pool.imap_unordered(worker_task, all_jobs), total=total_jobs, desc="Running backtest"))

    # --- Build report rows for filtering and DB update ---
    df_result = pd.DataFrame(result_array, columns=["Sharpe", "MDD", "Turnover", "NET Profit"])
    df_result["config"] = [f"{j[1]}_{j[2]}_{j[3]}_{j[4]}" for j in all_jobs]
    df_result = df_result.rename(
        columns={
            "Turnover": "tvr",
            "NET Profit": "total_net_profit",
            "Sharpe": "sharpe",
            "MDD": "mdd_percent",
        }
    )

    elapsed = time.time() - start_time
    print(f"[scan] completed {total_jobs} jobs in {elapsed:.2f}s ({total_jobs/elapsed:.2f} jobs/s)")
    
    shm.close()
    shm.unlink()
    return df_result.to_dict("records")

if __name__ == "__main__":
    print("Starting weekly scan from MongoDB...")

    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_collection = mongo_client["busd"]["busd_dynamic"]

    df = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    df = df[(df["day"] >= 20220101) & (df["day"] < 20270101)].copy()

    lst_day = df["day"].unique()
    group_dates_by_week(lst_day, busd_collection)
    configs_week = get_weeks_for_running(busd_collection)
    print(f"[main] weeks pending run: {len(configs_week)}")

    srcs = [
        {"source_column": "hose500", "mode": "ema"},
        # {"source_column": "hose500_vn30", "mode": "ema"},
        # {"source_column": "hose500_fhose500", "mode": "ema"},
        # {"source_column": "hose500_fhose500_vn30_fvn30", "mode": "ema"},
        # {"source_column": "vn30_fhose500", "mode": "ema"},
        # {"source_column": "hose500_fvn30", "mode": "ema"},
        # {"source_column": "hose500_vn30_fhose500", "mode": "ema"},
        # {"source_column": "vn30", "mode": "ema"},
        # {"source_column": "hose500_vn30_fvn30", "mode": "ema"},
        # {"source_column": "hose500_fhose500_fvn30", "mode": "ema"},
        # {"source_column": "vn30_fhose500_fvn30", "mode": "ema"},
        # {"source_column": "vn30_fvn30", "mode": "ema"},
    ]

    total_weeks = len(configs_week)
    total_srcs = len(srcs)
    for src_idx, src in enumerate(srcs, start=1):
        source_column = src["source_column"]
        mode = src["mode"]
        print(f"\n[main] src {src_idx}/{total_srcs}: {source_column} ({mode})")

        for week_idx, (week_key, week_info) in enumerate(configs_week.items(), start=1):
            start_day = int(week_info["start_day"])
            end_day = int(week_info["end_day"])
            done_src_modes = week_info.get("done_src_modes", set())
            src_mode_key = f"{source_column}__{mode}"

            if src_mode_key in done_src_modes:
                print(f"[main] skip week {week_key} for {source_column} ({mode}): already exists in db")
                continue

            print(f"[main] week {week_idx}/{total_weeks}: {week_key} ({start_day}-{end_day})")

            reports = scan(source_column, start_day, end_day, mode, df)
            week_results = {week_key: reports if reports else []}
            print(f"[main] week {week_key} total reports for {source_column} ({mode}): {len(week_results[week_key])}")

            # Filter and persist immediately after each week is done.
            filtered_results = filter_results_by_week(week_results, top_n=4000, max_tvr_mean=30)
            update_week_strategies(busd_collection, filtered_results, source_column, mode)

            if week_key in filtered_results:
                done_src_modes.add(src_mode_key)

    print("[main] done")
    