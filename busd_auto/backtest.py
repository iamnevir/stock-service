from datetime import datetime
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import numpy as np
import  time
import pandas as pd
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from busd_auto.utils import get_mongo_uri, load_data, make_key, setup_logger
from math import erf, sqrt
from scipy.stats import skew, kurtosis
from datetime import datetime
from typing import  Tuple, Dict
from multiprocessing import shared_memory

GLOBAL_DF_MA = None
GLOBAL_DF_RAW = None
GLOBAL_SHM = None
GLOBAL_SHAPE = None
def compute_range_info(values):
    """
    Trả về dict:
    {
        "start": min,
        "end": max,
        "step": step
    }
    """
    uniq = np.sort(np.unique(values))

    if len(uniq) > 1:
        diffs = np.diff(uniq)
        positive = diffs[diffs > 0]
        step = int(positive.min()) if len(positive) else 0
    else:
        step = 0

    return {
        "start": int(uniq.min()),
        "end": int(uniq.max()),
        "step": step
    }

def importdb(df, id):
    batch_size = 1000

    # =========================
    # 1. MongoDB
    # =========================
    mongo_client = MongoClient(get_mongo_uri())
    busd_db = mongo_client["busd"]
    coll = busd_db["backtest_results"]
    busd_collection = busd_db["busd_collection"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_doc với id này.")
        return False

    backtest_params = busd_doc.get("backtest", {})
    start = backtest_params.get("start")
    end = backtest_params.get("end")
    source = busd_doc.get("source", "")

    # =========================
    # 2. CLEAN DATAFRAME
    # =========================

    # mapping cột Excel -> schema MongoDB
    REQUIRED_MAP = {
        "Sharpe": "sharpe",
        "MDD": "mddPct",
        "Turnover": "tvr",
        "PPC": "ppc",
        "NET Profit %": "profitPct",
        "NET Profit": "netProfit",
        "PSR": "psr",
        "HHI": "hhi",
    }

    # chỉ giữ Config + các cột cần
    keep_cols = ["Config"] + [c for c in REQUIRED_MAP if c in df.columns]
    df = df[keep_cols].copy()

    # rename theo schema DB
    df = df.rename(columns=REQUIRED_MAP)

    # các cột numeric THỰC SỰ tồn tại
    numeric_cols = [c for c in REQUIRED_MAP.values() if c in df.columns]

    # clean số: bỏ ',' và '%'
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )

    # convert sang numeric
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # bỏ dòng mà toàn bộ numeric đều NaN
    df = df.dropna(how="all", subset=numeric_cols)

    if df.empty:
        print("❌ Không có dữ liệu hợp lệ để import")
        return False

    # =========================
    # 3. SPLIT CONFIG
    # =========================
    df[['ma1', 'ma2', 'th', 'es']] = (
        df['Config'].str.split('_', expand=True).astype(int)
    )

    th_list = df['th'].tolist()
    es_list = df['es'].tolist()

    threshold_info = compute_range_info(th_list)
    exit_strength_info = compute_range_info(es_list)

    # =========================
    # 4. INSERT MONGODB (WHITELIST)
    # =========================

    # schema DB ổn định
    DB_SCHEMA = [
        "sharpe",
        "mddPct",
        "tvr",
        "ppc",
        "profitPct",
        "netProfit",
        "psr",
        "hhi",
    ]

    total = len(df)
    print(f"⏳ Importing {total} rows into MongoDB (batch={batch_size})...")

    documents = []
    inserted = 0

    for _, row in df.iterrows():
        doc = {
            "_id": make_key(row["Config"], start, end, source),
            "strategy": row["Config"],
        }

        # whitelist + insert mềm
        for field in DB_SCHEMA:
            if field in df.columns and pd.notna(row[field]):
                doc[field] = float(row[field])

        documents.append(doc)

        if len(documents) >= batch_size:
            coll.insert_many(documents, ordered=False)
            inserted += len(documents)
            documents = []

    if documents:
        coll.insert_many(documents, ordered=False)
        inserted += len(documents)

    # =========================
    # 5. UPDATE STATUS
    # =========================
    busd_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "backtest.status": "done",
            "backtest.total": total,
            "backtest.threshold": threshold_info,
            "backtest.exit_strength": exit_strength_info,
            "backtest.process": inserted,
            "backtest.finished_at": datetime.now(),
        }}
    )

    print(f"✅ Import thành công {inserted}/{total} rows")
    return True
   
def calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")
    total_days = (end_date - start_date).days
    working_days = total_days * workdays_per_year / 365.0
    return round(working_days)

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
            grossProfit=("grossProfit", "sum"),
            turnover=("turnover", "sum"),
            netProfit=("netProfit", "sum")
        )
        .round(2)
    )
    df_1d["cumGrossProfit"] = df_1d["grossProfit"].cumsum()
    df_1d["cumTurnover"] = df_1d["turnover"].cumsum()
    df_1d["cumNetProfit"] = df_1d["netProfit"].cumsum()
    
    return df_1d

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
    
# --- Reporting ---
def compute_report(df_1d, start, end):
    report = {
        "sharpe": None,
        "tvr": 0,
        "numdays": len(df_1d),
        "start_day": int(df_1d.index[0]),
        "end_day": int(df_1d.index[-1]),
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
    daily_sharpe = df_1d["netProfit"].mean() / std
    ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
    profitPct = df_1d["netProfit"].sum() / equity*100
    working_day = calculate_working_days(start,end)
    returns = df_1d['netProfit'] / 300
    winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
    total_profit = winning_profits.sum()
    shares = winning_profits / total_profit
    hhi = (shares ** 2).sum()
    report.update({
        "sharpe": float(round(df_1d["netProfit"].mean() / std * np.sqrt(working_day), 3)),
        "tvr": float(tvr),
        "psr": float(round(psr(returns, daily_sharpe), 2)),
        "hhi": float(round(hhi, 4)),
        "netProfit": float(round(df_1d["netProfit"].sum(), 2)),
        "mddPct": float(round(df_1d["mdd1"].values[-1], 2)),
        "ppc": float(round(ppc, 3)),
        "profitPct": float(round(profitPct, 2)),
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

def worker_task_batch(job_batch):
    """Worker chạy 1 batch ≈ 1000 configs và trả về danh sách idx đã xử lý"""
    np_array = np.ndarray(GLOBAL_SHAPE, dtype=np.float64, buffer=GLOBAL_SHM.buf)
    completed = []

    for args in job_batch:
        (
            idx, ma1, ma2, th, es,
            fee, delay, config_id, _id,
            start, end
        ) = args

        completed.append(idx)

        rpt = run_one_config(ma1, ma2, th, es, fee, delay,
                             GLOBAL_DF_MA, GLOBAL_DF_RAW,
                             start, end)

        np_array[idx, 0] = rpt.get("sharpe", np.nan)
        np_array[idx, 1] = rpt.get("mddPct", np.nan)
        np_array[idx, 2] = rpt.get("tvr", np.nan)
        np_array[idx, 3] = rpt.get("ppc", np.nan)
        np_array[idx, 4] = rpt.get("profitPct", np.nan)
        np_array[idx, 5] = rpt.get("netProfit", np.nan)
        np_array[idx, 6] = rpt.get("psr", np.nan)
        np_array[idx, 7] = rpt.get("hhi", np.nan)

    return completed


def save_results_to_db(result_array, indices, backtest_results_collection, all_jobs, batch_size=1000):
    ops = []  # chứa các update operations

    for i, idx in enumerate(indices):
        row = result_array[idx]
        ro = all_jobs[idx]
        _id = ro[8]
        config_id = ro[7]

        doc = {
            "_id": _id,
            "strategy": config_id,
            "sharpe": float(row[0]),
            "mddPct": float(row[1]),
            "tvr": float(row[2]),
            "ppc": float(row[3]),
            "profitPct": float(row[4]),
            "netProfit": float(row[5]),
            "psr": float(row[6]),
            "hhi": float(row[7]),
        }

        # thêm vào batch operation
        ops.append(
            UpdateOne(
                {"_id": _id},
                {"$set": doc},
                upsert=True
            )
        )

        # Nếu batch đủ
        if len(ops) >= batch_size:
            backtest_results_collection.bulk_write(ops, ordered=False)
            ops = []  # reset

    # Insert/update phần còn lại
    if ops:
        backtest_results_collection.bulk_write(ops, ordered=False)


def backtest(id):
    FEE, DELAY = 175, 1
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri())
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    coll = busd_db["backtest_results"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_collection với id này.")
        return

    source = busd_doc.get("source", "")
    name = busd_doc.get("name", "unknown_backtest")
    logger = setup_logger(name)

    backtest_params = busd_doc.get("backtest", {})
    start = backtest_params.get("start")
    end = backtest_params.get("end")
    threshold = backtest_params.get("threshold", {})
    exit_strength = backtest_params.get("exit_strength", {})
    
    df = load_data(start,end,source)
    ma_lengths = {ma for ma1 in range(6, 151) for ma2 in range(5, ma1) for ma in (ma1, ma2)}
    df_ma = precompute_ma(df, ma_lengths, based_col="based_col", mode="ema")

    all_jobs = []
    th_list = list(range(threshold['start'], threshold['end'] + threshold['step'], threshold['step']))
    es_list = list(range(exit_strength['start'], exit_strength['end'] + exit_strength['step'], exit_strength['step']))
    idx = 0
    for ma1 in range(6, 151):
        for ma2 in range(5, ma1):
            for th in th_list:
                for es in es_list:
                    config_id = f"{ma1}_{ma2}_{th}_{es}"
                    _id = make_key(config_id, start, end, source)
                    all_jobs.append((idx, ma1, ma2, th, es, FEE, DELAY, config_id, _id, start, end))
                    idx += 1

    keys = [job[8] for job in all_jobs]  # Lấy danh sách các key (_id)
    batch_size = 10000                     # tuỳ chỉnh theo nhu cầu

    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    exist_params = []

    for batch in chunks(keys, batch_size):
        docs = coll.find({"_id": {"$in": batch}}, {"_id": 1})
        exist_params.extend(docs)
        
    exist_keys = {e["_id"] for e in exist_params}
    all_jobs = [job for job in all_jobs if job[8] not in exist_keys]
    logger.info(f"🔍 Đã bỏ qua {len(exist_keys)} configs đã có trong DB.")
    total_jobs = len(all_jobs)
    if total_jobs == 0:
        logger.info("✅ Tất cả configs đã có trong DB. Kết thúc backtest.")
        mongo_client.close()
        return
    logger.info(f"🚀 Running backtest with {total_jobs} parameter combinations...")

    busd_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "backtest.process": 0,
            "backtest.total": total_jobs,
            "backtest.started_at": datetime.now(),
            "backtest.status": "running"
        }},
    )

    # ----------------- Chuẩn bị batch -----------------
    BATCH_SIZE = 1000

    batches = [
        all_jobs[i:i + BATCH_SIZE]
        for i in range(0, total_jobs, BATCH_SIZE)
    ]

    n_metrics = 8  # sharpe, mdd, tvr, ppc, profit%, net profit, psr, hhi
    shm = shared_memory.SharedMemory(create=True, size=total_jobs * n_metrics * 8)
    result_array = np.ndarray((total_jobs, n_metrics), dtype=np.float64, buffer=shm.buf)
    result_array[:] = np.nan
    completed_indices = []
    last_save_time = time.time()
    save_interval = 15  # giây
    inserted = 0
    with mp.Pool(
        processes=40,
        initializer=init_worker,
        initargs=(df_ma, df, shm.name, result_array.shape)
    ) as pool:

        for batch_completed in pool.imap_unordered(worker_task_batch, batches):

            completed_indices.extend(batch_completed)

            # --- Lưu DB mỗi 15 giây ---
            if time.time() - last_save_time >= save_interval:
                if completed_indices:
                    save_results_to_db(result_array, completed_indices, coll, all_jobs)
                    inserted += len(completed_indices)
                    logger.info(f"⏳ Final progress update: {inserted}/{total_jobs}")
                    completed_indices.clear()
                last_save_time = time.time()

    # --- Lưu lần cuối ---
    if completed_indices:
        save_results_to_db(result_array, completed_indices, coll, all_jobs)
        inserted += len(completed_indices)
    

    # ----------------- Hoàn tất -----------------
    busd_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "backtest.status": "done",
            "backtest.process": inserted,
            "backtest.finished_at": datetime.now(),
            "backtest.finished_in": time.time() - start_time
        }}
    )

    mongo_client.close()
    shm.close()
    shm.unlink()
    logger.info(f"🎯 Backtest hoàn tất cho {inserted} configs in {time.time() - start_time:.2f}s.")


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/backtest.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    backtest(_id)

if __name__ == "__main__":
    main()