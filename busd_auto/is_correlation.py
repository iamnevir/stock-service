from datetime import datetime
from itertools import combinations, islice
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
from typing import Dict, Tuple
import numpy as np
import  time
import pandas as pd
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from busd_auto.utils import get_mongo_uri, sanitize_for_bson, setup_logger
from busd_auto.backtest import compute_report, compute_single_position

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
    df_trade = df_alpha[df_alpha["action"] != 0][["action"]]
    return df_1d, df_trade

def run_one_config(ma1, ma2, th,es, fee, delay, df_ma, df, start, end) -> Tuple[str, Dict]:
    # start_time = time.time()
    position = compute_single_position(df_ma, ma1, ma2, th, es, delay)
    # print(f"compute_single_position: {time.time()-start_time}")
    # start_time = time.time()
    df_1d, df_trade = compute_profit(position, df, fee)
    # print(f"compute_profit: {time.time()-start_time}")
    # start_time = time.time()
    report = compute_report(df_1d, start, end)
    
    return report, df_trade

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
    np_array = np.ndarray(GLOBAL_SHAPE, dtype=np.float64, buffer=GLOBAL_SHM.buf)

    results = []  # (idx, df_trade)

    for args in job_batch:
        (
            idx, ma1, ma2, th, es,
            fee, delay, config_id, _id,
            start, end
        ) = args

        rpt, df_trade = run_one_config(ma1, ma2, th, es, fee, delay,
                                       GLOBAL_DF_MA, GLOBAL_DF_RAW,
                                       start, end)

        # ghi metric vào shared memory
        np_array[idx, 0] = rpt.get("sharpe", np.nan)
        np_array[idx, 1] = rpt.get("mddPct", np.nan)
        np_array[idx, 2] = rpt.get("tvr", np.nan)
        np_array[idx, 3] = rpt.get("ppc", np.nan)
        np_array[idx, 4] = rpt.get("profitPct", np.nan)
        np_array[idx, 5] = rpt.get("netProfit", np.nan)
        np_array[idx, 6] = rpt.get("psr", np.nan)
        np_array[idx, 7] = rpt.get("hhi", np.nan)

        results.append((idx, df_trade))

    return results

def save_results_to_db(result_array, results, backtest_results_collection, all_jobs, batch_size=1000):

    ops = []

    for idx, df_trade in results:
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
            "df_trade": df_trade.reset_index().to_dict("records"),
        }

        ops.append(
            UpdateOne(
                {"_id": _id},
                {"$set": doc},
                upsert=True
            )
        )

        if len(ops) >= batch_size:
            backtest_results_collection.bulk_write(ops, ordered=False)
            ops = []

    if ops:
        backtest_results_collection.bulk_write(ops, ordered=False)

def calculate_trade_correlation_vectorized(df1, df2):
    if df1.empty or df2.empty:
        return 0.0, 0.0

    time_tolerance = pd.Timedelta(seconds=3)

    df1 = df1.sort_values('datetime')
    df2 = df2.sort_values('datetime')

    merged = pd.merge_asof(
        df1[['datetime', 'action']],
        df2[['datetime', 'action']],
        on='datetime',
        direction='nearest',
        tolerance=time_tolerance,
        suffixes=('_1', '_2')
    )

    # Điều kiện vector hóa
    a1 = merged['action_1'].values
    a2 = merged['action_2'].values

    # Nếu không match sẽ là NaN → loại bỏ
    valid = ~np.isnan(a2)

    a1 = a1[valid]
    a2 = a2[valid]

    # Match signal logic
    matches = (
        (a1 == a2) |
        ((a1 == 1) & (a2 == 2)) |
        ((a1 == 2) & (a2 == 1)) |
        ((a1 == -1) & (a2 == -2)) |
        ((a1 == -2) & (a2 == -1))
    )

    matched_count = np.sum(matches)

    corr1 = round(matched_count / len(df1) * 100, 2)
    corr2 = round(matched_count / len(df2) * 100, 2)
    return max(corr1, corr2)

# ---- process_chunk cũng phải ở cấp module ----
def process_chunk(args):
    """Worker: xử lý 1 chunk, trả về list kết quả"""
    chunk, id_to_trade_df = args
    results = []
    for id1, id2 in chunk:
        x, y = str(id1), str(id2)
        df1, df2 = id_to_trade_df[id1], id_to_trade_df[id2]
        c = calculate_trade_correlation_vectorized(df1, df2)
        results.append({"x": x, "y": y, "c": round(c, 4)})
    return results

def insert_batch(coll, records, batch_size=500):
    """Insert records vào MongoDB theo batch_size"""
    try:
        for i in range(0, len(records), batch_size):
            coll.insert_many(records[i:i+batch_size])
    except Exception as e:
        print(f"Error inserting batch: {e}")
        
def calculate_combined_correlations(
    busd_id,
    stras=None,
    logger=None,
    max_workers=20,
    chunk_size=100000,
    type="ios",
    start=None, end=None
):
    """
    Tính toán tương quan busd, sử dụng kiến trúc Producer-Consumer (Worker/Queue/Writer)
    giống như hàm tính tương quan stock chuẩn.
    """
    
    # === 0️⃣ Khởi tạo kết nối DB ===
    # Các process con sẽ kế thừa kết nối này, 
    # nhưng chỉ writer_worker thực sự dùng nó để ghi.
    db = MongoClient(get_mongo_uri("mgc3"))['busd']
    busd_collection = db["busd_collection"]
    db_local = MongoClient(get_mongo_uri())['busd']
    correlation_coll = db_local["correlation_results"]
    # === 1️⃣ Chuẩn bị dữ liệu ===
    logger.info("⏳ Đang chuẩn bị dữ liệu (parsing trades)...")
    id_to_trade_df = {}

    def parse_trade_doc(doc):
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            return None
        # Kiểm tra định dạng của busd
        if not all(isinstance(t, dict) and "datetime" in t and "action" in t for t in trades):
            return None
        df = pd.DataFrame(trades)
        df.dropna(subset=["datetime"], inplace=True)
        return df if not df.empty else None

    for doc in stras:
        _id = doc["_id"]
        trade_df = parse_trade_doc(doc)
        if trade_df is not None:
            id_to_trade_df[_id] = trade_df

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                break
            yield chunk

    valid_ids = list(id_to_trade_df.keys())
    
    # === 2️⃣ Lọc các cặp đã tồn tại (Giống code chuẩn) ===
    logger.info("🔎 Đang lấy danh sách cặp đã tồn tại trong MongoDB...")
    str_ids = [str(i) for i in valid_ids]
    existing_pairs = set()
    
    # Query hiệu quả hơn thay vì dùng 2 $in lớn
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            {"x": 1, "y": 1, "_id": 0}
        )
        for doc in cursor:
            existing_pairs.add(tuple(sorted((doc["x"], doc["y"]))))

    logger.info(f"✅ Đã có sẵn {len(existing_pairs):,} cặp trong DB — sẽ bỏ qua.")

    # === 3️⃣ Sinh danh sách cặp cần xử lý (Giống code chuẩn) ===
    all_pairs = []
    total_combinations = 0
    for id1, id2 in combinations(valid_ids, 2):
        total_combinations += 1
        key = tuple(sorted((str(id1), str(id2))))
        if key not in existing_pairs:
            all_pairs.append((id1, id2))

    logger.info(f"🧮 Còn lại {len(all_pairs):,} cặp cần tính mới (trên tổng số {total_combinations:,} cặp).")
    chunk_size = min(100000, len(all_pairs) // max_workers)
    chunks = list(chunked_iterable(all_pairs, chunk_size))
    total_chunks = len(chunks)
    total_pairs_to_process = len(all_pairs)

    logger.info(f"🔢 Tổng số cặp cần xử lý: {total_pairs_to_process}")
    logger.info(f"📦 Tổng số chunk (mỗi chunk ~{chunk_size} cặp): {total_chunks}")

    if type == "wfa":
        busd_collection.update_one(
            {"_id": ObjectId(busd_id),"wfa.is.start": start,"wfa.is.end": end},
            {"$set": {f"wfa.$.correlation.process": len(existing_pairs), f"wfa.$.correlation.total": total_combinations,f"wfa.$.correlation.status": "running"}}
        )
    else:
        busd_collection.update_one(
            {"_id": ObjectId(busd_id)},
            {"$set": {f"{type}.correlation.process": len(existing_pairs), f"{type}.correlation.total": total_combinations,f"{type}.correlation.status": "running"}}
        )

    if not all_pairs:
        logger.info("🏁 Không còn cặp nào để xử lý. Kết thúc.")
        if type == "wfa":
            busd_collection.update_one(
                {"_id": ObjectId(busd_id),"wfa.is.start": start,"wfa.is.end": end},
                {"$set": {f"wfa.$.correlation.status": "done", "wfa.$.correlation.process": total_combinations, "wfa.$.correlation.total": total_combinations}}
            )
        else:
            busd_collection.update_one(
                {"_id": ObjectId(busd_id)},
                {"$set": {f"{type}.correlation.status": "done", f"{type}.correlation.process": total_combinations, f"{type}.correlation.total": total_combinations}}
            )
        return
    total = sum(len(chunk) for chunk in chunks)
    inserted = 0
    temp_batch = []
    last_update = time.time()

    args_list = [(chunk, id_to_trade_df) for chunk in chunks]

    logger.info(f"🚀 Bắt đầu xử lý {len(chunks)} chunks với tối đa {max_workers} workers...")

    with mp.Pool(processes=max_workers) as pool:
        for batch_results in pool.imap_unordered(process_chunk, args_list, chunksize=1):
            temp_batch.extend(batch_results)
            inserted += len(batch_results)

            # ✅ Cứ sau mỗi update_interval giây thì insert và cập nhật tiến độ
            if time.time() - last_update >= 10:
                if temp_batch:
                    insert_batch(correlation_coll, temp_batch, batch_size=10000)
                    temp_batch.clear()
                if type == "wfa":
                    busd_collection.update_one(
                        {"_id": ObjectId(busd_id),"wfa.is.start": start,"wfa.is.end": end},
                        {"$set": {f"wfa.$.correlation.process": inserted}}
                    )
                else:
                    busd_collection.update_one(
                        {"_id": ObjectId(busd_id)},
                        {"$set": {f"{type}.correlation.process": inserted}}
                    )
                logger.info(f"⏳ Progress update: {inserted}/{total}")
                last_update = time.time()

    # 🔚 Xử lý phần còn lại
    if temp_batch:
        insert_batch(correlation_coll, temp_batch, batch_size=10000)
        temp_batch.clear()


    logger.info(f"✅ Hoàn tất: {inserted}/{total} pairs đã xử lý.")

    # === 8️⃣ Cập nhật process.done chính xác (Giống code chuẩn) ===
    logger.info("🔄 Đang cập nhật lại số lượng chính xác cuối cùng...")
    seen = set()
    projection = {"x": 1, "y": 1, "_id": 0}
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            projection
        )
        for doc in cursor:
            seen.add(tuple(sorted((doc["x"], doc["y"]))))

    unique_pair_count = len(seen)
    if type == "wfa":
        busd_collection.update_one(
            {"_id": ObjectId(busd_id),"wfa.is.start": start,"wfa.is.end": end},
            {"$set": {f"wfa.$.correlation.process": unique_pair_count, f"wfa.$.correlation.status": "done"}}
        )
    else:
        busd_collection.update_one(
            {"_id": ObjectId(busd_id)},
            {"$set": {f"{type}.correlation.process": unique_pair_count, f"{type}.correlation.status": "done"}}
        )
    logger.info(f"✅ Đã cập nhật lại chính xác process = {unique_pair_count} và status = 'done'")
  

