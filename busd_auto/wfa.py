from datetime import datetime
import multiprocessing as mp
from itertools import product
from multiprocessing import shared_memory
import pickle
import re
import sys
import numpy as np
import hashlib, json, time
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from busd_auto.backtest import compute_range_info, worker_task_batch, precompute_ma, init_worker, save_results_to_db
from busd_auto.utils import get_mongo_uri, load_data, make_key, send_telegram_message, setup_logger
from busd_auto.wfo import gen_strategies

def import_wfa(df, id, start, end):
    batch_size = 1000

    # =========================
    # 1. MongoDB
    # =========================
    mongo_client = MongoClient(get_mongo_uri())
    busd_db = mongo_client["busd"]
    coll = busd_db["wfa_results"]
    busd_collection = busd_db["busd_collection"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_doc với id này.")
        return False

    
    
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
    print(f"⏳ Importing {total} rows into MongoDB (batch={batch_size})... {start} -> {end} | Source: {source}")

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
            try:
                coll.insert_many(documents, ordered=False)
            except Exception:
                pass  
            inserted += len(documents)
            documents = []

    if documents:
        try:
            coll.insert_many(documents, ordered=False)
        except Exception:
            pass
        inserted += len(documents)

    # =========================
    # 5. UPDATE STATUS
    # =========================
    try:
        busd_collection.update_one(
            {
                "_id": ObjectId(id),
                "wfa.is.start": start,
                "wfa.is.end": end
            },
            {
                "$set": {
                    "wfa.$.status": "done",
                    "wfa.$.process": inserted,
                    "wfa.$.total": total,
                    "wfa.$.threshold": threshold_info,
                    "wfa.$.exit_strength": exit_strength_info
                }
            }
        )


        print(f"✅ Import thành công {inserted}/{total} rows")
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật trạng thái wfa: {e}")
    mongo_client.close()
    return True
   

def wfa(id):
    start_time = time.time()
    FEE, DELAY = 175, 1
    mongo_client = MongoClient(get_mongo_uri())
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    coll = busd_db["wfa_results"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_collection với id này.")
        return

    source = busd_doc.get("source", "")
    name = busd_doc.get("name", "unknown_backtest")
    logger = setup_logger(f"{name}_wfa")
    wfa = busd_doc.get("wfa", {})
    threshold = wfa.get("threshold", {})
    exit_strength = wfa.get("exit_strength", {})
    period = wfa.get("period", [])
    strategies = wfa.get("strategies",[])
    logger.info(f"🚀 Bắt đầu wfa cho busd_collection: {name} | Source: {source} | ID: {id}")
    logger.info(f"Preparing data...")
    df_all = load_data(20220101,20251114,source)
    ma_lengths = {ma for ma1 in range(6, 151) for ma2 in range(5, ma1) for ma in (ma1, ma2)}
    df_ma_all = precompute_ma(df_all, ma_lengths, based_col="based_col", mode="ema")
    for p in period:
        p_start_time = time.time()
        insample = p.get("is", [])
        start = int(insample[0])
        end = int(insample[1])
        df = df_all[(df_all["day"] >= start) & (df_all["day"] <= end)].copy()
        df_ma = df_ma_all[(df_ma_all["day"] >= start) & (df_ma_all["day"] <= end)].copy()
        if len(strategies) > 5:
            all_jobs = gen_strategies(strategies,source,start,end,FEE,DELAY)
        else:
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
        
        total = len(all_jobs)
        logger.info(
            f"🚀 Running scan {start} -> {end} | "
            f"{total} parameters to run | {len(exist_keys)} existed & skipped"
        )

        busd_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {
                "wfa.process": 0,
                "wfa.total": total,
                "wfa.started_at": datetime.now(),
                "wfa.status": "running"
            }}
        )
        if total == 0:
            logger.info("✅ Không có config mới để chạy wfa. Bỏ qua.")
            busd_collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": {
                    "wfa.process": 0,
                    "wfa.total": total,
                    "wfa.started_at": datetime.now(),
                    "wfa.status": "done"
                }}
            )
            continue
        # ----------------- Chuẩn bị batch -----------------
        
        BATCH_SIZE = 1000
        batches = []
        for start_idx in range(0, total, BATCH_SIZE):
            sub_jobs = all_jobs[start_idx:start_idx + BATCH_SIZE]

            batch = [
                (start_idx + i, *job[1:])   # new_idx + phần còn lại
                for i, job in enumerate(sub_jobs)
            ]

            batches.append(batch)
        n_workers = 40

        # tính số worker hợp lý
        if total <= BATCH_SIZE:
            n_workers_to_use = 1  # hoặc ít worker để tiết kiệm CPU
        else:
            n_workers_to_use = min(n_workers, (total + BATCH_SIZE - 1) // BATCH_SIZE)  # ceil(total / max_batch_size)
        n_metrics = 8  # sharpe, mdd, tvr, ppc, profit%, net profit, psr, hhi
        shm = shared_memory.SharedMemory(create=True, size=total * n_metrics * 8)
        result_array = np.ndarray((total, n_metrics), dtype=np.float64, buffer=shm.buf)
        result_array[:] = np.nan
        completed_indices = []
        last_save_time = time.time()
        save_interval = 15  # giây
        inserted = 0
        logger.info(f"Chạy với {n_workers_to_use} processes, tổng {len(batches)} batches mỗi batch {BATCH_SIZE} configs.")
        with mp.Pool(
            processes=n_workers_to_use,
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
                        logger.info(f"⏳ Final progress update: {inserted}/{total}")
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
                "wfa.status": "done",
                "wfa.process": inserted,
                "wfa.finished_at": datetime.now(),
                "wfa.finished_in": time.time() - p_start_time
            }}
        )

        
        logger.info(f"🎯 Backtest hoàn tất cho {total} configs in {time.time() - p_start_time:.2f}s.")
    send_telegram_message(f"✅ wfa hoàn tất cho alpha: {source} trong {time.time() - start_time:.2f}s.")
    mongo_client.close()
    logger.info(f"🎯 wfa hoàn tất trong {time.time() - start_time:.2f}s.")

def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/wfa.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    wfa(_id)

if __name__ == "__main__":
    main()