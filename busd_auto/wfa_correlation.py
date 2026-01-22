from datetime import datetime
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import  time
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from busd_auto.backtest import precompute_ma
from busd_auto.is_correlation import calculate_combined_correlations, init_worker, worker_task_batch, save_results_to_db
from busd_auto.mega import os_wfa_backtest
from busd_auto.utils import get_mongo_uri, load_data, make_key, setup_logger
from busd_auto.view_correl import view_wfa_correlation
from busd_auto.wfo import gen_strategies


def correlation(id, start, end):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    mongo_local = MongoClient(get_mongo_uri())['busd']
    coll = mongo_local["correlation_backtest"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_collection với id này.")
        return

    source = busd_doc.get("source", "")
    name = busd_doc.get("name", "unknown_backtest")
    logger = setup_logger(f"{name}_{start}_{end}_wfa_correlation")
    wfa = busd_doc.get("wfa", [])
    for fa in wfa:
        if fa.get("is").get("start") == start and fa.get("is").get("end") == end:
            need_configs = fa.get("filter_report", {}).get("strategies", [])
            break
    df = load_data(start,end,source)
    
    list_ids = [make_key(
            config=config,
            data_start=start,
            data_end=end,
            source=source
        ) for config in need_configs]
    exist_stra = list(coll.find(
        {"_id": {"$in": list_ids}},
        {"_id": 1,"strategy":1}
    ))
    logger.info(f"🔎 Found {len(exist_stra)} existing backtest results in DB.")
    run_configs = list(set(need_configs) - set([stra['strategy'] for stra in exist_stra]))
    if run_configs:
        all_jobs = gen_strategies(run_configs, source, start, end, fee=175, delay=1)
        ma_lengths = set([int(param) for job in all_jobs for param in job[1:3]])
        df_ma = precompute_ma(df, ma_lengths, based_col="based_col", mode="ema")
        total = len(all_jobs)
        logger.info(f"🚀 Running correlation with {total} parameter combinations...")
        
        busd_collection.update_one(
            {"_id": ObjectId(id),
             "wfa.is.start": start,
             "wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.process": 0,
                "wfa.$.correlation.total": total,
                "wfa.$.correlation.status": "running"
            }}
        )

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

        n_metrics = 8  # sharpe, mdd, tvr, ppc, profit%, net profit, psr, hhi
        shm = shared_memory.SharedMemory(create=True, size=total * n_metrics * 8)
        result_array = np.ndarray((total, n_metrics), dtype=np.float64, buffer=shm.buf)
        result_array[:] = np.nan
        completed_indices = []
        last_save_time = time.time()
        save_interval = 15  # giây
        inserted = 0
        with mp.Pool(
            processes=1,
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

        logger.info(f"⏳ Final progress update: {inserted}/{total}")
        # ----------------- Hoàn tất -----------------
        logger.info(f"🎯correlation hoàn tất cho {total} in {time.time() - start_time:.2f}s.")
    else:
        logger.info("✅ Không có config nào cần chạy thêm.")
        
    #--------------------- Tính correlation ---------------------
    logger.info("🚀 Bắt đầu tính correlation...")
    exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
    calculate_combined_correlations(
        busd_id=id,
        stras=exist_stra,
        logger=logger,
        type="wfa",
        start=start,
        end=end
    )
    busd_collection.update_one(
        {"_id": ObjectId(id),"wfa.is.start": start,"wfa.is.end": end},
        {"$set": {
            "wfa.$.correlation.status": "done",
        }}
    )
    mongo_client.close()
    
def main():
    if len(sys.argv) < 4:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/wfa_correlation.py <_id> <start> <end>")
        sys.exit(1)

    _id = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    correlation(_id, start, end)

if __name__ == "__main__":
    main()