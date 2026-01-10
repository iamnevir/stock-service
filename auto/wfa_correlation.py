from datetime import datetime
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import  time
import traceback
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from is_correlation import calculate_combined_correlations, worker_task_batch
from mega import os_wfa_backtest
from utils import get_mongo_uri, insert_batch, load_dic_freqs, setup_logger, send_telegram_message, make_key_alpha
from view_correl import view_wfa_correlation
from wfo import gen_strategies
from gen.alpha_func_lib import Domains


def correlation(id, start, end):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    mongo_local = MongoClient(get_mongo_uri())['alpha']
    coll = mongo_local["correlation_backtest"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return
    alpha_name = alpha_doc.get("alpha_name", "")
    name = alpha_doc.get("name", "")
    logger = setup_logger(f"{name}_{start}_{end}_wfa_correlation")
    wfa = alpha_doc.get("wfa", [])
    fa = None
    for item in wfa:
        if item.get("is").get("start") == start and item.get("is").get("end") == end:
            need_configs = item.get("filter_report", {}).get("strategies", [])
            fa = item
            break
    if not fa:
        logger.error("❌ Không tìm thấy WFA với khoảng thời gian này.")
        return
    
    source = alpha_doc.get("source", None)
    gen = alpha_doc.get("gen")
    fee = fa.get("fee")
    filter_report = fa.get("filter_report")
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()
    need_configs = filter_report.get("strategies", [])
    try:
        list_ids = [make_key_alpha(
                config=config,
                fee=fee,
                start=start,
                end=end,
                alpha_name=alpha_name,
                gen=gen,
                source=source
            ) for config in need_configs]
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        logger.info(f"🔎 Found {len(exist_stra)} existing backtest results in DB.")
        
        run_configs = list(set(need_configs) - set([stra['config'] for stra in exist_stra]))
        total = len(run_configs)
        logger.info(f"🛠️ Need to run backtests for {total} configurations.")
        if total > 0:
            logger.info(f"🚀 Running correlation with {total} parameter combinations...")
            
            alpha_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {
                    "wfa.$.correlation.process": 0,
                    "wfa.$.correlation.total": total,
                    "wfa.$.correlation.status": "running"
                }}
            )

            # ----------------- Chuẩn bị batch -----------------
            n_workers = 40
            batch_size_configs = total if total < 1000 else 1000
            batches = [run_configs[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
            
            args_list = [(batch, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source) for batch in batches]
            
            logger.info(f"Chạy với {n_workers} processes, tổng {len(batches)} batches mỗi batch {batch_size_configs} configs.")
            temp_batch = []
            inserted = 0
            last_update = time.time()
            insert_batch_size = 500
            update_interval = 15  # giây

            # ----------------- Multiprocessing -----------------
            with mp.Pool(processes=n_workers) as pool:
                for batch_results in pool.imap_unordered(worker_task_batch, args_list, chunksize=1):
                    temp_batch.extend(batch_results)
                    inserted += len(batch_results)

                    # ✅ Insert và update mỗi 15s
                    if time.time() - last_update >= update_interval:
                        if temp_batch:
                            insert_batch(coll, temp_batch, batch_size=insert_batch_size)
                            temp_batch.clear()
                        alpha_collection.update_one(
                            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                            {"$set": {"wfa.$.correlation.process": inserted}}
                        )
                        logger.info(f"⏳ Progress update: {inserted}/{total}")
                        last_update = time.time()

            # ----------------- Insert phần còn lại -----------------
            if temp_batch:
                insert_batch(coll, temp_batch, batch_size=insert_batch_size)
                temp_batch.clear()

            logger.info(f"⏳ Final progress update: {inserted}/{total}")
            alpha_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {"wfa.$.correlation.process": inserted}}
            )
            # ----------------- Hoàn tất -----------------
            logger.info(f"🎯correlation hoàn tất cho {total} in {time.time() - start_time:.2f}s.")
            #--------------------- Tính correlation ---------------------
        logger.info("🚀 Bắt đầu tính correlation...")
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        calculate_combined_correlations(
            alpha_id=id,
            stras=exist_stra,
            logger=logger,
            type="wfa"
        )
        alpha_collection.update_one(
            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.status": "done",
            }}
        )
        
        mongo_client.close()
        logger.info(f"✅ Hoàn tất chạy correlation.")  
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy correlation: {e}")
        print(traceback.format_exc())
        alpha_collection.update_one(
            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {"wfa.$.correlation.status": "error"}}
        )
        mongo_client.close()
    # view_wfa_correlation(id=id,start=start,end=end)
    # os_wfa_backtest(id=id, start=fa['os']['start'], end=fa['os']['end'])
    send_telegram_message(f"✅ Correlation hoàn tất cho alpha: {name} {start}-{end} trong {time.time() - start_time:.2f}s.")
    
    
def main():
    if len(sys.argv) < 4:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfa_correlation.py <_id> <start> <end>")
        sys.exit(1)

    _id = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]

    correlation(_id, start, end)

if __name__ == "__main__":
    main()