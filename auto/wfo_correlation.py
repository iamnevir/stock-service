from datetime import datetime
import multiprocessing as mp
import sys
import  time
from pymongo import MongoClient
from bson import ObjectId
from auto.is_correlation import calculate_combined_correlations, worker_task_batch
from auto.utils import get_mongo_uri,load_dic_freqs, make_key_alpha, setup_logger, insert_batch, send_telegram_message
from gen.alpha_func_lib import Domains

def correlation(id):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    coll = alpha_db["correlation_backtest"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    alpha_name = alpha_doc.get("alpha_name", "")
    name = alpha_doc.get("name", "")
    logger = setup_logger(f"{name}_wfo_correlation")
    try:
        wfo = alpha_doc.get("wfo", {})
        gen = alpha_doc.get("gen")
        fee = wfo.get("fee")
        correlation = wfo.get("correlation")
        DIC_ALPHAS = Domains.get_list_of_alphas()
        dic_freqs = load_dic_freqs()
        start = "2024_01_01"
        end  = "2024_07_01"
        need_configs = correlation.get("strategies", [])
        list_ids = [make_key_alpha(
                config=config,
                fee=fee,
                start=start,
                end=end,
                alpha_name=alpha_name,
                gen=gen
            ) for config in need_configs]
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        logger.info(f"🔎 Found {len(exist_stra)} existing backtest results in DB.")
        
        run_configs = list(set(need_configs) - set([stra['config'] for stra in exist_stra]))
        total = len(run_configs)
        if total > 0:
            logger.info(f"🚀 Running correlation with {total} parameter combinations...")
            
            alpha_collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": {
                    "wfo.correlation.process": 0,
                    "wfo.correlation.total": total,
                    "wfo.correlation.status": "running"
                }}
            )

            # ----------------- Chuẩn bị batch -----------------
            n_workers = 40
            batch_size_configs = total if total < 1000 else 1000
            batches = [run_configs[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
            
            args_list = [(batch, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end) for batch in batches]
            
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
                            {"_id": ObjectId(id)},
                            {"$set": {"wfo.correlation.process": inserted}}
                        )
                        logger.info(f"⏳ Progress update: {inserted}/{total}")
                        last_update = time.time()

            # ----------------- Insert phần còn lại -----------------
            if temp_batch:
                insert_batch(coll, temp_batch, batch_size=insert_batch_size)
                temp_batch.clear()

            logger.info(f"⏳ Final progress update: {inserted}/{total}")
            # ----------------- Hoàn tất -----------------
            logger.info(f"🎯correlation hoàn tất cho {total} in {time.time() - start_time:.2f}s.")
        #--------------------- Tính correlation ---------------------
        logger.info("🚀 Bắt đầu tính correlation...")
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        calculate_combined_correlations(
            alpha_id=id,
            stras=exist_stra,
            logger=logger,
            type="wfo"
        )
        alpha_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {
                "wfo.correlation.status": "done",
            }}
        )
        send_telegram_message(f"✅ Correlation hoàn tất cho alpha: {alpha_name}")
        mongo_client.close()
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy correlation: {e}")
        alpha_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"ios.correlation.status": "error"}}
        )
        mongo_client.close()
    
def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfo_correlation.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    correlation(_id)

if __name__ == "__main__":
    main()