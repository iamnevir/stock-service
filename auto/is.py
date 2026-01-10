from datetime import datetime
import multiprocessing as mp
import sys
import time
from pymongo import MongoClient
from bson import ObjectId
from auto.backtest import ScanParams, insert_batch, setup_logger, worker_task_batch
from auto.utils import get_mongo_uri, load_dic_freqs
from gen.alpha_func_lib import Domains


def insample(id):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    coll = alpha_db["is_results"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    alpha_name = alpha_doc.get("alpha_name", "")
    logger = setup_logger(f"{alpha_name}_is")

    ios = alpha_doc.get("ios", {})
    gen = alpha_doc.get("gen")
    start = ios.get("start")
    end = ios.get("end")
    params = ios.get("params", {})
    freq = ios.get("freq")
    fee = ios.get("fee")

    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()

    scan_params = ScanParams(
        lst_alpha_names=[alpha_name],
        alpha_params=params,
        freq=freq,
        fee=fee,
        gen=gen,
    ).lst_reports

    total = len(scan_params)
    logger.info(f"🚀 Running insample {start} => {end} with {total} parameter...")

    alpha_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "ios.process": 0,
            "ios.total": total,
            "ios.started_at": datetime.now(),
            "ios.status": "running"
        }}
    )

    # ----------------- Chuẩn bị batch -----------------
    n_workers = 4
    batch_size_configs = min(1000, total // n_workers)
    batches = [scan_params[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
    args_list = [(batch, dic_freqs, DIC_ALPHAS, gen, start, end) for batch in batches]
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
                    {"$set": {"ios.process": inserted}}
                )
                logger.info(f"⏳ Progress update: {inserted}/{total}")
                last_update = time.time()

    # ----------------- Insert phần còn lại -----------------
    if temp_batch:
        insert_batch(coll, temp_batch, batch_size=insert_batch_size)
        temp_batch.clear()

    logger.info(f"⏳ Final progress update: {inserted}/{total}")

    # ----------------- Hoàn tất -----------------
    alpha_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "ios.status": "done",
            "ios.process": inserted,
            "ios.finished_at": datetime.now(),
            "ios.finished_in": time.time() - start_time
        }}
    )

    mongo_client.close()
    logger.info(f"🎯 Chạy Insample hoàn tất cho {total} configs in {time.time() - start_time:.2f}s.")


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/envs/backtest/bin/python /home/ubuntu/nevir/auto/is.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    insample(_id)

if __name__ == "__main__":
    main()