from datetime import datetime
import multiprocessing as mp
from itertools import product
import pickle
import re
import sys
import numpy as np
import hashlib, json, time
from pymongo import MongoClient
from bson import ObjectId
from auto.backtest import ScanParams, worker_task_batch
from auto.utils import get_mongo_uri, load_dic_freqs, make_key_alpha, send_telegram_message, setup_logger,insert_batch
from gen.alpha_func_lib import Domains

def gen_strategies(strategies,alpha_name,gen,fee):
    lst = []
    for config in strategies:
        if gen == "1_1":
            freq, threshold, halflife, *rest = config.split("_")
            freq, threshold, halflife = int(freq), float(threshold), float(halflife)
            factor = float(rest[1]) if len(rest) > 1 else None
            window = int(rest[0]) if rest else None
            gen_params = {
                "threshold": threshold,
                "halflife": halflife
            }
            params = {}
            if factor is not None:
                params["factor"] = factor
            if window is not None:
                params["window"] = window
        elif gen == "1_2":
            if alpha_name == "alpha_075":
                freq, upper, lower, *rest = config.split("_")
                freq, upper, lower = int(freq), float(upper), float(lower)
                params = {}
                window = int(rest[0]) if rest else None
                window_corr_vwap = float(rest[1]) if len(rest) >= 2 else None
                window_corr_volume = float(rest[2]) if len(rest) >= 3 else None
                if window is not None:
                    params["window"] = window
                if window_corr_vwap is not None:
                    params["window_corr_vwap"] = window_corr_vwap
                if window_corr_volume is not None:
                    params["window_corr_volume"] = window_corr_volume
                gen_params = {
                    "upper": upper,
                    "lower": lower
                }
            freq, upper, lower, *rest = config.split("_")
            freq, upper, lower = int(freq), float(upper), float(lower)
            params = {}
            window = int(rest[0]) if rest else None
            factor = float(rest[1]) if len(rest) >= 2 else None
            if window is not None:
                params["window"] = window
            if factor is not None:
                params["factor"] = factor
            gen_params = {
                "upper": upper,
                "lower": lower
            }
        elif gen == "1_3":
            freq, score, entry, exit, *rest = config.split("_")
            freq, score, entry, exit = int(freq), int(score), int(entry), int(exit)
            params = {}
            window = int(rest[0]) if rest else None
            factor = float(rest[1]) if len(rest) >= 2 else None
            if window is not None:
                params["window"] = window
            if factor is not None:
                params["factor"] = factor
            gen_params = {
                "score":score,
                "entry":entry,
                "exit":exit
            }
        
        elif gen == "1_4":
            freq, entry, exit, smooth, *rest = config.split("_")
            freq, entry, exit, smooth = int(freq), float(entry), float(exit), int(smooth)
            params = {}
            window = int(rest[0]) if rest else None
            factor = float(rest[0]) if rest else None
            if window is not None:
                params["window"] = window
            if factor is not None:
                params["factor"] = factor
            gen_params = {
                "entry":entry,
                "exit":exit,
                "smooth":smooth
            }
            
        lst.append({"alphaName": alpha_name, "freq": freq, "fee": fee, "params": params , "cfg":config , **gen_params}) 
        
    return lst

    

def wfo(id):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    coll = alpha_db["wfo_results"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen")
    name = alpha_doc.get("name", "")
    logger = setup_logger(f"{name}_wfo")

    wfo = alpha_doc.get("wfo", {})
    params = wfo.get("params", {})
    freq = wfo.get("freq")
    fee = wfo.get("fee")
    
    period = wfo.get("period", [])
    
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()
    strategies = wfo.get("strategies",[])
    for p in period:
        p_start_time = time.time()
        insample = p.get("is", {})
        start = insample.split("-")[0]
        end = insample.split("-")[1]
        if len(strategies) > 5:
            scan_params = gen_strategies(strategies,alpha_name,gen,fee)
        else:
            scan_params = ScanParams(
                lst_alpha_names=[alpha_name],
                alpha_params=params,
                freq=freq,
                fee=fee,
                gen=gen,
            ).lst_reports
        list_keys = [
            make_key_alpha(
                config=cfg['cfg'],
                alpha_name=cfg["alphaName"],
                fee=cfg["fee"],
                start=start,
                end=end,
                gen=gen,
            )
            for cfg in scan_params
        ]
        exist_params = list(coll.find({"_id": {"$in": list_keys}}))
        exist_keys = {e["_id"] for e in exist_params}
        filtered_scan_params = [
            cfg for cfg, key in zip(scan_params, list_keys)
            if key not in exist_keys
        ]
        
        total = len(filtered_scan_params)
        logger.info(
            f"🚀 Running scan {start} -> {end} | "
            f"{total} parameters to run | {len(exist_keys)} existed & skipped"
        )

        alpha_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {
                "wfo.process": 0,
                "wfo.total": total,
                "wfo.started_at": datetime.now(),
                "wfo.status": "running"
            }}
        )
        if total == 0:
            logger.info("✅ Không có config mới để chạy WFO. Bỏ qua.")
            alpha_collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": {
                    "wfo.process": 0,
                    "wfo.total": total,
                    "wfo.started_at": datetime.now(),
                    "wfo.status": "done"
                }}
            )
            continue
        # ----------------- Chuẩn bị batch -----------------
        n_workers = 40
        max_batch_size = 1000

        # tính số worker hợp lý
        if total <= max_batch_size:
            n_workers_to_use = 1  # hoặc ít worker để tiết kiệm CPU
        else:
            n_workers_to_use = min(n_workers, (total + max_batch_size - 1) // max_batch_size)  # ceil(total / max_batch_size)

        batch_size_configs = (total + n_workers_to_use - 1) // n_workers_to_use  # ceil chia đều
        batches = [filtered_scan_params[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
        args_list = [(batch, dic_freqs, DIC_ALPHAS, gen, start, end) for batch in batches]
        logger.info(f"Chạy với {n_workers_to_use} processes, tổng {len(batches)} batches mỗi batch {batch_size_configs} configs.")

        temp_batch = []
        inserted = 0
        last_update = time.time()
        insert_batch_size = 500
        update_interval = 15  # giây
        # ----------------- Multiprocessing -----------------
        with mp.Pool(processes=n_workers_to_use) as pool:
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
                        {"$set": {"wfo.process": inserted}}
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
                "wfo.status": "done",
                "wfo.process": inserted,
                "wfo.finished_at": datetime.now(),
                "wfo.finished_in": time.time() - p_start_time
            }}
        )

        
        logger.info(f"🎯 Backtest hoàn tất cho {total} configs in {time.time() - p_start_time:.2f}s.")
    send_telegram_message(f"✅ WFO hoàn tất cho alpha: {alpha_name}")
    mongo_client.close()
    logger.info(f"🎯 WFO hoàn tất trong {time.time() - start_time:.2f}s.")

def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfo.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    wfo(_id)

if __name__ == "__main__":
    main()