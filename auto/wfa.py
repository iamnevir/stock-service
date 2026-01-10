from datetime import datetime
import multiprocessing as mp
from itertools import product
from multiprocessing import shared_memory
import pickle
import re
import sys
import traceback
import numpy as np
import hashlib, json, time
import pandas as pd
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from auto.backtest import ScanParams, worker_task_batch
from auto.utils import get_mongo_uri, load_dic_freqs, make_key_alpha, send_telegram_message, setup_logger
from auto.wfo import gen_strategies
from busd_auto.is_correlation import insert_batch
from gen.alpha_func_lib import Domains


def import_wfa(df,id,start,end):
    try:
        mongo_client = MongoClient(get_mongo_uri())
        alpha_db = mongo_client["alpha"]
        coll = alpha_db["wfa_results"]
        alpha_collection = alpha_db["alpha_collection"]
        alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
        if not alpha_doc:
            print("❌ Không tìm thấy alpha_collection với id này.")
            return False
        gen = alpha_doc.get("gen")
        alpha_name = alpha_doc.get("alpha_name", "")
        source = alpha_doc.get("source", None)
        fa = None
        wfa_list = alpha_doc.get("wfa", [])
        for item in wfa_list:
            if item.get("is", {}).get("start") == start and item.get("is", {}).get("end") == end:
                fa = item
                break
        fee = fa.get("fee")
        df.columns = [col.strip() for col in df.columns]
        print(f"Importing WFA results to DB... {len(df)} records found.")
        records = []
        for _, row in df.iterrows():
            
            record = {
                "_id": make_key_alpha(row["Strategy"], alpha_name, fee, start, end,0, gen,source),
                "alphaName": alpha_name,
                "strategy": row["Strategy"],
                "sharpe": float(row["Sharpe Ratio"]),
                "freq": int(row["Frequency"]),
                "mdd": float(row["MDD"]),
                "mddPct": float(row["MDD (%)"]),
                "ppc": float(row["PPC"]),
                "tvr": float(row["TVR"]),
                "netProfit": float(row["Net Profit"]),
                "start": start,
                "end": end,
                "fee": fee,
            }
            if gen == "1_1":
                record["halflife"] = float(row["Halflife"])
                record["threshold"] = float(row["Threshold"])
            elif gen == "1_2":
                record["upper"] = float(row["Upper"])
                record["lower"] = float(row["Lower"])
            elif gen == "1_3":
                record["score"] = float(row["Score"])
                record["entry"] = float(row["Entry"])
                record["exit"] = float(row["Exit"])
            elif gen == "1_4":
                record["entry"] = float(row["Entry"])
                record["exit"] = float(row["Exit"])
                record["smooth"] = float(row["Smooth"])
            if "Param: Factor" in row.index:
                record["param_factor"] = float(row["Param: Factor"])
            if "Param: Window" in row.index:
                record["param_window"] = int(row["Param: Window"])
            if "Param: Window" in row.index:
                record["param_window"] = int(row["Param: Window"])
            if "Param: Window_Corr_Vwap" in row.index:
                record["param_window_corr_vwap"] = int(row["Param: Window_Corr_Vwap"])
            if "Param: Window_Corr_Volume" in row.index:
                record["param_window_corr_volume"] = int(row["Param: Window_Corr_Volume"])
            records.append(record)
        
        
        BATCH_SIZE = 1000

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]

            requests = [
                UpdateOne(
                    {"_id": r["_id"]},
                    {"$setOnInsert": r},
                    upsert=True
                )
                for r in batch
            ]

            coll.bulk_write(requests, ordered=False)

        min_freq = int(df['Frequency'].min())
        max_freq = int(df['Frequency'].max())
        # step_freq = abs(int(df['Frequency'][1] - df['Frequency'][0])) if len(df['Frequency']) >1 else 0
        step_freq = 1
        params= {}

        if "Param: Factor" in df.columns:
            col = df['Param: Factor'].astype(float)
            min_factor = float(col.min())
            max_factor = float(col.max())

            uniq = np.sort(col.unique())
            if len(uniq) > 1:
                diffs = np.diff(uniq)
                positive_diffs = diffs[diffs > 0]
                factor_step = round(float(positive_diffs.min()), 4) if len(positive_diffs) > 0 else 0.0
            else:
                factor_step = 0.0

            params["factor"] = {
                "start": min_factor,
                "end": max_factor,
                "step": factor_step
            }


        if "Param: Window" in df.columns:
            col = df['Param: Window'].astype(int)
            min_window = int(df['Param: Window'].min())
            max_window = int(df['Param: Window'].max())
            uniq = np.sort(col.unique())
            if len(uniq) > 1:
                diffs = np.diff(uniq)
                positive_diffs = diffs[diffs > 0]
                window_step = int(positive_diffs.min()) if len(positive_diffs) > 0 else 0
            else:
                window_step = 0

            params["window"] = {
                "start": min_window,
                "end": max_window,
                "step": window_step
            }
        if "Param: Window_Corr_Vwap" in df.columns:
            col = df['Param: Window_Corr_Vwap'].astype(int)
            min_window = int(col.min())
            max_window = int(col.max())

            uniq = np.sort(col.unique())
            if len(uniq) > 1:
                diffs = np.diff(uniq)
                positive_diffs = diffs[diffs > 0]
                window_step = int(positive_diffs.min()) if len(positive_diffs) > 0 else 0
            else:
                window_step = 0

            params["window_corr_vwap"] = {
                "start": min_window,
                "end": max_window,
                "step": window_step
            }

        # --- Param: Window_Corr_Volume ---
        if "Param: Window_Corr_Volume" in df.columns:
            col = df['Param: Window_Corr_Volume'].astype(int)
            min_window = int(col.min())
            max_window = int(col.max())

            uniq = np.sort(col.unique())
            if len(uniq) > 1:
                diffs = np.diff(uniq)
                positive_diffs = diffs[diffs > 0]
                window_step = int(positive_diffs.min()) if len(positive_diffs) > 0 else 0
            else:
                window_step = 0

            params["window_corr_volume"] = {
                "start": min_window,
                "end": max_window,
                "step": window_step
            }
        print(f"Inserted {len(records)} WFA records into wfa_results collection.")
        alpha_collection.update_one(
            {
                "_id": ObjectId(id),
                "wfa.is.start": start,
                "wfa.is.end": end
            },
            {
                "$set": {
                    "wfa.$.status": "done",
                    "wfa.$.freq": {"start": min_freq, "end": max_freq, "step": step_freq},
                    "wfa.$.process": len(records),
                    "wfa.$.total": len(records),
                    "wfa.$.finished_at":  datetime.now(),
                    "wfa.$.params": params
                }
            }
        )
        print("✅ Import WFA vào DB thành công.")
        mongo_client.close()
        return True
    except Exception as e:
        print(f"Error importing to DB: {e}")
        print(traceback.format_exc())
        return False   

def wfa(_id,start=None,end=None):
    try:
        start_time = time.time()
        mongo_client = MongoClient(get_mongo_uri())
        alpha_db = mongo_client["alpha"]
        alpha_collection = alpha_db["alpha_collection"]
        coll = alpha_db["wfa_results"]
        alpha_doc = alpha_collection.find_one({"_id": ObjectId(_id)})
        if not alpha_doc:
            print("❌ Không tìm thấy alpha_collection với id này.")
            return
        wfa_list = alpha_doc.get("wfa", [])
        fa = None
        for item in wfa_list:
            if item.get("is", {}).get("start") == start and item.get("is", {}).get("end") == end:
                if item.get("status") == "done":
                    print("✅ WFA đã được xử lý trước đó.")
                    return
                fa = item
                break
        name = alpha_doc.get("name", "")
        logger = setup_logger(name)
        alpha_name = alpha_doc.get("alpha_name", "")
        gen = alpha_doc.get("gen")
        _is = fa.get("is", {})
        start = _is.get("start")
        end = _is.get("end")
        params = fa.get("params", {})
        freq = fa.get("freq")
        fee = fa.get("fee")
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
        logger.info(f"🚀 Running backtest with {total} parameter combinations...")

        alpha_collection.update_one(
            {"_id": ObjectId(id),"wfa.is.start": start,"wfa.is.end": end},
            {"$set": {
                "wfa.$.process": 0,
                "wfa.$.total": total,
                "wfa.$.started_at": datetime.now(),
                "wfa.$.status": "running"
            }}
        )

        # ----------------- Chuẩn bị batch -----------------
        n_workers = 20
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
                        {"_id": ObjectId(id),"wfa.is.start": start,"wfa.is.end": end},
                        {"$set": {"wfa.$.process": inserted}}
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
                "backtest.status": "done",
                "wfa.$.process": inserted,
                "wfa.$.finished_at": datetime.now(),
                "wfa.$.finished_in": time.time() - start_time
            }}
        )

        mongo_client.close()
        logger.info(f"🎯 Backtest hoàn tất cho {total} configs in {time.time() - start_time:.2f}s.")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý WFA: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfa.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    wfa(_id)

if __name__ == "__main__":
    main()