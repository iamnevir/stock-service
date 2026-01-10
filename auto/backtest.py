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
from auto.utils import get_mongo_uri, sanitize_for_bson, make_key_alpha, load_dic_freqs, setup_logger, insert_batch
from gen.alpha_func_lib import Domains
from gen.core import Simulator

# ----------------- PARAM GENERATOR -----------------
class ScanParams:
    def __init__(self, lst_alpha_names, alpha_params, freq, fee, gen=None):
        self.alpha_params = alpha_params or {}
        self.min_freq, self.max_freq, self.step_freq = freq["start"], freq["end"], freq["step"]
        self.fee, self.gen = fee, gen
        self.lst_reports = []

        if lst_alpha_names:
            self.lst_reports = self.gen_lst_reports(lst_alpha_names)

    @staticmethod
    def gen_list():
        return [round(i * 0.1, 1) for i in range(1, 11)]

    def gen_band_list(self):
        vals = [round(i * 0.1, 1) for i in range(1, 10)]
        return [(x, y) for x, y in product(vals + [1.0], vals) if x > y]

    def gen_smooth_list(self):
        vals = self.gen_list()
        return [(x, y) for x, y in product(vals, vals) if x > y]
    
    def gen_params_combinations(self):
        priority = {
            "window": 0,
            "window_corr_vwap": 1,
            "window_corr_volume": 2,
        }

        # Lấy các keys và sắp xếp theo priority trước, sau đó theo tên cho ổn định
        keys = sorted(self.alpha_params.keys(), key=lambda k: (priority.get(k, 999), k))

        names, ranges = [], []

        for k in keys:
            v = self.alpha_params[k]

            if isinstance(v, dict):
                values = np.arange(v["start"], v["end"] + v["step"], v["step"])
            elif isinstance(v, list):
                values = v
            else:
                values = [v]

            names.append(k)
            ranges.append(values)

        return [dict(zip(names, c)) for c in product(*ranges)]

    def gen_score_list(self):
        scores = [3, 4, 5, 6, 7, 8]
        entries = [1, 2, 3, 4]
        exits = [0, 1, 2]
        return [(s, e1, e2) for s, e1, e2 in product(scores, entries, exits) if e1 > e2]

    def gen_lst_reports(self, lst_alpha_names):
        freqs = range(self.min_freq, self.max_freq + self.step_freq, self.step_freq)
        params = self.gen_params_combinations()
        gl = self.gen_list
        map_gen = {
            "1_1": (["threshold", "halflife"], [gl(), [0.0] if "factor" in self.alpha_params and ("hose" in lst_alpha_names[0] or "vn30" in lst_alpha_names[0]) else [0.0]+gl()]),
            "1_2": (["upper_lower"], [self.gen_band_list()]),
            "1_3": (["score_entry_exit"], [self.gen_score_list()]),
            "1_4": (["entry_exit", "smooth"], [self.gen_smooth_list(), [1,2,3,4]]),
        }

        if self.gen not in map_gen:
            raise ValueError(f"Unsupported gen mode: {self.gen}")

        keys, values = map_gen[self.gen]
        combos = product(lst_alpha_names, freqs, *values, params)
        fee = self.fee
        gen = self.gen

        reports = []
        append = reports.append
        for c in combos:
            base = {"alphaName": c[0], "freq": c[1], "fee": fee, "params": c[-1]}

            if gen == "1_1":
                cfg = f"{c[1]}_{c[2]}_{c[3]}"
                for k, v in c[-1].items():
                    cfg += f"_{v}"
                base.update({"threshold": c[2], "halflife": c[3], "cfg": cfg})

            elif gen == "1_2":
                upper, lower = c[2]
                cfg = f"{c[1]}_{upper}_{lower}"
                for k, v in c[-1].items():
                    cfg += f"_{v}"
                base.update({"upper": upper, "lower": lower, "cfg": cfg})
            
            elif gen == "1_3":
                score, entry, exit = c[2]
                cfg = f"{c[1]}_{score}_{entry}_{exit}"
                for k, v in c[-1].items():
                    cfg += f"_{v}"
                base.update({"score": score, "entry": entry, "exit": exit, "cfg": cfg})
                
            elif gen == "1_4":
                entry, exit = c[2]
                cfg = f"{c[1]}_{entry}_{exit}_{c[3]}"
                for k, v in c[-1].items():
                    cfg += f"_{v}"
                base.update({"entry": entry, "exit": exit, "smooth": c[3], "cfg": cfg})

            append(base)
        return reports

# ----------------- BACKTEST CORE -----------------
def run_single_backtest(config, dic_freqs, DIC_ALPHAS, gen=None, start=None, end=None):
    gen_params = {}
    if gen == "1_3":
        gen_params["score"] = config["score"]
        gen_params["entry"] = config["entry"]
        gen_params["exit"] = config["exit"]
    elif gen == "1_2":
        gen_params["upper"] = config["upper"]
        gen_params["lower"] = config["lower"]
    elif gen == "1_1":
        gen_params["halflife"] = config["halflife"]
        gen_params["threshold"] = config["threshold"]
    elif gen == "1_4":
        gen_params["entry"] = config["entry"]
        gen_params["exit"] = config["exit"]
        gen_params["smooth"] = config["smooth"]
    bt = Simulator(
        alpha_name=config["alphaName"],
        freq=config["freq"],
        gen_params=gen_params,
        fee=config["fee"],
        df_alpha=dic_freqs[config["freq"]].copy(),
        params=config.get("params", {}),
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=None,
        gen=gen,
        start=start,
        end=end,
    )
    bt.compute_signal()
    bt.compute_position()
    bt.compute_tvr_and_fee()
    bt.compute_profits()
    bt.compute_performance(start=start, end=end)

    rpt = sanitize_for_bson(bt.report.copy())
    params = config.get("params", {})
    for k, v in params.items():
        rpt[f"param_{k}"] = round(float(v),4)
    rpt['strategy'] = config['cfg']
    rpt['_id'] = make_key_alpha(config['cfg'], config["alphaName"], config["fee"], start, end, gen=gen)
    keys_to_delete = ["aroe", "cdd", "cddPct","lastProfit","max_loss","max_gross","num_trades"]
    for key in keys_to_delete:
        rpt.pop(key, None)
    return rpt


   
def worker_task_batch(args):
    """
    Mỗi worker xử lý 1 batch (1000 configs)
    """
    batch_configs, dic_freqs, DIC_ALPHAS, gen, start, end = args
    results = []
    for cfg in batch_configs:
        rpt = run_single_backtest(cfg, dic_freqs, DIC_ALPHAS, gen, start, end)
        results.append(rpt)
    return results

def importdb(df,id):
    try:
        mongo_client = MongoClient(get_mongo_uri())
        alpha_db = mongo_client["alpha"]
        coll = alpha_db["backtest_results"]
        alpha_collection = alpha_db["alpha_collection"]
        alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
        if not alpha_doc:
            print("❌ Không tìm thấy alpha_collection với id này.")
            return False
        gen = alpha_doc.get("gen")
        alpha_name = alpha_doc.get("alpha_name", "")
        backtest_params = alpha_doc.get("backtest", {})
        start = backtest_params.get("start")
        end = backtest_params.get("end")
        fee = float(backtest_params.get("fee",0.175))
        if not start or not end or not fee:
            print("❌ Thiếu thông tin start, end hoặc fee trong backtest params.")
            return False
        df.columns = [col.strip() for col in df.columns]
        records = []
        for _, row in df.iterrows():
            
            record = {
                "_id": make_key_alpha(row["Strategy"], alpha_name, fee, start, end,0, gen),
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
        
        insert_batch(coll, records, batch_size=500)
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
            min_window = int(df['Param: Window'].min())
            max_window = int(df['Param: Window'].max())
            window_step = int(df['Param: Window'][1] - df['Param: Window'][0]) if len(df['Param: Window']) >1 else 0
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
        alpha_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {
                "backtest.freq": {"start": min_freq, "end": max_freq, "step": step_freq},
                "backtest.status": "done",
                "backtest.process": len(records),
                "backtest.total":  len(records),
                "backtest.finished_at":  datetime.now(),
                "backtest.params": params
            }}
        )
        mongo_client.close()
        return True
    except Exception as e:
        print(f"Error importing to DB: {e}")
        return False

def backtest(id):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    coll = alpha_db["backtest_results"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    name = alpha_doc.get("name", "")
    logger = setup_logger(name)
    alpha_name = alpha_doc.get("alpha_name", "")
    backtest_params = alpha_doc.get("backtest", {})
    gen = alpha_doc.get("gen")
    start = backtest_params.get("start")
    end = backtest_params.get("end")
    params = backtest_params.get("params", {})
    freq = backtest_params.get("freq")
    fee = backtest_params.get("fee")

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
        {"_id": ObjectId(id)},
        {"$set": {
            "backtest.process": 0,
            "backtest.total": total,
            "backtest.started_at": datetime.now(),
            "backtest.status": "running"
        }}
    )

    # ----------------- Chuẩn bị batch -----------------
    n_workers = 1
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
                    {"$set": {"backtest.process": inserted}}
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
            "backtest.process": inserted,
            "backtest.finished_at": datetime.now(),
            "backtest.finished_in": time.time() - start_time
        }}
    )

    mongo_client.close()
    logger.info(f"🎯 Backtest hoàn tất cho {total} configs in {time.time() - start_time:.2f}s.")


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/backtest.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    backtest(_id)

if __name__ == "__main__":
    main()