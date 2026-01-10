import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from urllib.parse import quote_plus

import numpy as np
from pymongo import MongoClient
from bson import ObjectId

from gen.core import Simulator 
from gen.alpha_func_lib import Domains



def get_mongo_uri():
    username_escaped = quote_plus("administrator")
    password_escaped = quote_plus("Adm1n@!#$123")
    return f"mongodb://{username_escaped}:{password_escaped}@localhost:27017/?authSource=admin"


def get_mongo_collection(collection_name="stock"):
    client = MongoClient(get_mongo_uri())
    return client["gen1_2"][collection_name]


def make_key_corr(config, fee, start, end, alpha_name,gen="gen1_2"):
    """Tạo khóa định danh duy nhất từ config và tham số"""
    identity = {
        "config": config,
        "fee": fee,
        "start": start,
        "end": end,
        "alpha_name": alpha_name,
    }
    if gen == "gen1_1":
        identity["gen"] = "gen1_1"
    identity_str = json.dumps(identity, sort_keys=True)
    return hashlib.md5(identity_str.encode()).hexdigest()


def load_dic_freqs():
    with open("/home/ubuntu/duy/new_strategy/gen1_2/dic_freqs.pkl", "rb") as file:
        return pickle.load(file)


def run_single_backtest(config, dic_freqs, DIC_ALPHAS, df_tick=None, alpha_name="",gen="", start="", end=""):
    def convert_np(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(x) for x in obj]
        return obj
    fee = 0.175
    _id = make_key_corr(config, fee, start, end, alpha_name,gen)
    coll = get_mongo_collection()
    if coll.find_one({"_id": _id}):
        return None
    
    if gen == "gen1_1":
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
    elif gen == "gen1_2":
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
    elif gen == "gen1_3":
        freq, score, entry, exit, *rest = config.split("_")
        freq, score, entry, exit = int(freq), float(score), float(entry), float(exit)
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
        
    bt = Simulator(
        alpha_name=alpha_name,
        freq=freq,
        gen_params=gen_params,
        fee=fee,
        df_alpha=dic_freqs[freq],
        params=params,
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=df_tick,
        start=start,
        end=end,
        gen=gen.replace("gen",""),
    )
    
    bt.compute_signal()
    bt.compute_position()
    bt.compute_tvr_and_fee()
    bt.compute_profits()
    bt.compute_performance(start,end)
    bt.compute_df_trade()
    report = bt.report

    df_trade = bt.df_trade.reset_index(drop=True).to_dict(orient="records")
    coll.replace_one(
        {"_id": _id},
        {"_id": _id, "df_trade": df_trade, "report":convert_np(report) , "config": config},
        upsert=True,
    )
    return report


def run_backtest_batch(configs_batch, dic_freqs, DIC_ALPHAS, df_tick=None, alpha_name="", gen="", start="", end=""):
    results = []
    for cfg in configs_batch:
        res = run_single_backtest(cfg, dic_freqs, DIC_ALPHAS, df_tick, alpha_name=alpha_name, gen=gen, start=start, end=end)
        if res:
            results.append(res)
    return results


def split_list_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


class MultiThreadScanner:
    def __init__(self, n_workers=None, use_processes=True, batch_size=10, alpha_correl_coll=None, id=None):
        self.n_workers = n_workers or cpu_count()
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        self.batch_size = batch_size
        self.use_processes = use_processes
        self.alpha_correl_coll = alpha_correl_coll
        self.id = id

    def run_parallel_backtest(self, lst_configs, dic_freqs, DIC_ALPHAS, df_tick=None, alpha_name="", gen="", start="", end=""):
        print(f"Running with {self.n_workers} {'processes' if self.use_processes else 'threads'}")
        print(f"Total configs: {len(lst_configs)}")

        batches = list(split_list_into_batches(lst_configs, self.batch_size))
        print(f"Split into {len(batches)} batches")

        results = []
        with self.executor_class(max_workers=self.n_workers) as executor:
            future_map = {
                executor.submit(run_backtest_batch, batch, dic_freqs, DIC_ALPHAS, df_tick, alpha_name, gen, start, end): batch
                for batch in batches
            }
            for future in as_completed(future_map):
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)
                    self.alpha_correl_coll.update_one(
                        {"_id": self.id},
                        {"$inc": {"process.done": len(batch_results)}}
                    )
                    

        print(f"Completed {len(results)} new configs")
        return results

def run(id):
    client = MongoClient(get_mongo_uri())
    db = client['gen1_2']
    alpha_correl_coll = db["alpha_correl"]
    dic_freqs = load_dic_freqs()
    DIC_ALPHAS = Domains.get_list_of_alphas()
    alpha_correl = alpha_correl_coll.find_one({"_id": ObjectId(id)})
    lst_configs = alpha_correl.get("configs", [])
    alpha_name = alpha_correl.get("alpha_name", "")
    gen = alpha_correl.get("gen", "gen1_2")
    start = alpha_correl.get("start", "2024_01_01")
    end = alpha_correl.get("end", "2025_01_01")
    alpha_correl_coll.update_one(
        {"_id": id},
        {"$set": {"process": {"done": 0, "total": len(lst_configs)}}}
    )
    scanner = MultiThreadScanner(n_workers=20, use_processes=True, batch_size=100, alpha_correl_coll=alpha_correl_coll, id=id)
    scanner.run_parallel_backtest(lst_configs, dic_freqs, DIC_ALPHAS, alpha_name=alpha_name, gen=gen, start=start, end=end)



    

    
