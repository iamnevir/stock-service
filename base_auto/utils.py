
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import product
import json
import pickle
import sys
from urllib.parse import quote_plus
import os
import pandas as pd
from pymongo.errors import BulkWriteError
import numpy as np
USER_MONGO = os.environ.get("USER_MONGO","administrator")
PASS_MONGO = os.environ.get("PASS_MONGO","Adm1n@!#$123")

SERVER_MAP = {
    "local": "localhost",
    "mgc3": "103.253.20.31",
}
def get_mongo_uri(server="local") -> str:
    server_host = SERVER_MAP.get(server, "localhost")

    # user / pass theo server
    if server == "mgc3":
        username = "stocknora"
        password = "St0ck@#!1123"
    else:
        username = USER_MONGO
        password = PASS_MONGO

    # escape ký tự đặc biệt
    username_escaped = quote_plus(username)
    password_escaped = quote_plus(password)

    return (
        f"mongodb://{username_escaped}:{password_escaped}"
        f"@{server_host}:27017/"
        f"?authSource=admin&authMechanism=SCRAM-SHA-256"
    )

def sanitize_for_bson(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: sanitize_for_bson(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_bson(x) for x in obj]
        return obj

def make_key_base(config, base_name, fee, start, end, source, gen, stop_loss=0):
    identity = {
        "config": config,
        "base_name": base_name,
        "fee": fee,
        "start": start,
        "end": end,
        "stop_loss": stop_loss,
        "gen": gen,
        "source":source
    }
    identity_str = json.dumps(identity, sort_keys=True)
    return hashlib.md5(identity_str.encode()).hexdigest()

def make_key_mega(configs, base_name, fee, start, end, stop_loss=0, gen=None):
    identity = {
        "config": sorted(configs),
        "base_name": base_name,
        "fee": fee,
        "start": start,
        "end": end,
        "stop_loss": stop_loss,
        "gen": gen,
    }
    identity_str = json.dumps(identity, sort_keys=True)
    return hashlib.md5(identity_str.encode()).hexdigest()

def load_dic_freqs(source):
    fn = "/home/ubuntu/nevir/gen_spot/dic_freqs_alpha_base.pkl"
    df_dict = pd.read_pickle(fn)
    mapping = {
        "hose500": "hose_Busd",
        "fhose500": "F_hose_Busd",
        "vn30": "vn30_Busd",
        "fvn30": "F_vn30_Busd"
    }

    for key in df_dict.keys():
        df = df_dict[key].copy()
        df["based_col"] = 0

        for col in source.split("_"):
            if col in mapping:
                df["based_col"] += df[mapping[col]]

        df_dict[key] = df

    return df_dict
    
import logging
UTC_PLUS_7 = timezone(timedelta(hours=7))

class UTC7Formatter(logging.Formatter):
    """Formatter ép timestamp sang UTC+7."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, UTC_PLUS_7)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_logger(base_name):
    log_folder = "/home/ubuntu/nevir/base_auto/logs/"
    os.makedirs(log_folder, exist_ok=True)
    log_file = f"{log_folder}{base_name}.log"

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)

    formatter = UTC7Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def insert_batch(coll, records, batch_size=1000):
    for i in range(0, len(records), batch_size):
        batch = [r for r in records[i:i+batch_size]]

        if not batch:
            continue

        try:
            coll.insert_many(batch, ordered=False)
        except BulkWriteError:
            # ignore duplicate key
            pass
    
import requests

# === CONFIG TELEGRAM ===


def send_telegram_message(text: str):
    TELEGRAM_TOKEN = "7386833040:AAE5Z05xuKDu1w9y3oar_CxKpoJMme-8zu0"
    CHAT_ID = "-4874762893"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Telegram error {r.status_code}: {r.text}")
        else:
            print("📨 Đã gửi Telegram thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khi gửi Telegram: {e}")
        
    
class ScanParams:
    """
    Ultra-optimized generator for base parameter scans.
    """

    def __init__(self, base_name, base_params, freq, fee, gen=None):
        self.base_params = base_params or {}
        self.min_freq, self.max_freq, self.step_freq = freq['start'], freq['end'], freq['step']
        self.fee, self.gen= fee, gen
        self.lst_reports = []
        

        self.lst_reports = self.gen_lst_reports(base_name)

    # ---------- Param utilities ----------
    @staticmethod
    def gen_list():
        return [round(i * 0.1, 1) for i in range(1, 10)]

    def gen_band_list(self):
        vals = self.gen_list()
        return [(x, y) for x, y in product(vals + [1.0], vals) if x > y]
    
    def gen_smooth_list(self):
        vals = [round(i * 0.1, 1) for i in range(1, 10)]
        return [(x, y) for x, y in product(vals, vals) if x > y]
    
    def gen_score_list(self):
        scores = [3, 4, 5, 6, 7, 8]
        entries = [1, 2, 3, 4]
        exits = [0, 1, 2]
        return [(s, e1, e2) for s, e1, e2 in product(scores, entries, exits) if e1 > e2]
        
    def gen_params_combinations(self):
        priority = {
            "window": 0,
            "window_corr_vwap": 1,
            "window_corr_volume": 2,
        }

        # Lấy các keys và sắp xếp theo priority trước, sau đó theo tên cho ổn định
        keys = sorted(self.base_params.keys(), key=lambda k: (priority.get(k, 999), k))

        names, ranges = [], []

        for k in keys:
            v = self.base_params[k]

            if isinstance(v, dict):
                values = np.arange(v["start"], v["end"] + v["step"], v["step"])
            elif isinstance(v, list):
                values = v
            else:
                values = [v]

            names.append(k)
            ranges.append(values)

        return [dict(zip(names, c)) for c in product(*ranges)]

    # ---------- Core generator ----------
    def gen_lst_reports(self, base_name):
        freqs = range(self.min_freq, self.max_freq + self.step_freq,self.step_freq)
        params = self.gen_params_combinations()
        gl = self.gen_list

        # map structure: gen -> (extra_fields, value_generators)
        map_gen = {
            "1_1": (["threshold"], [gl()]),
            "1_2": (["upper_lower"], [self.gen_band_list()]),
            "1_3": (["score_entry_exit"], [self.gen_score_list()]),
            "1_4": (["entry_exit", "smooth"], [self.gen_smooth_list(), [1,2,3,4]]),
        }

        if self.gen not in map_gen:
            raise ValueError(f"Unsupported gen mode: {self.gen}")

        keys, values = map_gen[self.gen]
        combos = product([base_name], freqs, *values, params)
        fee = self.fee
        gen = self.gen

        # Use list comprehension + tuple unpacking for speed
        reports = []
        append = reports.append
        for c in combos:
            base = {"baseName": c[0], "freq": c[1], "fee": fee, "params": c[-1]}

            if gen == "1_1":
                cfg = f"{c[1]}_{c[2]}"
                for k, v in c[-1].items():
                    cfg += f"_{round(v,3)}"
                
                base.update({"threshold": c[2], "cfg": cfg})

            elif gen == "1_2":
                upper, lower = c[2]
                cfg = f"{c[1]}_{upper}_{lower}"
                for k, v in c[-1].items():
                    cfg += f"_{round(v,3)}"
                base.update({"upper": upper, "lower": lower, "cfg": cfg})
            
            elif gen == "1_3":
                score, entry, exit = c[2]
                cfg = f"{c[1]}_{score}_{entry}_{exit}"
                for k, v in c[-1].items():
                    cfg += f"_{round(v,3)}"
                base.update({"score": score, "entry": entry, "exit": exit, "cfg": cfg})
                
            elif gen == "1_4":
                entry, exit = c[2]
                cfg = f"{c[1]}_{entry}_{exit}_{c[3]}"
                for k, v in c[-1].items():
                    cfg += f"_{round(v,3)}"
                base.update({"entry": entry, "exit": exit, "smooth": c[3], "cfg": cfg})
            append(base)

        return reports