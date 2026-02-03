
from datetime import datetime, timedelta, timezone
import hashlib
import json
import pickle
import sys
from urllib.parse import quote_plus
import os
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

def make_key_alpha(config, alpha_name, fee, start, end, stop_loss=0, gen=None,source=None,overnight=False,cut_time=None):
    identity = {
        "config": config,
        "alpha_name": alpha_name,
        "fee": fee,
        "start": start,
        "end": end,
        "stop_loss": stop_loss,
        "gen": gen,
    }
    if source is not None:
        identity["source"] = source
    if overnight:
        identity["overnight"] = overnight
    if cut_time is not None:
        identity["cut_time"] = cut_time
    identity_str = json.dumps(identity, sort_keys=True)
    return hashlib.md5(identity_str.encode()).hexdigest()

def make_key_mega(configs, alpha_name, fee, start, end, stop_loss=0, gen=None):
    identity = {
        "config": sorted(configs),
        "alpha_name": alpha_name,
        "fee": fee,
        "start": start,
        "end": end,
        "stop_loss": stop_loss,
        "gen": gen,
    }
    identity_str = json.dumps(identity, sort_keys=True)
    return hashlib.md5(identity_str.encode()).hexdigest()

def load_dic_freqs(source,overnight=False):
    if source == "dollar_bar":
        fn = "/home/ubuntu/nevir/gen/dic_freqs_dollar_bar.pickle"
    elif source == "volume_bar":
        fn = "/home/ubuntu/nevir/gen/dic_freqs_volume_bar.pickle"
    elif source == "ha":
        fn = "/home/ubuntu/nevir/gen/dic_freqs_ha.pickle"
    elif source == "ha_confirm":
        fn = "/home/ubuntu/nevir/gen/dic_freqs_comfirm_ha.pickle"
    else:  
        fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    with open(fn, 'rb') as file:
        dic_freqs = pickle.load(file)
    if overnight:
        for freq in dic_freqs.keys():
            df = dic_freqs[freq].copy()
            
            df.loc[df['executionTime'] == '14:45:00', 'exitPrice'] = df['open'].shift(-2)
            df['priceChange'] = df['exitPrice'] - df['entryPrice']
            
            dic_freqs[freq] = df
    return dic_freqs
    
import logging
UTC_PLUS_7 = timezone(timedelta(hours=7))

class UTC7Formatter(logging.Formatter):
    """Formatter ép timestamp sang UTC+7."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, UTC_PLUS_7)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_logger(alpha_name):
    log_file = f"/home/ubuntu/nevir/auto/logs/{alpha_name}.log"

    logger = logging.getLogger(alpha_name)
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
        payload = {"chat_id": CHAT_ID, "text": text}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Telegram error {r.status_code}: {r.text}")
        else:
            print("📨 Đã gửi Telegram thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khi gửi Telegram: {e}")