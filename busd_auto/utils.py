
from datetime import datetime, timedelta, timezone
import hashlib
import json
import pickle
import sys
import time
from urllib.parse import quote_plus
import os

import numpy as np
import pandas as pd
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


def make_key(config, data_start, data_end,source):
    identity = {
        "config": config,
        "data_start": data_start,
        "data_end": data_end,
        "source": source
    }
    return hashlib.md5(json.dumps(identity, sort_keys=True).encode()).hexdigest()
def sanitize_for_bson(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: sanitize_for_bson(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_bson(x) for x in obj]
        return obj
def calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")
    total_days = (end_date - start_date).days
    working_days = total_days * workdays_per_year / 365.0
    return round(working_days)

def load_data(start, end, source):
    df = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    df = df[(df["day"] >= start) & (df["day"] <= end)]
    df["based_col"] = 0
    for col in source.split("_"):
        if col == "hose500": df["based_col"] += df["aggBusd"]
        elif col == "fhose500": df["based_col"] += df["aggFBusd"]
        elif col == "vn30": df["based_col"] += df["aggBusdVn30"]
        elif col == "fvn30": df["based_col"] += df["aggFBusdVn30"]
    df = df[df["based_col"].diff() != 0]
    df = df[df["time_int"] >= 10091500]
    df["priceChange"] = df.groupby("day")["last"].diff(1).shift(-1).fillna(0)
    return df
    
import logging
UTC_PLUS_7 = timezone(timedelta(hours=7))

class UTC7Formatter(logging.Formatter):
    """Formatter ép timestamp sang UTC+7."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, UTC_PLUS_7)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_logger(name):
    log_file = f"/home/ubuntu/nevir/busd_auto/logs/{name}.log"

    logger = logging.getLogger(name)
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