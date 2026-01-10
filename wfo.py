import hashlib
import json
import multiprocessing
import os
import time
from urllib.parse import quote_plus
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from datetime import datetime
from typing import List, Tuple, Dict
from multiprocessing import shared_memory
from pymongo import MongoClient, ReplaceOne
from tqdm import tqdm

USERNAME = "administrator"
PASSWORD = "Adm1n@!#$123"
AUTH_DB = "admin"
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "stock_backtest"
GLOBAL_DF_MA = None
GLOBAL_DF_RAW = None
GLOBAL_SHM = None
GLOBAL_SHAPE = None

def create_mongo_uri():
    user = quote_plus(USERNAME)
    pwd = quote_plus(PASSWORD)
    return f"mongodb://{user}:{pwd}@{MONGO_HOST}:{MONGO_PORT}/?authSource={AUTH_DB}"

def make_key(config, fee, delay, data_start, data_end):
    identity = {
        "config": config,
        "fee": fee,
        "delay": delay,
        "data_start": data_start,
        "data_end": data_end,
    }
    return hashlib.md5(json.dumps(identity, sort_keys=True).encode()).hexdigest()
# --- Utils ---
def calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")
    total_days = (end_date - start_date).days
    working_days = total_days * workdays_per_year / 365.0
    return round(working_days)

# --- MA precomputation ---
def precompute_ma(
    df_alpha: pd.DataFrame,
    ma_lengths: set,
    based_col: str = "based_col",
    mode: str = "sma"  # Options: "sma", "ema", "dema", "wma"
) -> pd.DataFrame:
    df = df_alpha.copy()
    gb = df.groupby("day")
    ma_dfs = []

    def wma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.copy()
        weights = np.arange(1, length + 1)
        divisor = weights.sum()
        wma_values = np.convolve(series.fillna(0).to_numpy(), weights[::-1], mode='valid') / divisor
        result = np.full(series.shape, np.nan)
        result[length - 1:] = wma_values
        return pd.Series(result, index=series.index)

    for ma in ma_lengths:
        if mode == "sma":
            ma_series = gb[based_col].transform(lambda x: x.rolling(window=ma, min_periods=1).mean())
        elif mode == "ema":
            ma_series = gb[based_col].transform(lambda x: x.ewm(span=ma, adjust=True).mean())
        elif mode == "dema":
            def calc_dema(x):
                ema1 = x.ewm(span=ma, adjust=True).mean()
                ema2 = ema1.ewm(span=ma, adjust=True).mean()
                return 2 * ema1 - ema2
            ma_series = gb[based_col].transform(calc_dema)
        elif mode == "wma":
            ma_series = gb[based_col].transform(lambda x: wma(x, ma))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'sma', 'ema', 'dema', 'wma'.")
        
        ma_dfs.append(ma_series.rename(f"ma_{ma}"))
        

    ma_all = pd.concat(ma_dfs, axis=1)
    df = pd.concat([df, ma_all], axis=1)

    return df.copy()


# --- Position generation ---
def compute_single_position(df_ma: pd.DataFrame, ma1: int, ma2: int, threshold: int, exit_strength: int = 0, delay: int = 1) -> pd.Series:
    diff = df_ma[f"ma_{ma2}"] - df_ma[f"ma_{ma1}"]
    position = pd.Series(index=diff.index, dtype=np.float32)

    position[diff > threshold] = 1
    position[diff < -threshold] = -1

    if exit_strength != 0:
        take_profit = threshold * exit_strength / 100
        position[np.abs(diff) < take_profit] = 0

    position = position.shift(delay, fill_value=0)
    position[df_ma["time_int"] >= 10142930] = np.nan
    position[df_ma["time_int"] >= 10144500] = 0
    position[df_ma["time_int"] <= 10091500] = 0
    day_vals = df_ma['day'].values
    is_new_day = np.concatenate(([True], day_vals[1:] != day_vals[:-1]))
    new_day_indices = np.where(is_new_day)[0]
    pos_copy = position.values
    n = len(position)

    for start_idx in new_day_indices:
        end_idx = min(start_idx + delay, n)
        pos_copy[start_idx:end_idx] = 0.0
    position = pd.Series(pos_copy, index=position.index)
    position = position.ffill().fillna(0)
    return position

# --- Profit computation ---
def compute_profit(position: pd.Series, df_alpha: pd.DataFrame, fee: int) -> pd.DataFrame:
    df_alpha["position"] = position
    df_alpha["grossProfit"] = df_alpha["position"] * df_alpha["priceChange"]
    df_alpha["action"] = df_alpha["position"].diff().fillna(df_alpha["position"])
    df_alpha["turnover"] = df_alpha["action"].abs()
    df_alpha["fee"] = df_alpha["turnover"] * fee / 1000
    df_alpha["netProfit"] = df_alpha["grossProfit"] - df_alpha["fee"]

    df_1d = (
        df_alpha.groupby("day", sort=False, observed=True)
        .agg(
            grossProfit=("grossProfit", "sum"),
            turnover=("turnover", "sum"),
            netProfit=("netProfit", "sum")
        )
        .round(2)
    )
    df_1d["cumGrossProfit"] = df_1d["grossProfit"].cumsum()
    df_1d["cumTurnover"] = df_1d["turnover"].cumsum()
    df_1d["cumNetProfit"] = df_1d["netProfit"].cumsum()
    
    return df_1d

def psr(returns,sharpe):
    try:
        sample_size = len(returns)
        skewness = skew(returns)
        _kurtosis = kurtosis(returns, fisher=True, bias=False)
        sigma_sr = np.sqrt((1 - skewness*sharpe + (_kurtosis + 2)/4*sharpe**2) / (sample_size - 1))
        z = sharpe / sigma_sr
        return 0.5 * (1 + erf(z / sqrt(2))) * 100
    except Exception as e:
        return 0
    
# --- Reporting ---
def compute_report(df_1d, start, end):
    report = {
        "sharpe": None,
        "tvr": 0,
        "numdays": len(df_1d),
        "start_day": int(df_1d.index[0]),
        "end_day": int(df_1d.index[-1]),
    }
    equity = 300
    if "cdd" not in df_1d:
        cummax = df_1d["cumNetProfit"].cummax()
        df_1d["cdd"] = (cummax - df_1d["cumNetProfit"]).round(2)
        df_1d["cdd1"] = ((equity+ cummax - (equity + df_1d["cumNetProfit"])) / (equity + cummax) * 100).round(2)
        df_1d["mdd"] = df_1d["cdd"].cummax()
        df_1d["mdd1"] = df_1d["cdd1"].cummax()
        df_1d["cumNetProfit1"] = df_1d["cumNetProfit"] + equity

    tvr = round(df_1d["turnover"].mean(), 3)
    std = df_1d["netProfit"].std()
    
    if tvr == 0 or std == 0:
        return report
    daily_sharpe = df_1d["netProfit"].mean() / std
    ppc = round(df_1d['netProfit'].sum() / df_1d['turnover'].sum(), 3)
    profit_percent = df_1d["netProfit"].sum() / equity*100
    working_day = calculate_working_days(start,end)
    returns = df_1d['netProfit'] / 300
    winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
    total_profit = winning_profits.sum()
    shares = winning_profits / total_profit
    hhi = (shares ** 2).sum()
    report.update({
        "sharpe": float(round(df_1d["netProfit"].mean() / std * np.sqrt(working_day), 3)),
        "tvr": float(tvr),
        "psr": float(round(psr(returns, daily_sharpe), 2)),
        "hhi": float(round(hhi, 4)),
        "total_net_profit": float(round(df_1d["netProfit"].sum(), 2)),
        "mdd_percent": float(round(df_1d["mdd1"].values[-1], 2)),
        "ppc": float(round(ppc, 3)),
        "profit_percent": float(round(profit_percent, 2)),
    })

    return report 

# --- Master runner ---
def run_one_config(ma1, ma2, th,es, fee, delay, df_ma, df, start, end) -> Tuple[str, Dict]:
    # start_time = time.time()
    position = compute_single_position(df_ma, ma1, ma2, th, es, delay)
    # print(f"compute_single_position: {time.time()-start_time}")
    # start_time = time.time()
    df_1d = compute_profit(position, df, fee)
    # print(f"compute_profit: {time.time()-start_time}")
    # start_time = time.time()
    report = compute_report(df_1d, start, end)
    # print(f"compute_report: {time.time()-start_time}")
    return report

# --- Entry point ---

# === Worker init / task ===
def init_worker(df_ma_, df_raw_, shm_name_, shape_):
    """Mỗi worker ánh xạ shared memory và dữ liệu đọc-only"""
    global GLOBAL_DF_MA, GLOBAL_DF_RAW, GLOBAL_SHM, GLOBAL_SHAPE
    GLOBAL_DF_MA = df_ma_
    GLOBAL_DF_RAW = df_raw_
    GLOBAL_SHM = shared_memory.SharedMemory(name=shm_name_)
    GLOBAL_SHAPE = shape_



def worker_task(args):
    """Mỗi worker chạy 1 cấu hình và ghi kết quả trực tiếp vào shared memory"""
    idx, ma1, ma2, th, es, fee, delay, config_id, _id, start, end = args

    # === gọi hàm thực tế của bạn ===
    rpt = run_one_config(ma1, ma2, th, es, fee, delay, GLOBAL_DF_MA, GLOBAL_DF_RAW, start, end)
    # ví dụ rpt = {"sharpe": 1.23, "mdd_percent": -5.2, "tvr": 0.31, "ppc": 0.88, "profit_percent": 17.3, "total_net_profit": 12000}

    np_array = np.ndarray(GLOBAL_SHAPE, dtype=np.float64, buffer=GLOBAL_SHM.buf)
    np_array[idx, 0] = rpt.get("sharpe", np.nan)
    np_array[idx, 1] = rpt.get("mdd_percent", np.nan)
    np_array[idx, 2] = rpt.get("tvr", np.nan)
    np_array[idx, 3] = rpt.get("ppc", np.nan)
    np_array[idx, 4] = rpt.get("profit_percent", np.nan)
    np_array[idx, 5] = rpt.get("total_net_profit", np.nan)
    np_array[idx, 6] = rpt.get("psr", np.nan)
    np_array[idx, 7] = rpt.get("hhi", np.nan)

    
def scan(source_column:str, start_date, end_date, mode):
    FEE, DELAY = 175, 1
    start_time = time.time()
    os.makedirs("/home/ubuntu/nevir/results/", exist_ok=True)
    # --- Load data thực ---
    df = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    df = df[(df["day"] >= start_date) & (df["day"] <= end_date)]
    df["based_col"] = 0
    for col in source_column.split("_"):
        if col == "hose500": df["based_col"] += df["aggBusd"]
        elif col == "fhose500": df["based_col"] += df["aggFBusd"]
        elif col == "vn30": df["based_col"] += df["aggBusdVn30"]
        elif col == "fvn30": df["based_col"] += df["aggFBusdVn30"]

    df = df[df["based_col"].diff() != 0]
    df = df[df["time_int"] >= 10091500]
    df["priceChange"] = df.groupby("day")["last"].diff().shift(-1).fillna(0)

    # --- Precompute MA ---
    ma_lengths = {ma for ma1 in range(6, 151) for ma2 in range(5, ma1) for ma in (ma1, ma2)}
    df_ma = precompute_ma(df, ma_lengths, based_col="based_col", mode=mode)

    # --- Build jobs ---
    all_jobs = []
    th_list = list(range(5, 20))
    es_list = list(range(2, 9, 3))
    idx = 0
    for ma1 in range(6, 151):
        for ma2 in range(5, ma1):
            for th in th_list:
                for es in es_list:
                    config_id = f"{ma1}_{ma2}_{th}_{es}"
                    _id = make_key(config_id, FEE, DELAY, start_date, end_date)
                    all_jobs.append((idx, ma1, ma2, th, es, FEE, DELAY, config_id, _id, start_date, end_date))
                    idx += 1

    total_jobs = len(all_jobs)
    print(f"Total jobs: {total_jobs}")

    # --- Shared memory cho 6 chỉ tiêu ---
    n_metrics = 8  # sharpe, mdd, tvr, ppc, profit%, net profit
    shm = shared_memory.SharedMemory(create=True, size=total_jobs * n_metrics * 8)
    result_array = np.ndarray((total_jobs, n_metrics), dtype=np.float64, buffer=shm.buf)
    result_array[:] = np.nan

    # --- Run Pool ---
    with multiprocessing.Pool(
        processes=40,
        initializer=init_worker,
        initargs=(df_ma, df, shm.name, result_array.shape)
    ) as pool:
        list(tqdm(pool.imap_unordered(worker_task, all_jobs), total=total_jobs, desc="Running backtest"))

    # --- Làm phẳng & merge metadata ---
    df_result = pd.DataFrame(result_array, columns=["Sharpe", "MDD", "Turnover", "PPC", "NET Profit %", "NET Profit", "PSR", "HHI"])
    df_result["Config"] = [f"{j[1]}_{j[2]}_{j[3]}_{j[4]}" for j in all_jobs]
    df_result["TH"] = [j[3] for j in all_jobs]
    df_result["ES"] = [j[4] for j in all_jobs]
    df_result["P1"] = [j[1] for j in all_jobs]
    df_result["P2"] = [j[2] for j in all_jobs]
    cols = ["Config"] + [c for c in df_result.columns if c != "Config"]
    df_result = df_result[cols]
    # --- Sắp xếp & loại cột thừa ---
    df_result.sort_values(by=["TH", "P1", "P2", "ES"], inplace=True)
    df_result.drop(columns=["P1", "P2", "TH", "ES"], inplace=True, errors="ignore")

    # --- Export .pkl ---
    file_name = f"{source_column.upper()}_{start_date}_{end_date}_{mode.upper()}.xlsx"
    file_path = f"/home/ubuntu/nevir/results/{file_name}"
    
    df_result.to_excel(file_path, index=False)


    elapsed = time.time() - start_time
    print(f"✅ Completed {total_jobs} jobs in {elapsed:.2f}s ({total_jobs/elapsed:.2f} jobs/s)")
    try:
        # --- Upload 1 file duy nhất lên Google Drive ---
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive'])
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("/home/ubuntu/nevir/credentials.json", ['https://www.googleapis.com/auth/drive'])
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        drive_service = build('drive', 'v3', credentials=creds)
        file_metadata = {
            'name': os.path.splitext(file_name)[0],
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': ['1ZI65HWxDFaPcXQYyJFdu47cLQtwJ_4Iw']
        }
        media = MediaFileUpload(file_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        uploaded = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = uploaded.get('id')
        sheet_url = f"https://docs.google.com/spreadsheets/d/{file_id}"
        print(f"📤 Uploaded to Google Drive: {sheet_url}")
        result_log = "/home/ubuntu/nevir/result.txt"
        with open(result_log, "a") as f:
            f.write(f"{sheet_url}\n")
    except Exception as e:
        print(f"⚠️ Lỗi khi upload lên Google Drive: {e}")
    
    shm.close()
    shm.unlink()
    return df_result

if __name__ == "__main__":
    print("Starting scan by period...")

    periods = [
        # (20220101, 20230101),
        # (20220401, 20230401),
        # (20220701, 20230701),
        # (20221001, 20231001),
        # (20230101, 20240101),
        # (20230401, 20240401),
        # (20230701, 20240701),
        # (20231001, 20241001),
        # (20240101, 20250101),
        # (20240401, 20250401),
        # (20240701, 20250701),
        # (20241001, 20251001),
        ############################
        (20220101, 20251001),
        # (20220101, 20220401),
        # (20220401, 20220701),
        # (20220701, 20221001),
        # (20221001, 20230101),
        # (20230101, 20230401),
        # (20230401, 20230701),
        # (20230701, 20231001),
        # (20231001, 20240101),
        # (20240101, 20240401),
        # (20240401, 20240701),
        # (20240701, 20241001),
        # (20241001, 20250101),
        # (20250101, 20250401),
        # (20250401, 20250701),
        # (20250701, 20251001),
        # (20251001, 20251114),
        
        
    ]

    srcs = [
        {"source_column": "fvn30_fhose500", "mode": "ema"},
    ]

    for start_date, end_date in periods:
        for src in srcs:
            print(f"\n📆 Running {src['source_column']} from {start_date} to {end_date} ({src['mode']})")
            scan(src['source_column'], start_date, end_date, src['mode'])
    