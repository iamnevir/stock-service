from datetime import datetime
from itertools import combinations, islice
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
from typing import Dict, Tuple
import numpy as np
import  time
import pandas as pd
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from busd_auto.utils import get_mongo_uri, sanitize_for_bson, setup_logger
from busd_auto.backtest import compute_report, compute_single_position

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
    df_trade = df_alpha[df_alpha["action"] != 0][["action"]]
    return df_1d, df_trade

def run_one_config(ma1, ma2, th,es, fee, delay, df_ma, df, start, end) -> Tuple[str, Dict]:
    # start_time = time.time()
    position = compute_single_position(df_ma, ma1, ma2, th, es, delay)
    # print(f"compute_single_position: {time.time()-start_time}")
    # start_time = time.time()
    df_1d, df_trade = compute_profit(position, df, fee)
    # print(f"compute_profit: {time.time()-start_time}")
    # start_time = time.time()
    report = compute_report(df_1d, start, end)
    
    return report, df_trade

# --- Entry point ---

# === Worker init / task ===
def init_worker(df_ma_, df_raw_, shm_name_, shape_):
    """Mỗi worker ánh xạ shared memory và dữ liệu đọc-only"""
    global GLOBAL_DF_MA, GLOBAL_DF_RAW, GLOBAL_SHM, GLOBAL_SHAPE
    GLOBAL_DF_MA = df_ma_
    GLOBAL_DF_RAW = df_raw_
    GLOBAL_SHM = shared_memory.SharedMemory(name=shm_name_)
    GLOBAL_SHAPE = shape_

def worker_task_batch(job_batch):
    np_array = np.ndarray(GLOBAL_SHAPE, dtype=np.float64, buffer=GLOBAL_SHM.buf)

    results = []  # (idx, df_trade)

    for args in job_batch:
        (
            idx, ma1, ma2, th, es,
            fee, delay, config_id, _id,
            start, end
        ) = args

        rpt, df_trade = run_one_config(ma1, ma2, th, es, fee, delay,
                                       GLOBAL_DF_MA, GLOBAL_DF_RAW,
                                       start, end)

        # ghi metric vào shared memory
        np_array[idx, 0] = rpt.get("sharpe", np.nan)
        np_array[idx, 1] = rpt.get("mddPct", np.nan)
        np_array[idx, 2] = rpt.get("tvr", np.nan)
        np_array[idx, 3] = rpt.get("ppc", np.nan)
        np_array[idx, 4] = rpt.get("profitPct", np.nan)
        np_array[idx, 5] = rpt.get("netProfit", np.nan)
        np_array[idx, 6] = rpt.get("psr", np.nan)
        np_array[idx, 7] = rpt.get("hhi", np.nan)

        results.append((idx, df_trade))

    return results

def save_results_to_db(result_array, results, backtest_results_collection, all_jobs, batch_size=1000):

    ops = []

    for idx, df_trade in results:
        row = result_array[idx]
        ro = all_jobs[idx]

        _id = ro[8]
        config_id = ro[7]

        doc = {
            "_id": _id,
            "strategy": config_id,
            "sharpe": float(row[0]),
            "mddPct": float(row[1]),
            "tvr": float(row[2]),
            "ppc": float(row[3]),
            "profitPct": float(row[4]),
            "netProfit": float(row[5]),
            "psr": float(row[6]),
            "hhi": float(row[7]),
            "df_trade": df_trade.reset_index().to_dict("records"),
        }

        ops.append(
            UpdateOne(
                {"_id": _id},
                {"$set": doc},
                upsert=True
            )
        )

        if len(ops) >= batch_size:
            backtest_results_collection.bulk_write(ops, ordered=False)
            ops = []

    if ops:
        backtest_results_collection.bulk_write(ops, ordered=False)

def calc_block_pair(args):
    A, valid, idx_i, idx_j, lens_i, lens_j = args
    Ti = len(idx_i)
    Tj = len(idx_j)

    match = np.zeros((Ti, Tj), dtype=np.int32)
    count = np.zeros((Ti, Tj), dtype=np.int32)

    for t in range(A.shape[0]):
        ai = A[t, idx_i][:, None]
        aj = A[t, idx_j][None, :]
        v = valid[t, idx_i][:, None] & valid[t, idx_j][None, :]

        same = (ai == aj)
        match += same & v
        count += v

    results = {}
    for ii, i in enumerate(idx_i):
        for jj, j in enumerate(idx_j):
            if i >= j:
                continue
            if match[ii, jj] == 0:
                continue

            c1 = match[ii, jj] / lens_i[ii] * 100
            c2 = match[ii, jj] / lens_j[jj] * 100
            results[(i, j)] = round(max(c1, c2), 4)

    return results

def build_action_matrix(id_to_trade_df):
    dfs = []
    for sid, df in id_to_trade_df.items():
        tmp = df[['datetime', 'action']].copy()
        tmp['action'] = np.where(tmp['action'] > 1, 1,
                        np.where(tmp['action'] < 1, -1, 0))
        tmp = tmp.set_index('datetime')

        # 👇 GOM THEO 3 GIÂY
        bucket = tmp.index.floor("3S")
        tmp = tmp.groupby(bucket).last()

        tmp.rename(columns={'action': str(sid)}, inplace=True)
        dfs.append(tmp)

    action_df = pd.concat(dfs, axis=1).sort_index()
    return action_df

def calculate_all_correlations_block_mp(action_df, block_size=500, n_core=20):
    ids = action_df.columns.to_list()
    A_raw = action_df.values

    valid = ~np.isnan(A_raw)
    A = np.nan_to_num(A_raw).astype(np.int8)

    N = A.shape[1]
    lens = valid.sum(axis=0)

    blocks = [
        np.arange(i, min(i + block_size, N))
        for i in range(0, N, block_size)
    ]

    jobs = []
    for bi, idx_i in enumerate(blocks):
        for bj, idx_j in enumerate(blocks):
            if bj < bi:
                continue
            jobs.append((
                A,
                valid,
                idx_i,
                idx_j,
                lens[idx_i],
                lens[idx_j],
            ))

    print(f"🚀 Running {len(jobs)} block-pairs on {n_core} cores")

    results = {}
    with mp.Pool(processes=n_core) as pool:
        for res in pool.imap_unordered(calc_block_pair, jobs):
            results.update(res)

    return ids, results

def calculate_combined_correlations(
    busd_id,
    stras=None,
    logger=None,
    max_workers=20,
    block_size=500,
    type="ios",
    start=None,
    end=None
):
    """
    Final clean version:
    - Block-based correlation engine
    - No existing_pairs
    - MongoDB auto-skip duplicates
    - Minimal insert schema: {x, y, c}
    """

    start_time = time.time()

    # === 0️⃣ MongoDB ===
    db = MongoClient(get_mongo_uri("mgc3"))['busd']
    busd_collection = db["busd_collection"]
    db_local = MongoClient(get_mongo_uri())['busd']
    correlation_coll = db_local["correlation_results"]

    # === 1️⃣ Parse trades ===
    logger.info("⏳ Parsing trades...")
    id_to_trade_df = {}

    for doc in stras:
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            continue

        df = pd.DataFrame(trades)
        if "datetime" not in df or "action" not in df:
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)

        if not df.empty:
            id_to_trade_df[doc["_id"]] = df

    if len(id_to_trade_df) < 2:
        logger.info("⚠️ Not enough strategies to compute correlations.")
        return

    # === 2️⃣ Update status: running ===
    total_combinations = len(id_to_trade_df) * (len(id_to_trade_df) - 1) // 2
    projection = {"_id": 0, "x": 1, "y": 1}
    existing_pairs = set()

    list_ids = [str(i) for i in id_to_trade_df.keys()]
    total_combinations = len(list_ids) * (len(list_ids) - 1) // 2

    logger.info("🔍 Loading existing correlation pairs from DB...")
    def chunkify(seq, size=1000):
        for i in range(0, len(seq), size):
            yield seq[i:i+size]
    for x_chunk in chunkify(list_ids, size=1200):
        cursor = (correlation_coll
                .find({"x": {"$in": x_chunk}, "y": {"$in": list_ids}}, projection)
                .hint([("x",1),("y",1)])
                .batch_size(10_000))

        for d in cursor:
            existing_pairs.add((d["x"], d["y"]))

    logger.info(f"📌 Existing pairs: {len(existing_pairs)}/{total_combinations}")
    if len(existing_pairs) >= total_combinations:
        logger.info("✅ Correlations already complete. Skip computing.")

        if type == "wfa":
            busd_collection.update_one(
                {"_id": ObjectId(busd_id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {"wfa.$.correlation.status": "done"}}
            )
        else:
            busd_collection.update_one(
                {"_id": ObjectId(busd_id)},
                {"$set": {f"{type}.correlation.status": "done"}}
            )
        return
    if type == "wfa":
        busd_collection.update_one(
            {"_id": ObjectId(busd_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.status": "running",
                "wfa.$.correlation.process": 0,
                "wfa.$.correlation.total": total_combinations
            }}
        )
    else:
        busd_collection.update_one(
            {"_id": ObjectId(busd_id)},
            {"$set": {
                f"{type}.correlation.status": "running",
                f"{type}.correlation.process": 0,
                f"{type}.correlation.total": total_combinations
            }}
        )
    
    # === 3️⃣ Build action matrix ===
    logger.info("🧮 Building action matrix...")
    action_df = build_action_matrix(id_to_trade_df)

    # === 4️⃣ Compute correlations (BLOCK ENGINE) ===
    logger.info("⚡ Computing correlations (block-based)...")
    ids, corr_dict = calculate_all_correlations_block_mp(
        action_df,
        block_size=block_size,
        n_core=max_workers
    )

    logger.info(f"🧠 Computed {len(corr_dict):,} correlation pairs")

    # === 5️⃣ Prepare Mongo documents (MINIMAL) ===
    docs = []
    for (i, j), c in corr_dict.items():
        x = str(ids[i])
        y = str(ids[j])
        if x > y:
            x, y = y, x

        docs.append({
            "x": x,
            "y": y,
            "c": round(c, 4)
        })
    logger.info("🧹 Filtering existing pairs before insert...")

    new_docs = []
    for d in docs:
        if (d["x"], d["y"]) not in existing_pairs:
            new_docs.append(d)

    logger.info(f"📦 New pairs to insert: {len(new_docs)}")

    # === 6️⃣ Insert Mongo (skip duplicates) ===
    logger.info(f"📦 Inserting {len(new_docs):,} correlations...")
    inserted = 0
    last_update = time.time()

    from pymongo.errors import BulkWriteError

    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    for batch in chunked(new_docs, 10000):
        try:
            correlation_coll.insert_many(batch, ordered=False)
            inserted += len(batch)
        except BulkWriteError as e:
            inserted += e.details.get("nInserted", 0)

        if time.time() - last_update >= 10:
            if type == "wfa":
                busd_collection.update_one(
                    {"_id": ObjectId(busd_id), "wfa.is.start": start, "wfa.is.end": end},
                    {"$set": {"wfa.$.correlation.process": inserted}}
                )
            else:
                busd_collection.update_one(
                    {"_id": ObjectId(busd_id)},
                    {"$set": {f"{type}.correlation.process": inserted}}
                )

            logger.info(f"⏳ Progress: {inserted}/{len(new_docs)}")
            last_update = time.time()

    # === 7️⃣ Done ===
    if type == "wfa":
        busd_collection.update_one(
            {"_id": ObjectId(busd_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {"wfa.$.correlation.status": "done"}}
        )
    else:
        busd_collection.update_one(
            {"_id": ObjectId(busd_id)},
            {"$set": {f"{type}.correlation.status": "done"}}
        )

    logger.info(
        f"✅ Done correlations | inserted={inserted:,} | "
        f"time={time.time() - start_time:.2f}s"
    )

