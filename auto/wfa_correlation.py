from datetime import datetime
import multiprocessing as mp
import sys
import  time
import traceback
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from utils import get_mongo_uri, insert_batch, load_dic_freqs, setup_logger, make_key_alpha
from gen.alpha_func_lib import Domains
from gen.core import Simulator


from pymongo.errors import BulkWriteError

def run_single_backtest(config, alpha_name,fee, dic_freqs, DIC_ALPHAS, gen=None, start=None, end=None, source=None, overnight=False,cut_time=None):
    gen_params = {}
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
        freq, score, entry, exit = int(freq), int(score), float(entry), float(exit)
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
        freq, entry, exit, smooth = int(freq), float(entry), float(exit), float(smooth)
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
        
    bt = Simulator(
        alpha_name=alpha_name,
        freq=freq,
        gen_params=gen_params,
        fee=fee,
        df_alpha=dic_freqs[freq].copy(),
        params=params,
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=None,
        gen=gen,
        start=start,
        end=end,
        source=source,
        overnight=overnight
    )
    bt.compute_signal()
    bt.compute_position()
    bt.compute_tvr_and_fee()
    bt.compute_profits()
    bt.compute_performance(start=start, end=end)
    bt.compute_df_trade()
    df_trade = bt.df_trade.reset_index(drop=True).to_dict(orient="records")
    rpt = sanitize_for_bson(bt.report.copy())
    for k, v in params.items():
        rpt[f"param_{k}"] = round(float(v),4)

    _id = make_key_alpha(config=config,
                fee=fee,
                start=start,
                end=end,
                alpha_name=alpha_name,
                gen=gen,
                source=source,
                overnight=overnight,
                cut_time=cut_time)

    keys_to_delete = ["aroe", "cdd", "cddPct","lastProfit","max_loss","max_gross","num_trades"]
    for key in keys_to_delete:
        rpt.pop(key, None)
    return {"report": rpt, "df_trade": df_trade, "_id": _id,"config": config}

def worker_task_batch(args):
    """
    Mỗi worker xử lý 1 batch (1000 configs)
    """
    batch_configs, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source, overnight, cut_time = args
    results = []
    for cfg in batch_configs:
        rpt = run_single_backtest(cfg, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source, overnight, cut_time)
        results.append(rpt)
    return results

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
        tmp = df[['executionT', 'action']].copy()
        tmp['action'] = np.where(tmp['action'] > 1, 1,
                        np.where(tmp['action'] < 1, -1, 0))
        tmp = tmp.set_index('executionT')
        tmp.rename(columns={'action': str(sid)}, inplace=True)
        dfs.append(tmp)

    # Align theo executionT
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
    alpha_id,
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
    db = MongoClient(get_mongo_uri("mgc3"))["alpha"]
    alpha_collection = db["alpha_collection"]
    local_db = MongoClient(get_mongo_uri())["alpha"]
    correlation_coll = local_db["correlation_results"]

    # === 1️⃣ Parse trades ===
    logger.info("⏳ Parsing trades...")
    id_to_trade_df = {}

    for doc in stras:
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            continue

        df = pd.DataFrame(trades)
        if "executionT" not in df or "action" not in df:
            continue

        df["executionT"] = pd.to_datetime(df["executionT"], errors="coerce")
        df.dropna(subset=["executionT"], inplace=True)

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
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {"wfa.$.correlation.status": "done"}}
            )
        else:
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id)},
                {"$set": {f"{type}.correlation.status": "done"}}
            )
        return
    if type == "wfa":
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.status": "running",
                "wfa.$.correlation.process": 0,
                "wfa.$.correlation.total": total_combinations
            }}
        )
    else:
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id)},
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
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id), "wfa.is.start": start, "wfa.is.end": end},
                    {"$set": {"wfa.$.correlation.process": inserted}}
                )
            else:
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id)},
                    {"$set": {f"{type}.correlation.process": inserted}}
                )

            logger.info(f"⏳ Progress: {inserted}/{len(new_docs)}")
            last_update = time.time()

    # === 7️⃣ Done ===
    if type == "wfa":
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {"wfa.$.correlation.status": "done"}}
        )
    else:
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id)},
            {"$set": {f"{type}.correlation.status": "done"}}
        )

    logger.info(
        f"✅ Done correlations | inserted={inserted:,} | "
        f"time={time.time() - start_time:.2f}s"
    )

def correlation(id, start, end):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    mongo_local = MongoClient(get_mongo_uri())['alpha']
    coll = mongo_local["correlation_backtest"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return
    alpha_name = alpha_doc.get("alpha_name", "")
    name = alpha_doc.get("name", "")
    logger = setup_logger(f"{name}_{start}_{end}_wfa_correlation")
    wfa = alpha_doc.get("wfa", [])
    fa = None
    for item in wfa:
        if item.get("is").get("start") == start and item.get("is").get("end") == end:
            need_configs = item.get("filter_report", {}).get("strategies", [])
            fa = item
            break
    if not fa:
        logger.error("❌ Không tìm thấy WFA với khoảng thời gian này.")
        return
    
    source = alpha_doc.get("source", None)
    gen = alpha_doc.get("gen")
    overnight = alpha_doc.get("overnight",False)
    cut_time = alpha_doc.get("cut_time",None)
    fee = fa.get("fee")
    filter_report = fa.get("filter_report")
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs(source, overnight)
    need_configs = filter_report.get("strategies", [])
    try:
        list_ids = [make_key_alpha(
                config=config,
                fee=fee,
                start=start,
                end=end,
                alpha_name=alpha_name,
                gen=gen,
                source=source,
                overnight=overnight,
                cut_time=cut_time
            ) for config in need_configs]
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        logger.info(f"🔎 Found {len(exist_stra)} existing backtest results in DB.")
        
        run_configs = list(set(need_configs) - set([stra['config'] for stra in exist_stra]))
        total = len(run_configs)
        logger.info(f"🛠️ Need to run backtests for {total} configurations.")
        if total > 0:
            logger.info(f"🚀 Running correlation with {total} parameter combinations...")
            
            alpha_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {
                    "wfa.$.correlation.process": 0,
                    "wfa.$.correlation.total": total,
                    "wfa.$.correlation.status": "running"
                }}
            )

            # ----------------- Chuẩn bị batch -----------------
            n_workers = 40
            batch_size_configs = total if total < 1000 else 1000
            batches = [run_configs[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
            
            args_list = [(batch, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source, overnight, cut_time) for batch in batches]
            
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
                            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                            {"$set": {"wfa.$.correlation.process": inserted}}
                        )
                        logger.info(f"⏳ Progress update: {inserted}/{total}")
                        last_update = time.time()

            # ----------------- Insert phần còn lại -----------------
            if temp_batch:
                insert_batch(coll, temp_batch, batch_size=insert_batch_size)
                temp_batch.clear()

            logger.info(f"⏳ Final progress update: {inserted}/{total}")
            alpha_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {"wfa.$.correlation.process": inserted}}
            )
            # ----------------- Hoàn tất -----------------
            logger.info(f"🎯correlation hoàn tất cho {total} in {time.time() - start_time:.2f}s.")
            #--------------------- Tính correlation ---------------------
        logger.info("🚀 Bắt đầu tính correlation...")
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        calculate_combined_correlations(
            alpha_id=id,
            stras=exist_stra,
            logger=logger,
            type="wfa"
        )
        alpha_collection.update_one(
            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.status": "done",
            }}
        )
        
        mongo_client.close()
        logger.info(f"✅ Hoàn tất chạy correlation.")  
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy correlation: {e}")
        print(traceback.format_exc())
        alpha_collection.update_one(
            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {"wfa.$.correlation.status": "error"}}
        )
        mongo_client.close()
    
    
def main():
    if len(sys.argv) < 4:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfa_correlation.py <_id> <start> <end>")
        sys.exit(1)

    _id = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]

    correlation(_id, start, end)

if __name__ == "__main__":
    main()