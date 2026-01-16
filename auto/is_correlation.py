from datetime import datetime
from itertools import combinations, islice
import multiprocessing as mp
import sys
import numpy as np
import  time
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from auto.utils import get_mongo_uri,load_dic_freqs, make_key_alpha, sanitize_for_bson, setup_logger, insert_batch
from gen.alpha_func_lib import Domains
from gen.core import Simulator
from pymongo.errors import BulkWriteError

def run_single_backtest(config, alpha_name,fee, dic_freqs, DIC_ALPHAS, gen=None, start=None, end=None, source=None, overnight=False):
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
                overnight=overnight)

    keys_to_delete = ["aroe", "cdd", "cddPct","lastProfit","max_loss","max_gross","num_trades"]
    for key in keys_to_delete:
        rpt.pop(key, None)
    return {"report": rpt, "df_trade": df_trade, "_id": _id,"config": config}

def worker_task_batch(args):
    """
    Mỗi worker xử lý 1 batch (1000 configs)
    """
    batch_configs, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source, overnight = args
    results = []
    for cfg in batch_configs:
        rpt = run_single_backtest(cfg, alpha_name, fee, dic_freqs, DIC_ALPHAS, gen, start, end, source, overnight)
        results.append(rpt)
    return results

def calculate_trade_correlation_vectorized(df1, df2):
    if df1.empty or df2.empty:
        return 0.0, 0.0

    time_tolerance = pd.Timedelta(seconds=0)

    df1 = df1.sort_values('executionT')
    df2 = df2.sort_values('executionT')

    merged = pd.merge_asof(
        df1[['executionT', 'action']],
        df2[['executionT', 'action']],
        on='executionT',
        direction='nearest',
        tolerance=time_tolerance,
        suffixes=('_1', '_2')
    )

    # Lấy action
    a1 = merged['action_1'].values
    a2 = merged['action_2'].values

    # Loại bỏ các dòng không match
    valid = ~np.isnan(a2)
    a1 = a1[valid]
    a2 = a2[valid]

    # 🔥 Chuẩn hoá action về {-1, 0, 1}
    a1 = np.where(a1 > 1, 1, np.where(a1 < 1, -1, 0))
    a2 = np.where(a2 > 1, 1, np.where(a2 < 1, -1, 0))

    # Match khi cùng direction
    matches = (a1 == a2)

    matched_count = np.sum(matches)

    corr1 = round(matched_count / len(df1) * 100, 2)
    corr2 = round(matched_count / len(df2) * 100, 2)

    return max(corr1, corr2)

# ---- process_chunk cũng phải ở cấp module ----
def process_chunk(args):
    """Worker: xử lý 1 chunk, trả về list kết quả"""
    chunk, id_to_trade_df = args
    results = []
    for id1, id2 in chunk:
        x, y = str(id1), str(id2)
        df1, df2 = id_to_trade_df[id1], id_to_trade_df[id2]
        c = calculate_trade_correlation_vectorized(df1, df2)
        results.append({"x": x, "y": y, "c": round(c, 4)})
    return results

def calculate_combined_correlations(
    alpha_id,
    stras=None,
    logger=None,
    max_workers=20,
    chunk_size=100000,
    type="ios",
    start=None,
    end=None
):
    """
    Tính toán tương quan alpha, sử dụng kiến trúc Producer-Consumer (Worker/Queue/Writer)
    giống như hàm tính tương quan stock chuẩn.
    """
    
    # === 0️⃣ Khởi tạo kết nối DB ===
    # Các process con sẽ kế thừa kết nối này, 
    # nhưng chỉ writer_worker thực sự dùng nó để ghi.
    db = MongoClient(get_mongo_uri("mgc3"))['alpha']
    alpha_collection = db["alpha_collection"]
    db_local = MongoClient(get_mongo_uri())['alpha']
    correlation_coll = db_local["correlation_results"]

    # === 1️⃣ Chuẩn bị dữ liệu ===
    logger.info("⏳ Đang chuẩn bị dữ liệu (parsing trades)...")
    id_to_trade_df = {}

    def parse_trade_doc(doc):
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            return None
        # Kiểm tra định dạng của alpha
        if not all(isinstance(t, dict) and "executionT" in t and "action" in t for t in trades):
            return None
        df = pd.DataFrame(trades)
        df["executionT"] = pd.to_datetime(df["executionT"], errors="coerce")
        df.dropna(subset=["executionT"], inplace=True)
        return df if not df.empty else None

    for doc in stras:
        _id = doc["_id"]
        trade_df = parse_trade_doc(doc)
        if trade_df is not None:
            id_to_trade_df[_id] = trade_df

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                break
            yield chunk

    valid_ids = list(id_to_trade_df.keys())
    
    # === 2️⃣ Lọc các cặp đã tồn tại (Giống code chuẩn) ===
    logger.info("🔎 Đang lấy danh sách cặp đã tồn tại trong MongoDB...")
    str_ids = [str(i) for i in valid_ids]
    existing_pairs = set()
    
    # Query hiệu quả hơn thay vì dùng 2 $in lớn
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            {"x": 1, "y": 1, "_id": 0}
        )
        for doc in cursor:
            existing_pairs.add(tuple(sorted((doc["x"], doc["y"]))))

    logger.info(f"✅ Đã có sẵn {len(existing_pairs):,} cặp trong DB — sẽ bỏ qua.")

    # === 3️⃣ Sinh danh sách cặp cần xử lý (Giống code chuẩn) ===
    all_pairs = []
    total_combinations = 0
    for id1, id2 in combinations(valid_ids, 2):
        total_combinations += 1
        key = tuple(sorted((str(id1), str(id2))))
        if key not in existing_pairs:
            all_pairs.append((id1, id2))

    logger.info(f"🧮 Còn lại {len(all_pairs):,} cặp cần tính mới (trên tổng số {total_combinations:,} cặp).")
    chunk_size = min(100000, len(all_pairs) // max_workers)
    chunks = list(chunked_iterable(all_pairs, chunk_size))
    total_chunks = len(chunks)
    total_pairs_to_process = len(all_pairs)

    logger.info(f"🔢 Tổng số cặp cần xử lý: {total_pairs_to_process}")
    logger.info(f"📦 Tổng số chunk (mỗi chunk ~{chunk_size} cặp): {total_chunks}")
    if type == "wfa":
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id), f"wfa.is.start": start, f"wfa.is.end": end},
            {"$set": {"wfa.$.correlation.process": len(existing_pairs), f"wfa.$.correlation.total": total_combinations,f"wfa.$.correlation.status": "running"}}
        )
    else:
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id)},
            {"$set": {f"{type}.correlation.process": len(existing_pairs), f"{type}.correlation.total": total_combinations,f"{type}.correlation.status": "running"}}
        )

    if not all_pairs:
        logger.info("🏁 Không còn cặp nào để xử lý. Kết thúc.")
        if type == "wfa":
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id), f"wfa.is.start": start, f"wfa.is.end": end},
                {"$set": {f"{type}.$.correlation.status": "done"}}
            )
        else:
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id)},
                {"$set": {f"{type}.correlation.status": "done"}}
            )
        return
    total = sum(len(chunk) for chunk in chunks)
    inserted = 0
    temp_batch = []
    last_update = time.time()

    args_list = [(chunk, id_to_trade_df) for chunk in chunks]

    logger.info(f"🚀 Bắt đầu xử lý {len(chunks)} chunks với tối đa {max_workers} workers...")

    with mp.Pool(processes=max_workers) as pool:
        for batch_results in pool.imap_unordered(process_chunk, args_list, chunksize=1):
            temp_batch.extend(batch_results)
            inserted += len(batch_results)

            # ✅ Cứ sau mỗi update_interval giây thì insert và cập nhật tiến độ
            if time.time() - last_update >= 10:
                if temp_batch:
                    insert_batch(correlation_coll, temp_batch, batch_size=10000)
                    temp_batch.clear()
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
                logger.info(f"⏳ Progress update: {inserted}/{total}")
                last_update = time.time()

    # 🔚 Xử lý phần còn lại
    if temp_batch:
        insert_batch(correlation_coll, temp_batch, batch_size=10000)
        temp_batch.clear()


    logger.info(f"✅ Hoàn tất: {inserted}/{total} pairs đã xử lý.")

    # === 8️⃣ Cập nhật process.done chính xác (Giống code chuẩn) ===
    logger.info("🔄 Đang cập nhật lại số lượng chính xác cuối cùng...")
    seen = set()
    projection = {"x": 1, "y": 1, "_id": 0}
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            projection
        )
        for doc in cursor:
            seen.add(tuple(sorted((doc["x"], doc["y"]))))

    unique_pair_count = len(seen)
    if type == "wfa":
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {f"{type}.$.correlation.process": unique_pair_count, f"{type}.$.correlation.status": "done"}}
        )
    else:
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id)},
            {"$set": {f"{type}.correlation.process": unique_pair_count, f"{type}.correlation.status": "done"}}
        )
    logger.info(f"✅ Đã cập nhật lại chính xác process = {unique_pair_count} và status = 'done'")
  

