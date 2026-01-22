from datetime import datetime
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import  time
import traceback
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from base_auto.utils import get_mongo_uri, insert_batch, load_dic_freqs, sanitize_for_bson, setup_logger, make_key_base
from gen_spot.base_func_lib import Domains
from itertools import combinations, islice

from gen_spot.core import Simulator

def run_single_backtest(config, base_name,fee, dic_freqs, DIC_BASES, gen=None, start=None, end=None, source=None):
    def parse_common_params(rest, base_name):
        params = {}

        if len(rest) > 0:
            params["window"] = int(rest[0])
        if len(rest) > 1:
            if base_name == "base_003":
                params["window_rank"] = int(rest[1])
            else:
                params["factor"] = float(rest[1])
        if base_name == "base_005" and len(rest) > 2:
            params["window_rank"] = int(rest[2])

        return params
    gen_params = {}
    params = {}

    parts = config.split("_")

    if gen == "1_1":
        freq, threshold, *rest = parts
        freq = int(freq)
        threshold = float(threshold)

        params = parse_common_params(rest, base_name)
        gen_params = {"threshold": threshold}

    elif gen == "1_2":
        freq, upper, lower, *rest = parts
        freq = int(freq)
        upper, lower = float(upper), float(lower)

        params = parse_common_params(rest, base_name)
        gen_params = {
            "upper": upper,
            "lower": lower
        }

    elif gen == "1_3":
        freq, score, entry, exit, *rest = parts
        freq = int(freq)
        score = int(score)
        entry, exit = float(entry), float(exit)

        params = parse_common_params(rest, base_name)
        gen_params = {
            "score": score,
            "entry": entry,
            "exit": exit
        }

    elif gen == "1_4":
        freq, entry, exit, smooth, *rest = parts
        freq = int(freq)
        entry, exit, smooth = float(entry), float(exit), float(smooth)

        params = parse_common_params(rest, base_name)
        gen_params = {
            "entry": entry,
            "exit": exit,
            "smooth": smooth
        }
    bt = Simulator(
        base_name=base_name,
        freq=freq,
        gen_params=gen_params,
        fee=fee,
        df_base=dic_freqs[freq].copy(),
        params=params,
        DIC_BASES=DIC_BASES,
        df_tick=None,
        gen=gen,
        start=start,
        end=end
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

    _id = make_key_base(config=config,
                fee=fee,
                start=start,
                end=end,
                base_name=base_name,
                gen=gen,
                source=source
                )

    keys_to_delete = ["aroe", "cdd", "cddPct","lastProfit","max_loss","max_gross","num_trades"]
    for key in keys_to_delete:
        rpt.pop(key, None)
    return {"report": rpt, "df_trade": df_trade, "_id": _id,"config": config}

def worker_task_batch(args):
    """
    Mỗi worker xử lý 1 batch (1000 configs)
    """
    batch_configs, base_name, fee, dic_freqs, DIC_BASES, gen, start, end, source = args
    results = []
    for cfg in batch_configs:
        rpt = run_single_backtest(cfg, base_name, fee, dic_freqs, DIC_BASES, gen, start, end, source)
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
    base_id,
    stras=None,
    logger=None,
    max_workers=20,
    chunk_size=100000,
    type="ios",
    start=None,
    end=None
):
    """
    Tính toán tương quan base, sử dụng kiến trúc Producer-Consumer (Worker/Queue/Writer)
    giống như hàm tính tương quan stock chuẩn.
    """
    
    # === 0️⃣ Khởi tạo kết nối DB ===
    # Các process con sẽ kế thừa kết nối này, 
    # nhưng chỉ writer_worker thực sự dùng nó để ghi.
    db = MongoClient(get_mongo_uri("mgc3"))['base']
    base_collection = db["base_collection"]
    local_db = MongoClient(get_mongo_uri())['base']
    correlation_coll = local_db["correlation_results"]
    
    # === 1️⃣ Chuẩn bị dữ liệu ===
    logger.info("⏳ Đang chuẩn bị dữ liệu (parsing trades)...")
    id_to_trade_df = {}

    def parse_trade_doc(doc):
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            return None
        # Kiểm tra định dạng của base
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
        base_collection.update_one(
            {"_id": ObjectId(base_id), f"wfa.is.start": start, f"wfa.is.end": end},
            {"$set": {"wfa.$.correlation.process": len(existing_pairs), f"wfa.$.correlation.total": total_combinations,f"wfa.$.correlation.status": "running"}}
        )
    else:
        base_collection.update_one(
            {"_id": ObjectId(base_id)},
            {"$set": {f"{type}.correlation.process": len(existing_pairs), f"{type}.correlation.total": total_combinations,f"{type}.correlation.status": "running"}}
        )

    if not all_pairs:
        logger.info("🏁 Không còn cặp nào để xử lý. Kết thúc.")
        if type == "wfa":
            base_collection.update_one(
                {"_id": ObjectId(base_id), f"wfa.is.start": start, f"wfa.is.end": end},
                {"$set": {f"{type}.$.correlation.status": "done"}}
            )
        else:
            base_collection.update_one(
                {"_id": ObjectId(base_id)},
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
                    base_collection.update_one(
                        {"_id": ObjectId(base_id), "wfa.is.start": start, "wfa.is.end": end},
                        {"$set": {"wfa.$.correlation.process": inserted}}
                    )
                else:
                    base_collection.update_one(
                        {"_id": ObjectId(base_id)},
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
        base_collection.update_one(
            {"_id": ObjectId(base_id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {f"{type}.$.correlation.process": unique_pair_count, f"{type}.$.correlation.status": "done"}}
        )
    else:
        base_collection.update_one(
            {"_id": ObjectId(base_id)},
            {"$set": {f"{type}.correlation.process": unique_pair_count, f"{type}.correlation.status": "done"}}
        )
    logger.info(f"✅ Đã cập nhật lại chính xác process = {unique_pair_count} và status = 'done'")
  


def correlation(id, start, end):
    start_time = time.time()
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    base_db = mongo_client["base"]
    base_collection = base_db["base_collection"]
    local_mongo = MongoClient(get_mongo_uri())['base']
    coll = local_mongo["correlation_backtest"]

    base_doc = base_collection.find_one({"_id": ObjectId(id)})
    if not base_doc:
        print("❌ Không tìm thấy base_collection với id này.")
        return
    base_name = base_doc.get("base_name", "")
    name = base_doc.get("name", "")
    logger = setup_logger(f"{name}_{start}_{end}_wfa_correlation")
    wfa = base_doc.get("wfa", [])
    fa = None
    for item in wfa:
        if item.get("is").get("start") == start and item.get("is").get("end") == end:
            need_configs = item.get("filter_report", {}).get("strategies", [])
            fa = item
            break
    if not fa:
        logger.error("❌ Không tìm thấy WFA với khoảng thời gian này.")
        return
    

    gen = base_doc.get("gen")
    fee = fa.get("fee")
    filter_report = fa.get("filter_report")
    DIC_BASES = Domains.get_list_of_bases()
    source = base_doc.get("source", "hose500")
    dic_freqs = load_dic_freqs(source=source)
    # print(dic_freqs[1])
    need_configs = filter_report.get("strategies", [])
    try:
        list_ids = [make_key_base(
                config=config,
                fee=fee,
                start=start,
                end=end,
                base_name=base_name,
                gen=gen,
                source=source
            ) for config in need_configs]
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        logger.info(f"🔎 Found {len(exist_stra)} existing backtest results in DB.")
        
        run_configs = list(set(need_configs) - set([stra['config'] for stra in exist_stra]))
        total = len(run_configs)
        logger.info(f"🛠️ Need to run backtests for {total} configurations.")
        if total > 0:
            logger.info(f"🚀 Running correlation with {total} parameter combinations...")
            
            base_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {
                    "wfa.$.correlation.process": 0,
                    "wfa.$.correlation.total": total,
                    "wfa.$.correlation.status": "running"
                }}
            )

            # ----------------- Chuẩn bị batch -----------------
            n_workers = 30
            batch_size_configs = total if total < 1000 else 1000
            batches = [run_configs[i:i + batch_size_configs] for i in range(0, total, batch_size_configs)]
            
            args_list = [(batch, base_name, fee, dic_freqs, DIC_BASES, gen, start, end, source) for batch in batches]
            
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
                        base_collection.update_one(
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
            base_collection.update_one(
                {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
                {"$set": {"wfa.$.correlation.process": inserted}}
            )
            # ----------------- Hoàn tất -----------------
            logger.info(f"🎯correlation hoàn tất cho {total} in {time.time() - start_time:.2f}s.")
            #--------------------- Tính correlation ---------------------
        logger.info("🚀 Bắt đầu tính correlation...")
        exist_stra = list(coll.find({"_id": {"$in": list_ids}}))
        calculate_combined_correlations(
            base_id=id,
            stras=exist_stra,
            logger=logger,
            type="wfa"
        )
        base_collection.update_one(
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
        base_collection.update_one(
            {"_id": ObjectId(id), "wfa.is.start": start, "wfa.is.end": end},
            {"$set": {"wfa.$.correlation.status": "error"}}
        )
        mongo_client.close()
    
    
def main():
    if len(sys.argv) < 4:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/base_auto/wfa_correlation.py <_id> <start> <end>")
        sys.exit(1)

    _id = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]

    correlation(_id, start, end)

if __name__ == "__main__":
    main()