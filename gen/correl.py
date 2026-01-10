
import hashlib
import json
from time import time
from urllib.parse import quote_plus
from bson import ObjectId
import numpy as np
import pandas as pd
from pymongo import MongoClient
import os
import multiprocessing
from itertools import combinations, islice
from pymongo.errors import BulkWriteError, PyMongoError
from gen.scan import get_mongo_uri,make_key_corr
from bson import ObjectId

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

    # Điều kiện vector hóa
    a1 = merged['action_1'].values
    a2 = merged['action_2'].values

    # Nếu không match sẽ là NaN → loại bỏ
    valid = ~np.isnan(a2)

    a1 = a1[valid]
    a2 = a2[valid]

    # Match signal logic
    matches = (
        (a1 == a2) |
        ((a1 == 1) & (a2 == 2)) |
        ((a1 == 2) & (a2 == 1)) |
        ((a1 == -1) & (a2 == -2)) |
        ((a1 == -2) & (a2 == -1))
    )

    matched_count = np.sum(matches)

    corr1 = round(matched_count / len(df1) * 100, 2)
    corr2 = round(matched_count / len(df2) * 100, 2)
    return max(corr1, corr2)

def calculate_combined_correlations(
    config_id,  # Đổi tên 'id' thành 'config_id' cho nhất quán
    stras=None,
    max_workers=20,
    chunk_size=100000
):
    """
    Tính toán tương quan alpha, sử dụng kiến trúc Producer-Consumer (Worker/Queue/Writer)
    giống như hàm tính tương quan stock chuẩn.
    """
    
    # === 0️⃣ Khởi tạo kết nối DB ===
    # Các process con sẽ kế thừa kết nối này, 
    # nhưng chỉ writer_worker thực sự dùng nó để ghi.
    db = MongoClient(get_mongo_uri())['gen1_2']
    alpha_correl_coll = db["alpha_correl"]
    correlation_coll = db["correlation_results"]

    # === 1️⃣ Chuẩn bị dữ liệu ===
    print("⏳ Đang chuẩn bị dữ liệu (parsing trades)...")
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
    print("🔎 Đang lấy danh sách cặp đã tồn tại trong MongoDB...")
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

    print(f"✅ Đã có sẵn {len(existing_pairs):,} cặp trong DB — sẽ bỏ qua.")

    # === 3️⃣ Sinh danh sách cặp cần xử lý (Giống code chuẩn) ===
    all_pairs = []
    total_combinations = 0
    for id1, id2 in combinations(valid_ids, 2):
        total_combinations += 1
        key = tuple(sorted((str(id1), str(id2))))
        if key not in existing_pairs:
            all_pairs.append((id1, id2))

    print(f"🧮 Còn lại {len(all_pairs):,} cặp cần tính mới (trên tổng số {total_combinations:,} cặp).")
    
    chunks = list(chunked_iterable(all_pairs, chunk_size))
    total_chunks = len(chunks)
    total_pairs_to_process = len(all_pairs)

    print(f"🔢 Tổng số cặp cần xử lý: {total_pairs_to_process}")
    print(f"📦 Tổng số chunk (mỗi chunk ~{chunk_size} cặp): {total_chunks}")

    if config_id:
        alpha_correl_coll.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"process": {"done": len(existing_pairs), "total": total_combinations}}}
        )

    if not all_pairs:
        print("🏁 Không còn cặp nào để xử lý. Kết thúc.")
        alpha_correl_coll.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"status": "done"}}
        )
        return

    # === 4️⃣ Writer process (Giống code chuẩn) ===
    def writer_worker(q: multiprocessing.Queue, initial_done_count: int):
        total_written = initial_done_count
        MAX_BATCH_SIZE = 10000

        while True:
            results = q.get()
            if results == "STOP":
                print("🧾 Writer nhận tín hiệu dừng, kết thúc.")
                break

            for i in range(0, len(results), MAX_BATCH_SIZE):
                sub_batch = results[i:i + MAX_BATCH_SIZE]
                try:
                    correlation_coll.insert_many(sub_batch, ordered=False)
                    total_written += len(sub_batch)
                    # Cập nhật tiến độ
                    if config_id:
                        alpha_correl_coll.update_one(
                            {"_id": ObjectId(config_id)},
                            {"$set": {"process.done": total_written}}
                        )
                except BulkWriteError as bwe:
                    # Bỏ qua lỗi trùng lặp (11000) nhưng vẫn đếm
                    valid_writes = len(sub_batch) - len(bwe.details.get("writeErrors", []))
                    total_written += valid_writes # Chỉ đếm số lượng ghi thành công
                    if config_id:
                         alpha_correl_coll.update_one(
                            {"_id": ObjectId(config_id)},
                            {"$set": {"process.done": total_written}}
                        )
                    print(f"⚠️ Writer: Bỏ qua {len(bwe.details.get('writeErrors', []))} lỗi (ví dụ: trùng lặp).")
                    continue
                except Exception as e:
                    print(f"❌ Writer lỗi MongoDB (batch nhỏ): {e}")

        print(f"✅ Writer hoàn tất, tổng ghi mới: {total_written - initial_done_count}")

    # === 5️⃣ Worker process: Chỉ tính toán, gửi kết quả ===
    def process_chunk(q, chunk):
        results = []
        for id1, id2 in chunk:
            x, y = str(id1), str(id2)
            
            # Không cần kiểm tra existing_pairs ở đây nữa
            df1, df2 = id_to_trade_df[id1], id_to_trade_df[id2]
            
            # Logic tính toán của alpha
            c = calculate_trade_correlation_vectorized(df1, df2)
            
            results.append({
                "x": x,
                "y": y,
                "c": round(c, 4), # Format của alpha
            })

        if results:
            q.put(results) # Gửi kết quả vào queue

    # === 6️⃣ Khởi tạo queue và writer ===
    q = multiprocessing.Queue(maxsize=max_workers * 4)
    writer = multiprocessing.Process(
        target=writer_worker, 
        args=(q, len(existing_pairs)) # Truyền số lượng đã hoàn thành ban đầu
    )
    writer.start()

    # === 7️⃣ Khởi tạo các worker (Giống code chuẩn) ===
    processes = []
    print(f"🚀 Bắt đầu xử lý {total_chunks} chunks với tối đa {max_workers} workers...")
    
    for chunk in chunks:
        while len(processes) >= max_workers:
            # Chờ một process con kết thúc trước khi bắt đầu process mới
            for p in processes[:]:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
            import time
            time.sleep(0.05) # Ngủ 1 chút để tránh busy-waiting
            
        p = multiprocessing.Process(target=process_chunk, args=(q, chunk))
        p.start()
        processes.append(p)

    # Đợi tất cả worker xong
    for p in processes:
        p.join()
    
    print("✅ Tất cả workers đã hoàn thành tính toán.")

    # Gửi tín hiệu dừng writer
    q.put("STOP")
    writer.join()

    # === 8️⃣ Cập nhật process.done chính xác (Giống code chuẩn) ===
    if config_id:
        print("🔄 Đang cập nhật lại số lượng chính xác cuối cùng...")
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
        alpha_correl_coll.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"process.done": unique_pair_count, "status": "done"}}
        )
        print(f"✅ Đã cập nhật lại chính xác process.done = {unique_pair_count} và status = 'done'")
        
def run_corr(id):
    db =  MongoClient(get_mongo_uri())['gen1_2']
    alpha_correl_coll = db["alpha_correl"]
    alpha_correl = alpha_correl_coll.find_one({"_id": ObjectId(id)})
    alpha_name = alpha_correl.get("alpha_name", "")
    gen = alpha_correl.get("gen", "gen1_2")
    lst_configs = alpha_correl.get("configs", [])
    fee = 0.175
    start = alpha_correl.get("start", "2024_01_01")
    end = alpha_correl.get("end", "2025_01_01")
    list_ids = [make_key_corr(
            config=config,
            fee=fee,
            start=start,
            end=end,
            alpha_name=alpha_name,
            gen=gen
        ) for config in lst_configs]
    
    mongo_coll = db["stock"]
    exist_stra = list(mongo_coll.find({"_id": {"$in": list_ids}}))
    print(f"Found {len(exist_stra)} strategies in the database.")
    calculate_combined_correlations(
        config_id=id,
        stras=exist_stra,
        max_workers=20,
    ) 
    


