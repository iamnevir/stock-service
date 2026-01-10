


import time
from bson import ObjectId
import numpy as np
import pandas as pd
from pymongo import MongoClient
from itertools import combinations, islice
from auto.is_correlation import calculate_trade_correlation_vectorized
from auto.utils import get_mongo_uri, load_dic_freqs, make_key_mega, sanitize_for_bson
from gen.alpha_func_lib import Domains
from gen.core_mega import Simulator

from itertools import combinations

def fetch_existing_correlations(correlation_coll, ids):
    
    str_ids = [str(i) for i in ids]

    cursor = correlation_coll.find(
        {
            "x": {"$in": str_ids},
            "y": {"$in": str_ids},
        },
        {"_id": 0}
    )

    results = list(cursor)
    print(f"✅ Tìm thấy {len(results):,} cặp correlation đã có trong DB.")

    existing_pairs = {
        tuple(sorted((doc["x"], doc["y"])))
        for doc in results
    }

    expected_pairs = {
        tuple(sorted((str(i), str(j))))
        for i, j in combinations(ids, 2)
    }

    missing_pairs = expected_pairs - existing_pairs

    return results, missing_pairs


def process_chunk_wrapper(id_to_trade_df, id_to_profit_map, id1, id2):
    x, y = str(id1), str(id2)
    if x == y:
        return None
    df1, df2 = id_to_trade_df[id1], id_to_trade_df[id2]
    c = calculate_trade_correlation_vectorized(df1, df2)
    ct = round(c, 4)

    prof1, prof2 = id_to_profit_map[id1], id_to_profit_map[id2]
    common_days = set(prof1.keys()) & set(prof2.keys())
    if len(common_days) >= 2:
        list1 = [prof1[d] for d in sorted(common_days)]
        list2 = [prof2[d] for d in sorted(common_days)]
        corr = np.corrcoef(list1, list2)[0, 1]
        max_days = max(len(prof1), len(prof2))
        cp = round(corr * len(common_days) / max_days * 100, 2)
    else:
        cp = 0
    return {"x": x, "y": y, "ct": ct, "cp": cp}

def calculate_trade_dfs_and_profits(all_alpha_docs, df1d_coll):
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()
    start = "2024_01_01"
    end = "2024_06_01"
    id_to_trade_df = {}
    id_to_profit_map = {}
    id_to_report = {}
    for alpha_doc in all_alpha_docs:
        alpha_id = alpha_doc["_id"]
        alpha_name = alpha_doc.get("alpha_name", "")
        gen = alpha_doc.get("gen", "1_2")
        wfo = alpha_doc.get("wfo",{})
        configs = wfo.get("correlation", {}).get("results", {}).get("strategies", [])
        
        _id = make_key_mega(
                configs=configs,
                alpha_name=alpha_name,
                start=start,
                end=end,
                fee=0.175,
                stop_loss=0,
                gen=gen,
            )
        df_doc = df1d_coll.find_one({"_id": _id})
        if df_doc is None:
            df_doc = {}
        if df_doc.get("df_1d", None) is not None and df_doc.get("df_trade", None) is not None:
            df_trade = pd.DataFrame(df_doc["df_trade"])
            df_trade["executionT"] = pd.to_datetime(df_trade["executionT"], errors="coerce")
            df_trade.dropna(subset=["executionT"], inplace=True)
            id_to_trade_df[alpha_id] = df_trade
            profit_data = {
                row['day']: row['netProfit']
                for row in df_doc.get('df_1d', [])
                if 'day' in row and 'netProfit' in row
            }
            id_to_profit_map[alpha_id] = profit_data
            report = df_doc.get("report", {})
            report['name'] = alpha_doc.get("name","")
            id_to_report[alpha_id] = report
            continue
        if not configs and len(configs) == 0:
            continue
        bt = Simulator(
            alpha_name=alpha_name,
            configs=configs,
            dic_freqs=dic_freqs,
            DIC_ALPHAS=DIC_ALPHAS,
            df_tick=None,
            start=start,
            end=end,
            fee=0.175,
            stop_loss=0,
            gen=gen,
            booksize=len(configs)
        )
        bt.compute_mega()
        bt.compute_performance()
        bt.compute_df_trade()
        report = sanitize_for_bson(bt.report)
        df_1d = bt.df_1d.astype(object).to_dict(orient="records")
        df_trade = bt.df_trade.astype(object).to_dict(orient="records")
        df1d_coll.update_one(
            {"_id": _id},
            {
                "$set": {
                    "report": report,
                    "df_trade": df_trade,
                    "df_1d": df_1d,
                }
            },
            upsert=True
        )
        df_trade = pd.DataFrame(df_trade)
        df_trade["executionT"] = pd.to_datetime(df_trade["executionT"], errors="coerce")
        df_trade.dropna(subset=["executionT"], inplace=True)
        id_to_trade_df[alpha_id] = df_trade
        profit_data = {
            row['day']: row['netProfit']
            for row in df_1d    
            if 'day' in row and 'netProfit' in row
        }
        id_to_profit_map[alpha_id] = profit_data
        id_to_report[alpha_id] = report
    return id_to_trade_df, id_to_profit_map, id_to_report

def calculate_correlation(id_to_trade_df, id_to_profit_map, correlation_coll):
    results = []
    valid_ids = list(id_to_trade_df.keys())
    
    # === 2️⃣ Lọc các cặp đã tồn tại (Giống code chuẩn) ===
    print("🔎 Đang lấy danh sách cặp đã tồn tại trong MongoDB...")
    str_ids = [str(i) for i in valid_ids]
    existing_pairs = set()
    
    # Query hiệu quả hơn thay vì dùng 2 $in lớn
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            {"x": 1, "y": 1, "_id": 0, "ct": 1, "cp": 1}
        )
        for doc in cursor:
            results.append(doc)
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


    if not all_pairs:
        print("🏁 Không còn cặp nào để xử lý. Kết thúc.")
        return


    
    for pair in all_pairs:
        result = process_chunk_wrapper(id_to_trade_df, id_to_profit_map, pair[0], pair[1])
        # Insert batch if it reaches 5000 records
        if result is not None:
            results.append(result)

    if results:
        correlation_coll.insert_many(results)
        print(f"✅ Đã chèn thêm {len(results):,} cặp mới vào DB.")
    return results

def load_reports_only(all_alpha_docs, df1d_coll):
    start = "2024_01_01"
    end = "2024_06_01"

    id_to_report = {}

    for alpha_doc in all_alpha_docs:
        alpha_id = alpha_doc["_id"]
        name = alpha_doc.get("name", "")
        alpha_name = alpha_doc.get("alpha_name", "")
        gen = alpha_doc.get("gen", "1_2")
        configs = alpha_doc.get("wfo", {}).get("correlation", {}).get("results", {}).get("strategies", [])

        if not configs:
            continue

        _id = make_key_mega(
            configs=configs,
            alpha_name=alpha_name,
            start=start,
            end=end,
            fee=0.175,
            stop_loss=0,
            gen=gen,
        )

        df_doc = df1d_coll.find_one(
            {"_id": _id},
            {"report": 1}
        )
        df_doc['report']['name'] = name
        if df_doc and "report" in df_doc:
            id_to_report[alpha_id] = df_doc["report"]

    return id_to_report


def mega_correlation(ids):
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    df1d_coll = alpha_db["df1d"]
    correlation_coll = alpha_db["mega_correlation"]
    ids = [ObjectId(i) for i in ids]
    all_alpha_docs = list(alpha_collection.find({"_id": {"$in": ids}},{"_id":1,"name":1, "alpha_name":1, "gen":1, "wfo.correlation.results.strategies":1}))
    # all_alpha_docs = list(alpha_collection.find({"wfo.correlation.results.strategies":{"$exists": True}},{"_id":1,"name":1, "alpha_name":1, "gen":1, "wfo.correlation.results.strategies":1}))
    for doc in all_alpha_docs:
        doc['_id'] = str(doc['_id'])
    list_ids = [str(d['_id']) for d in all_alpha_docs]
    # ✅ 1. Check correlation trước
    existing_results, missing_pairs = fetch_existing_correlations(
        correlation_coll, list_ids
    )

    if not missing_pairs and len(existing_results) > 0:
        print("✅ Correlation đã đầy đủ — trả về luôn, không chạy Simulator.")
        id_to_report = load_reports_only(all_alpha_docs, df1d_coll)
        return existing_results, id_to_report

    print(f"⚠️ Thiếu {len(missing_pairs)} cặp — bắt đầu tính thêm...")

    # ❌ CHỈ CHẠY KHI THIẾU
    id_to_trade_df, id_to_profit_map, id_to_report = \
        calculate_trade_dfs_and_profits(all_alpha_docs, df1d_coll)

    correlations = calculate_correlation(
        id_to_trade_df=id_to_trade_df,
        id_to_profit_map=id_to_profit_map,
        correlation_coll=correlation_coll
    )

    # Gộp kết quả cũ + mới
    all_results = existing_results + (correlations or [])

    return all_results, id_to_report

    

def main():
    ids = []  
    all_results, id_to_report = mega_correlation(ids)
    # print(all_results)
if __name__ == "__main__":
    main()
    
    
 