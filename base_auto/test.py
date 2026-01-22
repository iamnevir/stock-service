
import multiprocessing as mp
import sys
import  time
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from base_auto.utils import get_mongo_uri, setup_logger, make_key_base
from itertools import combinations, islice

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

def calculate_correlations_core(
    stras,
    logger,
    max_workers=20,
    chunk_size=100000,
):
    """
    PURE FUNCTION:
    - input: stras
    - output: dict {(x, y): c}
    - KHÔNG DB
    """

    logger.info("⏳ Parsing trades...")
    id_to_trade_df = {}

    def parse_trade_doc(doc):
        trades = doc.get("df_trade")
        if not isinstance(trades, list) or not trades:
            return None
        if not all("executionT" in t and "action" in t for t in trades):
            return None
        df = pd.DataFrame(trades)
        df["executionT"] = pd.to_datetime(df["executionT"], errors="coerce")
        df.dropna(subset=["executionT"], inplace=True)
        return df if not df.empty else None

    for doc in stras:
        df = parse_trade_doc(doc)
        if df is not None:
            id_to_trade_df[doc["_id"]] = df

    valid_ids = list(id_to_trade_df.keys())
    all_pairs = list(combinations(valid_ids, 2))

    logger.info(f"🧮 Total pairs: {len(all_pairs)}")

    # chunk
    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                break
            yield chunk

    chunk_size = min(chunk_size, max(1, len(all_pairs) // max_workers))
    chunks = list(chunked(all_pairs, chunk_size))

    args_list = [(chunk, id_to_trade_df) for chunk in chunks]

    results = {}

    with mp.Pool(processes=max_workers) as pool:
        for batch in pool.imap_unordered(process_chunk, args_list):
            for item in batch:
                key = tuple(sorted((item["x"], item["y"])))
                results[key] = item["c"]

    logger.info(f"✅ Finished computing {len(results)} pairs")
    return results

def load_existing_correlations( ids):
    db = MongoClient(get_mongo_uri())['base']
    correlation_coll = db["correlation_results"]
    str_ids = [str(i) for i in ids]

    existing = {}
    for x in str_ids:
        cursor = correlation_coll.find(
            {"x": x, "y": {"$in": str_ids}},
            {"x": 1, "y": 1,"c":1, "_id": 0}
        )
        for doc in cursor:
            key = tuple(sorted((doc["x"], doc["y"])))
            existing[key] = doc["c"]
    return existing

def compare_results(calculated, existing, logger, tolerance=0.01):
    mismatched = []
    missing = []
    extra = []

    for key, c_new in calculated.items():
        if key not in existing:
            missing.append(key)
        else:
            c_old = existing[key]
            if abs(c_new - c_old) > tolerance:
                mismatched.append((key, c_old, c_new))

    for key in existing:
        if key not in calculated:
            extra.append(key)

    logger.info(f"❌ Mismatch: {len(mismatched)}")
    logger.info(f"⚠️ Missing in DB: {len(missing)}")
    logger.info(f"⚠️ Extra in DB: {len(extra)}")

    if mismatched:
        logger.warning("🔍 Sample mismatches:")
        for k, old, new in mismatched[:10]:
            logger.warning(f"{k}: db={old}, calc={new}")

    return {
        "mismatched": mismatched,
        "missing": missing,
        "extra": extra
    }


def correlation_test(base_id, start, end):
    mongo = MongoClient(get_mongo_uri("mgc3"))
    db = mongo["base"]

    base_doc = db["base_collection"].find_one({"_id": ObjectId(base_id)})
    logger = setup_logger(f"TEST_CORR_{start}_{end}")
    local_db = MongoClient(get_mongo_uri())['base']
    wfa = next(
        x for x in base_doc["wfa"]
        if x["is"]["start"] == start and x["is"]["end"] == end
    )

    need_configs = wfa["filter_report"]["strategies"]
    list_ids = [
        make_key_base(
            config=c,
            fee=wfa["fee"],
            start=start,
            end=end,
            base_name=base_doc["base_name"],
            gen=base_doc["gen"],
            source=base_doc.get("source", "hose500")
        )
        for c in need_configs
    ]

    stras = list(local_db["correlation_backtest"].find({"_id": {"$in": list_ids}}))

    calculated = calculate_correlations_core(stras, logger)
    existing = load_existing_correlations(list_ids)

    report = compare_results(calculated, existing, logger)

    logger.info("✅ TEST DONE")
    return report

   
def main():
    if len(sys.argv) < 4:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/base_auto/test.py <_id> <start> <end>")
        sys.exit(1)

    _id = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]

    correlation_test(_id, start, end)

if __name__ == "__main__":
    main()