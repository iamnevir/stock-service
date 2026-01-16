
import random
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

from utils import get_mongo_uri, make_key_alpha

def view_wfa_correlation(id, start, end):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    local_db = MongoClient(get_mongo_uri())['alpha']
    coll = local_db["correlation_backtest"]
    correlation_results = local_db["correlation_results"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    alpha_name = alpha_doc.get("alpha_name", "")
    wfa = alpha_doc.get("wfa", [])
    source = alpha_doc.get("source", None)
    overnight = alpha_doc.get("overnight", False)
    fa = None
    for item in wfa:
        is_data = item.get("is", {})
        if is_data.get("start") == start and is_data.get("end") == end:
            fa = item
            break
    filter_report = fa.get("filter_report", {})
    correl = fa.get("correlation", {})
    strategy_count = correl.get("strategy_count", 20)
    restarts = correl.get("restarts", 30)
    seed_size = correl.get("seed_size", 5)
    threshold = correl.get("threshold", 95)
    gen = alpha_doc.get("gen")
    fee = fa.get("fee", 0.175)
    results = []
    need_configs = filter_report.get("strategies", [])
    
    list_ids = [make_key_alpha(
                config=config,
                fee=fee,
                start=start,
                end=end,
                alpha_name=alpha_name,
                gen=gen,
                source=source,
                overnight=overnight
            ) for config in need_configs]
    total_config = len(list_ids)
    print(f"Total strategies to consider: {total_config}")
    def chunkify(seq, size=1000):
        for i in range(0, len(seq), size):
            yield seq[i:i+size]

    projection = {"_id": 0, "x": 1, "y": 1, "c": 1}
    records = []

    # 2) Chunk theo X (800–1200 là hợp lý cho 2000 IDs)
    for x_chunk in chunkify(list_ids, size=1200):
        cursor = (correlation_results
                .find({"x": {"$in": x_chunk}, "y": {"$in": list_ids}}, projection)
                .hint([("x",1),("y",1),("c",1)])  # nếu dùng index cover
                .batch_size(10_000))
        # Stream để giảm RAM; gom kết quả dần
        records.extend(cursor)

    all_ids = sorted(list({r["x"] for r in records} | {r["y"] for r in records}))
    id_to_idx = {v: i for i, v in enumerate(all_ids)}
    n = len(all_ids)
    corr_matrix = np.zeros((n, n), dtype=np.float32)
    c_matrix = np.zeros((n, n), dtype=np.float32)

    for r in records:
        i, j = id_to_idx[r["x"]], id_to_idx[r["y"]]
        c = r.get("c", 0.0)
        corr_matrix[i, j] = corr_matrix[j, i] = c
        c_matrix[i, j] = c_matrix[j, i] = c

    def score_subset(matrix, indices):
        return sum(matrix[i, j] for i in indices for j in indices if i < j)

    def greedy_once(matrix, strategy_count, seed_size, n):
        selected = set(random.sample(range(n), seed_size))
        while len(selected) < strategy_count:
            remaining = [i for i in range(n) if i not in selected]
            best_candidate = min(remaining, key=lambda c: score_subset(matrix, selected | {c}))
            selected.add(best_candidate)
        return list(selected), score_subset(matrix, selected)
    
    def greedy_with_restart(matrix, strategy_count, restarts, seed_size):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(greedy_once)(matrix, strategy_count, seed_size, matrix.shape[0])
            for _ in range(restarts)
        )
        return min(results, key=lambda x: x[1])[0]
    
    selected_indices = greedy_with_restart(corr_matrix, strategy_count, restarts, seed_size)
    
    selected_arr = np.array(selected_indices)
    i_mat, j_mat = np.triu_indices(len(selected_indices), k=1)
    to_remove = set()
    for i, j in zip(i_mat, j_mat):
        if corr_matrix[selected_arr[i], selected_arr[j]] > threshold:
            to_remove.update([selected_arr[i], selected_arr[j]])
    
    final_selected = [i for i in selected_indices if i not in to_remove]
    if len(final_selected) < 2:
        return [], 0.0, 0.0, 0.0
    
    selected_ids = [all_ids[i] for i in final_selected]
    docs = list(coll.find({"_id": {"$in": selected_ids}}, {"_id": 1, "config": 1, "report": 1}))
    id_to_label = {doc["_id"]: (doc["config"] if doc.get("config") else str(doc["_id"])) for doc in docs}
    
    result = []
    for i in range(len(final_selected)):
        for j in range(len(final_selected)):
            if i != j:
                idx_i, idx_j = final_selected[i], final_selected[j]
                x_id, y_id = all_ids[idx_i], all_ids[idx_j]
                result.append({
                    "x": id_to_label.get(x_id, str(x_id)),
                    "y": id_to_label.get(y_id, str(y_id)),
                    "c": float(c_matrix[idx_i, idx_j]),
                })

    triangle_ids = {all_ids[i] for i in final_selected}
    triangle_docs = [doc for doc in docs if doc["_id"] in triangle_ids]
    # Lấy list các config duy nhất
    unique_configs = list({doc.get("config") for doc in triangle_docs if doc.get("config") is not None})

    sharpe_values = [
        doc.get("report", {}).get("sharpe", 0.0)
        for doc in triangle_docs if doc.get("report", {}).get("sharpe") is not None
    ]
    # print(sharpe_values)
    avg_sharpe = float(np.mean(sharpe_values)) if sharpe_values else 0.0

    idx_pairs = np.triu_indices(len(final_selected), k=1)
    sub_c = c_matrix[np.ix_(final_selected, final_selected)][idx_pairs]
    tvr_values = [
        doc.get("report", {}).get("tvr", 0.0)
        for doc in triangle_docs if doc.get("report", {}).get("tvr") is not None
    ]
    avg_tvr = float(np.mean(tvr_values)) if tvr_values else 0.0
    avg_corr = float(np.mean(sub_c))
    results = {
        "strategies": unique_configs,
        "result": result,
        "avg_sharpe": avg_sharpe,
        "avg_tvr": avg_tvr,
        "avg_corr": avg_corr,
    }
    alpha_collection.update_one(
            {"_id": ObjectId(id),"wfa.is.start": start,"wfa.is.end": end},
            {"$set": {
                "wfa.$.correlation.results": results,
                "wfa.$.book_size": len(unique_configs),
            }}
        )       
    return True