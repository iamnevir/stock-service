import re
import sys
import numpy as np
from pymongo import MongoClient
from bson import ObjectId

from busd_auto.utils import get_mongo_uri, make_key
from busd_auto.wfo import gen_strategies


def apply_filter(result_docs, is_filter):
    def check_condition(value, op, thr):
        if value is None:
            return False
        if op == ">": return value > thr
        if op == ">=": return value >= thr
        if op == "<": return value < thr
        if op == "<=": return value <= thr
        if op == "==": return value == thr
        if op == "!=": return value != thr
        return False

    passed_strategies = []

    for doc in result_docs:
        ok = True
        for field, cond in is_filter.items():
            value = doc.get(field)
            for op, thr in cond.items():
                if not check_condition(value, op, thr):
                    ok = False
                    break
            if not ok:
                break
        if ok and "strategy" in doc:
            passed_strategies.append(doc["strategy"])

    return passed_strategies
def auto_tune_netprofit_stepwise(
    result_docs,
    is_filter,
    target,
    max_iter=500
):
    """
    Stepwise + Adaptive + Cache
    Always returns closest result
    """
    t_min = target.get("min", 3800)
    t_max = target.get("max", 4200)
    min_profit = target.get("min_profit", 0)
    base_step = target.get("step_profit", 2)

    net_cond = is_filter.get("profitPct", {})
    sharpe_cond = is_filter.get("sharpe", None)
    mdd_cond = is_filter.get("mddPct", None)
    tvr_cond = is_filter.get("tvr", None)
    profit = net_cond[">"]   # start at hi
    
    if sharpe_cond:
        is_filter["sharpe"] = sharpe_cond
    if mdd_cond:
        is_filter["mddPct"] = mdd_cond
    if tvr_cond:
        is_filter["tvr"] = tvr_cond
        
    
    
    if ">" not in net_cond:
        print("❌ Missing profitPct '>' filter.")
        return {
            "success": False,
            "profit": profit,
            "reason": "missing profitPct filter"
        }

    
    cache = {}
    best = None

    for i in range(max_iter):
        if profit < min_profit:
            break

        # ---- CACHE ----
        if profit in cache:
            n = cache[profit]
        else:
            is_filter["profitPct"][">"] = profit
            n = len(apply_filter(result_docs, is_filter))
            cache[profit] = n

        diff = 0
        if n < t_min:
            diff = t_min - n
        elif n > t_max:
            diff = n - t_max

        # update best (closest to target band)
        if best is None or diff < best["diff"]:
            best = {
                "profit": profit,
                "n": n,
                "diff": diff
            }

        # 🎯 SUCCESS
        if t_min <= n <= t_max:
            print(f"✅ Auto-tune success after {i+1} iterations: Profit={profit}, Strategies={n}")
            return {
                "success": True,
                "profit": profit,
                "n": n,
                "iterations": i + 1
            }

        # ---- ADAPTIVE STEP ----
        if diff > 20000:
            step = base_step * 10
        elif diff > 5000:
            step = base_step * 4
        elif diff > 1000:
            step = base_step * 2
        else:
            step = base_step

        # ---- DIRECTION ----
        if n > t_max:
            profit += step  # siết
        else:
            profit -= step  # nới

    # ❌ FAIL → trả về best gần nhất
    print(f"❌ Auto-tune failed after {max_iter} iterations. Best Profit={best['profit']}, Strategies={best['n']}.")
    return {
        "success": False,
        "profit": best["profit"],
        "n": best["n"],
        "diff": best["diff"],
        "iterations": max_iter
    }


def filter_all_wfa_auto(id):
    FEE, DELAY = 175, 1
    mongo = MongoClient(get_mongo_uri())
    db = mongo["busd"]

    busd_coll = db["busd_collection"]
    busd_results = db["wfa_results"]

    busd_doc = busd_coll.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd.")
        return None
    wfa_list = busd_doc.get("wfa", [])
    busd_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set":{"wfa_filtering": True}}
    )
    for idx, fa in enumerate(wfa_list):
        is_filter = fa.get("filter", {})
        target = fa.get("filter_target", {})
        is_range = fa.get("is", {})
        prev_report = fa.get("filter_report", {})
        th = fa.get("threshold", {})
        es = fa.get("exit_strength", {})
        if prev_report.get("passed", 0) > target.get("min", 3800) and prev_report.get("passed", 0) < target.get("max", 4200):
            print(f"Skipping WFA {idx+1}/{len(wfa_list)}: already has suitable filter")
            continue
        start, end = is_range.get("start"), is_range.get("end")
        if not is_filter:
            print(f"Skipping WFA {idx+1}/{len(wfa_list)}: missing filter or target")
            continue
        start = int(is_range["start"])
        end = int(is_range["end"])
        th_list = list(range(th['start'], th['end'] + th['step'], th['step']))
        es_list = list(range(es['start'], es['end'] + es['step'], es['step']))

        list_keys = []
        for ma1 in range(6, 151):
            for ma2 in range(5, ma1):
                for th in th_list:
                    for es in es_list:
                        cfg = f"{ma1}_{ma2}_{th}_{es}"
                        _id = make_key(cfg, start, end,source=busd_doc.get("source",""))
                        list_keys.append(_id)

        print("Total keys:", len(list_keys))

        result_docs = []
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        for chunk in chunks(list_keys, 100_000):
            cursor = busd_results.find({"_id": {"$in": chunk}})
            result_docs.extend(list(cursor))

        total = len(result_docs)
        print("Found results:", total)
        
        if total == 0:
            busd_coll.update_one(
                {"_id": ObjectId(id)},
                {
                    "$set": {
                        f"wfa.{idx}.filter_message": "Filter failed: No result documents found.",
                    }
                }
            )
            continue 

        result = auto_tune_netprofit_stepwise(
            result_docs=result_docs,
            is_filter=is_filter,
            target=target
        )

        if result["success"]:
            is_filter["profitPct"][">"] = result["profit"]
            status = "success"
        else:
            # vẫn dùng kết quả gần nhất
            is_filter["profitPct"][">"] = result["profit"]
            status = "fallback"
        # ❌ không tìm được threshold phù hợp
        if  status == "fallback":
            print("❌ Auto filter failed.")
            report = {
                "total": len(result_docs),
                "passed": result["n"],
                "pass_rate": round(result["n"] / len(result_docs) * 100, 2),
                "strategies": apply_filter(result_docs, is_filter),
            }
            busd_coll.update_one(
                {"_id": ObjectId(id)},
                {
                    "$set": {
                        f"wfa.{idx}.filter_message": f"Filter failed: Last Profit={result['profit']}, Strategies={result['n']}.",
                        f"wfa.{idx}.filter": is_filter,
                        f"wfa.{idx}.filter_report": report,
                        
                    }
                }
            )
            continue

        # ✅ Thành công
        strategies = apply_filter(result_docs, is_filter)
        n = len(strategies)

        report = {
            "total": len(result_docs),
            "passed": n,
            "pass_rate": round(n / len(result_docs) * 100, 2),
            "strategies": strategies,
        }
        
        busd_coll.update_one(
            {"_id": ObjectId(id)},
            {
                "$set": {
                    f"wfa.{idx}.filter_report": report,
                    f"wfa.{idx}.filter_message": f"Auto filter success Profit={result['profit']}, Strategies={n}.",
                    f"wfa.{idx}.filter": is_filter,
                    
                }
            }
        )
        
        print(f"✅ WFA {idx+1}/{len(wfa_list)} filter applied: Profit={result['profit']}, Strategies={n}.")
        
    busd_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set":{"wfa_filtering": False}}
    )
    print("✅ Done filtering all WFA")


def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/filter_all_wfa.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    filter_all_wfa_auto(_id)

if __name__ == "__main__":
    main()
