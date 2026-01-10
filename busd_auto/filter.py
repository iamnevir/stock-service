
import re
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from busd_auto.utils import get_mongo_uri, make_key
from busd_auto.wfo import gen_strategies

def filter_backtest_busd(id):
    """
    Lọc kết quả backtest BUSD theo filter người dùng (AND logic):
        - %(field > x)
        - count(field > x)
        - avg(field)
    """
    fee, delay = 175, 1
    mongo = MongoClient(get_mongo_uri())
    db = mongo["busd"]

    busd_coll = db["busd_collection"]
    busd_results = db["backtest_results"]

    # --- 1️⃣ Lấy document backtest BUSD ---
    busd_doc = busd_coll.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_backtest với id này.")
        return None

    params = busd_doc.get("backtest", {})
    start = params.get("start")
    end = params.get("end")
    backtest_filter = params.get("filter", {})

    if not backtest_filter:
        print("⚠️ Không có filter trong backtest.")
        return None

    # --- 2️⃣ Sinh toàn bộ list keys từ grid MA, TH, ES ---
    th = params["threshold"]
    es = params["exit_strength"]

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

    # --- 3️⃣ Load full results theo chunks 100k ---
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
        return None

    # --- Helper: lấy giá trị numeric an toàn ---
    def safe_values(field):
        vals = [
            doc.get(field) for doc in result_docs
            if isinstance(doc.get(field), (int, float))
        ]
        return np.array(vals) if len(vals) else np.array([])

    # --- 4️⃣ Xử lý từng filter ---
    results = {}
    pass_all = True

    for expr, cond in backtest_filter.items():
        expr = expr.strip()
        percent_value = None
        count_value = None

        # % (field > x)
        if expr.startswith("%("):
            match = re.match(r"%\((\w+)\s*([<>!=]+)\s*([\d\.\-]+)\)", expr)
            field, op, thr = match.groups()
            thr = float(thr)
            vals = safe_values(field)

            if len(vals) == 0:
                count_value, percent_value = 0, 0
            else:
                ops = {">": vals > thr, ">=": vals >= thr,
                       "<": vals < thr, "<=": vals <= thr,
                       "==": vals == thr, "!=": vals != thr}
                count_value = np.sum(ops[op])
                percent_value = count_value / len(vals) * 100

            value = percent_value

        # count(field > x)
        elif expr.startswith("count("):
            match = re.match(r"count\((\w+)\s*([<>!=]+)\s*([\d\.\-]+)\)", expr)
            field, op, thr = match.groups()
            thr = float(thr)
            vals = safe_values(field)

            if len(vals) == 0:
                count_value = 0
            else:
                ops = {">": vals > thr, ">=": vals >= thr,
                       "<": vals < thr, "<=": vals <= thr,
                       "==": vals == thr, "!=": vals != thr}
                count_value = np.sum(ops[op])

            value = count_value

        # avg(field)
        elif expr.startswith("avg("):
            field = expr[4:-1]
            vals = safe_values(field)
            value = float(np.mean(vals)) if len(vals) else 0

        else:
            raise ValueError(f"❌ Cú pháp không hỗ trợ: {expr}")

        # kiểm tra điều kiện
        pass_flag = True
        for op, thr in cond.items():
            thr = float(thr)
            if op == ">" and not (value > thr): pass_flag = False
            if op == ">=" and not (value >= thr): pass_flag = False
            if op == "<" and not (value < thr): pass_flag = False
            if op == "<=" and not (value <= thr): pass_flag = False
            if op == "==" and not (value == thr): pass_flag = False
            if op == "!=" and not (value != thr): pass_flag = False

        results[expr] = {
            "value": float(round(value, 4)),
            "condition": cond,
            "pass": pass_flag
        }
        if percent_value is not None:
            results[expr]["percent"] = round(percent_value, 2)
        if count_value is not None:
            results[expr]["count"] = int(count_value)

        if not pass_flag:
            pass_all = False

    # --- 5️⃣ Trả về danh sách chiến thuật thỏa ALL filters dạng count() ---
    only_count = all(expr.startswith("count(") for expr in backtest_filter)

    strategies_pass = []
    if only_count:
        for doc in result_docs:
            ok = True
            for expr, cond in backtest_filter.items():
                match = re.match(r"count\((\w+)\s*([<>!=]+)\s*([\d\.\-]+)\)", expr)
                field, op, thr = match.groups()
                thr = float(thr)

                val = doc.get(field)
                if not isinstance(val, (int, float)): 
                    ok = False
                    break

                if op == ">" and not (val > thr): ok = False
                if op == ">=" and not (val >= thr): ok = False
                if op == "<" and not (val < thr): ok = False
                if op == "<=" and not (val <= thr): ok = False
                if op == "==" and not (val == thr): ok = False
                if op == "!=" and not (val != thr): ok = False

                if not ok:
                    break

            if ok:
                strategies_pass.append(doc["strategy"])

    report = {
        "total": total,
        "results": results,
        "strategies": strategies_pass,
        "strategy_count": len(strategies_pass),
        "summary": {"passed": pass_all}
    }

    busd_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"backtest.report": report,"wfo.strategies":strategies_pass}}
    )

    return report

def filter_wfo_results(id):
    """
    Lọc kết quả backtest (AND nhiều điều kiện), 
    chỉ trả về số lượng chiến thuật thỏa tất cả điều kiện.
    Ví dụ filter:
        {"netProfit": {">": 0}, "sharpe": {">": 1}}
    """
    FEE, DELAY = 175, 1
    mongo_client = MongoClient(get_mongo_uri())
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    coll = busd_db["wfo_results"]

    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_collection với id này.")
        return

    source = busd_doc.get("source", "")

    wfo = busd_doc.get("wfo", {})
    threshold = wfo.get("threshold", {})
    exit_strength = wfo.get("exit_strength", {})
    
    periods = wfo.get("period", [])
    
    strategies = wfo.get("strategies",[])
    
    reports = []
    for period in periods:
        insample = period.get("is", {})
        is_filter = period.get("filter", {})
        start = int(insample[0])
        end = int(insample[1])
        if len(strategies) > 5:
            all_jobs = gen_strategies(strategies,source,start,end,FEE,DELAY)
        else:
            all_jobs = []
            th_list = list(range(threshold['start'], threshold['end'] + threshold['step'], threshold['step']))
            es_list = list(range(exit_strength['start'], exit_strength['end'] + exit_strength['step'], exit_strength['step']))
            idx = 0
            for ma1 in range(6, 151):
                for ma2 in range(5, ma1):
                    for th in th_list:
                        for es in es_list:
                            config_id = f"{ma1}_{ma2}_{th}_{es}"
                            _id = make_key(config_id, FEE, DELAY, start, end, source)
                            all_jobs.append((idx, ma1, ma2, th, es, FEE, DELAY, config_id, _id, start, end))
                            idx += 1

        keys = [job[8] for job in all_jobs]  

        cursor = list(coll.find({"_id": {"$in": keys}}))
        total = len(cursor)
        if total == 0:
            print("⚠️ Không tìm thấy kết quả insample tương ứng.")
            return None

        # 3️⃣ Áp dụng filter: tất cả điều kiện đều phải đúng
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

        passed = 0
        passed_strategies = []

        for doc in cursor:
            ok = True
            for field, cond in is_filter.items():
                value = doc.get(field)
                for op, thr in cond.items():
                    if not check_condition(value, op, thr):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                passed += 1
                if "strategy" in doc:
                    passed_strategies.append(doc["strategy"])

        # 4️⃣ Ghi kết quả gọn + danh sách passed_strategies
        report = {
            "total": total,
            "passed": passed,
            "strategies": passed_strategies,
        }
        reports.append({
            "is": insample,
            "report": report
        })
    all_strategies = [set(r["report"].get("strategies", [])) for r in reports]

    # Lấy những strategy có trong tất cả các báo cáo (giao nhau)
    common_strategies = set.intersection(*all_strategies) if all_strategies else set()

    # Sau đó xóa key 'strategies' khỏi mỗi báo cáo
    for r in reports:
        r["report"].pop("strategies", None)
    
    busd_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "wfo.report": reports,
            "wfo.correlation.strategies":list(common_strategies),
            "wfo.correlation.strategy_length":len(common_strategies)
        }}
    )
    return reports

def filter_wfa(id,start,end):
    """
    Lọc kết quả backtest BUSD theo filter người dùng (AND logic):
        - %(field > x)
        - count(field > x)
        - avg(field)
    """
    mongo = MongoClient(get_mongo_uri())
    db = mongo["busd"]

    busd_coll = db["busd_collection"]
    busd_results = db["wfa_results"]

    # --- 1️⃣ Lấy document backtest BUSD ---
    busd_doc = busd_coll.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_backtest với id này.")
        return None

    wfa = busd_doc.get("wfa", [])
    for fa in wfa:
        if fa.get("is",{}) == {"start":start,"end":end}:
            params = fa
            break
    else:
        print("❌ Không tìm thấy wfa với is này.")
        return None
    is_filter = params.get("filter", {})

    if not is_filter:
        print("⚠️ Không có filter trong backtest.")
        return None

    # --- 2️⃣ Sinh toàn bộ list keys từ grid MA, TH, ES ---
    th = params["threshold"]
    es = params["exit_strength"]

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

    # --- 3️⃣ Load full results theo chunks 100k ---
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
        return None

    # --- 4️⃣ Xử lý từng filter ---
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

    passed = 0
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
        if ok:
            passed += 1
            if "strategy" in doc:
                passed_strategies.append(doc["strategy"])

    # 4️⃣ Ghi kết quả gọn + danh sách passed_strategies
    report = {
        "total": total,
        "passed": passed,
        "strategies": passed_strategies,
        "pass_rate": round(passed/total*100,2) if total>0 else 0,
    }

    busd_coll.update_one(
        {
            "_id": ObjectId(id),
            "wfa.is.start": start,
            "wfa.is.end": end
        },
        {
            "$set": {
                "wfa.$.filter_report": report,
            }
        }
    )

    return report
