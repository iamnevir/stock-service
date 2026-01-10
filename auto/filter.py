
import re
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from auto.backtest import ScanParams
from auto.utils import get_mongo_uri, make_key_alpha, sanitize_for_bson
from auto.wfo import gen_strategies

def filter_backtest_results(id):
    """
    Lọc & đánh giá kết quả backtest dựa trên filter (AND):
    - Hỗ trợ:
        "percent(profit > 0)": {">": 80}
        "mean(sharpe)": {">": 1.0}
    """
    mongo_client = MongoClient(get_mongo_uri())
    db = mongo_client["alpha"]
    alpha_coll = db["alpha_collection"]
    backtest_results = db["backtest_results"]

    # --- 1️⃣ Lấy thông tin alpha_doc ---
    alpha_doc = alpha_coll.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return None

    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen")
    backtest_params = alpha_doc.get("backtest", {})
    start = backtest_params.get("start")
    end = backtest_params.get("end")
    fee = backtest_params.get("fee")
    freq = backtest_params.get("freq")
    params = backtest_params.get("params", {})
    backtest_filter = backtest_params.get("filter", {})

    if not backtest_filter:
        print("⚠️ Không có filter trong backtest.")
        return None

    # --- 2️⃣ Sinh toàn bộ list keys từ ScanParams ---
    
    scan_params = ScanParams(
        lst_alpha_names=[alpha_name],
        alpha_params=params,
        freq=freq,
        fee=fee,
        gen=gen
    ).lst_reports
 
    list_keys = [
        make_key_alpha(
            config=cfg['cfg'],
            alpha_name=cfg["alphaName"],
            fee=cfg["fee"],
            start=start,
            end=end,
            gen=gen,
        )
        for cfg in scan_params
    ]
    print("len",len(list_keys))
    result_docs = []
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    for key_chunk in chunks(list_keys, 100_000):
        cursor = backtest_results.find({"_id": {"$in": key_chunk}})
        result_docs.extend([doc for doc in cursor ])
    total = len(result_docs)

    print("Total:", total)
    print("Filter:", result_docs[0] )
    if total == 0:
        print("⚠️ Không tìm thấy kết quả backtest tương ứng.")
        return None

    print(f"📊 Tổng số kết quả: {total}")

    # --- 3️⃣ Helper: trích giá trị ---
    def safe_values(field):
        vals = [doc.get(field) for doc in result_docs if isinstance(doc.get(field), (int, float, np.number))]
        return np.array(vals) if len(vals) > 0 else np.array([])

    # --- 4️⃣ Xử lý từng điều kiện ---
    results = {}
    pass_all = True

    for expr, cond in backtest_filter.items():
        expr = expr.strip()
        percent_value = None
        count_value = None

        # percent(field > x)
        if expr.startswith("%("):
            match = re.match(r"%\((\w+)\s*([<>!=]+)\s*([0-9\.\-]+)\)", expr)
            if not match:
                raise ValueError(f"❌ Sai cú pháp: {expr}")
            field, op, thr = match.groups()
            thr = float(thr)

            vals = safe_values(field)
            if len(vals) == 0:
                count_value = 0
                percent_value = 0
            else:
                if op == ">":     count_value = np.sum(vals > thr)
                elif op == ">=":  count_value = np.sum(vals >= thr)
                elif op == "<":   count_value = np.sum(vals < thr)
                elif op == "<=":  count_value = np.sum(vals <= thr)
                elif op == "==":  count_value = np.sum(vals == thr)
                elif op == "!=":  count_value = np.sum(vals != thr)
                else:             count_value = 0
                percent_value = count_value / len(vals) * 100
            value = percent_value  # value chính bằng %

        # count(field > x)
        elif expr.startswith("count("):
            match = re.match(r"count\((\w+)\s*([<>!=]+)\s*([0-9\.\-]+)\)", expr)
            if not match:
                raise ValueError(f"❌ Sai cú pháp: {expr}")
            field, op, thr = match.groups()
            thr = float(thr)

            vals = safe_values(field)
            if len(vals) == 0:
                count_value = 0
            else:
                if op == ">":     count_value = np.sum(vals > thr)
                elif op == ">=":  count_value = np.sum(vals >= thr)
                elif op == "<":   count_value = np.sum(vals < thr)
                elif op == "<=":  count_value = np.sum(vals <= thr)
                elif op == "==":  count_value = np.sum(vals == thr)
                elif op == "!=":  count_value = np.sum(vals != thr)
                else:             count_value = 0
            value = count_value  # value chính bằng số lượng

        # avg(field)
        elif expr.startswith("avg("):
            field = expr[4:-1]
            vals = safe_values(field)
            value = float(np.mean(vals)) if len(vals) > 0 else 0

        else:
            raise ValueError(f"❌ Không hỗ trợ biểu thức {expr}")

        # kiểm tra điều kiện
        pass_flag = True
        for op, thr in cond.items():
            if op == ">" and not (value > thr): pass_flag = False
            elif op == ">=" and not (value >= thr): pass_flag = False
            elif op == "<" and not (value < thr): pass_flag = False
            elif op == "<=" and not (value <= thr): pass_flag = False
            elif op == "==" and not (value == thr): pass_flag = False
            elif op == "!=" and not (value != thr): pass_flag = False

        # thêm key percent/count nếu có
        result_entry = {"value": round(value, 4), "condition": f"{cond}", "pass": pass_flag}
        if percent_value is not None:
            result_entry["percent"] = round(percent_value, 2)
        if count_value is not None:
            result_entry["count"] = int(count_value)

        results[expr] = result_entry
        if not pass_flag:
            pass_all = False
        # --- 5️⃣ Nếu tất cả điều kiện đều là dạng count(), lấy danh sách chiến thuật thỏa mãn ---
    
    only_count_filters = all(expr.startswith("count(") for expr in backtest_filter.keys())

    strategies_pass = []
    if only_count_filters:
        for doc in result_docs:
            ok = True
            for expr, cond in backtest_filter.items():
                match = re.match(r"count\((\w+)\s*([<>!=]+)\s*([0-9\.\-]+)\)", expr)
                if not match:
                    ok = False
                    break
                field, op, thr = match.groups()
                thr = float(thr)

                val = doc.get(field)
                if not isinstance(val, (int, float)):
                    ok = False
                    break

                if op == ">" and not (val > thr): ok = False
                elif op == ">=" and not (val >= thr): ok = False
                elif op == "<" and not (val < thr): ok = False
                elif op == "<=" and not (val <= thr): ok = False
                elif op == "==" and not (val == thr): ok = False
                elif op == "!=" and not (val != thr): ok = False

                if not ok:
                    break

            if ok:
                strategies_pass.append(doc.get("strategy"))

    # Thêm kết quả vào report
    count_strategies = len(strategies_pass)

    # --- Tạo report tổng hợp ---
    report = sanitize_for_bson({
        "total": int(total),
        "results": results,
        "strategies":strategies_pass,
        "strategy_count": count_strategies,
        "summary": {
            "passed": pass_all,
        }
    })

    # --- Ghi report vào backtest_params ---
    
    alpha_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"backtest.report": report,"wfo.strategies":strategies_pass}}
    )
    return report

def filter_is_results(id):
    """
    Lọc kết quả backtest (AND nhiều điều kiện), 
    chỉ trả về số lượng chiến thuật thỏa tất cả điều kiện.
    Ví dụ filter:
        {"netProfit": {">": 0}, "sharpe": {">": 1}}
    """

    mongo_client = MongoClient(get_mongo_uri())
    db = mongo_client["alpha"]
    alpha_coll = db["alpha_collection"]
    is_results = db["is_results"]

    # 1️⃣ Lấy thông tin alpha_doc
    alpha_doc = alpha_coll.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return None

    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen")
    ios = alpha_doc.get("ios", {})
    start, end, fee, freq = ios.get("start"), ios.get("end"), ios.get("fee"), ios.get("freq")
    params = ios.get("params", {})
    is_filter = ios.get("filter", {})

    if not is_filter:
        print("⚠️ Không có filter trong insample.")
        return None

    # 2️⃣ Lấy danh sách keys từ ScanParams
    scan_params = ScanParams(
        lst_alpha_names=[alpha_name],
        alpha_params=params,
        freq=freq,
        fee=fee,
        gen=gen
    ).lst_reports

    list_keys = [
        make_key_alpha(
            config=cfg['cfg'],
            alpha_name=cfg["alphaName"],
            fee=cfg["fee"],
            start=start,
            end=end,
            gen=gen,
        )
        for cfg in scan_params
    ]

    cursor = list(is_results.find({"_id": {"$in": list_keys}}))
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
        "pass_rate": round(passed / total * 100, 2) if total > 0 else 0
    }

    alpha_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "ios.report": report,
            "ios.passed_strategies": passed_strategies
        }}
    )

    print(f"✅ Số chiến thuật đạt yêu cầu: {passed}/{total} ({report['pass_rate']}%)")
    print(f"🧩 Đã lưu {len(passed_strategies)} chiến thuật vào ios.passed_strategies")
    return report


def filter_wfo_results(id):
    """
    Lọc kết quả backtest (AND nhiều điều kiện), 
    chỉ trả về số lượng chiến thuật thỏa tất cả điều kiện.
    Ví dụ filter:
        {"netProfit": {">": 0}, "sharpe": {">": 1}}
    """

    mongo_client = MongoClient(get_mongo_uri())
    db = mongo_client["alpha"]
    alpha_coll = db["alpha_collection"]
    wfo_results = db["wfo_results"]

    # 1️⃣ Lấy thông tin alpha_doc
    alpha_doc = alpha_coll.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return None

    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen")
    wfo = alpha_doc.get("wfo", {})
    fee, freq = wfo.get("fee"), wfo.get("freq")
    params = wfo.get("params", {})
    
    reports = []
    for period in wfo.get("period", []):
        insample = period.get("is", "")
        start, end = insample.split("-")
        is_filter = period.get("filter",None)
        if not is_filter:
            print("⚠️ Không có filter trong wfo.")
            continue
        # 2️⃣ Lấy danh sách keys từ ScanParams
        if len(wfo.get("strategies",[])) > 5:
            scan_params = gen_strategies(wfo.get("strategies",[]),alpha_name,gen,fee)
        else:
            scan_params = ScanParams(
                lst_alpha_names=[alpha_name],
                alpha_params=params,
                freq=freq,
                fee=fee,
                gen=gen
            ).lst_reports

        list_keys = [
            make_key_alpha(
                config=cfg['cfg'],
                alpha_name=cfg["alphaName"],
                fee=cfg["fee"],
                start=start,
                end=end,
                gen=gen,
            )
            for cfg in scan_params
        ]

        cursor = list(wfo_results.find({"_id": {"$in": list_keys}}))
        total = len(cursor)
        if total == 0:
            print("⚠️ Không tìm thấy kết quả insample tương ứng.")
            return None

        print(f"📊 Tổng số kết quả: {total}")

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
    
    alpha_coll.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "wfo.report": reports,
            "wfo.correlation.strategies":list(common_strategies),
            "wfo.correlation.strategy_length":len(common_strategies)
        }}
    )
    return reports

def filter_wfa_results(id,start,end):
    """
    Lọc kết quả backtest alpha theo filter người dùng (AND logic):
        - %(field > x)
        - count(field > x)
        - avg(field)
    """
    mongo = MongoClient(get_mongo_uri())
    db = mongo["alpha"]

    alpha_coll = db["alpha_collection"]
    alpha_results = db["wfa_results"]

    # --- 1️⃣ Lấy document backtest alpha ---
    alpha_doc = alpha_coll.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_backtest với id này.")
        return None
    params = None
    wfa = alpha_doc.get("wfa", [])
    for fa in wfa:
        if fa.get("is",{}) == {"start":start,"end":end}:
            params = fa
            break
    else:
        print("❌ Không tìm thấy wfa với is này.")
        return None
    is_filter = params.get("filter", {})
    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen")
    fee, freq = params.get("fee"), params.get("freq")
    params = params.get("params", {})
    if not is_filter:
        print("⚠️ Không có filter trong backtest.")
        return None
    else:
        scan_params = ScanParams(
            lst_alpha_names=[alpha_name],
            alpha_params=params,
            freq=freq,
            fee=fee,
            gen=gen
        ).lst_reports

    list_keys = [
        make_key_alpha(
            config=cfg['cfg'],
            alpha_name=cfg["alphaName"],
            fee=cfg["fee"],
            start=start,
            end=end,
            gen=gen,
        )
        for cfg in scan_params
    ]

    print("Total keys:", len(list_keys))

    # --- 3️⃣ Load full results theo chunks 100k ---
    result_docs = []
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for chunk in chunks(list_keys, 100_000):
        cursor = alpha_results.find({"_id": {"$in": chunk}})
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

    alpha_coll.update_one(
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
