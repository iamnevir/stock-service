

from time import sleep
from bson import ObjectId
from pymongo import MongoClient

from api.utils import send_telegram_message
from base_auto.mega import os_wfa_backtest
from base_auto.utils import get_mongo_uri, setup_logger
from base_auto.view_correl import view_wfa_correlation
from base_auto.wfa_correlation import correlation as run_correlation

def get_next_wfa(base):
    for idx, fa in enumerate(base.get("wfa", [])):
        report = fa.get("filter_report")
        if not report or not report.get("strategies"):
            continue

        correlation = fa.get("correlation", {})
        if correlation.get("status") != "done":
            return idx, fa

        if fa.get("report", {}).get("tvr") is None and fa.get("os",{}).get("end") !="2026_07_01" and fa.get("os",{}).get("end") !="2026_04_01":
            return idx, fa

    return None, None

def run_all_wfa(base_id: str):
    
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    base_collection = mongo_client["base"]["base_collection"]
    base = base_collection.find_one({"_id": ObjectId(base_id)})
    if not base:
        raise ValueError("base not found")
    base_collection.update_one(
        {"_id": ObjectId(base_id)},
        {"$set": {"wfa_status": "running"}},
    )
    logger = setup_logger(f"{base.get('name', 'unknown')}_run_all_wfa")
    try:
        while True:
            # 🔁 LUÔN query lại DB
            base = base_collection.find_one({"_id": ObjectId(base_id)})
            if not base:
                raise ValueError("base not found")

            idx, fa = get_next_wfa(base)
            if fa is None:
                logger.info("No more WFA to process")
                break

            start = fa.get("is", {}).get("start")
            end = fa.get("is", {}).get("end")

            logger.info(
                "Processing WFA idx=%d | start=%s | end=%s",
                idx+1, start, end
            )

            correlation = fa.get("correlation", {})
            correlation_status = correlation.get("status")
            correlation_results = correlation.get("results", {}).get("strategies", [])

            # ---- Correlation ----
            if correlation_status != "done":
                logger.info("Run correlation")
                run_correlation(id=base_id, start=start, end=end)

            # ---- View correlation ----
            if not correlation_results:
                logger.info("View correlation result")
                view_wfa_correlation(id=base_id, start=start, end=end)

            sleep(2)

            # ---- OS Backtest ----
            if fa.get("report", {}).get("tvr") is None:
                logger.info("Run OS WFA backtest")
                os_wfa_backtest(id=base_id, start=start, end=end)

            logger.info("WFA idx=%d done", idx)

    except Exception:
        logger.exception("run_all_wfa failed")
        raise

    finally:
        base_collection.update_one(
            {"_id": ObjectId(base_id)},
            {"$set": {"wfa_status": "done"}},
        )
        logger.info("Set wfa_status=done")
        msg = (
            f"WFA Group {base.get('group','').replace('%20',' ')} Name: {base.get('name','')} :\n"
            "Import: ✅\n"
            "Filter: ✅\n"
            "Running: Done ✅"
        )
        send_telegram_message(msg)



        
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python /home/ubuntu/nevir/base_auto/run_all_wfa.py <base_id>")
        return
    base_id = sys.argv[1]
    run_all_wfa(base_id)

if __name__ == "__main__":
    main()