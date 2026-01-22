

from time import sleep
from bson import ObjectId
from pymongo import MongoClient

from api.utils import send_telegram_message
from busd_auto.mega import os_wfa_backtest
from busd_auto.utils import get_mongo_uri, setup_logger
from busd_auto.view_correl import view_wfa_correlation
from busd_auto.wfa_correlation import correlation as run_correlation

def get_next_wfa(busd):
    for idx, fa in enumerate(busd.get("wfa", [])):
        report = fa.get("filter_report")
        if not report or not report.get("strategies"):
            continue

        correlation = fa.get("correlation", {})
        if correlation.get("status") != "done":
            return idx, fa

        if fa.get("report", {}).get("tvr") is None and fa.get("os",{}).get("end") <= 20260101:
            return idx, fa

    return None, None

def run_all_wfa(busd_id: str):
    
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_collection = mongo_client["busd"]["busd_collection"]
    busd = busd_collection.find_one({"_id": ObjectId(busd_id)})
    if not busd:
        raise ValueError("busd not found")
    busd_collection.update_one(
        {"_id": ObjectId(busd_id)},
        {"$set": {"wfa_status": "running"}},
    )
    logger = setup_logger(f"{busd.get('name', 'unknown')}_run_all_wfa")
    try:
        while True:
            # 🔁 LUÔN query lại DB
            busd = busd_collection.find_one({"_id": ObjectId(busd_id)})
            if not busd:
                raise ValueError("busd not found")

            idx, fa = get_next_wfa(busd)
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
                run_correlation(id=busd_id, start=start, end=end)

            # ---- View correlation ----
            if not correlation_results:
                logger.info("View correlation result")
                view_wfa_correlation(id=busd_id, start=start, end=end)

            sleep(2)

            # ---- OS Backtest ----
            if fa.get("report", {}).get("tvr") is None:
                logger.info("Run OS WFA backtest")
                os_wfa_backtest(id=busd_id, start=start, end=end)

            logger.info("WFA idx=%d done", idx)

    except Exception:
        logger.exception("run_all_wfa failed")
        raise

    finally:
        busd_collection.update_one(
            {"_id": ObjectId(busd_id)},
            {"$set": {"wfa_status": "done"}},
        )
        logger.info("Set wfa_status=done")
        msg = (
            f"WFA Group {busd.get('group','').replace('%20',' ')} Name: {busd.get('name','')} :\n"
            "Import: ✅\n"
            "Filter: ✅\n"
            "Running: Done ✅"
        )
        send_telegram_message(msg)


        
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/busd_auto/run_all_wfa.py <busd_id>")
        return
    busd_id = sys.argv[1]
    run_all_wfa(busd_id)

if __name__ == "__main__":
    main()