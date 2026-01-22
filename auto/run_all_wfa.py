

from time import sleep
from bson import ObjectId
from pymongo import MongoClient

from auto.wfa_cpcv import cpcv
from auto.mega import os_wfa_backtest
from auto.utils import get_mongo_uri, send_telegram_message, setup_logger
from auto.view_correl import view_wfa_correlation
from auto.wfa_correlation import correlation as run_correlation

def get_next_wfa(alpha):
    for idx, fa in enumerate(alpha.get("wfa", [])):
        report = fa.get("filter_report")
        if not report or not report.get("strategies"):
            continue

        correlation = fa.get("correlation", {})
        if correlation.get("status") != "done":
            return idx, fa

        if fa.get("report", {}).get("tvr") is None and fa.get("os",{}).get("end") !="2026_07_01" and fa.get("os",{}).get("end") !="2026_04_01":
            return idx, fa

    return None, None

def run_all_wfa(alpha_id: str):
    
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_collection = mongo_client["alpha"]["alpha_collection"]
    alpha = alpha_collection.find_one({"_id": ObjectId(alpha_id)})
    if not alpha:
        raise ValueError("alpha not found")
    alpha_collection.update_one(
        {"_id": ObjectId(alpha_id)},
        {"$set": {"wfa_status": "running"}},
    )
    logger = setup_logger(f"{alpha.get('name', 'unknown')}_run_all_wfa")
    try:
        while True:
            # 🔁 LUÔN query lại DB
            alpha = alpha_collection.find_one({"_id": ObjectId(alpha_id)})
            if not alpha:
                raise ValueError("alpha not found")

            idx, fa = get_next_wfa(alpha)
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
                run_correlation(id=alpha_id, start=start, end=end)

            # ---- View correlation ----
            if not correlation_results:
                logger.info("View correlation result")
                view_wfa_correlation(id=alpha_id, start=start, end=end)

            sleep(2)

            # ---- OS Backtest ----
            if fa.get("report", {}).get("tvr") is None:
                logger.info("Run OS WFA backtest")
                os_wfa_backtest(id=alpha_id, start=start, end=end)

            logger.info("WFA idx=%d done", idx)
        logger.info("Run CPCV backtest", idx)
        cpcv(alpha_id) 
    except Exception:
        logger.exception("run_all_wfa failed")
        raise

    finally:
        alpha_collection.update_one(
            {"_id": ObjectId(alpha_id)},
            {"$set": {"wfa_status": "done"}},
        )
        logger.info("Set wfa_status=done")
        send_telegram_message(
            f"WFA Alpha {alpha.get("group","").replace("%20"," ")} {alpha.get("name","")} :\n"
            "Import: ✅\n"
            "Filter: ✅\n"
            "Running: Done ✅"
        )


        
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python /home/ubuntu/nevir/auto/run_all_wfa.py <alpha_id>")
        return
    alpha_id = sys.argv[1]
    run_all_wfa(alpha_id)

if __name__ == "__main__":
    main()