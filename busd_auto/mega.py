
from time import time
import pandas as pd
from busd_auto.utils import get_mongo_uri, load_data, sanitize_for_bson
from pymongo import MongoClient
from bson import ObjectId
from busd_auto.MegaBbAccV2 import MegaBbAccV2 as Simulator

def os_wfa_backtest(id, start, end):
    FEE, DELAY = 175, 1
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    busd_db = mongo_client["busd"]
    busd_collection = busd_db["busd_collection"]
    # df1d_coll = busd_db["df1d"]
    
    busd_doc = busd_collection.find_one({"_id": ObjectId(id)})
    if not busd_doc:
        print("❌ Không tìm thấy busd_collection với id này.")
        return False

    source = busd_doc.get("source", "")
    wfa = busd_doc.get("wfa",[])
    fa = {}
    insample = None
    out_sample = None
    for f in wfa:
        if f.get("is").get("start") == start and f.get("is").get("end") == end:
            fa = f
            insample = f.get("is", {})
            out_sample = f.get("os", {})
            
            break
    
        
    correlation = fa.get("correlation", {})
    
    stop_loss = fa.get(f"stop_loss", 0)
    book_size = fa.get(f"book_size", 1)
    is_sizing = fa.get(f"is_sizing",False)
    init_sizing = fa.get(f"init_sizing",30)
    configs = correlation.get("results", {}).get("strategies",[])
    if not configs and len(configs) == 0:
        print("❌ Không có cấu hình để chạy MEGA.")
        return False
    data_start = out_sample.get("start",0)
    data_end = out_sample.get("end",0)
    if data_start == 0 or data_end == 0:
        print("❌ Không có khoảng dữ liệu out-sample để chạy MEGA.")
        return False
    def mega_backtest(df,bt_start,bt_end):
        bt = Simulator(
            configs=configs,
            df_alpha=df,
            data_start=bt_start,
            data_end=bt_end,
            fee=FEE,
            delay=DELAY,
            stop_loss=stop_loss,
            book_size=book_size,
            is_sizing=is_sizing,
            init_sizing=init_sizing
        )
        bt.compute_mas()
        bt.compute_all_position()
        bt.compute_mega_position()
        bt.compute_profit_and_df_1d()
        report = sanitize_for_bson(bt.compute_report(bt.df_1d))
     
        return report
    in_report = {}
    if insample:
        df = load_data(start=start, end=end, source=source)
        in_report = mega_backtest(df, start, end)
    if data_end <= 20260101:
        df = load_data(start=data_start, end=data_end, source=source)
        report = mega_backtest(df, data_start, data_end)
    else:
        report = {}
        print("⚠️ Dữ liệu out-sample vượt quá ngày 2025-12-31, bỏ qua chạy MEGA out-sample.")
    busd_collection.update_one(
        {"_id": ObjectId(id),"wfa.is.start": start,"wfa.is.end": end},
        {
            "$set": {
                "wfa.$.report": report,
                "wfa.$.in_report": in_report,
            }
        }
    )
    print(f"✅ MEGA hoàn thành cho busd_collection id: {id}")
    return True

# if __name__ == '__main__':
#     os_mega_backtest("690db3922bd7be97ad347744")  # Example call


"""
{'sharpe': 1.127, 'mdd': 606.05, 'mddPct': 4.5415, 'ppc': 0.486, 'tvr': 14.7213, 'lastProfit': 0.0, 'netProfit': 872.9, 'profitPct': 7.274166666666666, 'max_loss': -254.95, 'max_gross': 425.4, 'winrate': 44.12, 'num_trades': 34}
"""