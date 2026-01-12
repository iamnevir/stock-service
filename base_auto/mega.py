
from time import time
import pandas as pd
from base_auto.utils import get_mongo_uri, load_dic_freqs, make_key_mega, sanitize_for_bson
from gen_spot.base_func_lib import Domains
from gen_spot.core_mega import Simulator
from pymongo import MongoClient
from bson import ObjectId

def os_wfa_backtest(id, start, end):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    base_db = mongo_client["base"]
    base_collection = base_db["base_collection"]
    base_doc = base_collection.find_one({"_id": ObjectId(id)})
    if not base_doc:
        print("❌ Không tìm thấy base_collection với id này.")
        return
    wfa = base_doc.get("wfa",[])
    _fa = None
    correlation = None
    for fa in wfa:
        if fa.get("is").get("start") == start and fa.get("is").get("end") == end:
            correlation = fa.get("correlation", {})
            _fa = fa
            break
    if not _fa or not correlation:
        print(start,end)
        print("❌ Không tìm thấy wfa với os.start và os.end này.")
        return
    base_name = base_doc.get("base_name", "")
    gen = base_doc.get("gen", "1_2")
    fee = _fa.get("fee", 0.175)
    DIC_BASES = Domains.get_list_of_bases()
    source = base_doc.get("source", "hose500")
    dic_freqs = load_dic_freqs(source=source)
    df_tick = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    
    _os = _fa.get("os", None)
    _is = _fa.get("is", None)
    start = _os.get("start")
    end = _os.get("end")
    stop_loss = _fa.get(f"stop_loss", 0)
    book_size = _fa.get(f"book_size", 1)
    is_sizing = _fa.get(f"is_sizing",False)
    init_sizing = _fa.get(f"init_sizing",30)
    configs = correlation.get("results", {}).get("strategies",[])
    if not configs and len(configs) == 0:
        return
    
    def mega_backtest(start,end):
        bt = Simulator(
            base_name=base_name,
            configs=configs,
            dic_freqs=dic_freqs,
            DIC_BASES=DIC_BASES,
            df_tick=df_tick,
            start=start,
            end=end,
            fee=fee,
            stop_loss=stop_loss,
            gen=gen,
            booksize=book_size,
            is_sizing=is_sizing,
            init_sizing=init_sizing
        )
        bt.compute_mega()
        bt.compute_performance()
        report_with_params = sanitize_for_bson(bt.report) 
        return report_with_params
    is_report = None
    if _is:
        is_report = mega_backtest(_is.get("start"),_is.get("end"))
    if start == "2026_01_01" or end == "2026_07_01":
        os_report = {}
    else:
        os_report = mega_backtest(start,end)
    base_collection.update_one(
        {"_id": ObjectId(id),"wfa.os.start": start,"wfa.os.end": end},
        {
            "$set": {
                "wfa.$.report": os_report,
                "wfa.$.in_report": is_report if is_report else {},
            }
        }
    )
    print(f"✅ MEGA hoàn thành cho wfa {start} - {end} id: {id}")
    

