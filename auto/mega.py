
from time import time
import pandas as pd
from utils import get_mongo_uri, load_dic_freqs, make_key_mega, sanitize_for_bson
from gen.alpha_func_lib import Domains
from gen.core_mega import Simulator
from pymongo import MongoClient
from bson import ObjectId

def os_wfa_backtest(id, start, end):
    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]
    # df1d_coll = alpha_db["df1d"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return
    wfa = alpha_doc.get("wfa",[])
    _fa = None
    correlation = None
    for fa in wfa:
        if fa.get("is").get("start") == start and fa.get("is").get("end") == end:
            correlation = fa.get("correlation", {})
            _fa = fa
            break
    if not _fa or not correlation:
        print("❌ Không tìm thấy wfa với os.start và os.end này.")
        return
    alpha_name = alpha_doc.get("alpha_name", "")
    source = alpha_doc.get("source", None)
    overnight = alpha_doc.get("overnight", False)
    gen = alpha_doc.get("gen", "1_2")
    fee = _fa.get("fee", 0.175)
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs(source, overnight)
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
            alpha_name=alpha_name,
            configs=configs,
            dic_freqs=dic_freqs,
            DIC_ALPHAS=DIC_ALPHAS,
            df_tick=df_tick,
            start=start,
            end=end,
            fee=fee,
            stop_loss=stop_loss,
            gen=gen,
            booksize=book_size,
            is_sizing=is_sizing,
            init_sizing=init_sizing,
            source=source,
            overnight=overnight
        )
        bt.compute_mega()
        bt.compute_performance()
        # bt.compute_df_trade()
        # _id = make_key_mega(
        #     configs=configs,
        #     alpha_name=alpha_name,
        #     start=start,
        #     end=end,
        #     fee=fee,
        #     stop_loss=stop_loss,
        #     gen=gen,
        # )
        report_with_params = sanitize_for_bson(bt.report) 
        # df_1d = bt.df_1d.astype(object).to_dict(orient="records")
        # df_trade = bt.df_trade.astype(object).to_dict(orient="records")
        # df1d_coll.update_one(
        #     {"_id": _id},
        #     {
        #         "$set": {
        #             "df_trade": df_trade,
        #             "df_1d": df_1d,
        #             "report": report_with_params,
        #         }
        #     },
        #     upsert=True
        # )
        return report_with_params
    is_report = None
    if _is:
        is_report = mega_backtest(_is.get("start"),_is.get("end"))
    if start == "2026_01_01" or end == "2026_07_01":
        os_report = {}
    else:
        os_report = mega_backtest(start,end)
    alpha_collection.update_one(
        {"_id": ObjectId(id),"wfa.os.start": start,"wfa.os.end": end},
        {
            "$set": {
                "wfa.$.report": os_report,
                "wfa.$.in_report": is_report if is_report else {},
            }
        }
    )
    print(f"✅ MEGA hoàn thành cho wfa {start} - {end} id: {id}")
    


# if __name__ == '__main__':
#     os_mega_backtest("690db3922bd7be97ad347744")  # Example call


"""
{'sharpe': 1.127, 'mdd': 606.05, 'mddPct': 4.5415, 'ppc': 0.486, 'tvr': 14.7213, 'lastProfit': 0.0, 'netProfit': 872.9, 'profitPct': 7.274166666666666, 'max_loss': -254.95, 'max_gross': 425.4, 'winrate': 44.12, 'num_trades': 34}
"""