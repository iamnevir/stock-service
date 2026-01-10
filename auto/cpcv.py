
import sys
from time import time
import pandas as pd
from auto.utils import get_mongo_uri, load_dic_freqs, sanitize_for_bson
from gen.alpha_func_lib import Domains
from gen.core_mega import Simulator
from pymongo import MongoClient
from bson import ObjectId
from itertools import combinations
from datetime import datetime
from math import erf, sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count
from functools import partial
from pymongo import InsertOne

def get_cp_list_from_months(start, end, window_month):
    """
    Chia khoảng từ start đến end thành các đoạn liên tiếp, 
    mỗi đoạn dài window_month tháng, 
    start và end của mỗi đoạn phải là ngày 1 của tháng
    """
    from dateutil.relativedelta import relativedelta
    
    start_dt = datetime.strptime(start, "%Y_%m_%d")
    end_dt = datetime.strptime(end, "%Y_%m_%d")
    
    # Làm tròn start_dt lên ngày 1 của tháng tiếp theo nếu không phải ngày 1
    if start_dt.day != 1:
        start_dt = start_dt.replace(day=1) + relativedelta(months=1)
    
    cp_list = []
    current = start_dt
    
    while current < end_dt:
        # Tính ngày kết thúc của khoảng hiện tại (ngày 1 của tháng sau window_month)
        period_end = current + relativedelta(months=window_month)
        
        # Nếu period_end vượt quá end_dt, dừng lại
        if period_end > end_dt:
            break
            
        cp_list.append(f"{current.strftime('%Y_%m_%d')}-{period_end.strftime('%Y_%m_%d')}")
        
        # Di chuyển sang khoảng tiếp theo
        current = period_end
    
    return cp_list

def calculate_working_days(cv_tuple, workdays_per_year: int = 250) -> int:
    """
    Tính số ngày làm việc từ tuple các khoảng thời gian.

    Args:
        cp_tuple: Tuple các khoảng thời gian, ví dụ ('2024_10_01-2025_01_01', '2025_04_01-2025_07_01')
                    Hoặc một chuỗi đơn '2024_10_01-2025_01_01'
                    Hoặc start, end dạng int/string
        workdays_per_year (int): Số ngày làm việc mỗi năm (mặc định 250).

    Returns:
        int: Số ngày làm việc theo chuẩn workdays_per_year.
    """
    total_days = 0
    
    # Xử lý nếu input là tuple/list các khoảng
    if isinstance(cv_tuple, (tuple, list)):
        for period in cv_tuple:
            if isinstance(period, str) and '-' in period:
                start_str, end_str = period.split('-')
                start_date = datetime.strptime(start_str, "%Y_%m_%d")
                end_date = datetime.strptime(end_str, "%Y_%m_%d")
                total_days += (end_date - start_date).days
    # Xử lý nếu input là chuỗi đơn
    elif isinstance(cv_tuple, str) and '-' in cv_tuple:
        start_str, end_str = cv_tuple.split('-')
        start_date = datetime.strptime(start_str, "%Y_%m_%d")
        end_date = datetime.strptime(end_str, "%Y_%m_%d")
        total_days = (end_date - start_date).days
    # Xử lý legacy format (2 tham số riêng biệt)
    else:
        # Giả sử cp_tuple là start và cần end từ tham số thứ 2
        start = cv_tuple
        end = workdays_per_year if isinstance(workdays_per_year, (int, str)) and not isinstance(workdays_per_year, bool) else None
        if end is None:
            return 0
        if type(start) is int:
            start_date = datetime.strptime(str(start), "%Y%m%d")
            end_date = datetime.strptime(str(end), "%Y%m%d")
        else:
            start_date = datetime.strptime(start, "%Y_%m_%d")
            end_date = datetime.strptime(end, "%Y_%m_%d")
        total_days = (end_date - start_date).days
        workdays_per_year = 250

    # Chuyển đổi sang ngày làm việc
    working_days = total_days * workdays_per_year / 365.0

    return round(working_days)

def dsr(returns,sharpe, sr_benchmark=0.18):
        try:
            def volatility_sharpe(returns):
                sample_size = len(returns)
                skewness = skew(returns)
                _kurtosis = kurtosis(returns, fisher=True, bias=False)
                return np.sqrt((1 - skewness*sharpe + (_kurtosis + 2)/4*sharpe**2) / (sample_size - 1)), sharpe
            
            sigma_sr, sr = volatility_sharpe(returns)
            z = (sr - sr_benchmark) / sigma_sr
            return 0.5 * (1 + erf(z / sqrt(2))) * 100
        except Exception as e:
            return 0
        
def compute_mdd_vectorized(df_1d, equity=300):
    """
    Tính Maximum Drawdown (MDD) có xét đến equity ban đầu.
    - MDD% được tính từ CDD% = (cummax - cumNetProfit) / (equity + cummax)
    """

    if 'cumNetProfit' in df_1d:
        net_profit = df_1d['cumNetProfit']
    else:
        net_profit = df_1d['netProfit'].cumsum()

    cummax = net_profit.cummax()
    cdd = cummax - net_profit
    cdd_pct = (cdd / (equity + cummax) * 100)

    mdd = cdd.cummax().iloc[-1]
    mdd_pct = cdd_pct.cummax()
    cdd_last = cdd.iloc[-1]
    cdd_pct_last = cdd_pct.iloc[-1]

    return mdd, mdd_pct, cdd_last, cdd_pct
    
def compute_performance_with_df1d(cv,equity =300,df_1d=None):
    lst_errs = []
    
    working_day = calculate_working_days(cv)
    try:
        mean = df_1d["netProfit"].mean()
        std = df_1d["netProfit"].std()
        if std and not np.isnan(std):
            sharpe = mean / std * working_day ** 0.5
            daily_sharpe = mean / std
        else:
            sharpe = np.nan
            daily_sharpe = np.nan
    except Exception as e:
        lst_errs.append(f"{e}")
        # U.report_error(e)
        sharpe = -999
    tvr = df_1d['turnover'].mean()
    ppc = df_1d['netProfit'].sum() / (df_1d['turnover'].sum() + 1e-8)

    mdd, mdd_pct, cdd, cdd_pct = compute_mdd_vectorized(df_1d,equity)
    
    returns = df_1d['netProfit'] / equity
    winning_profits = df_1d[df_1d['netProfit'] > 0]['netProfit']
    loss_profits = df_1d[df_1d['netProfit'] < 0]['netProfit']
    net_profit_pos = winning_profits.sum()
    net_profit_neg = loss_profits.sum()
    npf = net_profit_pos / abs(net_profit_neg) if net_profit_neg != 0 else net_profit_pos
    total_profit = winning_profits.sum()
    shares = winning_profits / total_profit
    hhi = (shares ** 2).sum()
    new_report = {
        'sharpe': round(sharpe, 3),
        'mdd': round(mdd, 3),
        'mddPct': round(mdd_pct.iloc[-1], 4),
        "hhi": round(hhi,3),
        "psr": round(dsr(returns, daily_sharpe,0),3),
        'ppc': round(ppc, 4),
        'tvr': round(tvr, 4),
        "npf":float(round(npf, 2)),
        "netProfit": round(df_1d['netProfit'].sum(), 2),
        "profitPct": round(df_1d['netProfit'].sum(), 2) / equity * 100,
        "max_loss": round(df_1d['netProfit'].min(), 2),
        "max_gross": round(df_1d['netProfit'].max(), 2),
    }
    return new_report

def precompute_df_1d_periods(df_1d, cp_list):
    result = {}
    for period in cp_list:
        start, end = period.split("-")
        df_period = df_1d[(df_1d.index >= start) & (df_1d.index < end)].copy()
        df_period["cumNetProfit"] = df_period["netProfit"].cumsum()
        result[period] = df_period
    return result

def get_df_1d_fast(cv, PRE):
    df_merged = pd.concat([PRE[p] for p in cv])
    df_merged["cumNetProfit"] = df_merged["netProfit"].cumsum()
    return df_merged

def run_chunk_cv(chunk, PRE, equity, id):
    out = []
    for cv in chunk:
        df_1d_cv = get_df_1d_fast(cv, PRE)
        report = compute_performance_with_df1d(
            cv,
            equity=equity,
            df_1d=df_1d_cv
        )
        out.append({"cv": cv, "alpha_id": id, **sanitize_for_bson(report)})
    return out


def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def cpcv(id):
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(id)})
    if not alpha_doc:
        print("❌ Không tìm thấy alpha_collection với id này.")
        return

    alpha_name = alpha_doc.get("alpha_name", "")
    gen = alpha_doc.get("gen", "1_2")
    configs = alpha_doc.get("wfo", {}).get("correlation", {}).get("results", {}).get("strategies",[])
    cpcv = alpha_doc.get("cpcv", {})
    start = "2018_01_01"
    end = "2025_09_08"
    cp_list = get_cp_list_from_months(start, end, cpcv.get("window_month",6))
    gen_path = cpcv["gen_path"]
    # gen_path =  3
    cv_list = list(combinations(cp_list, gen_path))
    print(f"🔍 CPCV for alpha: {alpha_name}, total CV combinations: {len(cv_list)}")
    stop_loss = cpcv.get(f"stop_loss", 0)
    book_size = cpcv.get(f"book_size", 1)
    is_sizing = cpcv.get(f"is_sizing",False)
    init_sizing = cpcv.get(f"init_sizing",30)
    fee = alpha_doc.get("wfo", {}).get("fee", 0.175)
    # fee = 0.175
    DIC_ALPHAS = Domains.get_list_of_alphas()
    dic_freqs = load_dic_freqs()
    df_tick = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    reports = []
    
    if not configs and len(configs) == 0:
        print("❌ Không có cấu hình chiến lược nào để chạy MEGA.")
        return
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
        init_sizing=init_sizing
    )
    bt.compute_mega()
    start_time = time()
    df_1d = bt.df_1d

    PRE = precompute_df_1d_periods(df_1d, cp_list)
    equity = 300 * book_size
    CHUNK_SIZE = 10000
    chunks = list(chunkify(cv_list, CHUNK_SIZE))
    print(f"🔧 Running {len(chunks)} chunks x {CHUNK_SIZE} CV per process")

    # --- multiprocessing ---
    with Pool(processes=20) as pool:
        fn = partial(run_chunk_cv, PRE=PRE, equity=equity, id=id)
        results = pool.map(fn, chunks)

    reports = [item for sub in results for item in sub]

    if reports:
        # Duyệt 1 lần qua reports → giảm 3 lần list comprehension xuống còn 1
        mdd_pcts = []
        profit_pcts = []
        sharpes = []

        for r in reports:
            if 'mddPct' in r:
                mdd_pcts.append(r['mddPct'])
            if 'profitPct' in r:
                profit_pcts.append(r['profitPct'])
            if 'sharpe' in r:
                sharpes.append(r['sharpe'])

        def stats(arr, mean_round=4, reverse=False):
            if not arr:
                return {"mean": 0, "min": 0, "max": 0, "75pct": 0, "95pct": 0, "99pct": 0}

            arr_np = np.asarray(arr)

            if reverse:  
                # Ví dụ MDD: cao là xấu → đảo mảng
                arr_sort = np.sort(arr_np)[::-1]
            else:
                # Ví dụ Profit: thấp là xấu
                arr_sort = np.sort(arr_np)

            return {
                "mean": round(arr_np.mean(), mean_round),
                "min": round(arr_np.min(), mean_round),
                "max": round(arr_np.max(), mean_round),
                "75pct": round(arr_sort[int(0.75 * (len(arr_sort)-1))], mean_round),
                "95pct": round(arr_sort[int(0.95 * (len(arr_sort)-1))], mean_round),
                "99pct": round(arr_sort[int(0.99 * (len(arr_sort)-1))], mean_round),
            }


        statistics = {
            "mddPct": stats(mdd_pcts, 4, reverse=False),   # MDD cao → xấu
            "profitPct": stats(profit_pcts, 4, reverse=True),  # Profit thấp → xấu
            "sharpe": {"mean": round(np.mean(sharpes), 3) if sharpes else 0},
        }


        print("📊 Thống kê CPCV:")
        m = statistics["mddPct"]
        p = statistics["profitPct"]
        s = statistics["sharpe"]

        print(f"  MDD%: mean={m['mean']}, min={m['min']}, max={m['max']}, 75pct={m['75pct']}, 95pct={m['95pct']}, 99pct={m['99pct']}")
        print(f"  Profit%: mean={p['mean']}, min={p['min']}, max={p['max']}, 75pct={p['75pct']}, 95pct={p['95pct']}, 99pct={p['99pct']}")
        print(f"  Sharpe: mean={s['mean']}")

    else:
        statistics = {}

    print(f"⏱️ CPCV completed in {time() - start_time:.2f} seconds.")
    # new collection
    cpcv_collection = alpha_db["alpha_cpcv"]
    cpcv_collection.delete_many({"alpha_id": id})
    # Lưu batch vào collection
    batch_size = 2000
    ops = []

    for report in reports:
        ops.append(InsertOne(report))

        if len(ops) == batch_size:
            cpcv_collection.bulk_write(ops, ordered=False)
            ops.clear()

    # Insert phần còn lại
    if ops:
        cpcv_collection.bulk_write(ops, ordered=False)

    # Cập nhật alpha document không chứa reports lớn nữa
    alpha_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "cpcv.path_count": len(cv_list),
            "cpcv.statistics": statistics
        }}
    )

    


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: /home/ubuntu/anaconda3/envs/backtest/bin/python /home/ubuntu/nevir/auto/cpcv.py <_id>")
#         sys.exit(1)

#     _id = sys.argv[1]

#     cpcv(_id)

# if __name__ == "__main__":
#     main()