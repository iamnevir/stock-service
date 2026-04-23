import os
import sys
import time
import json
import pickle
import signal
import pymongo
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from core import Simulator, Collector

try:
    sys.path.insert(0, "/home/ubuntu")
    from auto.utils import get_mongo_uri
except ImportError:
    pass

class TimeoutException(Exception):
    pass

class Scan_Params:
    def __init__(self,lst_alpha_names, min_freq, max_freq, fee,
                 shuffle=False, convert_reports_to_dicts=True):

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fee = fee
        self.lst_reports = []

        if lst_alpha_names is not None:
            self.lst_reports = self.gen_lst_reports(
                lst_alpha_names,
                convert_reports_to_dicts=convert_reports_to_dicts,
                shuffle=shuffle)

    def gen_threshold_list(self):
        return [(x, y) for x, y in product(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                if x > y]

    def gen_lst_reports(self, lst_alpha_names, shuffle=False, in_place=True,
                        convert_reports_to_dicts=True):
        from itertools import product
        lst_freqs = list(range(self.min_freq, self.max_freq + 1))
        lst_thresholds = self.gen_threshold_list()
        lst_reports = list(product(lst_alpha_names, lst_freqs, lst_thresholds))

        if shuffle:
            from random import shuffle as apply_shuffle
            apply_shuffle(lst_reports)

        if convert_reports_to_dicts:
            lst = []
            for dp in lst_reports:
                alpha_name, freq, (upper, lower) = dp
                report = {'alphaName': alpha_name, 'freq': freq, 'upper': upper, 'lower': lower, "fee": self.fee}
                lst.append(report)
            lst_reports = lst
        if in_place:
            self.lst_reports = lst_reports
        return lst_reports

def load_dic_freqs():
    fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    with open(fn, 'rb') as file:
        DIC_FREQS = pickle.load(file)
    for freq in DIC_FREQS.keys():
        df = DIC_FREQS[freq].copy()
        df.loc[df['executionTime'] == '14:45:00', 'exitPrice'] = df['open'].shift(-2)
        df['priceChange'] = df['exitPrice'] - df['entryPrice']
        DIC_FREQS[freq] = df
    return DIC_FREQS

def init_worker(freqs, alphas, years):
    global DIC_FREQS, DIC_ALPHAS, DIC_YEAR
    DIC_FREQS = freqs
    DIC_ALPHAS = alphas
    DIC_YEAR = years

def run_simulation(config):
    is_reverse = config.get('reverse', False)
    
    reports = []
    # Gán mặc định report lỗi nếu crash
    err_report = {k: 0 for k in ['sharpe', 'tvr', 'annual_return', 'drawdown', 'pnl']}
    err_report.update({'year': 'all', 'is_reverse': is_reverse})

    try:
        # Kích hoạt báo thức 60 giây cho mỗi lượt backtest lẻ
        import signal
        def local_timeout_handler(signum, frame):
            raise TimeoutException("Simulation Timeout")
        
        signal.signal(signal.SIGALRM, local_timeout_handler)
        signal.alarm(60)

        # Sử dụng trực tiếp hàm từ global DIC_ALPHAS
        alpha_func = DIC_ALPHAS[config['alphaName']]
        
        def execute_alpha(df, **kwargs):
            signal_vals = alpha_func(df, **kwargs)
            return -signal_vals if is_reverse else signal_vals

        bt = Simulator(
            alpha_name=config['alphaName'],
            freq=config['freq'],   
            upper=config['upper'],
            lower=config['lower'],
            fee=config['fee'],
            df_alpha=DIC_FREQS[config['freq']],
            DIC_ALPHAS={config['alphaName']: execute_alpha}
        )
        bt.compute_signal()
        bt.compute_position()
        bt.compute_tvr_and_fee()
        bt.compute_profits()
        
        for year in DIC_YEAR:
            bt.compute_performance(start=DIC_YEAR[year]['start'], end=DIC_YEAR[year]['end'])
            bt.report['year'] = year
            bt.report['is_reverse'] = is_reverse
            reports.append(bt.report.copy())

        signal.alarm(0) # Tắt báo thức nếu xong sớm
    except Exception as e:
        # Nếu có lỗi (Timeout/Syntax/Logic), trả về kết quả rỗng có Sharpe=0
        signal.alarm(0)
        for year in DIC_YEAR:
            tmp = err_report.copy()
            tmp['year'] = year
            reports.append(tmp)
            
    return reports

dic_year = {
    'all' : {'start' : None, 'end' : None},
    '2018' : {'start' : '2018_01_01', 'end' : '2019_01_01'},
    '2019' : {'start' : '2019_01_01', 'end' : '2020_01_01'},
    '2020' : {'start' : '2020_01_01', 'end' : '2021_01_01'},
    '2021' : {'start' : '2021_01_01', 'end' : '2022_01_01'},
    '2022' : {'start' : '2022_01_01', 'end' : '2023_01_01'},
    '2023' : {'start' : '2023_01_01', 'end' : '2024_01_01'},
    '2024' : {'start' : '2024_01_01', 'end' : '2025_01_01'},
    '2025' : {'start' : '2025_01_01', 'end' : '2026_01_01'},
}



def check_is_highly_stable(res):
    """Kiểm tra điều kiện: (s0 >= 50 và s1 >= 40 cho all-time) VÀ s0 hằng năm >= 50%"""
    if not res: return False
    
    # 1. Kiểm tra dòng "all" (s0 >= 50% và s1 >= 40%)
    all_m = res.get("all", [0, 0, 0])
    if all_m[0] < 50 or all_m[1] < 40: return False
    
    # 2. Kiểm tra tính ổn định hằng năm (BẮT BUỘC tất cả các năm từ 2018-2025 phải có s0 >= 50%)
    years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    for y in years:
        if y not in res or res[y][0] < 50:
            return False
    return True

if __name__ == '__main__': 
    alpha_dir = "/home/ubuntu/nevir/huy/Gen_Alpha"
    if alpha_dir not in sys.path:
        sys.path.insert(0, alpha_dir)
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", help="Lọc theo group alpha")
    args = parser.parse_args()

    mongo_client = pymongo.MongoClient(get_mongo_uri("mgc3"))
    db = mongo_client["alpha"]
    alpha_coll = db["gen_alpha"]
    
    # Tạo filter dựa trên group nếu có
    query_filter = {"scan": 0, "run_IC": 1, "is_error": 0}
    if args.group:
        query_filter["group"] = args.group
        print(f">>> Chỉ quét Alpha thuộc group: {args.group}")

    docs = list(alpha_coll.find(query_filter).limit(1))
    if not docs:
        if args.group:
            print(f"Không có Alpha nào thuộc group {args.group} đang chờ Scan Final.")
        else:
            print("Không có Alpha nào đang chờ Scan Final.")
        sys.exit(0)
        
    print(f"Bắt đầu Scan Final tuần tự cho {len(docs)} alpha (Tối ưu CPU)...")
    dic_freqs = load_dic_freqs()
    
    import importlib
    import alpha
    importlib.reload(alpha)
    from alpha import Alpha
    DIC_ALPHAS = {name: getattr(Alpha, name) for name in dir(Alpha) if name.startswith("alpha_")}

    def run_alpha_scan_with_timeout(configs, freqs, alphas, years, alpha_name):
        """Hàm bọc để chạy Pool trong Process riêng - GIA CỐ VẬN HÀNH"""
        num_processes = 20
        results = []
        pool = None
        try:
            pool = multiprocessing.Pool(
                processes=num_processes,
                initializer=init_worker,
                initargs=(freqs, alphas, years)
            )
            
            # Sử dụng imap_unordered và bọc lỗi đường ống (Broken Pipe)
            iterator = pool.imap_unordered(run_simulation, configs)
            
            with tqdm(total=len(configs), desc=f"  Scanning {alpha_name}", leave=False) as pbar:
                while True:
                    try:
                        result_list = next(iterator)
                        results.extend(result_list)
                        pbar.update(1)
                    except StopIteration:
                        break
                    except (BrokenPipeError, EOFError, ConnectionResetError) as pipe_err:
                        print(f"\n   [Scanner Critical] {alpha_name} làm sập đường ống dữ liệu (Broken Pipe).")
                        raise pipe_err # Ném ra ngoài để Pool dọn dẹp
            
            pool.close()
            pool.join()
        except Exception as e:
            if pool:
                try:
                    pool.terminate()
                    pool.join()
                except: pass
            print(f"   [Pool Alert] {alpha_name} bị dừng do lỗi hệ thống hoặc Timeout: {e}")
        return results

    def worker_wrapper(configs, freqs, alphas, years, fn, alpha_name):
        try:
            res = run_alpha_scan_with_timeout(configs, freqs, alphas, years, alpha_name)
            with open(fn, 'wb') as f:
                pickle.dump(res, f)
        except Exception as e:
            print(f"Lỗi trong worker cho {alpha_name}: {e}")

    for doc in docs:
        alpha_name = doc["alpha_name"]
        print(f"\n>>> [Bắt đầu Scan] {alpha_name}")
        
        scan_params = Scan_Params(lst_alpha_names=[alpha_name], min_freq=20, max_freq=100, fee=0.175, shuffle=True)
        lst_configs_all = [dict(c, reverse=False) for c in scan_params.lst_reports] + \
                          [dict(c, reverse=True) for c in scan_params.lst_reports]

        # Sử dụng File tạm để tránh Deadlock với Queue khi dữ liệu lớn
        temp_fn = f"/home/ubuntu/nevir/huy/scan/temp_{alpha_name}.pickle"
        if os.path.exists(temp_fn): os.remove(temp_fn)

        p = multiprocessing.Process(target=worker_wrapper, args=(lst_configs_all, dic_freqs, DIC_ALPHAS, dic_year, temp_fn, alpha_name))
        p.start()
        
        p.join(timeout=300)
        
        if p.is_alive():
            print(f"⚠️ [TIMEOUT] {alpha_name} bị dừng sau 10 phút.")
            p.terminate()
            p.join()
            if os.path.exists(temp_fn): os.remove(temp_fn)
            alpha_coll.update_one({"alpha_name": alpha_name}, {"$set": {"scan": 1, "is_error": 2, "error_detail": "Lỗi: Timeout (quá 5 phút)"}})
            continue

        if not os.path.exists(temp_fn):
            print(f"❌ [LỖI] {alpha_name} không tạo được file kết quả (có thể crash).")
            alpha_coll.update_one({"alpha_name": alpha_name}, {"$set": {"scan": 1, "is_error": 1, "error_detail": "Lỗi: Crash (sai logic/cú pháp)"}})
            continue

        with open(temp_fn, 'rb') as f:
            raw_result = pickle.load(f)
        os.remove(temp_fn)

        df = pd.DataFrame(raw_result)
        if df.empty:
            print(f"❌ [LỖI] {alpha_name} trả về kết quả rỗng (không có tín hiệu).")
            alpha_coll.update_one({"alpha_name": alpha_name}, {"$set": {"scan": 1, "is_error": 1, "error_detail": "Lỗi: Tín hiệu trống (có thể do NaN)"}})
            continue
        
        def agg_results(data_frame):
            res = {}
            for year in dic_year:
                df_year = data_frame[data_frame['year'] == year].copy()
                if df_year.empty: continue
                df_year = df_year[df_year['tvr'] != 0].copy()
                if df_year.empty: continue
                total_runs = len(df_year)
                r0 = round(len(df_year[df_year['sharpe'] > 0]) / total_runs * 100, 2)
                r1 = round(len(df_year[df_year['sharpe'] > 1]) / total_runs * 100, 2)
                r2 = round(len(df_year[df_year['sharpe'] > 2]) / total_runs * 100, 2)
                res[year] = [r0, r1, r2]
            return res

        results_orig = agg_results(df[df['is_reverse'] == False])
        results_rev = agg_results(df[df['is_reverse'] == True])

        if not results_orig and not results_rev:
            print(f"❌ [LỖI] {alpha_name} trả về kết quả rỗng sau khi lọc TVR (có thể do lỗi tính toán).")
            alpha_coll.update_one({"alpha_name": alpha_name}, {"$set": {"scan": 1, "is_error": 1, "error_detail": "Lỗi: Kết quả scan rỗng (tất cả TVR = 0 hoặc crash)"}})
            continue

        # Xác định hướng lưu dựa trên is_reverse đã có trong DB
        is_already_reversed = doc.get("is_reverse", False)
        
        if is_already_reversed:
            # Nếu code trong alpha.py ĐÃ CÓ dấu trừ, thì:
            # results_orig chính là hướng "Âm"
            # results_rev chính là hướng "Dương" (gốc)
            final_scan_result = results_rev
            final_scan_reverse_result = results_orig
        else:
            # Nếu code trong alpha.py CHƯA CÓ dấu trừ, thì:
            # results_orig chính là hướng "Dương" (gốc)
            # results_rev chính là hướng "Âm"
            final_scan_result = results_orig
            final_scan_reverse_result = results_rev

        # Kiểm tra xem Alpha có đạt chuẩn tốt không
        is_good = check_is_highly_stable(final_scan_result) or check_is_highly_stable(final_scan_reverse_result)
        
        update_data = {
            "scan": 1, 
            "scan_result": final_scan_result,
            "scan_reverse_result": final_scan_reverse_result,
            "is_reverse": is_already_reversed,
            "param_range": doc.get("param_range", {})
        }
        
        if is_good:
            update_data["run_all"] = 1

        alpha_coll.update_one({"alpha_name": alpha_name}, {"$set": update_data})
        print(f"✓ {alpha_name} DONE. (is_rev in code: {is_already_reversed})")

    print("\n>>> TẤT CẢ ALPHA TRONG BATCH NÀY ĐÃ ĐƯỢC QUÉT XONG.")

