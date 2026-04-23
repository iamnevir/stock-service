import numpy as np
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
from random import shuffle

import pickle
from core import Simulator, Collector
from alpha_func_lib import Domains
import pymongo
import sys
import os
try:
    sys.path.insert(0, "/home/ubuntu")
    from auto.utils import get_mongo_uri
except ImportError:
    pass
import time
from concurrent.futures import ThreadPoolExecutor

def process_single_alpha_ic(doc, dic_freqs, DIC_ALPHAS, alpha_coll):
    alphaName = doc["alpha_name"]
    print(f"Đang tính IC cho: {alphaName}")
    start_time_alpha = time.time()
    try:
        alpha_ic_result = {}
        alpha_ic_reverse_result = {} # Chứa kết quả đảo ngược
        local_lst_passed = []
        
        # Định nghĩa hàm đảo ngược
        orig_func = DIC_ALPHAS[alphaName]
        def alpha_func_reverse(df, **kwargs):
            return -orig_func(df, **kwargs)

        # Tính IC cho từng mốc freq
        for freq in range(20, 101, 10):
            # Kiểm tra Timeout 5 phút cho mỗi Alpha
            if time.time() - start_time_alpha > 300:
                raise Exception("Timeout: Thời gian tính toán vượt quá 300 giây")

            # --- Tính IC gốc ---
            bt_orig = Simulator(
                alpha_name=alphaName, freq=freq, upper=0.20, lower=0.10, fee=0.175,
                df_alpha=dic_freqs[freq], DIC_ALPHAS=DIC_ALPHAS, df_1m=dic_freqs[1], cutTime=None,
            )
            res_orig = bt_orig.compute_signal()
            
            # --- Tính IC đảo ngược ---
            # Tạm thời thay thế hàm trong DIC_ALPHAS để Simulator dùng hàm đảo
            temp_dic = DIC_ALPHAS.copy()
            temp_dic[alphaName] = alpha_func_reverse
            
            bt_rev = Simulator(
                alpha_name=alphaName, freq=freq, upper=0.20, lower=0.10, fee=0.175,
                df_alpha=dic_freqs[freq], DIC_ALPHAS=temp_dic, df_1m=dic_freqs[1], cutTime=None,
            )
            res_rev = bt_rev.compute_signal()

            if pd.isna(res_orig.get("ic_mean")) or np.isinf(res_orig.get("ic_mean")):
                raise Exception(f"Kết quả không hợp lệ tại freq {freq}")

            alpha_ic_result[str(freq)] = {
                "ic_mean": res_orig.get("ic_mean"),
                "is_positive_only": res_orig.get("is_positive_only")
            }
            alpha_ic_reverse_result[str(freq)] = {
                "ic_mean": res_rev.get("ic_mean"),
                "is_positive_only": res_rev.get("is_positive_only")
            }
            
            ic_data = { 'alpha_name' : alphaName, 'freq' : freq }
            ic_data.update(res_orig)
            local_lst_passed.append(ic_data)
        
        # Cập nhật kết quả thành công vào DB
        alpha_coll.update_one(
            {"alpha_name": alphaName}, 
            {"$set": {
                "run_IC": 1, 
                "ic_result": alpha_ic_result, 
                "ic_reverse_result": alpha_ic_reverse_result, 
                "is_error": 0
            }}
        )
        print(f"[IC Done] {alphaName} đã xong (cả gốc và đảo ngược).")
        return local_lst_passed
        print(f"[IC Done] {alphaName} đã xong.")
        return local_lst_passed
        
    except Exception as e:
        print(f"[IC Error] Bỏ qua {alphaName} do: {e}")
        alpha_coll.update_one(
            {"alpha_name": alphaName}, 
            {"$set": {"run_IC": 1, "is_error": 1, "error_detail": "Công thức lỗi"}}
        )
        return []



class Scan_Params:
    def __init__(self,lst_alpha_names, min_freq, max_freq, fee,
                 shuffle=False, convert_reports_to_dicts=True):

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fee = fee

        """PLACE-HOLDER"""
        self.lst_reports = []

        """AUTO-GEN"""
        if lst_alpha_names is not None:
            self.lst_reports = self.gen_lst_reports(
                lst_alpha_names,
                convert_reports_to_dicts=convert_reports_to_dicts,
                shuffle=shuffle)
    def gen_threshold_list(self):
        return [(x, y) for x, y in product(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                if x > y]

    def gen_lst_reports(self, lst_alpha_names, shuffle=False, in_place=True,
                        convert_reports_to_dicts=True):
        lst_freqs = list(range(self.min_freq, self.max_freq + 1))

        lst_thresholds = self.gen_threshold_list()
        lst_reports = list(product(
            lst_alpha_names,
            lst_freqs,
            lst_thresholds))

        if shuffle:
            from random import shuffle as apply_shuffle
            apply_shuffle(lst_reports)

        if convert_reports_to_dicts:
            lst = []
            for dp in lst_reports:
                alpha_name, freq, (upper, lower) = dp
                report = {
                    'alphaName': alpha_name,
                    'freq': freq,
                    'upper': upper,
                    'lower': lower,
                    "fee": self.fee,
                }
                lst.append(report)
            lst_reports = lst

        if in_place:
            self.lst_reports = lst_reports

        return lst_reports


def load_dic_freqs():
    # fn = "/home/ubuntu/duy/new_strategy/gen1_2/dic_freqs.pkl"
    fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    with open(fn, 'rb') as file:
        DIC_FREQS = pickle.load(file)
        
    for freq in DIC_FREQS.keys():
        df = DIC_FREQS[freq].copy()
        
        df.loc[df['executionTime'] == '14:45:00', 'exitPrice'] = df['open'].shift(-2)
        df['priceChange'] = df['exitPrice'] - df['entryPrice']
        
        DIC_FREQS[freq] = df

    return DIC_FREQS

def load_hardcoded_config_filtered():
    lst = [
        {
            "alphaName": "alpha_zscore",
            "freq": 25,
            "fee": 0.1,
            "lower": 0.3,
            "upper": 0.4,
            "name": "alpha_zscore"
        },
        {
            "alphaName": "alpha_zscore",
            "freq": 53,
            "fee": 0.1,
            "lower": 0.4,
            "upper": 0.7,
            "name": "alpha_zscore"
        }]
    return lst


dic_year = {
    'all' : {
        'start' : None,
        'end' : None
    },
    '2018' : {
        'start' : '2018_01_01',
        'end' : '2019_01_01'
    },
    '2019' : {
        'start' : '2019_01_01',
        'end' : '2020_01_01'
    },
    '2020' : {
        'start' : '2020_01_01',
        'end' : '2021_01_01'
    },
    '2021' : {
        'start' : '2021_01_01',
        'end' : '2022_01_01'
    },
    '2022' : {
        'start' : '2022_01_01',
        'end' : '2023_01_01'
    },
    '2023' : {
        'start' : '2023_01_01',
        'end' : '2024_01_01'
    },
    '2024' : {
        'start' : '2024_01_01',
        'end' : '2025_01_01'
    },
    '2025' : {
        'start' : '2025_01_01',
        'end' : '2026_01_01'
    },

}

    
if __name__ == '__main__': 
    dic_freqs = load_dic_freqs()
    
    try:
        # Thêm đường dẫn chứa file alpha.py
        alpha_dir = "/home/ubuntu/nevir/huy/Gen_Alpha"
        if alpha_dir not in sys.path:
            sys.path.insert(0, alpha_dir)

        # Refresh file alpha.py (để nhận diện hàm mới vừa append)
        import importlib
        import alpha 
        importlib.reload(alpha)
        from alpha import Alpha
        
        # Tự động lấy tất cả các hàm alpha_ trong class Alpha
        DIC_ALPHAS = {name: getattr(Alpha, name) for name in dir(Alpha) if name.startswith("alpha_")}
        
        # Kết nối db
        mongo_client = pymongo.MongoClient(get_mongo_uri("mgc3"))
        db = mongo_client["alpha"]
        alpha_coll = db["gen_alpha"]
        
        docs = list(alpha_coll.find({"run_IC": 0}))
        
        if docs:
            print(f"\nPhát hiện {len(docs)} alpha mới cần chạy IC.")
            lst_passed = []

            with ThreadPoolExecutor(max_workers=min(5, len(docs))) as executor:
                # Tạo map để theo dõi từng future gắn với doc nào
                future_to_doc = {executor.submit(process_single_alpha_ic, doc, dic_freqs, DIC_ALPHAS, alpha_coll): doc for doc in docs}
                
                for future in future_to_doc: # Đợi kết quả
                    doc = future_to_doc.get(future)
                    alpha_name = doc["alpha_name"] if doc else "Unknown"
                    try:
                        # Áp đặt giới hạn 5 phút (300 giây) cho mỗi con Alpha
                        result = future.result(timeout=300)
                        lst_passed.extend(result)
                    except Exception as e:
                        error_msg = f"Timeout/Error: {str(e)}"
                        if "TimeoutError" in str(type(e)):
                            error_msg = "Timeout: Tính toán IC vượt quá 5 phút (300 giây)"
                        
                        print(f"[IC Skip] {alpha_name} do: {error_msg}")
                        # Cập nhật trạng thái lỗi vào DB với thông tin chi tiết (Timeout hoặc Crash)
                        alpha_coll.update_one(
                            {"alpha_name": alpha_name}, 
                            {"$set": {"run_IC": 1, "is_error": 1, "error_detail": error_msg}}
                        )

            if lst_passed:
                df = pd.DataFrame(lst_passed).sort_values(['alpha_name', 'freq'])
                print("\nBảng tóm tắt kết quả IC:")
                print(df[['alpha_name', 'freq', 'ic_mean']].head(20))

        else:
            print("Không có Alpha nào đang chờ chạy IC.")
            
    except Exception as global_e:
        print(f"Lỗi hệ thống trong quy trình IC: {global_e}")
    


