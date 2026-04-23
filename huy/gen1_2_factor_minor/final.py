import numpy as np
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
from random import shuffle
import pickle
import multiprocessing
from functools import partial
from tqdm import tqdm

from core import Simulator, Collector
from alpha_func_lib import Domains


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
    fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    # fn = "/home/ubuntu/duy/dic_freqs_ha.pickle"
    # fn = "/home/ubuntu/duy/dic_freqs_comfirm_ha.pickle"
    with open(fn, 'rb') as file:
        DIC_FREQS = pickle.load(file)

    return DIC_FREQS

# This function will be called once for each worker process
def init_worker(freqs, alphas, years):
    global DIC_FREQS, DIC_ALPHAS, DIC_YEAR
    DIC_FREQS = freqs
    DIC_ALPHAS = alphas
    DIC_YEAR = years

# Worker function now uses global variables, avoiding data transfer for each task
def run_simulation(config):
    """
    Runs a single simulation for a given configuration.
    """
    bt = Simulator(
        alpha_name=config['alphaName'],
        freq=config['freq'],   
        upper=config['upper'],
        lower=config['lower'],
        fee=config['fee'],
        df_alpha=DIC_FREQS[config['freq']],
        DIC_ALPHAS=DIC_ALPHAS
    )
    bt.compute_signal()
    bt.compute_position()
    bt.compute_tvr_and_fee()
    bt.compute_profits()
    
    reports = []
    for year in DIC_YEAR:
        bt.compute_performance(start=DIC_YEAR[year]['start'], end=DIC_YEAR[year]['end'])

        bt.report['year'] = year
        reports.append(bt.report.copy())
    return reports


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
    
    alpha_name = 'alpha_068'
    scan_params = Scan_Params(
        lst_alpha_names=[alpha_name],
        min_freq=10, max_freq=80,
        fee = 0.175,
        shuffle=True)
    lst_configs = scan_params.lst_reports

    print("Loading data...")
    dic_freqs = load_dic_freqs()
    DIC_ALPHAS = Domains.get_list_of_alphas()


    lst_passed = []
    # Use all available CPU cores
    num_processes = multiprocessing.cpu_count()
    print(f"Starting simulation with {num_processes} processes for {len(lst_configs)} configs...")

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(dic_freqs, DIC_ALPHAS, dic_year)
    ) as pool:
        # Use tqdm for a progress bar, imap_unordered for efficiency
        results = tqdm(pool.imap_unordered(run_simulation, lst_configs), total=len(lst_configs))
        for result_list in results:
            lst_passed.extend(result_list)

    print("Simulation finished. Saving results...")
    df = pd.DataFrame(lst_passed)
    df.to_pickle("/home/ubuntu/duy/new_strategy/gen1_2/df_fee_1.pkl")
    
    print("Analyzing results...")
    for year in dic_year:
        print(f"--- Year: {year} ---")
        df_year = df[df['year'] == year].copy()

        if df_year.empty:
            print("No data for this year.")
            continue
            
        df_year = df_year[df_year['tvr'] != 0].copy()

        if df_year.empty:
            print("No trades made in this year (tvr=0).")
            continue

        total_runs = len(df_year)
        ratio_1 = round(len(df_year[df_year['sharpe'] > 0]) / total_runs  * 100 , 2)
        ratio_2 = round(len(df_year[df_year['sharpe'] > 1]) / total_runs  * 100 , 2)
        ratio_3 = round(len(df_year[df_year['sharpe'] > 2]) / total_runs  * 100 , 2)
        
        print(f"{ratio_1}, {ratio_2}, {ratio_3}")
        print(f"{len(df_year)}")
        
        
# --- Year: all ---
# 92.64, 34.12, 0.0
# 3195
# --- Year: 2018 ---
# 87.1, 59.75, 30.99
# 3195
# --- Year: 2019 ---
# 76.62, 37.72, 2.0
# 3195
# --- Year: 2020 ---
# 82.19, 36.28, 6.29
# 3195
# --- Year: 2021 ---
# 85.26, 52.39, 14.33
# 3195
# --- Year: 2022 ---
# 80.34, 42.16, 9.86
# 3195
# --- Year: 2023 ---
# 74.84, 29.17, 2.5
# 3195
# --- Year: 2024 ---
# 57.59, 19.94, 0.28
# 3195
# --- Year: 2025 ---
# 72.27, 38.22, 10.14
# 3195
        
        
        

 
