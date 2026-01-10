import numpy as np
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
import pickle


from core import Simulator
from alpha_func_lib import Domains


def load_dic_freqs():
    fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    with open(fn, 'rb') as file:
        DIC_FREQS = pickle.load(file)
        
    return DIC_FREQS

def run_single_backtest(config, dic_freqs, DIC_ALPHAS,df_tick=None,gen=None,start=None,end=None):
    """
    Chạy backtest cho một config duy nhất
    """
    # try:
    gen_params = {}
    if gen == "1_3":
        gen_params['score'] = config['score']
        gen_params['entry'] = config['entry']
        gen_params['exit'] = config['exit']
    elif gen == "1_2":
        gen_params['upper'] = config['upper']
        gen_params['lower'] = config['lower']
    elif gen == "1_1":
        gen_params['halflife'] = config['halflife']
        gen_params['threshold'] = config['threshold']
    elif gen == "1_4":
        gen_params['inertia'] = config['inertia']
        gen_params['threshold'] = config['threshold']
    elif gen == "1_5":
        gen_params['velocity'] = config['velocity']
    elif gen == "1_6":
        gen_params['velocity'] = config['velocity']
        gen_params['threshold'] = config['threshold']

    bt = Simulator(
        alpha_name=config['alphaName'],
        freq=config['freq'],
        gen_params=gen_params,
        fee=config['fee'],
        df_alpha=dic_freqs[config['freq']].copy(),
        params=config.get('params', {}),
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=None,
        gen=gen,
        start=start,
        end=end
    )
    bt.compute_signal()
    bt.compute_position()
    bt.compute_tvr_and_fee()
    bt.compute_profits()
    bt.compute_performance(start=start, end=end)
    
    report_with_params = bt.report.copy()
    

    
    params = config.get('params', {})
    for param_name, param_value in params.items():
        report_with_params[f"param_{param_name}"] = param_value
    print(f"Report", report_with_params['sharpe'],report_with_params['tvr'], report_with_params['mddPct'], report_with_params['netProfit'])
    return report_with_params

if __name__ == '__main__': 
    config = {
                    'alphaName': 'alpha_101_acceleration',
                    'freq': 34,
                    "threshold": 0.6,
                    "halflife": 0.9,
                    "fee": 0.175,
                    "stop_loss":0,
                   "start":"2022_01_01",
                   "end":"2025_12_31",
                     "params": {
                          "window": 1,
                     }
                }
    # halflife = 0 - 1 step 0.1
    dic_freqs = load_dic_freqs()
    DIC_ALPHAS = Domains.get_list_of_alphas()
    # df = pd.read_pickle("/home/ubuntu/nevir/data/df_2025.pkl")
    # df = df[(df["day"] >= 20250101) & (df["day"] <= 20250908)]
    run_single_backtest(
        config=config,
        dic_freqs=dic_freqs,
        DIC_ALPHAS=DIC_ALPHAS,
        df_tick=None,
        start = config["start"],
        end = config["end"],
        gen="1_1"
    )
  
    
