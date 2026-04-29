import numpy as np
from itertools import product
import pandas as pd
import pickle
import multiprocessing
from tqdm import tqdm
import json
import sys
import alpha_loop
from gen.core import Simulator
from gen.alpha_func_lib import Domains

sys.path.insert(0, "/home/ubuntu/nevir/huy/Gen_Alpha")
import advance


def prepare_alphas(alphas_str):
    dic_alphas = {}
    optimized_codes = {}
    best_params_dict = {}
    optuna_metrics = {}
    for name, func_text in alphas_str.items():
        code = func_text.replace('@staticmethod\n', '').replace('@staticmethod', '').strip()
        code = code.replace("'volume'", "'matchingVolume'")
        code = code.replace('"volume"', '"matchingVolume"')
        best_params = None
        print(f"--- Bắt đầu tối ưu hóa tham số cho {name} ---")
        try:
            results, is_timeout = advance.run_advance_scan(code)
            res = advance.find_best_combo(results, code)
            if res:
                code = res['new_code']
                best_params = res['combo']
                # Ánh xạ: advance.py (r0=Sharpe>0, r1=Sharpe>1) -> run_test.py (r1=Sharpe>0, r2=Sharpe>1)
                metrics = {'r1': res.get('r0',0), 'r2': res.get('r1',0)}
                print(f"✅ Tối ưu thành công {name}: {res['combo']} (Direction: {res['direction']})")
            else:
                print(f"❌ Không tìm thấy combo tốt hơn cho {name}, dùng mặc định.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Lỗi trong quá trình tối ưu {name}: {e}. Dùng hàm mặc định.")
            
        local_namespace = {}
        exec(code, globals(), local_namespace)
        dic_alphas[name] = local_namespace[name]
        optimized_codes[name] = code
        best_params_dict[name] = best_params
        if 'metrics' in locals() and metrics:
            optuna_metrics[name] = metrics
    return dic_alphas, optimized_codes, best_params_dict, optuna_metrics


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
    try:
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
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        reports = []
        for year in DIC_YEAR:
            reports.append({
                'alphaName': config['alphaName'],
                'freq': config['freq'],
                'upper': config['upper'],
                'lower': config['lower'],
                'fee': config['fee'],
                'year': year,
                'tvr': 0,
                'sharpe': 0,
                'netProfit': 0,
                'is_error': True,
                'error_msg': str(e)
            })
        return reports


dic_year = {
    'all' : {
        'start': "2018_01_01",
        'end' : "2026_01_01"
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
    import time
    start_time_global = time.time()

    import os
    import sys
    import signal
    import multiprocessing

    def sigint_handler(signum, frame):
        print("\n\n🛑 [CTRL+C] NGƯỜI DÙNG ĐÃ HỦY TIẾN TRÌNH. ĐANG DỪNG TOÀN BỘ HỆ THỐNG...")
        try:
            for p in multiprocessing.active_children():
                p.terminate()
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["flaw", "enhance"], default="flaw", help="Mode: 'flaw' (cải thiện alpha tệ) or 'enhance' (cải thiện alpha tốt)")
    parser.add_argument("--id", type=str, nargs="+", default=None, help="MongoDB ObjectId(s) of the alpha to evolve")
    args = parser.parse_args()
    RUN_MODE = args.mode
    
    json_path = '/home/ubuntu/nevir/huy/evolution/combined_alphas.json'
    history_dir = '/home/ubuntu/nevir/huy/evolution/history'
    os.makedirs(history_dir, exist_ok=True)
    
    if args.id and isinstance(args.id, list):
        if len(args.id) > 1:
            print(f"\n[BATCH MODE] Phát hiện {len(args.id)} IDs. Sẽ chạy tuần tự từng ID...\n")
            for single_id in args.id:
                cmd = f"python /home/ubuntu/nevir/huy/evolution/run_test.py --mode {RUN_MODE} --id {single_id}"
                print(f"\n{'*'*80}\nĐang chạy tiến trình cho ID: {single_id}\n{'*'*80}\n")
                os.system(cmd)
            print("\n[BATCH MODE] Hoàn thành toàn bộ tiến trình cho các IDs được yêu cầu.")
            sys.exit(0)
        else:
            args.id = args.id[0]

    if args.id:
        import pymongo
        from bson.objectid import ObjectId
        import sys
        sys.path.insert(0, "/home/ubuntu/nevir")
        from auto.utils import get_mongo_uri
        
        client = pymongo.MongoClient(get_mongo_uri("mgc3"))
        col = client["alpha"]["gen_alpha"]
        doc = col.find_one({"_id": ObjectId(args.id)})
        if not doc:
            raise ValueError(f"Alpha with ID {args.id} not found in DB.")
        
        alpha_name = doc.get("alpha_name", "alpha_from_db")
        alpha_code = doc.get("alpha_code")
        if not alpha_code:
            raise ValueError(f"Alpha {args.id} has no 'alpha_code' field.")
        
        current_alphas_str = {alpha_name: alpha_code}
        print(f"Loaded alpha '{alpha_name}' from DB (ID: {args.id})")
        run_uuid = args.id
    else:
        with open(json_path, 'r') as f:
            current_alphas_str = json.load(f)
        print(f"Loaded alphas from {json_path}")
        import uuid
        run_uuid = str(uuid.uuid4())
    
    iteration = 1
    MAX_ITERATIONS = 15  # Cấu hình số vòng lặp tối đa tiến hóa
    
    previous_state = {}
    
    history_file_path = os.path.join(history_dir, f'history_{run_uuid}.json')
    history_data = {}
    
    stagnation_counter = {}
    dropped_alphas = set()
    
    while iteration <= MAX_ITERATIONS:
        iter_start_time = time.time()
        print(f"\n{'='*60}\nITERATION {iteration}/{MAX_ITERATIONS}\n{'='*60}")
        print("Preparing alphas...")
        DIC_ALPHAS, optimized_codes, best_params_dict, optuna_metrics = prepare_alphas(current_alphas_str)
        alpha_names = list(DIC_ALPHAS.keys())

        skipped_alphas = {}
        alphas_to_simulate = []
        for name in alpha_names:
            if name in previous_state and optimized_codes[name] == previous_state[name]['code']:
                print(f"⏩ Đã bỏ qua mô phỏng {name} do mã code sinh ra giống hệt vòng lặp trước.")
                skipped_alphas[name] = previous_state[name]
            elif name in optuna_metrics and optuna_metrics[name]['r1'] < 70 and optuna_metrics[name]['r2'] < 10:
                print(f"⏩ Đã bỏ qua mô phỏng {name} do Optuna baseline quá thấp (r1={optuna_metrics[name]['r1']}%, r2={optuna_metrics[name]['r2']}%).")
                skipped_alphas[name] = {
                    "code": optimized_codes[name],
                    "params": best_params_dict[name],
                    "r1_all": optuna_metrics[name]['r1'],
                    "r2_all": optuna_metrics[name]['r2'],
                    "s0_by_year": {"all": optuna_metrics[name]['r1']}
                }
            else:
                alphas_to_simulate.append(name)
                
        scan_params = Scan_Params(
            lst_alpha_names=alphas_to_simulate,
            min_freq=20, max_freq=100,
            fee=0.175,
            shuffle=True)
        lst_configs = scan_params.lst_reports

        timeout_early_stop_alphas = set()
        iter_duration = 0
        if lst_configs:
            print("Loading data...")
            dic_freqs = load_dic_freqs()

            lst_passed = []
            num_processes = 30
            print(f"Starting simulation with {num_processes} processes for {len(lst_configs)} configs...")

            with multiprocessing.Pool(
                processes=num_processes,
                initializer=init_worker,
                initargs=(dic_freqs, DIC_ALPHAS, dic_year)
            ) as pool:
                results = tqdm(pool.imap_unordered(run_simulation, lst_configs), total=len(lst_configs))
                for result_list in results:
                    lst_passed.extend(result_list)

            print("Simulation finished. Analyzing results...")
            df = pd.DataFrame(lst_passed)
            
            iter_duration = time.time() - iter_start_time
            if iter_duration > 300:
                print(f"\n🛑 EARLY STOP: Quá trình backtest của iter này tốn quá lâu ({iter_duration:.2f}s > 300s). Loại bỏ các alpha nặng!")
                timeout_early_stop_alphas = set(alphas_to_simulate)
        else:
            print("No new configs to simulate. Skipping simulation phase.")
            df = pd.DataFrame()
        
        target_achieved = False
        alphas_for_diagnose = []

        for name in alpha_names:
            if name in skipped_alphas:
                r1_all = skipped_alphas[name]['r1_all']
                r2_all = skipped_alphas[name]['r2_all']
                s0_by_year = skipped_alphas[name].get('s0_by_year', {})
                has_error = False
                error_msg = ""
            else:
                df_alpha = df[df['alphaName'] == name] if not df.empty else pd.DataFrame()
                s0_by_year = {}
                r1_all = 0
                r2_all = 0
                has_error = False
                error_msg = ""
                
                if 'is_error' in df_alpha.columns and df_alpha['is_error'].any():
                    has_error = True
                    error_msg = df_alpha[df_alpha['is_error'] == True]['error_msg'].iloc[0]
                
                if not has_error:
                    for year in dic_year:
                        df_year = df_alpha[(df_alpha['year'] == year) & (df_alpha['tvr'] != 0)]
                        if df_year.empty:
                            continue
                        total = len(df_year)
                        if total > 0:
                            ratio_1 = round(len(df_year[df_year['sharpe'] > 0]) / total * 100, 2)
                            ratio_2 = round(len(df_year[df_year['sharpe'] > 1]) / total * 100, 2)
                        else:
                            ratio_1 = ratio_2 = 0
                        
                        s0_by_year[year] = ratio_1
                        if year == 'all':
                            r1_all = ratio_1
                            r2_all = ratio_2
                            print(f"Alpha: {name} | Year: all | Ratio 1 (>0): {ratio_1}% | Ratio 2 (>1): {ratio_2}%")

            # --- EARLY STOP LOGIC ---
            stop_reason_str = None
            if name in timeout_early_stop_alphas:
                stop_reason_str = f"EARLY STOP: Iteration took too long ({iter_duration:.2f}s > 300s)"
                print(f"🛑 EARLY STOP: {name} bị loại do timeout iter > 300s.")
                dropped_alphas.add(name)
            elif iteration >= 5 and (r2_all < 5 or r1_all <= 70):
                stop_reason_str = f"EARLY STOP: iter {iteration}, r1: {r1_all}%, r2: {r2_all}% (not meeting threshold)"
                print(f"🛑 EARLY STOP: {name} bị loại ở iter {iteration} do không đạt chuẩn tối thiểu (r1: {r1_all}%, r2: {r2_all}%).")
                dropped_alphas.add(name)
                
            if not stop_reason_str and name in previous_state:
                prev_r1 = previous_state[name]['r1_all']
                if abs(r1_all - prev_r1) < 5:
                    stagnation_counter[name] = stagnation_counter.get(name, 0) + 1
                    if stagnation_counter[name] >= 3:
                        stop_reason_str = "EARLY STOP: r1_all giao động < 5% trong 3 vòng liên tiếp"
                        print(f"🛑 EARLY STOP: {name} bị loại do r1_all giao động < 5% trong 3 vòng liên tiếp.")
                        dropped_alphas.add(name)
                else:
                    stagnation_counter[name] = 0
            # ------------------------

            if not has_error and r1_all > 80 and r2_all > 40:
                print(f"\n🎉 SUCCESS! {name} achieved Ratio 1 > 80% ({r1_all}%) and Ratio 2 > 40% ({r2_all}%)!")
                
                previous_state[name] = {
                    "code": optimized_codes[name], 
                    "params": best_params_dict[name],
                    "r1_all": r1_all, 
                    "r2_all": r2_all,
                    "s0_by_year": s0_by_year,
                    "stop_reason": "SUCCESS"
                }
                
                target_achieved = True
                break
                
            trend_note = ""
            if has_error:
                trend_note = f"FEEDBACK: This code CRASHED during execution with error: {error_msg}. You MUST FIX the syntax/logic error. Do NOT use functions on pd.Series that require scalars (like ewm(span=Series))."
                if name in previous_state:
                    trend_note += f"\nPrevious working code was:\n{previous_state[name]['code']}"
            elif name in previous_state:
                prev_r1 = previous_state[name]['r1_all']
                prev_code = previous_state[name]['code']
                if name in skipped_alphas or optimized_codes[name] == prev_code:
                    trend_note = "FEEDBACK: Performance is unchanged because you returned the EXACT SAME CODE. You MUST make meaningful modifications to the mathematical logic. DO NOT return the exact same code."
                elif r1_all > prev_r1:
                    trend_note = f"FEEDBACK: This code improved performance from Ratio 1 = {prev_r1}% to {r1_all}%. The previous modification direction is GOOD. Continue exploring this direction."
                elif r1_all < prev_r1:
                    trend_note = f"FEEDBACK: This code DEGRADED performance from Ratio 1 = {prev_r1}% to {r1_all}%. The previous modification direction is BAD. AVOID this direction and revert or try a completely different approach. Previous better code was:\n{prev_code}"
                else:
                    trend_note = "FEEDBACK: Performance is unchanged despite logic modifications. Try a completely different mathematical approach."

            
            if not has_error:
                previous_state[name] = {
                    "code": optimized_codes[name], 
                    "params": best_params_dict[name],
                    "r1_all": r1_all, 
                    "r2_all": r2_all,
                    "s0_by_year": s0_by_year
                }
            
            if stop_reason_str and name in previous_state:
                previous_state[name]["stop_reason"] = stop_reason_str
            
            if name not in dropped_alphas:
                alphas_for_diagnose.append({
                    "name": name,
                    "code": optimized_codes[name],
                    "metrics": {
                        "s0_all": r1_all,
                        "s1_all": r2_all,
                        "s0_by_year": s0_by_year,
                        "note": trend_note
                    },
                    "error": error_msg if has_error else None,
                    "error_type": 1 if has_error else None
                })

        # Lưu tiến trình của iter hiện tại vào history_data (chỉ lấy tier cuối cùng như yêu cầu)
        history_data = {
            "iteration": iteration,
            "final_tier": previous_state.copy(),
            "llm_usage": alpha_loop.get_token_usage_and_cost()
        }
        with open(history_file_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        print(f"Saved iteration {iteration} progress to {history_file_path}")

        if target_achieved:
            break

        if iteration >= MAX_ITERATIONS:
            for n in previous_state:
                if "stop_reason" not in previous_state[n]:
                    previous_state[n]["stop_reason"] = "MAX_ITERATIONS"
            history_data["final_tier"] = previous_state.copy()
            with open(history_file_path, 'w') as f:
                json.dump(history_data, f, indent=4)
            print(f"\nĐã đạt tới giới hạn số vòng lặp tối đa ({MAX_ITERATIONS}). Dừng quá trình tiến hóa.")
            print(f"Tiến trình tiến hóa hoàn chỉnh có thể tìm thấy trong {history_file_path}")
            break

        if not alphas_for_diagnose:
            print("\nTất cả các alpha đều đã bị loại bỏ (Early Stop) hoặc đạt target. Kết thúc sớm quá trình tiến hóa.")
            break

        max_retries = 3
        success_llm = False
        
        for attempt in range(max_retries):
            print(f"\nNone achieved the target. Proceeding to Diagnose and Evolve (Mode: {RUN_MODE}) [Attempt {attempt+1}/{max_retries}]...")
            try:
                diagnosis, _ = alpha_loop.diagnose(ids=[], intent=RUN_MODE, _alphas=alphas_for_diagnose, verbose=True)
                evolved = alpha_loop.evolve(alphas_for_diagnose, diagnosis, verbose=True)
                
                if not evolved:
                    print("Evolve failed to return new variants.")
                    import time
                    time.sleep(2)
                    continue
                    
                current_alphas_str = evolved
                with open(json_path, 'w') as f:
                    json.dump(evolved, f, indent=4)
                    
                iteration += 1
                success_llm = True
                break
                
            except Exception as e:
                print(f"Error during diagnose/evolve on attempt {attempt+1}: {e}")
                import traceback
                traceback.print_exc()
                import time
                time.sleep(2)
                
        if not success_llm:
            print(f"Failed to evolve after {max_retries} attempts. Exiting evolution loop.")
            break

    print(f"\n{'='*60}\nEVOLUTION COMPLETED\n{'='*60}")
    print(f"Total iterations run: {iteration}")
    
    final_usage = alpha_loop.get_token_usage_and_cost()
    print(f"LLM Usage: {final_usage['tokens']}")
    print(f"Total Cost: ${final_usage['total_cost_usd']:.4f}")
    
    import time
    end_time_global = time.time()
    total_time_seconds = end_time_global - start_time_global
    
    print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time_seconds))}")
    
    # Ensure final update of usage and time
    history_data["llm_usage"] = final_usage
    history_data["total_time_seconds"] = round(total_time_seconds, 2)
    history_data["total_time_formatted"] = time.strftime("%H:%M:%S", time.gmtime(total_time_seconds))
    
    with open(history_file_path, 'w') as f:
        json.dump(history_data, f, indent=4)
        
    print(f"Toàn bộ kết quả (code, params, metrics, usage) được lưu tại {history_file_path}")

