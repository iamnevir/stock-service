import re
import itertools
import pandas as pd
import numpy as np
import pickle
import multiprocessing
import textwrap
import sys
import os
import signal
import ast
from tqdm import tqdm

# --- CƠ CHẾ SIMULATOR (Tương tự final.py) ---
sys.path.insert(0, "/home/ubuntu/nevir/huy/scan")
from core import Simulator

# Khai báo các đối tượng toàn cục để dùng chung trong Pool (hiệu năng cao)
DIC_FREQS_SCAN = None
DIC_ALPHAS_SCAN = None

def init_worker(freqs, alphas):
    global DIC_FREQS_SCAN, DIC_ALPHAS_SCAN
    DIC_FREQS_SCAN = freqs
    DIC_ALPHAS_SCAN = alphas

def get_alpha_params(alpha_code):
    """Tự động tìm hàm đầu tiên và trích xuất tham số"""
    match_def = re.search(r'def\s+\w+\s*\((.*?)\):', alpha_code)
    if not match_def:
        return []
    params_raw = match_def.group(1)
    param_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    return [(name, float(val)) for name, val in param_pattern.findall(params_raw)]

def get_active_names_from_code(alpha_code):
    """Sử dụng AST để tìm các tham số thực sự được sử dụng trong logic hàm"""
    try:
        # Xóa các khoảng trắng thừa để parse chuẩn
        tree = ast.parse(textwrap.dedent(alpha_code))
        
        # Tìm node định nghĩa hàm (FunctionDef)
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
        
        if not func_def:
            return set()
            
        # Tìm tất cả các tên biến được sử dụng (Load)
        used_names = set()
        for node in ast.walk(func_def):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        return used_names
    except Exception as e:
        print(f"   [AST Warning] Không thể phân tích code: {e}")
        return set()

def is_temporal_param(alpha_code, param_name):
    """Kiểm tra tham số có dùng trong hàm window/thời gian không"""
    temporal_funcs = ['diff', 'shift', 'delay', 'quantile']
    for func in temporal_funcs:
        pattern = rf'({func}\s*\(.*?\b{param_name}\b|(\.{func}\s*\(\s*){param_name}\b)'
        if re.search(pattern, alpha_code):
            return True
    return False

def generate_advance_grid(alpha_code):
    """Sinh lưới tham số Advance: [5..100] hoặc [10..100] & [1..7]"""
    # 1. Lấy toàn bộ tham số từ signature
    all_params = get_alpha_params(alpha_code)
    if not all_params: return {}

    # 2. Lọc các tham số thực sự được sử dụng trong logic
    active_names = get_active_names_from_code(alpha_code)
    
    # Chỉ thực hiện lọc nếu parse được used names và danh sách lọc không bị rỗng hoàn toàn
    if active_names:
        active_params = [(name, val) for name, val in all_params if name in active_names]
        
        # Nếu sau khi lọc vẫn còn ít nhất 1 tham số thì mới áp dụng lọc
        if active_params:
            removed = [p[0] for p in all_params if p[0] not in active_names]
            if removed:
                print(f"   [Advance Optimizer] Đã loại bỏ tham số thừa: {removed}")
            params = active_params
        else:
            # Trường hợp hi hữu lọc xong không còn gì (có thể do lỗi parse), fallback về bộ gốc
            params = all_params
    else:
        params = all_params

    tuning_map = {}
    if not params: return {}

    if len(params) == 1:
        name, val = params[0]
        if 0 < val < 1:
            tuning_map[name] = [0.1, 0.3,  0.5, 0.7, 0.9]
        else:
            tuning_map[name] = list(range(5, 105, 5))
        return tuning_map

    if len(params) >= 2:
        p1_name, p1_val = params[0]
        p2_name, p2_val = params[1]
        
        if p1_val >= p2_val:
            x1_name, x1_val = p1_name, p1_val
            x2_name, x2_val = p2_name, p2_val
        else:
            x1_name, x1_val = p2_name, p2_val
            x2_name, x2_val = p1_name, p1_val
            
        # 1. Thiết lập Grid cho tham số thứ nhất (x1 - Thường là Window lớn)
        if 0 < x1_val < 1:
            tuning_map[x1_name] = [0.1, 0.3,  0.5, 0.7, 0.9]
        else:
            tuning_map[x1_name] = list(range(10, 110, 10))
        
        # 2. Thiết lập Grid cho tham số thứ hai (x2 - Thường là tham số phụ)
        if 0 < x2_val < 1:
            tuning_map[x2_name] = [0.1,  0.3,  0.5, 0.7, 0.9]
        elif is_temporal_param(alpha_code, x2_name):
            tuning_map[x2_name] = list(range(1, 8))
        else:
            tuning_map[x2_name] = [1, 3, 5, 7, 10, 20, 30, 40]

    return tuning_map

def load_scan_data():
    """Nạp dữ liệu thị trường"""
    fn = "/home/ubuntu/nevir/gen/alpha.pkl"
    with open(fn, 'rb') as file:
        data = pickle.load(file)
    for freq in data.keys():
        df = data[freq].copy()
        df.loc[df['executionTime'] == '14:45:00', 'exitPrice'] = df['open'].shift(-2)
        df['priceChange'] = df['exitPrice'] - df['entryPrice']
        data[freq] = df
    return data

def run_simulation_worker(config):
    """Hàm worker xử lý 1 lượt backtest đơn lẻ - CÓ CẦU CHÌ"""
    try:
        # Cầu chì 60 giây cho mỗi lượt quét
        def local_timeout(signum, frame): raise Exception("Timeout")
        signal.signal(signal.SIGALRM, local_timeout)
        signal.alarm(60)
        
        is_reverse = config.get('reverse', False)
        alpha_func = DIC_ALPHAS_SCAN[config['alphaName']]
        
        def execute_alpha(df, **kwargs):
            merged_params = {**config['combo'], **kwargs}
            signal_vals = alpha_func(df, **merged_params)
            return -signal_vals if is_reverse else signal_vals

        bt = Simulator(
            alpha_name=config['alphaName'], freq=config['freq'],   
            upper=config['upper'], lower=config['lower'], fee=config['fee'],
            df_alpha=DIC_FREQS_SCAN[config['freq']],
            DIC_ALPHAS={config['alphaName']: execute_alpha}
        )
        bt.compute_signal()
        bt.compute_position()
        bt.compute_tvr_and_fee()
        bt.compute_profits()
        
        reports = []
        dic_year = {
            'all' : {'start' : None, 'end' : None},
            # '2018' : {'start' : '2018_01_01', 'end' : '2019_01_01'},
            # '2019' : {'start' : '2019_01_01', 'end' : '2020_01_01'},
            # '2020' : {'start' : '2020_01_01', 'end' : '2021_01_01'},
            # '2021' : {'start' : '2021_01_01', 'end' : '2022_01_01'},
            # '2022' : {'start' : '2022_01_01', 'end' : '2023_01_01'},
            # '2023' : {'start' : '2023_01_01', 'end' : '2024_01_01'},
            # '2024' : {'start' : '2024_01_01', 'end' : '2025_01_01'},
            # '2025' : {'start' : '2025_01_01', 'end' : '2026_01_01'},
        }
        
        for year in dic_year:
            bt.compute_performance(start=dic_year[year]['start'], end=dic_year[year]['end'])
            bt.report.update({'year': year, 'is_reverse': is_reverse, 'combo_id': config['combo_id']})
            reports.append(bt.report.copy())
            
        signal.alarm(0) # Tắt báo thức
        return reports
    except:
        signal.alarm(0)
        return []

def gen_threshold_list():
        return [(x, y) for x, y in itertools.product(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                if x > y]

def agg_scan_results(df_results):
    """Gom nhóm kết quả giống final.py"""
    res = {}
    if df_results.empty: return res
    for year in df_results['year'].unique():
        df_year = df_results[(df_results['year'] == year) & (df_results['tvr'] != 0)].copy()
        if df_year.empty: continue
        total = len(df_year)
        res[year] = [
            round(len(df_year[df_year['sharpe'] > 0]) / total * 100, 2),
            round(len(df_year[df_year['sharpe'] > 1]) / total * 100, 2),
            round(len(df_year[df_year['sharpe'] > 2]) / total * 100, 2)
        ]
    return res

def run_advance_scan(alpha_code):
    """Quét toàn bộ tổ hợp tham số - PHIÊN BẢN GIA CỐ"""
    grid = generate_advance_grid(alpha_code)
    if not grid: return []
    
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    
    match_def = re.search(r'def\s+(\w+)\s*\(', alpha_code)
    alpha_name = match_def.group(1)
    
    loc = {}
    exec(textwrap.dedent(alpha_code), globals(), loc)
    alpha_func = loc[alpha_name]
    
    dic_freqs = load_scan_data()
    
    # Cấu hình quét (Giống final.py)
    # Rút gọn dải Threshold để tăng tốc: [0.2, 0.4, 0.6, 0.8, 1.0] vs [0.1, 0.3, 0.5, 0.7, 0.9]
    thresholds = gen_threshold_list()
    freqs = list(range(20, 101, 10))
    
    tasks = []
    for c_id, combo in enumerate(combos):
        for freq in freqs:
            for u, l in thresholds:
                for rev in [False, True]:
                    tasks.append({
                        'alphaName': alpha_name, 'freq': freq, 'upper': u, 'lower': l, 
                        'fee': 0.175, 'reverse': rev, 'combo': combo, 'combo_id': c_id
                    })
    
    print(f"\n>>> [Advance Scan] {alpha_name} | {len(combos)} tổ hợp | {len(tasks)} tasks...")
    
    # Sử dụng Pool với cơ chế bảo vệ Broken Pipe
    all_raw_reports = []
    pool = None
    import time
    start_time = time.time()
    timeout_limit = 300 
    is_timeout = False
    
    try:
        pool = multiprocessing.Pool(processes=30, initializer=init_worker, initargs=(dic_freqs, {alpha_name: alpha_func}))
        iterator = pool.imap_unordered(run_simulation_worker, tasks)
        
        with tqdm(total=len(tasks), desc="Scanning", leave=False, ncols=100) as pbar:
            while True:
                # Kiểm tra Timeout tổng quát
                if time.time() - start_time > timeout_limit:
                    print(f"\n   [Advance Alert] Đã chạm giới hạn timeout {timeout_limit}s. Đang dừng quét và lấy kết quả hiện tại...")
                    is_timeout = True
                    pool.terminate() # Ngắt ngay lập tức các worker
                    break
                    
                try:
                    # Chế độ non-blocking hoặc timeout ngắn để có thể kiểm tra start_time thường xuyên
                    # imap_unordered.next() mặc định là blocking, ta dùng timeout nhỏ (1s) để check loop
                    res_list = iterator.next(timeout=1)
                    all_raw_reports.extend(res_list)
                    pbar.update(1)
                except multiprocessing.TimeoutError:
                    continue # Qua vòng lặp mới để check start_time
                except StopIteration:
                    break
                except (BrokenPipeError, EOFError, Exception) as e:
                    print(f"\n   [Advance Alert] Phát hiện lỗi Worker: {e}")
                    break
        if pool:
            pool.close()
            pool.join()
    except Exception as e:
        if pool:
            try: pool.terminate(); pool.join()
            except: pass
        print(f"   [Advance Error] Scan thất bại: {e}")
            
    df_all = pd.DataFrame(all_raw_reports)
    final_output = []
    
    if not df_all.empty:
        for c_id, combo in enumerate(combos):
            df_combo = df_all[df_all['combo_id'] == c_id].copy()
            if df_combo.empty: continue
            final_output.append({
                'combo': combo,
                'scan_result': agg_scan_results(df_combo[df_combo['is_reverse'] == False]),
                'scan_reverse_result': agg_scan_results(df_combo[df_combo['is_reverse'] == True])
            })
        
    return final_output, is_timeout

def find_best_combo(results, alpha_code):
    """
    Tìm combo tốt nhất theo chuẩn All-time:
    - Chuẩn: Sharpe > 0 >= 90% và Sharpe > 1 >= 40%
    - Nếu có nhiều bộ đạt chuẩn: Ưu tiên Sharpe > 1, sau đó là Sharpe > 2
    - Nếu không có bộ đạt chuẩn: Lấy bộ tiệm cận (gần nhất) với chuẩn 90/40.
    """
    # Tự động gỡ Tuple nếu results được truyền trực tiếp từ run_advance_scan
    if isinstance(results, tuple):
        results = results[0]
        
    all_candidates = []
    
    for item in results:
        for res_type in ['scan_result', 'scan_reverse_result']:
            res = item[res_type]
            all_perf = res.get('all', [0, 0, 0])
            r0, r1, r2 = all_perf
            
            all_candidates.append({
                'combo': item['combo'],
                'direction': 'Normal' if res_type == 'scan_result' else 'Reverse',
                'r0': r0, 'r1': r1, 'r2': r2,
                'full_report': res
            })
            
    if not all_candidates: return None
    
    # 1. Lọc nhóm đạt chuẩn (S0 >= 90 and S1 >= 40)
    strict_group = [c for c in all_candidates if c['r0'] >= 90 and c['r1'] >= 40]
    if strict_group:
        # Sắp xếp: Ưu tiên r1 (S>1) giảm dần, sau đó r2 (S>2) giảm dần
        best_match = max(strict_group, key=lambda x: (x['r1'], x['r2'], x['r0']))
    else:
        # 2. Không có bộ nào đạt chuẩn -> Tìm anh gần nhất
        # Ưu tiên anh lỳ lợm nhất (r0 cao nhất), sau đó là r1
        best_match = max(all_candidates, key=lambda x: (x['r0'], x['r1'], x['r2']))
    
    best_combo = best_match['combo']
    direction = best_match['direction']
    
    # 2. Tiêm tham số vào mã nguồn (Inject Params) & Xóa tham số rác
    match_def = re.search(r'def\s+(\w+)\s*\((.*?)\):', alpha_code)
    if not match_def: return alpha_code
    
    alpha_name = match_def.group(1)
    params_raw = match_def.group(2)
    
    # Lấy danh sách tham số thực dùng (đã có từ bước 5 bên dưới nhưng ta gọi sớm ở đây)
    active_names = get_active_names_from_code(alpha_code)
    
    # Phân tách và lọc signature
    param_parts = [p.strip() for p in params_raw.split(',')]
    new_param_parts = []
    
    for p in param_parts:
        if p in ['df', 'self', 'cls']:
            new_param_parts.append(p)
            continue
            
        # Kiểm tra nếu là dạng name=val
        p_match = re.match(r'(\w+)\s*=\s*(.*)', p)
        if p_match:
            name = p_match.group(1)
            # Nếu không dùng (và có kết quả phân tích AST), bỏ qua luôn
            if active_names and name not in active_names:
                continue
            
            # Nếu dùng, cập nhật giá trị nếu có trong best_combo
            if name in best_combo:
                val = best_combo[name]
                target_val = int(val) if val == int(val) else round(val, 6)
                new_param_parts.append(f"{name}={target_val}")
            else:
                new_param_parts.append(p)
        else:
            # Tham số không có giá trị mặc định
            if active_names and p not in active_names:
                continue
            new_param_parts.append(p)

    new_params_str = ", ".join(new_param_parts)
    
    # Thay thế dòng def cũ bằng dòng def mới (chỉ chứa active params)
    new_code = alpha_code.replace(f"({params_raw}):", f"({new_params_str}):", 1)
    
    # 3. Xử lý hướng Nghịch đảo (Reverse)
    if direction == 'Reverse':
        # Tìm dòng return và thêm dấu trừ phía trước
        if "return " in new_code:
            new_code = new_code.replace("return ", "return -", 1)
            new_code = new_code.replace("return --", "return ", 1) # Chống bị double dấu trừ
            
    # 4. In các chỉ số để xem
    print(f"\n--- PHÂN TÍCH KẾT QUẢ ({best_match.get('status', 'N/A')}) ---")
    print(f"  * Combo tối ưu : {best_combo}")
    print(f"  * Hướng tối ưu: {direction}")
    print(f"  * Sharpe > 0  : {best_match['r0']}%")
    print(f"  * Sharpe > 1  : {best_match['r1']}%")
    print(f"  * Sharpe > 2  : {best_match['r2']}%")
    
    # 5. Tính toán param_range (Dạng Dictionary theo yêu cầu mới)
    param_range = {}
    
    # Cần lọc active params ở đây để param_range lưu vào DB chuẩn xác
    all_orig_params = get_alpha_params(alpha_code)
    active_names = get_active_names_from_code(alpha_code)
    if active_names:
        orig_params = [p for p in all_orig_params if p[0] in active_names]
        if not orig_params: orig_params = all_orig_params # Fallback
    else:
        orig_params = all_orig_params

    if len(orig_params) == 1:
        p1_name = orig_params[0][0]
        param_range[p1_name] = [5, 200, 5]
    elif len(orig_params) >= 2:
        p1_name = orig_params[0][0]
        p2_name = orig_params[1][0]
        
        p1_val = best_combo[p1_name]
        p2_val = best_combo[p2_name]

        if p1_val > p2_val:
            p_large = p1_name
            p_small = p2_name
        else:
            p_large = p2_name
            p_small = p1_name
        
        param_range[p_large] = [10, 100, 10]
        
        v2 = min(p1_val, p2_val)
        if 0 < v2 < 0.5:
            param_range[p_small] = [0.1, 0.7, 0.2]
        elif 0.5 <= v2 < 1:
            param_range[p_small] = [0.3, 0.9, 0.2]
        elif 1 <= v2 <= 4:
            param_range[p_small] = [1, 4, 1]
        elif 5 <= v2 <= 7:
            param_range[p_small] = [4, 7, 1]
        else:
            param_range[p_small] = [10, 40, 10]
            
    return {
        'new_code': new_code,
        'combo': best_combo,
        'direction': direction,
        'param_range': param_range
    }

# if __name__ == "__main__":

#     example_code = """
#     @staticmethod
#     def alpha_factor_miner_new_203_rank(df, window=43):
#         # 1. Khởi tạo dữ liệu cơ bản
#         # Tính toán giá trị giao dịch (Amount) và Khối lượng (Volume)
#         amount = df.get('amount', df['close'] * df.get('matchingVolume', 1))
#         volume = df.get('matchingVolume', df.get('volume', 1))
        
#         # 2. Tìm vị trí (Index) mà tại đó Amount đạt giá trị cao nhất trong cửa sổ quan sát
#         # Trả về giá trị từ 0 đến (window - 1)
#         idx_max = amount.rolling(window).apply(
#             lambda x: np.nanargmax(x) if not np.isnan(x).all() else np.nan, 
#             raw=True
#         )
        
#         # 3. Định nghĩa hàm tính phần dư hồi quy tuyến tính (Linear Regression Residual)
#         # Mục tiêu: Tìm phần biến động của Volume không giải thích được bằng xu hướng thời gian
#         t = np.arange(window)
#         t_mean = t.mean()
#         var_t = ((t - t_mean) ** 2).mean()

#         def linreg_resid(arr):
#             if np.isnan(arr).any():
#                 return np.nan
#             arr_mean = arr.mean()
#             # Tính Slope (Hệ số góc) và Intercept (Hệ số chặn)
#             cov = ((t - t_mean) * (arr - arr_mean)).mean()
#             slope = cov / var_t
#             intercept = arr_mean - slope * t_mean
#             # Dự báo giá trị cuối cùng dựa trên xu hướng
#             pred = intercept + slope * (window - 1)
#             # Trả về phần dư (Giá trị thực tế - Giá trị dự báo)
#             return arr[-1] - pred

#         # 4. Tính toán phần dư cho Volume
#         resid_volume = volume.rolling(window).apply(linreg_resid, raw=True)
        
#         # 5. Tính toán tương quan (Correlation) giữa vị trí đỉnh Amount và phần dư Volume
#         corr = idx_max.rolling(window).corr(resid_volume)
#         raw = -corr
        
#         # 6. Chuẩn hóa Rank về khoảng [-1, 1]
#         normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
#         normalized = normalized.ffill().fillna(0)
        
#         return normalized.clip(-1, 1)
#     """
#     results, _ = run_advance_scan(example_code)
    
#     res = find_best_combo(results, example_code)
#     if res:
    #     print(f"\n✅ ĐÃ TÌM THẤY SIÊU COMBO!")
    #     print(f"Combo: {res['combo']}")
    #     print(f"Param Range: {res['param_range']}")
    #     print(f"Direction: {res['direction']}")
        
    #     print("\n--- MÃ NGUỒN ALPHA ĐÃ TỐI ƯU ---")
    #     print(res['new_code'])
    #     print("--------------------------------")
    # else:
    #     print("\n❌ Không tìm thấy bộ tham số nào thỏa mãn.")
