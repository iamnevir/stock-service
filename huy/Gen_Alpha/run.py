import os
import json
import time
import re
import textwrap
import subprocess
import multiprocessing
from openai import OpenAI
import pymongo
import pickle
import sys
from datetime import datetime

# Khai báo các đường dẫn
INPUT_RAW_FORMULA_ALPHA = "/home/ubuntu/nevir/huy/Gen_Alpha/quanta.json"
PROMT = "/home/ubuntu/nevir/huy/Gen_Alpha/promt_v0.txt"
ALPHA_PATCH = "/home/ubuntu/nevir/huy/Gen_Alpha/alpha.py"
RUN_IC_SCRIPT = "/home/ubuntu/nevir/huy/gen1_2_factor_minor/run.py"
RUN_SCAN_SCRIPT = "/home/ubuntu/nevir/huy/scan/final.py"
QUEUE_DIR = "/home/ubuntu/nevir/huy/Gen_Alpha/queue/"

if not os.path.exists(QUEUE_DIR):
    os.makedirs(QUEUE_DIR)

def get_db():
    try:
        sys.path.insert(0, "/home/ubuntu")
        from auto.utils import get_mongo_uri
        client = pymongo.MongoClient(get_mongo_uri("mgc3"))
        return client["alpha"]["gen_alpha"]
    except Exception as e:
        print(f"Lỗi kết nối DB: {e}")
        return None

def chat_with_deepseek(system_promt, user_promt):
    # API KEY CỦA BẠN (Vui lòng kiểm tra lại key này)
    api_key = "sk-d2920fe91a98497eadcaf6bdd63506c8"
    client_ai = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client_ai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": user_promt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def get_next_alpha_index(file_path, prefix):
    if not os.path.exists(file_path): return 1
    max_idx = 0
    pattern = re.compile(rf"alpha_{re.escape(prefix)}_(\d+)")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx: max_idx = idx
    except: pass
    return max_idx + 1

def loop_generation(shared_state):
    shared_state['gen'] = "Đang khởi động..."
    print(">>> [Vòng lặp 1] Khởi động trình SINH Alpha...")
    prefix = os.path.splitext(os.path.basename(INPUT_RAW_FORMULA_ALPHA))[0]
    alpha_coll = get_db()
    if alpha_coll is None: return

    while True:
        try:
            with open(INPUT_RAW_FORMULA_ALPHA, "r", encoding="utf-8") as f:
                all_formulas = json.load(f)
            
            # Logic mới: Chỉ coi là xong nếu đã có ĐỦ 5 biến thể cho công thức đó (kiểm tra theo alpha_name)
            unprocessed_keys = []
            for k in all_formulas.keys():
                # Lấy số thứ tự từ key (ví dụ: factor_miner_new_153 -> 153)
                try:
                    idx = int(k.split('_')[-1])
                    # Đếm xem trong DB đã có đủ 5 biến thể của số thứ tự này chưa
                    variants_count = alpha_coll.count_documents({"alpha_name": {"$regex": f"alpha_{prefix}_{idx:03d}_"}})
                    if variants_count < 5:
                        unprocessed_keys.append(k)
                except:
                    unprocessed_keys.append(k)
            
            if unprocessed_keys:
                key = unprocessed_keys[0]
                shared_state['gen'] = key
                
                # --- CƠ CHẾ QUEUE BẰNG PICKLE ---
                queue_file = os.path.join(QUEUE_DIR, f"queue_{key}.pickle")
                response_json = {}

                if os.path.exists(queue_file):
                    print(f"🔄 [Resume] Tìm thấy hàng đợi tạm cho {key}. Đang khôi phục...")
                    with open(queue_file, 'rb') as f:
                        response_json = pickle.load(f)
                else:
                    print(f"\n[AI Progress] Đang xử lý công thức: {key}")
                    with open(PROMT, "r", encoding="utf-8") as f:
                        system_promt = f.read()
                    
                    alpha_input = json.dumps({key: all_formulas[key]})
                    response_text = chat_with_deepseek(system_promt, alpha_input)
                    
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:
                        response_json = json.loads(match.group())
                        with open(queue_file, 'wb') as f:
                            pickle.dump(response_json, f)
                        print(f"💾 Đã lưu 5 biến thể của {key} vào hàng đợi tạm.")

                if response_json:
                    # Logic lấy số thứ tự: 
                    start_idx = None
                    try:
                        match_idx = re.search(r'_(\d+)$', key)
                        if match_idx:
                            start_idx = int(match_idx.group(1))
                    except:
                        pass
                    
                    if start_idx is None:
                        start_idx = get_next_alpha_index(ALPHA_PATCH, prefix)

                    naming_map = {'A': 'rank', 'B': 'tanh', 'C': 'zscore', 'D': 'sign', 'E': 'wf'}
                    
                    for i, (original_id, alpha_code) in enumerate(response_json.items(), 1):
                        variant = original_id[-1]
                        norm_type = naming_map.get(variant, variant)
                        core_name = original_id[:-2]
                        new_name = f"alpha_{prefix}_{start_idx:03d}_{norm_type}"
                        new_original_id = f"{core_name}_{norm_type}"

                        shared_state['gen'] = f"{key} ({i}/5: {norm_type})"

                        # KIỂM TRA: Nếu con này đã có trong DB rồi thì bỏ qua luôn cho nhanh
                        if alpha_coll.find_one({"alpha_name": new_name}):
                            print(f"   [Skip] {new_name} đã tồn tại trong DB.")
                            continue

                        # alpha_code = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', f'def {new_name}(', alpha_code)
                        alpha_code = re.sub(r'def\s+([^\(]+)\s*\(', f'def {new_name}(', alpha_code)
                        formatted_code = textwrap.indent(textwrap.dedent(alpha_code.strip()), '    ')

                        # Kiểm tra cú pháp và TỐI ƯU HÓA THAM SỐ
                        try:
                            # 1. Kiểm tra cú pháp
                            test_code = f"class Alpha:\n{formatted_code}"
                            compile(test_code, '<string>', 'exec')
                            
                            import huy.Gen_Alpha.advance as advance
                            import importlib
                            importlib.reload(advance) # ÉP NẠP LẠI MÃ MỚI NHẤT
                            
                            is_rev = False
                            p_range = {}
                            is_err = 0
                            r_ic = 1
                            
                            try:
                                print(f"\n[Advance Optimizer] Đang tối ưu {new_name} ({i}/5)...")
                                shared_state['gen'] = f"{key} (Advance {i}/5)"
                                results, timed_out = advance.run_advance_scan(formatted_code)
                                
                                if timed_out:
                                    print(f"🚨 [TIMEOUT Advance] {new_name} -> Dùng tham số mặc định & chuyển sang Scan Sharpe")
                                else:
                                    best_info = advance.find_best_combo(results, formatted_code)
                                    if best_info:
                                        formatted_code = best_info['new_code']
                                        alpha_code = textwrap.dedent(best_info['new_code']).strip()
                                        is_rev = True if best_info.get('direction') == 'Reverse' else False
                                        p_range = best_info.get('param_range', {})
                                        print(f"✅ Tối ưu xong {new_name} (is_reverse: {is_rev})")
                                    else:
                                        print(f"⚠️ Không tìm thấy bộ tham số tốt cho {new_name}, giữ nguyên mặc định.")
                            except Exception as e:
                                print(f"⚠️ Lỗi khi tối ưu {new_name}: {e}")

                            # 3. GHI FILE alpha.py
                            with open(ALPHA_PATCH, "a", encoding="utf-8") as f:
                                f.write("\n\n" + formatted_code)
                            
                            # 4. LƯU TẤT CẢ VÀO DATABASE
                            alpha_coll.update_one(
                                {"alpha_name": new_name},
                                {"$set": {
                                    "original_id": new_original_id,
                                    "alpha_code": alpha_code,
                                    "group": prefix,
                                    "param_range": p_range,
                                    "is_reverse": is_rev,
                                    "run_IC": r_ic, 
                                    "scan": 0,
                                    "is_error": is_err,
                                    "create_at": time.time()
                                }},
                                upsert=True
                            )
                            print(f"   + Đã sinh và lưu DB: {new_name} (run_IC: {r_ic}, is_error: {is_err})")
                                
                        except Exception as syntax_err:
                            print(f"❌ [LỖI CÚ PHÁP AI] Bỏ qua {new_name} do code AI sinh không hợp lệ: {syntax_err}")
                            continue

                    # Kiểm tra cuối: Xóa file queue sau khi đã xử lý xong hết các biến thể bên trong
                    if os.path.exists(queue_file):
                        os.remove(queue_file)
                        print(f"🧹 Đã dọn dẹp hàng đợi tạm cho {key}.")
                time.sleep(1)
            else:
                shared_state['gen'] = "Đang đợi công thức mới..."
                time.sleep(5)
        except Exception as e:
            print(f"Lỗi Vòng lặp 1 (Gen): {e}")                          
            time.sleep(5)

def loop_ic(shared_state):
    shared_state['ic'] = "Đang khởi động..."
    print(">>> [Vòng lặp 2] Khởi động trình tính IC...")
    alpha_coll = get_db()
    if alpha_coll is None: return

    while True:
        try:
            pending_docs = list(alpha_coll.find({"run_IC": 0}))
            if pending_docs:
                names = [doc["alpha_name"] for doc in pending_docs]
                shared_state['ic'] = f"{names[0]} (+{len(names)-1})"
                print(f"\n[IC Queue] Phát hiện {len(names)} alpha mới: {', '.join(names)}")
                subprocess.run(["python", RUN_IC_SCRIPT], check=True)
                time.sleep(0.01)
            else:
                shared_state['ic'] = "N/A (Chờ Alpha)"
                time.sleep(1)
        except Exception as e:
            print(f"Lỗi Vòng lặp 2 (IC): {e}")
            time.sleep(0.1)

def loop_scan(shared_state):
    shared_state['scan'] = "Đang khởi động..."
    print(">>> [Vòng lặp 3] Khởi động trình QUÉT Sharpe...")
    alpha_coll = get_db()
    if alpha_coll is None: return

    # Lấy prefix để lọc đúng group hiện tại
    prefix = os.path.splitext(os.path.basename(INPUT_RAW_FORMULA_ALPHA))[0]

    while True:
        try:
            # Chỉ Scan những con ĐÃ XONG IC, KHÔNG BỊ LỖI và THUỘC GROUP HIỆN TẠI
            pending_docs = list(alpha_coll.find({"group": prefix, "scan": 0, "run_IC": 1, "is_error": 0}))
            if pending_docs:
                names = [doc["alpha_name"] for doc in pending_docs]
                shared_state['scan'] = f"{names[0]} (+{len(names)-1})"
                display_names = names[:5]
                suffix = f" và {len(names)-5} con khác" if len(names) > 5 else ""
                print(f"\n[Scan Queue] Bắt đầu đợt quét cho {len(names)} con: {', '.join(display_names)}{suffix}")
                
                # THỰC HIỆN SCAN + OPTIMIZE (Truyền đúng group hiện tại)
                cmd = ["python", RUN_SCAN_SCRIPT, "--group", prefix]
                
                try:
                    p = subprocess.Popen(cmd) # Chạy bình thường để nhận signal từ main
                    p.wait()
                except KeyboardInterrupt:
                    if p.poll() is None:
                        p.kill()
                    raise
                time.sleep(1)
            else:
                shared_state['scan'] = "N/A (Chờ IC)"
                # Log nhẹ để biết tiến trình vẫn sống
                if int(time.time()) % 60 < 5: 
                    print("... [Scan Loop] Đang đợi Alpha hoàn thành IC ...", end="\r")
                time.sleep(5)
        except Exception as e:
            print(f"Lỗi Vòng lặp 3 (Scan): {e}")
            time.sleep(5)

def loop_monitor(shared_state):
    print(">>> [Vòng lặp 4] Khởi động trình GIÁM SÁT...")
    alpha_coll = get_db()
    if alpha_coll is None: return

    # Lấy prefix để lọc đúng group hiện tại
    prefix = os.path.splitext(os.path.basename(INPUT_RAW_FORMULA_ALPHA))[0]

    while True:
        try:
            query = {"group": prefix}
            total = alpha_coll.count_documents(query)
            pending_ic = alpha_coll.count_documents({**query, "run_IC": 0, "is_error": 0})
            pending_scan = alpha_coll.count_documents({**query, "run_IC": 1, "scan": 0, "is_error": 0})
            done = alpha_coll.count_documents({**query, "scan": 1, "is_error": 0})
            error = alpha_coll.count_documents({**query, "is_error": 1})
            timeout = alpha_coll.count_documents({**query, "is_error": 2})

            now_str = datetime.now().strftime("%H:%M:%S")
            print("\n" + "="*60)
            print(f"📊 MONITOR [{now_str}] (Group: {prefix})")
            print(f"  [1] SINH CODE :  {shared_state.get('gen', 'N/A')}")
            print(f"  [2] TÍNH IC   :  {shared_state.get('ic', 'N/A')}")
            print(f"  [3] QUÉT SHARP:  {shared_state.get('scan', 'N/A')}")
            print("-" * 30)
            print(f"  TỔNG: {total} | ĐỢI IC: {pending_ic} | ĐỢI SCAN: {pending_scan} | XONG: {done} | LỖI: {error} | TIMEOUT: {timeout}")
            print("="*60 + "\n")
            
            time.sleep(15)
        except:
            time.sleep(10)

# Đã loại bỏ process_wrapper để giữ chung Process Group cho việc xử lý Ctrl+C

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    
    # Khởi tạo các tiến trình chung nhóm với main
    p1 = multiprocessing.Process(target=loop_generation, args=(shared_state,))
    p3 = multiprocessing.Process(target=loop_scan, args=(shared_state,))
    p4 = multiprocessing.Process(target=loop_monitor, args=(shared_state,))
    
    processes = [p1, p3, p4]
    
    try:
        for p in processes:
            p.start()
            
        print("\n🚀 HỆ THỐNG ALPHA PIPELINE ĐANG CHẠY...")
        
        # Luồng chính thực hiện giám sát trạng thái để tự động dừng
        empty_count = 0
        prefix = os.path.splitext(os.path.basename(INPUT_RAW_FORMULA_ALPHA))[0]
        
        while True:
            try:
                alpha_coll = get_db()
                if alpha_coll is not None:
                    # 1. Kiểm tra tiến độ Sinh code
                    try:
                        with open(INPUT_RAW_FORMULA_ALPHA, 'r', encoding='utf-8') as f:
                            all_formulas = json.load(f)
                        processed_oids = alpha_coll.distinct("original_id")
                        unprocessed_gen = 0
                        for k in all_formulas.keys():
                            if sum(1 for oid in processed_oids if k in oid) < 5:
                                unprocessed_gen += 1
                    except:
                        unprocessed_gen = 0
                    
                    # 2. Kiểm tra tiến độ Scan
                    pending_scan = alpha_coll.count_documents({"group": prefix, "run_IC": 1, "scan": 0, "is_error": 0})
                    
                    
                    
            except Exception as e:
                print(e)
                pass
            time.sleep(15)

    except KeyboardInterrupt:
        print("\n" + "!"*60)
        print("🛑 [HỆ THỐNG] ĐANG DỪNG KHẨN CẤP TOÀN BỘ PIPELINE...")
        print("!"*60)
        import signal
        # Tiêu diệt sạch sẽ toàn bộ group tiến trình
        os.killpg(0, signal.SIGKILL)
        sys.exit(0)
