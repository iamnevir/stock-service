import pymongo
import os
import sys
import json
import datetime

sys.path.insert(0, "/home/ubuntu/nevir")
from auto.utils import get_mongo_uri

def main():
    tracking_file = "/home/ubuntu/nevir/huy/evolution/processed_alphas.json"
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracking_data = {}
    all_processed_ids = set()
    
    if os.path.exists(tracking_file):
        with open(tracking_file, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    # Migration: old format was a list of IDs
                    tracking_data = {"historical": data}
                    all_processed_ids = set(data)
                elif isinstance(data, dict):
                    tracking_data = data
                    for ids_list in data.values():
                        if isinstance(ids_list, list):
                            all_processed_ids.update(ids_list)
            except (json.JSONDecodeError, ValueError):
                tracking_data = {}

    client = pymongo.MongoClient(get_mongo_uri())
    col = client["alpha"]["gen_alpha"]
    
    print("Đang tải toàn bộ alphas từ database để phân tích các biến thể...")
    docs = col.find({"alpha_name": {"$exists": True}})
    
    from collections import defaultdict
    groups = defaultdict(list)
    
    for doc in docs:
        alpha_name = doc.get("alpha_name", "")
        if not alpha_name:
            continue
        # Lấy base name bằng cách cắt phần đuôi (vd: _rank, _tanh, _zscore)
        base_name = alpha_name.rsplit('_', 1)[0]
        groups[base_name].append(doc)
        
    ids = []
    
    for base_name, variants in groups.items():
        best_doc = None
        best_score = -1 # Theo dõi điểm r1 cao nhất
        
        for doc in variants:
            score = 0
            is_error = doc.get("is_error", False)
            scan_result = doc.get("scan_result")
            
            if not is_error and scan_result and "all" in scan_result:
                score = scan_result["all"][0] # r1_all
            
            if score >= best_score:
                best_score = score
                best_doc = doc
                
        if best_doc:
            scan_result = best_doc.get("scan_result")
            is_error = best_doc.get("is_error", False)
            doc_id = str(best_doc["_id"])
            r1 = 0
            r2 = 0
            if scan_result and "all" in scan_result:
                r1 = scan_result["all"][0]
                r2 = scan_result["all"][1]
                
            # Kiểm tra nếu chưa từng chạy, và chưa đạt chuẩn hoặc bị lỗi
            if doc_id not in all_processed_ids and (is_error or not scan_result or r1 < 80 or r2 < 40):
                ids.append(doc_id)
        
    print(f"✅ Tìm thấy {len(ids)} alphas không đạt chuẩn và CHƯA TỪNG ĐƯỢC CHẠY.")
    
    if not ids:
        print("Tất cả các alpha đều đã đạt chuẩn hoặc đã được chạy trước đó!")
        return

    # Chạy theo từng batch (vd: 50 IDs / 1 lần gọi lệnh) để tránh lỗi lệnh quá dài trong Terminal
    batch_size = 50
    import subprocess
    
    try:
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            # Sử dụng list thay vì shell=True để xử lý tín hiệu Ctrl+C tốt hơn
            cmd = [sys.executable, "/home/ubuntu/nevir/huy/evolution/run_test.py", "--mode", "flaw", "--id"] + batch
            
            print(f"\n🚀 ĐANG CHẠY BATCH {i//batch_size + 1}/{(len(ids) - 1)//batch_size + 1} ({len(batch)} alphas)...")
            
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"\n⚠️ Batch {i//batch_size + 1} bị dừng hoặc gặp lỗi. Không lưu vào lịch sử.")
                break
                
            # Lưu lại lịch sử những ID vừa chạy xong
            if start_time not in tracking_data:
                tracking_data[start_time] = []
            tracking_data[start_time].extend(batch)
            all_processed_ids.update(batch)
            
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=4)
                
            print(f"✅ Hoàn thành batch {i//batch_size + 1} và đã lưu lịch sử!")
            
    except KeyboardInterrupt:
        print("\n🛑 Đã nhận lệnh dừng từ bàn phím (Ctrl+C). Đang thoát script...")

if __name__ == "__main__":
    main()
