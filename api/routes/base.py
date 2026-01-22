import subprocess
from time import sleep
import psutil
from flask import Blueprint, request, jsonify, Response
from collections import deque
from pymongo import MongoClient
from bson import ObjectId
from flask import Flask
from flask_cors import CORS
from base_auto.utils import get_mongo_uri
import os
import sys

base_bp = Blueprint('base_bp', __name__)

@base_bp.route('/get_base_log', methods=['POST'])
def get_base_log():
    try:
        data = request.json
        _id = data.get("_id")
        log_type = data.get("type")  # "is" hoặc "backtest"
        length = int(data.get("length", 100))

        if not _id:
            print("Missing _id")
            return jsonify({"message": "Missing _id"}), 400
        if log_type not in ["run_all_wfa", "wfa_correlation"]:
            print("Invalid log type")
            return jsonify({"message": "Invalid log type"}), 400

        # 1️⃣ Kết nối Mongo
        mongo_client = MongoClient(get_mongo_uri("mgc3"))
        db = mongo_client["base"]
        base_coll = db["base_collection"]
        doc = base_coll.find_one({"_id": ObjectId(_id)})

        if not doc:
            return jsonify({"message": "base not found"}), 404

        # 2️⃣ Xác định đường dẫn log file
        name = doc.get("name", "")
        if log_type == "run_all_wfa":
            log_file = f"/home/ubuntu/nevir/base_auto/logs/{name}_run_all_wfa.log"
        elif log_type == "wfa_correlation":
            _is = data.get("is", {})
            start = _is.get("start")
            end = _is.get("end")
            if not start or not end:
                return jsonify({"message": "Missing is start or end for wfa log"}), 400
           
            log_file = f"/home/ubuntu/nevir/base_auto/logs/{name}_{start}_{end}_wfa_correlation.log"
        else:
            return jsonify({"message": "Invalid log type"}), 400
 
        
        if not os.path.exists(log_file):
            print("no log file")
            return jsonify({"message": "Log file not found"}), 200

        # 3️⃣ Đọc N dòng cuối cùng trong file log
        with open(log_file, "r", encoding="utf-8") as f:
            last_lines = deque(f, maxlen=length)
        log_data = "".join(last_lines)

        # 4️⃣ Trả về toàn bộ log (plain text)
        return Response(log_data, mimetype="text/plain")

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@base_bp.route('/run_all_base_wfa', methods=['POST'])
def run_all_base_wfa():
    data = request.json
    base_id = data.get("id")
    if not base_id:
        return jsonify({"message": "Missing id"}), 400
    
    p = subprocess.Popen(
        ["/home/ubuntu/anaconda3/bin/python", "/home/ubuntu/nevir/base_auto/run_all_wfa.py", base_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"Started run_all_wfa with PID {p.pid} for id {base_id}")
    sleep(1)  # give some time for the process to start
    return jsonify({"message": f"Running all WFA for id {base_id}","result": True}), 200

@base_bp.route('/kill_base_process', methods=['POST'])
def kill_base_process():
    # try:
        data = request.json
        _id = data.get("_id")
        process_type = data.get("type")  # "is" hoặc "backtest"

        if not _id:
            print("Missing _id")
            return jsonify({"message": "Missing _id"}), 400
        if process_type not in ["run_all_wfa"]:
            print("Invalid process type")
            return jsonify({"message": "Invalid process type"}), 400

        # 1️⃣ Kết nối Mongo
        mongo_client = MongoClient(get_mongo_uri("mgc3"))
        db = mongo_client["base"]
        base_coll = db["base_collection"]

        doc = base_coll.find_one({"_id": ObjectId(_id)})
        if not doc:
            return jsonify({"message": "base not found"}), 404

        # 2️⃣ Xác định script path theo loại
        script_map = {
            "run_all_wfa": "/home/ubuntu/nevir/base_auto/run_all_wfa.py"
        }
        script_path = script_map[process_type]

        # 3️⃣ Lọc process theo _id và script
        target_procs = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = proc.info["cmdline"]
                if not cmd:
                    continue
                if ("/home/ubuntu/anaconda3/bin/python" in cmd[0] and
                    script_path in cmd[1] and
                    _id in cmd):
                    target_procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 4️⃣ Nếu không có process → cleanup DB
        if not target_procs:
            if process_type == "run_all_wfa":
                base_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {"$set": {f"wfa_status": "not_running"}}
                )
            else:
                base_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {"$set": {f"{process_type}.status": "not_running"},
                    "$unset": {f"{process_type}.pid": ""}}
                )
            return jsonify({"message": f"No {process_type} process found for {_id}"}), 200

        # 5️⃣ SIGTERM các process
        for proc in target_procs:
            try:
                proc.terminate()
            except Exception:
                continue

        # 6️⃣ Đợi tối đa 5s
        gone, alive = psutil.wait_procs(target_procs, timeout=5)

        # 7️⃣ SIGKILL nếu vẫn còn
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                continue

        # 8️⃣ Cleanup MongoDB
        if  process_type == "run_all_wfa":
            base_coll.update_one(
                {"_id": ObjectId(_id)},
                {"$set": {f"wfa_status": "not_running"}}
            )
        else:
            base_coll.update_one(
                {"_id": ObjectId(_id)},
                {"$set": {f"{process_type}.status": "not_running"},
                "$unset": {f"{process_type}.pid": ""}}
            )

        return jsonify({
            "message": f"{process_type.capitalize()} {_id} processes terminated successfully",
            "result": True
        }), 200

    # except Exception as e:
    #     return jsonify({"message": str(e)}), 500
     