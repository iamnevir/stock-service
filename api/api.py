

import os
import sys



sys.path.insert(0, "/home/ubuntu/nevir/")
import subprocess
from time import sleep
import psutil
from flask import Blueprint, request, jsonify, Response
from collections import deque
from pymongo import MongoClient
from bson import ObjectId
from flask import Flask
from flask_cors import CORS
from auto.utils import get_mongo_uri
from routes.base import base_bp

stock_bp = Blueprint('stock_bp', __name__)

@stock_bp.route('/get_alpha_log', methods=['POST'])
def get_alpha_log():
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
        db = mongo_client["alpha"]
        alpha_coll = db["alpha_collection"]
        doc = alpha_coll.find_one({"_id": ObjectId(_id)})

        if not doc:
            return jsonify({"message": "Alpha not found"}), 404

        # 2️⃣ Xác định đường dẫn log file
        name = doc.get("name", "")
        if log_type == "run_all_wfa":
            log_file = f"/home/ubuntu/nevir/auto/logs/{name}_run_all_wfa.log"
        elif log_type == "wfa_correlation":
            _is = data.get("is", {})
            start = _is.get("start")
            end = _is.get("end")
            if not start or not end:
                return jsonify({"message": "Missing is start or end for wfa log"}), 400
           
            log_file = f"/home/ubuntu/nevir/auto/logs/{name}_{start}_{end}_wfa_correlation.log"
        else:
            return jsonify({"message": "Invalid log type"}), 400
        

        if not os.path.exists(log_file):
            return jsonify({"message": "Log file not found"}), 404

        # 3️⃣ Đọc N dòng cuối cùng trong file log
        with open(log_file, "r", encoding="utf-8") as f:
            last_lines = deque(f, maxlen=length)
        log_data = "".join(last_lines)

        # 4️⃣ Trả về toàn bộ log (plain text)
        return Response(log_data, mimetype="text/plain")

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@stock_bp.route('/run_all_alpha_wfa', methods=['POST'])
def run_all_alpha_wfa():
    data = request.json
    alpha_id = data.get("id")
    if not alpha_id:
        return jsonify({"message": "Missing id"}), 400
    
    p = subprocess.Popen(
        ["/home/ubuntu/anaconda3/bin/python", "/home/ubuntu/nevir/auto/run_all_wfa.py", alpha_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"Started run_all_wfa with PID {p.pid} for id {alpha_id}")
    sleep(1)  # give some time for the process to start
    return jsonify({"message": f"Running all WFA for id {alpha_id}","result": True}), 200

@stock_bp.route('/get_process_running', methods=['POST'])
def get_process_running() -> bool:
    """
    Kiểm tra có tiến trình 'is.py' hoặc 'backtest.py' với _id đang chạy hay không.

    Args:
        _id (str): ID alpha cần kiểm tra.
        process_type (str): "is" hoặc "backtest".

    Returns:
        bool: True nếu process đang chạy, False nếu không.
    """
    data = request.json
    _id = data.get("_id")
    process_type = data.get("process_type")  # "is" hoặc "backtest
    backtest_type = data.get("backtest_type","")  # None hoặc "os", "os_wfa"
    extra_params = data.get("extra_params")  # None hoặc {"start": "2020-01-01", "end": "2020-12-31"}
    if process_type not in ["run_all_wfa"]:
        raise ValueError("process_type must be 'is' or 'backtest'")
    if backtest_type:
        backtest_type = backtest_type + "_"
    script_map = {
        "run_all_wfa": f"/home/ubuntu/nevir/{backtest_type}auto/run_all_wfa.py"
    }
    script_path = script_map[process_type]
    print(f"Checking process for type: {process_type}, script: {script_path}, id: {_id}, extra_params: {extra_params}")
    # print(f"Checking for process: {script_path} with ID: {_id}")
    if extra_params is not None:
        start = extra_params.get("start")
        end = extra_params.get("end")
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = proc.info["cmdline"]
                if not cmd:
                    continue
                if (
                    "/home/ubuntu/anaconda3/bin/python" in cmd[0]
                    and script_path in cmd[1]
                    and _id in cmd 
                    and str(start) in cmd
                    and str(end) in cmd
                ):
                    print("Process running" )
                    return jsonify({
                    "status": "running",
                }), 200
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    else:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = proc.info["cmdline"]
                if not cmd:
                    continue
                if (
                    "/home/ubuntu/anaconda3/bin/python" in cmd[0]
                    and script_path in cmd[1]
                    and _id in cmd 
                ):
                    print("Process running" )
                    return jsonify({
                    "status": "running",
                }), 200
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    print("Process not running" )
    return jsonify({
                    "status": "not running",
                }), 200

@stock_bp.route('/kill_alpha_process', methods=['POST'])
def kill_alpha_process():
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
        db = mongo_client["alpha"]
        alpha_coll = db["alpha_collection"]

        doc = alpha_coll.find_one({"_id": ObjectId(_id)})
        if not doc:
            return jsonify({"message": "Alpha not found"}), 404

        # 2️⃣ Xác định script path theo loại
        script_map = {
            "run_all_wfa": "/home/ubuntu/nevir/auto/run_all_wfa.py"
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
                alpha_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {"$set": {f"wfa_status": "not_running"}}
                )
            else:
                alpha_coll.update_one(
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
            alpha_coll.update_one(
                {"_id": ObjectId(_id)},
                {"$set": {f"wfa_status": "not_running"}}
            )
        else:
            alpha_coll.update_one(
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
     
@stock_bp.route('/get_busd_log', methods=['POST'])
def get_busd_log():
    try:
        data = request.json
        _id = data.get("_id")
        log_type = data.get("type")  # "is" hoặc "backtest"
        length = int(data.get("length", 100))

        if not _id:
            print("Missing _id")
            return jsonify({"message": "Missing _id"}), 400
        if log_type not in ["run_all_wfa","wfa_correlation"]:
            print("Invalid log type")
            return jsonify({"message": "Invalid log type"}), 400

        # 1️⃣ Kết nối Mongo
        mongo_client = MongoClient(get_mongo_uri("mgc3"))
        db = mongo_client["busd"]
        busd_coll = db["busd_collection"]
        doc = busd_coll.find_one({"_id": ObjectId(_id)})

        if not doc:
            return jsonify({"message": "busd not found"}), 404

        # 2️⃣ Xác định đường dẫn log file
        name = doc.get("name", "")
        if log_type == "run_all_wfa":
            log_file = f"/home/ubuntu/nevir/busd_auto/logs/{name}_run_all_wfa.log"
        elif log_type == "wfa_correlation":
            _is = data.get("is", {})
            start = _is.get("start")
            end = _is.get("end")
            if not start or not end:
                return jsonify({"message": "Missing is start or end for wfa log"}), 400
           
            log_file = f"/home/ubuntu/nevir/busd_auto/logs/{name}_{start}_{end}_wfa_correlation.log"
        else:
            return jsonify({"message": "Invalid log type for busd"}), 400
            
        if not os.path.exists(log_file):
            return jsonify({"message": "Log file not found"}), 404

        # 3️⃣ Đọc N dòng cuối cùng trong file log
        with open(log_file, "r", encoding="utf-8") as f:
            last_lines = deque(f, maxlen=length)
        log_data = "".join(last_lines)

        # 4️⃣ Trả về toàn bộ log (plain text)
        return Response(log_data, mimetype="text/plain")

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@stock_bp.route('/run_all_wfa', methods=['POST'])
def run_all_wfa():
    data = request.json
    busd_id = data.get("id")
    if not busd_id:
        return jsonify({"message": "Missing id"}), 400
    
    p = subprocess.Popen(
        ["/home/ubuntu/anaconda3/bin/python", "/home/ubuntu/nevir/busd_auto/run_all_wfa.py", busd_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"Started run_all_wfa with PID {p.pid} for id {busd_id}")
    return jsonify({"message": f"Running all WFA for id {busd_id}","result": True}), 200

@stock_bp.route('/kill_busd_process', methods=['POST'])
def kill_busd_process():
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
        db = mongo_client["busd"]
        busd_coll = db["busd_collection"]

        doc = busd_coll.find_one({"_id": ObjectId(_id)})
        if not doc:
            return jsonify({"message": "Busd not found"}), 404

        # 2️⃣ Xác định script path theo loại
        script_map = {
            "run_all_wfa": "/home/ubuntu/nevir/busd_auto/run_all_wfa.py"
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
                busd_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {"$set": {f"wfa_status": "not_running",}}
                )
                busd_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {
                        "$set": {
                            "wfa.$[elem].correlation.status": "not_running"
                        }
                    },
                    array_filters=[
                        {"elem.correlation.status": "running"}
                    ]
                )
            else:
                busd_coll.update_one(
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
        if process_type == "run_all_wfa":
            busd_coll.update_one(
                {"_id": ObjectId(_id)},
                {"$set": {f"wfa_status": "not_running"
                         }}
            )
            busd_coll.update_one(
                    {"_id": ObjectId(_id)},
                    {
                        "$set": {
                            "wfa.$[elem].correlation.status": "not_running"
                        }
                    },
                    array_filters=[
                        {"elem.correlation.status": "running"}
                    ]
                )
        else:
            busd_coll.update_one(
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
     
@stock_bp.route('/get_server_status', methods=['GET'])
def get_server_status():
    cmd = "ps -C python -o pid= | wc -l"
    count = int(subprocess.check_output(cmd, shell=True))
    server_status = {"running":count}
    return jsonify({"message": f"Get success.","result":server_status}), 200

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(stock_bp, url_prefix="/")
    app.register_blueprint(base_bp, url_prefix="/")

    @app.route("/", methods=["GET"])
    def on():
        return "API On"

    return app

app = create_app()
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", 8057))
    )

