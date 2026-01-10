
import os
import subprocess
import datetime
import logging

from auto.utils import get_mongo_uri

# ================== CONFIG ==================
BACKUP_BASE_DIR = "/home/ubuntu/nevir/mongo_bak"
LOG_FILE = "/home/ubuntu/nevir/mongo_bak/backup.log"

BACKUP_DBS = [
    "alpha",
    "busd",
    "stock_backtest",
    "gen1_2"
]


EXCLUDE_COLLECTIONS = {
    "alpha": ["wfa_results", "correlation_results"],
    "busd": ["wfa_results", "correlation_results"],
    "stock_backtest": ["correlation_results"],
    "gen1_2": ["correlation_results"]
}

# ================== LOGGING ==================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
def cleanup_old_backups(base_dir, keep_dir):
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if name != keep_dir and os.path.isdir(path):
            try:
                logging.info(f"Deleting old backup: {path}")
                subprocess.run(["rm", "-rf", path], check=True)
            except Exception as e:
                logging.error(f"Failed to delete {path}: {e}")

def main():
    mongo_uri = get_mongo_uri("mgc3")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    backup_dir = os.path.join(BACKUP_BASE_DIR, today)
    os.makedirs(backup_dir, exist_ok=True)

    logging.info("=== START BACKUP ===")
    cleanup_old_backups(BACKUP_BASE_DIR, today)
    for db in BACKUP_DBS:
        

        output_file = f"{db}.archive.gz"
        output_path = os.path.join(backup_dir, output_file)

        cmd = [
            "mongodump",
            f"--uri={mongo_uri}",
            "--db", db,
            "--archive",
            "--gzip"
        ]

        # Thêm exclude collection nếu có
        for col in EXCLUDE_COLLECTIONS.get(db, []):
            cmd.extend(["--excludeCollection", col])

        logging.info(f"Dumping database: {db}")
        try:
            with open(output_path, "wb") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)

            logging.info(f"Saved backup: {output_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Backup failed for {db}: {e.stderr.decode()}")

    logging.info("=== BACKUP COMPLETED ===")

# ================= RUN =================
if __name__ == "__main__":
    main()
