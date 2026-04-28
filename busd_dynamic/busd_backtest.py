"""
busd_backtest.py
================
Đọc output của busd_correlation (play_strategy) từ collection busd_dynamic,
sau đó chạy MegaBbAccV2 trên dữ liệu của next_week (tuần tiếp theo),
và lưu kết quả play_report vào từng srcs element trong DB.

Flow:
    1. Lấy các doc có `play_strategy` trong srcs.
    2. Với mỗi doc, xác định next_week và lst_day của tuần đó.
    3. Load dữ liệu intaday cho next_week.
    4. Chạy Simulator (MegaBbAccV2) với play_strategy configs.
    5. Lưu report vào `srcs.$[elem].play_report`.
"""

import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd
from pymongo import MongoClient

from busd_auto.MegaBbAccV2 import MegaBbAccV2 as Simulator
from busd_auto.utils import get_mongo_uri, sanitize_for_bson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEE = 175
DELAY = 1
DATA_PKL = "/home/ubuntu/nevir/data/busd.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df_base() -> pd.DataFrame:
    """Load toàn bộ dữ liệu intraday từ file pickle."""
    df = pd.read_pickle(DATA_PKL)
    return df


def _prepare_df_for_src(df_base: pd.DataFrame, start_day: int, end_day: int, src: str) -> pd.DataFrame:
    """Filter và chuẩn bị df cho một src/tuần cụ thể."""
    df = df_base[(df_base["day"] >= start_day) & (df_base["day"] <= end_day)].copy()
    if df.empty:
        return df

    df["based_col"] = 0
    for col in src.split("_"):
        if col == "hose500":
            df["based_col"] += df["aggBusd"]
        elif col == "fhose500":
            df["based_col"] += df["aggFBusd"]
        elif col == "vn30":
            df["based_col"] += df["aggBusdVn30"]
        elif col == "fvn30":
            df["based_col"] += df["aggFBusdVn30"]

    df = df[df["based_col"].diff() != 0]
    df = df[df["time_int"] >= 10091500]
    if df.empty:
        return df

    df["priceChange"] = df.groupby("day")["last"].diff().shift(-1).fillna(0)
    return df


def _run_mega(df: pd.DataFrame, configs: List[str], src: str, mode: str,
              start_day: int, end_day: int) -> dict:
    """Chạy Simulator (MegaBbAccV2) và trả về report dict."""
    bt = Simulator(
        configs=configs,
        df_alpha=df,
        data_start=start_day,
        data_end=end_day,
        fee=FEE,
        delay=DELAY,
        moving_average_type=mode,
        book_size=50
    )
    bt.compute_mas()
    bt.compute_all_position()
    bt.compute_mega_position()
    bt.compute_profit_and_df_1d()
    report = bt.compute_report(bt.df_1d)
    return sanitize_for_bson(report)


def _build_next_week_map(collection) -> dict:
    """
    Truy vấn toàn bộ week trong collection, sort theo thứ tự tăng dần,
    và trả về dict: {week -> next_week} dựa trên vị trí liên tiếp.
    """
    all_weeks = sorted(
        doc["week"]
        for doc in collection.find({"week": {"$exists": True}}, {"week": 1})
        if doc.get("week")
    )
    week_map = {}
    for i, w in enumerate(all_weeks):
        week_map[w] = all_weeks[i + 1] if i + 1 < len(all_weeks) else None
    print(f"[backtest] Đã build next_week map: {len(all_weeks)} tuần")
    return week_map


def _get_next_week_days(collection, next_week: str) -> Optional[List[int]]:
    """Lấy lst_day của next_week từ DB."""
    if not next_week:
        return None
    doc = collection.find_one({"week": next_week}, {"lst_day": 1})
    if not doc:
        return None
    lst_day = sorted(int(d) for d in doc.get("lst_day", []))
    return lst_day if lst_day else None


# ---------------------------------------------------------------------------
# Main processing per doc/src
# ---------------------------------------------------------------------------

def process_doc_src(
    collection,
    doc: dict,
    src_item: dict,
    df_base: pd.DataFrame,
    next_week_map: dict,
    force: bool = False,
) -> None:
    week = doc.get("week")
    # Ưu tiên field next_week trong doc (nếu có), fallback sang map tính từ collection
    next_week = doc.get("next_week") or next_week_map.get(week)
    src = src_item.get("src")
    mode = src_item.get("mode")
    play_strategy: List[str] = src_item.get("play_strategy", [])

    # Validate
    if not week or not src or not mode:
        print(f"[backtest] skip {week} {src} {mode}: thiếu thông tin week/src/mode")
        return

    if not play_strategy:
        print(f"[backtest] skip {week} {src} {mode}: không có play_strategy")
        return

    if src_item.get("play_report") and not force:
        print(f"[backtest] skip {week} {src} {mode}: đã có play_report (dùng --force để rebuild)")
        return

    # Không có next_week thì vẫn giữ luồng xử lý tổng thể, nhưng bỏ qua backtest/report.
    if not next_week:
        print(f"[backtest] {week} {src} {mode}: không có next_week, bỏ qua chạy backtest/play_report")
        return

    # Lấy lst_day của next_week
    next_week_days = _get_next_week_days(collection, next_week)
    if not next_week_days:
        print(f"[backtest] skip {week} {src} {mode}: không tìm thấy lst_day cho next_week={next_week}")
        return

    start_day = next_week_days[0]
    end_day = next_week_days[-1]

    print(f"[backtest] {week} → next_week={next_week} ({start_day}-{end_day}) src={src} mode={mode} configs={len(play_strategy)}")

    # Chuẩn bị dữ liệu
    df = _prepare_df_for_src(df_base, start_day, end_day, src)
    if df.empty:
        print(f"[backtest] skip {week} {src} {mode}: df rỗng cho next_week {start_day},{end_day}")
        return

    # Chạy mega
    try:
        report = _run_mega(df, play_strategy, src, mode, start_day, end_day)
    except Exception as e:
        print(f"[backtest] ❌ lỗi khi chạy mega cho {week} {src} {mode}: {e}")
        return

    # Ghi kết quả vào DB
    collection.update_one(
        {
            "_id": doc["_id"],
            "srcs": {"$elemMatch": {"src": src, "mode": mode}},
        },
        {
            "$set": {
                "srcs.$[elem].play_report": report,
                "srcs.$[elem].play_report_week": next_week,
                "srcs.$[elem].play_report_updated_at": datetime.utcnow(),
            }
        },
        array_filters=[{"elem.src": src, "elem.mode": mode}],
    )

    sharpe = report.get("sharpe")
    net = report.get("total_net_profit")
    print(f"[backtest] ✅ {week} {src} {mode} → next_week={next_week} sharpe={sharpe} net={net}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chạy MEGA backtest cho next_week dựa trên play_strategy từ busd_correlation"
    )
    parser.add_argument("--week", type=str, default=None, help="Chỉ xử lý 1 tuần, vd: 2026-W02")
    parser.add_argument("--src", type=str, default=None, help="Chỉ xử lý 1 src, vd: hose500")
    parser.add_argument("--mode", type=str, default=None, help="Chỉ xử lý 1 mode, vd: ema")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild kể cả khi đã có play_report"
    )
    args = parser.parse_args()

    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    collection = mongo_client["busd"]["busd_dynamic"]

    # Load dữ liệu một lần
    print("[backtest] Đang load dữ liệu intraday...")
    df_base = _load_df_base()
    print(f"[backtest] Loaded df_base: {len(df_base)} rows, days: {df_base['day'].min()} - {df_base['day'].max()}")

    # Query các doc có play_strategy
    query: dict = {
        "srcs": {
            "$elemMatch": {
                "play_strategy": {"$exists": True, "$not": {"$size": 0}},
            }
        }
    }
    if args.week:
        query["week"] = args.week

    projection = {"week": 1, "next_week": 1, "lst_day": 1, "srcs": 1}
    docs = list(collection.find(query, projection).sort("week", 1))
    print(f"[backtest] Tìm thấy {len(docs)} doc(s) có play_strategy")

    # Build next_week map từ toàn bộ collection (không phụ thuộc vào field next_week trong doc)
    next_week_map = _build_next_week_map(collection)

    total_processed = 0
    for doc in docs:
        week = doc.get("week")
        if not week:
            continue

        for src_item in doc.get("srcs", []):
            src = src_item.get("src")
            mode = src_item.get("mode")

            if not src_item.get("play_strategy"):
                continue
            if args.src and src != args.src:
                continue
            if args.mode and mode != args.mode:
                continue

            process_doc_src(
                collection=collection,
                doc=doc,
                src_item=src_item,
                df_base=df_base,
                next_week_map=next_week_map,
                force=args.force,
            )
            total_processed += 1

    mongo_client.close()
    print(f"[backtest] Done. Đã xử lý {total_processed} (week, src, mode) combination(s).")


if __name__ == "__main__":
    main()
