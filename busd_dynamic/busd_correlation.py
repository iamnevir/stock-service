import argparse
import multiprocessing as mp
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient

from busd_auto.is_correlation import build_action_matrix, calculate_all_correlations_block_mp
from busd_auto.utils import get_mongo_uri
from busd_dynamic import compute_single_position, precompute_ma

_MP_DF_MA = None
_MP_DF_RAW = None


def _calculate_working_days(start: int, end: int, workdays_per_year: int = 250) -> int:
    start_date = datetime.strptime(str(start), "%Y%m%d")
    end_date = datetime.strptime(str(end), "%Y%m%d")
    total_days = (end_date - start_date).days
    working_days = total_days * workdays_per_year / 365.0
    return max(1, round(working_days))


def _init_trade_worker(df_ma: pd.DataFrame, df_raw: pd.DataFrame):
    global _MP_DF_MA, _MP_DF_RAW
    _MP_DF_MA = df_ma
    _MP_DF_RAW = df_raw


def _compute_trade_for_config(config: str) -> Tuple[str, Optional[pd.DataFrame], Optional[float], Optional[float]]:
    try:
        ma1, ma2, th, es = [int(x) for x in config.split("_")[:4]]
    except Exception:
        return config, None, None, None

    position = compute_single_position(_MP_DF_MA, ma1, ma2, th, es, delay=1)
    action = position.diff().fillna(position)
    mask = action != 0

    if not mask.any():
        return config, None, None, None

    if "datetime" in _MP_DF_RAW.columns:
        dt = pd.to_datetime(_MP_DF_RAW.loc[mask, "datetime"], errors="coerce")
    else:
        dt = pd.to_datetime(_MP_DF_RAW.index[mask], errors="coerce")

    df_trade = pd.DataFrame({
        "datetime": dt.values,
        "action": action.loc[mask].values,
    })
    df_trade = df_trade.dropna(subset=["datetime"])

    if df_trade.empty:
        return config, None, None, None

    # Build daily pnl/turnover for sharpe/tvr.
    df_local = _MP_DF_RAW.copy()
    df_local["position"] = position
    df_local["grossProfit"] = df_local["position"] * df_local["priceChange"]
    df_local["turnover"] = action.abs()
    df_local["fee"] = df_local["turnover"] * 175 / 1000
    df_local["netProfit"] = df_local["grossProfit"] - df_local["fee"]

    df_1d = (
        df_local.groupby("day", sort=False, observed=True)
        .agg(
            turnover=("turnover", "sum"),
            netProfit=("netProfit", "sum"),
        )
        .round(2)
    )

    if df_1d.empty:
        return config, df_trade[["datetime", "action"]], None, None

    tvr = float(round(df_1d["turnover"].mean(), 3))
    std = df_1d["netProfit"].std()
    if std is None or pd.isna(std) or std == 0:
        sharpe = None
    else:
        working_day = _calculate_working_days(int(df_1d.index[0]), int(df_1d.index[-1]))
        sharpe = float(round(df_1d["netProfit"].mean() / std * np.sqrt(working_day), 3))

    return config, df_trade[["datetime", "action"]], sharpe, tvr


def _prepare_df_for_src(df_base: pd.DataFrame, start_day: int, end_day: int, src: str) -> pd.DataFrame:
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


def _valid_configs(configs: List[str]) -> List[str]:
    good = []
    for cfg in configs:
        if not isinstance(cfg, str):
            continue
        parts = cfg.split("_")
        if len(parts) < 4:
            continue
        try:
            int(parts[0])
            int(parts[1])
            int(parts[2])
            int(parts[3])
        except Exception:
            continue
        good.append(f"{int(parts[0])}_{int(parts[1])}_{int(parts[2])}_{int(parts[3])}")
    return list(dict.fromkeys(good))


def _compute_trades(
    configs: List[str],
    df_ma: pd.DataFrame,
    df_src: pd.DataFrame,
    workers: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Optional[float]]]]:
    id_to_trade: Dict[str, pd.DataFrame] = {}
    metric_map: Dict[str, Dict[str, Optional[float]]] = {}

    with mp.Pool(
        processes=max(1, workers),
        initializer=_init_trade_worker,
        initargs=(df_ma, df_src),
    ) as pool:
        for cfg, trade_df, sharpe, tvr in pool.imap_unordered(_compute_trade_for_config, configs, chunksize=20):
            if trade_df is not None and not trade_df.empty:
                id_to_trade[cfg] = trade_df
                metric_map[cfg] = {
                    "sharpe": sharpe,
                    "tvr": tvr,
                }

    return id_to_trade, metric_map


def _greedy_low_corr(
    corr_matrix: np.ndarray,
    target_size: int,
    restarts: int,
    seed_size: int,
) -> List[int]:
    n = corr_matrix.shape[0]
    if n == 0:
        return []

    target_size = min(target_size, n)
    seed_size = min(seed_size, target_size)

    best_set = None
    best_score = float("inf")

    for _ in range(max(1, restarts)):
        selected: Set[int] = set(random.sample(range(n), seed_size))

        while len(selected) < target_size:
            remaining = [i for i in range(n) if i not in selected]
            sel_list = list(selected)

            def candidate_score(cand: int) -> float:
                if not sel_list:
                    return 0.0
                return float(corr_matrix[cand, sel_list].sum())

            best_cand = min(remaining, key=candidate_score)
            selected.add(best_cand)

        idx = sorted(selected)
        if len(idx) <= 1:
            score = 0.0
        else:
            tri = corr_matrix[np.ix_(idx, idx)]
            i_u, j_u = np.triu_indices(len(idx), k=1)
            score = float(tri[i_u, j_u].sum())

        if score < best_score:
            best_score = score
            best_set = idx

    return best_set or []


def _apply_threshold_and_refill(
    corr_matrix: np.ndarray,
    selected: List[int],
    threshold: float,
    target_size: int,
) -> List[int]:
    if not selected:
        return []

    selected_arr = np.array(selected, dtype=int)
    i_mat, j_mat = np.triu_indices(len(selected_arr), k=1)
    to_remove = set()

    for i, j in zip(i_mat, j_mat):
        if corr_matrix[selected_arr[i], selected_arr[j]] > threshold:
            to_remove.add(int(selected_arr[i]))
            to_remove.add(int(selected_arr[j]))

    filtered = [i for i in selected if i not in to_remove]

    if len(filtered) >= target_size:
        return filtered[:target_size]

    n = corr_matrix.shape[0]
    remaining = [i for i in range(n) if i not in filtered]

    while len(filtered) < min(target_size, n) and remaining:
        if not filtered:
            next_i = remaining.pop(0)
            filtered.append(next_i)
            continue

        best_cand = min(remaining, key=lambda c: float(corr_matrix[c, filtered].mean()))
        filtered.append(best_cand)
        remaining.remove(best_cand)

    return filtered


def select_play_strategies(
    id_to_trade_df: Dict[str, pd.DataFrame],
    top_n: int = 50,
    restarts: int = 30,
    seed_size: int = 5,
    threshold: float = 95,
    corr_workers: int = 20,
) -> Tuple[List[str], float]:
    if len(id_to_trade_df) < 2:
        return list(id_to_trade_df.keys())[:top_n], 0.0

    action_df = build_action_matrix(id_to_trade_df)
    ids, corr_dict = calculate_all_correlations_block_mp(
        action_df,
        block_size=500,
        n_core=max(1, corr_workers),
    )

    n = len(ids)
    corr_matrix = np.zeros((n, n), dtype=np.float32)
    for (i, j), c in corr_dict.items():
        corr_matrix[i, j] = c
        corr_matrix[j, i] = c

    selected_indices = _greedy_low_corr(corr_matrix, top_n, restarts, seed_size)
    final_indices = _apply_threshold_and_refill(corr_matrix, selected_indices, threshold, top_n)

    selected = [str(ids[i]) for i in final_indices]

    if len(final_indices) <= 1:
        mean_corr = 0.0
    else:
        tri = corr_matrix[np.ix_(final_indices, final_indices)]
        iu, ju = np.triu_indices(len(final_indices), k=1)
        vals = tri[iu, ju]
        mean_corr = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return selected, mean_corr


def run_for_week_src(
    collection,
    doc_id,
    week: str,
    start_day: int,
    end_day: int,
    src: str,
    mode: str,
    lst_strategy: List[str],
    df_base: pd.DataFrame,
    backtest_workers: int,
    corr_workers: int,
    top_n: int,
    restarts: int,
    seed_size: int,
    threshold: float,
) -> None:
    print(f"[corr] week={week} src={src} mode={mode} strategies={len(lst_strategy)}")

    configs = _valid_configs(lst_strategy)
    if len(configs) < 2:
        print(f"[corr] skip {week} {src} {mode}: not enough valid strategies")
        return

    df_src = _prepare_df_for_src(df_base, start_day, end_day, src)
    if df_src.empty:
        print(f"[corr] skip {week} {src} {mode}: empty source data")
        return

    ma_lengths = set()
    for cfg in configs:
        p = cfg.split("_")
        ma_lengths.add(int(p[0]))
        ma_lengths.add(int(p[1]))

    df_ma = precompute_ma(df_src, ma_lengths, based_col="based_col", mode=mode)
    id_to_trade_df, metric_map = _compute_trades(configs, df_ma, df_src, workers=backtest_workers)

    if len(id_to_trade_df) < 2:
        print(f"[corr] skip {week} {src} {mode}: not enough trades for correlation")
        return

    play_strategy, play_correlation = select_play_strategies(
        id_to_trade_df,
        top_n=top_n,
        restarts=restarts,
        seed_size=seed_size,
        threshold=threshold,
        corr_workers=corr_workers,
    )

    selected_sharpes = [metric_map[cfg].get("sharpe") for cfg in play_strategy if cfg in metric_map]
    selected_tvrs = [metric_map[cfg].get("tvr") for cfg in play_strategy if cfg in metric_map]

    selected_sharpes = [x for x in selected_sharpes if x is not None and not pd.isna(x)]
    selected_tvrs = [x for x in selected_tvrs if x is not None and not pd.isna(x)]

    play_mean_sharpe = float(np.mean(selected_sharpes)) if selected_sharpes else 0.0
    play_mean_tvr = float(np.mean(selected_tvrs)) if selected_tvrs else 0.0

    collection.update_one(
        {
            "_id": doc_id,
            "srcs": {"$elemMatch": {"src": src, "mode": mode}},
        },
        {
            "$set": {
                "srcs.$[elem].play_strategy": play_strategy,
                "srcs.$[elem].play_strategy_updated_at": datetime.utcnow(),
                "srcs.$[elem].play_strategy_size": len(play_strategy),
                "srcs.$[elem].play_correlation": round(play_correlation, 4),
                "srcs.$[elem].play_mean_sharpe": round(play_mean_sharpe, 4),
                "srcs.$[elem].play_mean_tvr": round(play_mean_tvr, 4),
            }
        },
        array_filters=[{"elem.src": src, "elem.mode": mode}],
    )

    print(f"[corr] done {week} {src} {mode}: play_strategy={len(play_strategy)}")


def main():
    parser = argparse.ArgumentParser(description="Build low-correlation play strategies for busd_dynamic")
    parser.add_argument("--week", type=str, default=None, help="Process only one week, e.g. 2026-W02")
    parser.add_argument("--src", type=str, default=None, help="Process only one src, e.g. hose500")
    parser.add_argument("--mode", type=str, default=None, help="Process only one mode, e.g. ema")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=95)
    parser.add_argument("--backtest-workers", type=int, default=20)
    parser.add_argument("--corr-workers", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Rebuild even if play_strategy already exists")
    args = parser.parse_args()

    mongo_client = MongoClient(get_mongo_uri("mgc3"))
    collection = mongo_client["busd"]["busd_dynamic"]

    df_base = pd.read_pickle("/home/ubuntu/nevir/data/busd.pkl")
    df_base = df_base[(df_base["day"] >= 20220101) & (df_base["day"] < 20270101)].copy()

    query = {"srcs": {"$elemMatch": {"lst_strategy": {"$exists": True, "$ne": []}}}}
    if args.week:
        query["week"] = args.week

    projection = {"week": 1, "lst_day": 1, "srcs": 1}
    docs = list(collection.find(query, projection).sort("week", 1))

    print(f"[main] docs found: {len(docs)}")

    for doc in docs:
        week = doc.get("week")
        lst_day = sorted(int(d) for d in doc.get("lst_day", []))
        if not week or not lst_day:
            continue

        start_day, end_day = lst_day[0], lst_day[-1]

        for src_item in doc.get("srcs", []):
            src = src_item.get("src")
            mode = src_item.get("mode")
            lst_strategy = src_item.get("lst_strategy", [])

            if not src or not mode or not lst_strategy:
                continue
            if args.src and src != args.src:
                continue
            if args.mode and mode != args.mode:
                continue
            if src_item.get("play_strategy") and not args.force:
                print(f"[main] skip {week} {src} {mode}: already has play_strategy")
                continue

            run_for_week_src(
                collection=collection,
                doc_id=doc["_id"],
                week=week,
                start_day=start_day,
                end_day=end_day,
                src=src,
                mode=mode,
                lst_strategy=lst_strategy,
                df_base=df_base,
                backtest_workers=args.backtest_workers,
                corr_workers=args.corr_workers,
                top_n=args.top_n,
                restarts=args.restarts,
                seed_size=args.seed_size,
                threshold=args.threshold,
            )

    mongo_client.close()
    print("[main] done")


if __name__ == "__main__":
    main()
