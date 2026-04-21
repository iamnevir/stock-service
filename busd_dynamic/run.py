import pandas as pd
from Simulator import MegaBbAccV2
from datetime import datetime
import multiprocessing
from multiprocessing import cpu_count
from typing import Optional
from busd_auto.utils import get_mongo_uri
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

_MP_DF_DATA = None
_MP_WEEK_KEY = None
_MP_WEEK_DAYS = None

def group_dates_by_week(date_list, collection: Optional[object] = None):
    groups = {}
    
    for date_str in date_list:
        s_date = str(date_str).strip()
        date_obj = datetime.strptime(s_date, '%Y%m%d')
        year, week_num, _ = date_obj.isocalendar()
        week_key = f"{year}-W{week_num:02d}"
        
        if week_key not in groups:
            groups[week_key] = {
                'lst_day': [],
                'lst_strategy' : [],
                'df_result_1D' : pd.DataFrame()
            }
        groups[week_key]['lst_day'].append(date_str)
    

    sorted_keys = sorted(groups.keys())

    for i in range(len(sorted_keys)):
        current_key = sorted_keys[i]
        if i < len(sorted_keys) - 1:
            groups[current_key]['next_week'] = sorted_keys[i+1]
        else:
            groups[current_key]['next_week'] = None

    if collection is not None and sorted_keys:
        ops = []
        for week in sorted_keys:
            lst_day = [int(day) for day in groups[week]['lst_day']]
            ops.append(
                UpdateOne(
                    {'week': week},
                    {
                        '$setOnInsert': {'week': week},
                        '$addToSet': {'lst_day': {'$each': lst_day}},
                    },
                    upsert=True,
                )
            )

        if ops:
            collection.bulk_write(ops, ordered=False)
   
    return None


def get_weeks_for_running(collection):
    query = {
        'lst_day': {'$exists': True, '$ne': []},
        '$or': [
            {'lst_strategy': {'$exists': False}},
            {'lst_strategy': {'$eq': []}},
        ],
    }

    weeks = list(collection.find(query, {'_id': 0, 'week': 1, 'lst_day': 1}).sort('week', 1))
    cofigs_week = {}
    for item in weeks:
        week = item.get('week')
        lst_day = item.get('lst_day', [])
        if not week or not lst_day:
            continue
        cofigs_week[week] = {
            'lst_day': [int(day) for day in lst_day],
            'lst_strategy': [],
            'df_result_1D': pd.DataFrame(),
        }

    return cofigs_week


def _init_running_worker(df_data, week_key, week_days):
    global _MP_DF_DATA, _MP_WEEK_KEY, _MP_WEEK_DAYS
    _MP_DF_DATA = df_data
    _MP_WEEK_KEY = week_key
    _MP_WEEK_DAYS = set(int(day) for day in week_days)


def _run_single_config(config):
    self = MegaBbAccV2(
        alpha_name="mega_bb_acc",
        configs=[config],
        fee=175,
        df_alpha=_MP_DF_DATA.copy(),
        # data_start=20240101,
        # data_end=20250101,
        busd_source="hose500",
        # busd_source="vn30",

        #NN
        foreign_multiplier=0,
        foreign_add_column="aggFBusdVn30",  # "aggFBusdVn30" | "aggFBusd"

        #FNN
        filter_col="aggFBusdVn30",
        foreign_policy=0,
        replace_filter_value=0,
        moving_average_type="sma",

        init_budget = 1,
        booksize_sizing = False
    )
    self.compute_based_col()
    self.compute_mas()
    self.compute_all_position()
    self.compute_mega_position()
    self.compute_profit_and_df_1d()

    df_week = self.df_1d[self.df_1d.index.isin(_MP_WEEK_DAYS)].copy()
    if df_week.empty:
        return None

    df_week['cumNetProfit'] = df_week['netProfit'].cumsum()
    report = self.compute_report(df_week, config)

    if isinstance(report, pd.Series):
        report = report.to_dict()
    elif hasattr(report, 'to_dict') and not isinstance(report, dict):
        report = report.to_dict()

    if not isinstance(report, dict):
        return None

    report['week'] = _MP_WEEK_KEY
    report['config'] = config
    return report


def worker_task_batch(config_batch):
    batch_reports = []
    for config in config_batch:
        report = _run_single_config(config)
        if report is not None:
            batch_reports.append(report)
    return batch_reports


def running(df_data, week_key, week_info, lst_params, num_workers=None, chunk_size=20):
    dic_result = {week_key: []}
    week_days = week_info.get('lst_day', []) if week_info else []

    if not week_days or not lst_params:
        print(f"[running] skip week {week_key}: no days or no configs")
        return dic_result

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"[running] week {week_key}: {len(lst_params)} configs, workers={num_workers}")

    batches = [lst_params[i:i + chunk_size] for i in range(0, len(lst_params), chunk_size)]
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_running_worker,
        initargs=(df_data, week_key, week_days),
    ) as pool:
        for batch_completed in tqdm(
            pool.imap_unordered(worker_task_batch, batches),
            total=len(batches),
            desc=f"[running] {week_key}",
            unit="batch",
        ):
            if batch_completed:
                dic_result[week_key].extend(batch_completed)

    print(f"[running] week {week_key} done: {len(dic_result[week_key])} reports")
    return dic_result


def filter_results_by_week(dic_result, top_n=4000, max_tvr_mean=30):
    filtered_results = {}
    total_weeks = len(dic_result)
    print(f"[filter] start: {total_weeks} weeks")

    for week_idx, (week, reports) in enumerate(dic_result.items(), start=1):
        if week_idx == 1 or week_idx % 10 == 0 or week_idx == total_weeks:
            pct_week = (week_idx / total_weeks * 100) if total_weeks else 100
            print(f"[filter] week {week_idx}/{total_weeks} ({pct_week:.2f}%) - {week}")

        if not reports:
            continue

        normalized_reports = []
        for report in reports:
            if isinstance(report, pd.Series):
                normalized_reports.append(report.to_dict())
            elif isinstance(report, dict):
                normalized_reports.append(report)
            elif hasattr(report, 'to_dict'):
                normalized_reports.append(report.to_dict())

        if not normalized_reports:
            continue

        df_week = pd.DataFrame(normalized_reports)
        if (
            df_week.empty
            or 'cumNetProfit' not in df_week.columns
            or 'tvr_mean' not in df_week.columns
            or 'config' not in df_week.columns
        ):
            continue

        df_week['cumNetProfit'] = pd.to_numeric(df_week['cumNetProfit'], errors='coerce')
        df_week['tvr_mean'] = pd.to_numeric(df_week['tvr_mean'], errors='coerce')

        df_week = df_week.dropna(subset=['cumNetProfit', 'tvr_mean'])
        df_week = df_week[df_week['tvr_mean'] < max_tvr_mean]
        df_week = df_week.sort_values(by='cumNetProfit', ascending=False).head(top_n)

        if not df_week.empty:
            configs = df_week['config'].dropna().astype(str).tolist()
            # Keep order while removing duplicates in the same week.
            filtered_results[week] = list(dict.fromkeys(configs))

    return filtered_results


def update_week_strategies(collection, filtered_results):
    if not filtered_results:
        print("[update] no filtered results to write")
        return

    print(f"[update] preparing writes for {len(filtered_results)} weeks")
    ops = []
    for week, configs in filtered_results.items():
        ops.append(
            UpdateOne(
                {'week': week},
                {'$set': {'lst_strategy': configs}},
            )
        )

    if ops:
        collection.bulk_write(ops, ordered=False)
        print(f"[update] done: wrote lst_strategy for {len(ops)} weeks")

if __name__ == "__main__":
    mongo_client = MongoClient(get_mongo_uri())
    busd_collection = mongo_client["busd"]["busd_dynamic"]
    df = pd.read_pickle("/home/ubuntu/nevir/data/df_2025.pkl")
    
    # weeks = list(busd_collection.find({}))
    # print(weeks)
    df = df[df['day'] >= 20260101] 
    df = df[df['day'] < 20270101]
    lst_day = df['day'].unique()
    group_dates_by_week(lst_day, busd_collection)
    configs_week = get_weeks_for_running(busd_collection)
    print(f"[main] weeks pending run: {len(configs_week)}")
    print(configs_week)

    # if not cofigs_week:
    #     print("No week with lst_day and empty lst_strategy to run.")
    #     raise SystemExit(0)
    
    # # lst_params = ["47_7_19_5", "8_5_5_8", "41_6_18_2", "25_5_14_2", "35_5_17_2", "15_5_19_8", "12_5_8_2", "33_18_14_8", "21_9_18_8", "20_5_12_2", "15_10_8_8", "22_5_13_2", "25_14_14_8", "31_19_11_8", "53_7_19_8", "51_7_19_8", "20_11_13_8", "9_5_9_8", "10_5_6_2", "9_5_6_8", "17_8_7_2", "11_5_7_2", "10_6_5_8", "21_15_8_8", "24_14_13_8", "22_14_10_8", "15_7_7_2", "12_5_15_8", "26_15_9_5", "8_5_7_8", "48_7_18_5", "49_6_19_5", "16_12_6_8", "29_5_15_2", "26_19_6_8", "26_18_6_8", "36_18_17_8", "33_18_15_8", "50_6_19_5", "25_16_8_8", "32_16_15_5", "29_21_8_8", "25_13_10_5", "26_15_10_8", "35_17_18_8", "28_7_17_2", "8_5_5_2", "17_7_17_8", "14_5_9_2", "13_6_7_2"]
    # lst_params = []
    # th_list = list(range(5, 6))
    # es_list = list(range(2, 5, 3))
    # for ma1 in range(6, 151):
    #     for ma2 in range(5, ma1):
    #         for th in th_list:
    #             for es in es_list:
    #                 lst_params.append(f"{ma1}_{ma2}_{th}_{es}")
    # print(len(lst_params))
    # total_weeks = len(cofigs_week)
    # done_summary = {}
    # for week_idx, (week_key, week_info) in enumerate(cofigs_week.items(), start=1):
    #     print(f"[main] week {week_idx}/{total_weeks}: {week_key}")

    #     results = running(
    #         df_data=df,
    #         week_key=week_key,
    #         week_info=week_info,
    #         lst_params=lst_params,
    #         num_workers=40
    #     )
    #     filtered_results = filter_results_by_week(results, top_n=4000, max_tvr_mean=30)
    #     update_week_strategies(busd_collection, filtered_results)

    #     done_summary[week_key] = len(filtered_results.get(week_key, []))
    #     print(f"[main] week {week_key} strategies: {done_summary[week_key]}")

    # print(done_summary)
    

    

