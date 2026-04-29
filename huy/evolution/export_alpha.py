import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def export_successful_alphas(history_dir, output_file, ids=None, include_failed=False):
    if not os.path.exists(history_dir):
        print(f"Directory {history_dir} does not exist.")
        return

    if ids is None:
        # Default behavior: all files in directory
        json_files = [f for f in os.listdir(history_dir) if f.endswith('.json')]
    else:
        # Filtered by IDs
        json_files = []
        for alpha_id in ids:
            filename = f"history_{alpha_id}.json"
            if os.path.exists(os.path.join(history_dir, filename)):
                json_files.append(filename)
            else:
                # Some IDs might not have history files
                pass
    
    if not json_files:
        print("No files to process.")
        return

    all_data = []
    
    # Statistics variables
    stats = {
        'Total Files': len(json_files),
        'Total Cost (USD)': 0,
        'Input Tokens (Prompt)': 0,
        'Output Tokens (Completion)': 0,
        'Total Tokens': 0,
        'Total Time (s)': 0,
        'Success Count': 0,
        'Failed Count': 0,
        'Total Alphas': 0
    }
    
    print(f"Processing {len(json_files)} files...")
    for filename in tqdm(json_files):
        file_path = os.path.join(history_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Global stats from file
            llm_usage = data.get('llm_usage', {})
            tokens = llm_usage.get('tokens', {})
            
            input_tokens = tokens.get('prompt_cache_hit_tokens', 0) + tokens.get('prompt_cache_miss_tokens', 0)
            output_tokens = tokens.get('completion_tokens', 0)
            
            stats['Total Cost (USD)'] += llm_usage.get('total_cost_usd', 0)
            stats['Input Tokens (Prompt)'] += input_tokens
            stats['Output Tokens (Completion)'] += output_tokens
            stats['Total Tokens'] += tokens.get('total_tokens', 0)
            stats['Total Time (s)'] += data.get('total_time_seconds', 0)
            
            final_tier = data.get('final_tier', {})
            for alpha_name, alpha_info in final_tier.items():
                stats['Total Alphas'] += 1
                stop_reason = alpha_info.get('stop_reason')
                
                is_success = stop_reason == 'SUCCESS'
                if is_success:
                    stats['Success Count'] += 1
                else:
                    stats['Failed Count'] += 1

                if is_success or include_failed:
                    # Prepare performance string
                    perf_lines = []
                    
                    # 'all' row
                    r1_all = alpha_info.get('r1_all', 0)
                    r2_all = alpha_info.get('r2_all', 0)
                    r3_all = alpha_info.get('r3_all', 0) 
                    perf_lines.append(f"all: {r1_all}|{r2_all}|{r3_all}")
                    
                    # Yearly rows
                    s0_by_year = alpha_info.get('s0_by_year', {})
                    years = sorted([y for y in s0_by_year.keys() if y != 'all'])
                    for yr in years:
                        val = s0_by_year[yr]
                        if isinstance(val, list):
                            v1 = val[0] if len(val) > 0 else 0
                            v2 = val[1] if len(val) > 1 else 0
                            v3 = val[2] if len(val) > 2 else 0
                            perf_lines.append(f"{yr}: {v1}|{v2}|{v3}")
                        else:
                            perf_lines.append(f"{yr}: {val}|0.0|0.0")
                    
                    performance_str = "\n".join(perf_lines)

                    row = {
                        'Alpha Name': alpha_name,
                        'Iteration': data.get('iteration'),
                        'r1_all': r1_all,
                        'r2_all': r2_all,
                        'Performance': performance_str,
                        'Stop Reason': stop_reason if stop_reason else "N/A",
                        'Total Time (s)': data.get('total_time_seconds'),
                        'File': filename
                    }
                    
                    # Add params
                    params = alpha_info.get('params', {})
                    row['Parameters'] = str(params)
                    
                    # Add code
                    row['Code'] = alpha_info.get('code')
                    
                    all_data.append(row)
                else:
                    pass
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Prepare Stats DataFrame
    stats['Success Rate (%)'] = round((stats['Success Count'] / stats['Total Alphas'] * 100), 2) if stats['Total Alphas'] > 0 else 0
    stats['Average Time per Run (s)'] = round(stats['Total Time (s)'] / stats['Total Files'], 2) if stats['Total Files'] > 0 else 0
    
    # Convert total time to formatted string
    total_sec = stats['Total Time (s)']
    stats['Total Time (Formatted)'] = f"{int(total_sec // 3600)}h {int((total_sec % 3600) // 60)}m {int(total_sec % 60)}s"

    df_stats = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])

    if not all_data:
        print("No matching alphas found.")
        # Still save stats if any
        if stats['Total Alphas'] > 0:
            with pd.ExcelWriter(output_file) as writer:
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            return
        return

    df_main = pd.DataFrame(all_data)
    
    # Reorder columns
    cols = ['Alpha Name', 'r1_all', 'r2_all', 'Iteration', 'Performance', 'Parameters']
    other_cols = [c for c in df_main.columns if c not in cols and c != 'Code']
    final_cols = cols + other_cols + ['Code']
    df_main = df_main[final_cols]

    # Save to Excel with two sheets
    with pd.ExcelWriter(output_file) as writer:
        df_main.to_excel(writer, sheet_name='Alphas', index=False)
        df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        
    print(f"Exported {len(df_main)} alphas to {output_file} with Statistics tab.")

def get_ids_from_processed_alphas(processed_alphas_path):
    if not os.path.exists(processed_alphas_path):
        print(f"File {processed_alphas_path} not found.")
        return {}
    with open(processed_alphas_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    HISTORY_DIR = "/home/ubuntu/nevir/huy/evolution/history"
    PROCESSED_ALPHAS_PATH = "/home/ubuntu/nevir/huy/evolution/processed_alphas.json"
    OUTPUT_FILE = "/home/ubuntu/nevir/huy/evolution/alphas_success.xlsx"
    
    parser = argparse.ArgumentParser(description='Export successful alphas to Excel.')
    parser.add_argument('--mode', type=str, choices=['all', 'keys', 'interactive'], default=None, help='Export mode')
    parser.add_argument('--keys', type=str, help='Comma separated keys to export (if mode is keys)')
    parser.add_argument('--include-failed', action='store_true', help='Include alphas that did not reach SUCCESS')
    args = parser.parse_args()

    data = get_ids_from_processed_alphas(PROCESSED_ALPHAS_PATH)
    all_keys = list(data.keys())
    
    selected_ids = []
    include_failed = args.include_failed
    
    # Determine mode
    mode = args.mode
    if mode is None:
        if args.keys:
            mode = 'keys'
        else:
            mode = 'interactive'
            
    if mode == 'interactive':
        print("\nAvailable export keys/modes:")
        print("0: [ALL] Export everything in processed_alphas.json")
        for i, key in enumerate(all_keys):
            print(f"{i+1}: {key} ({len(data[key])} IDs)")
        
        choice = input("\nEnter choice (e.g. 0 or 1,2 or all): ").strip().lower()
        
        if choice == '0' or choice == 'all':
            for key in all_keys:
                selected_ids.extend(data[key])
        else:
            try:
                choices = [int(c.strip()) for c in choice.split(',')]
                for c in choices:
                    if 1 <= c <= len(all_keys):
                        key = all_keys[c-1]
                        selected_ids.extend(data[key])
                        print(f"Selected key: {key}")
                    else:
                        print(f"Invalid choice: {c}")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas or 'all'.")
                exit(1)
        
        # Ask for include_failed in interactive mode if not already set via flag
        if not include_failed:
            inc_f = input("Include FAILED alphas? (y/n, default n): ").strip().lower()
            if inc_f == 'y':
                include_failed = True
    elif mode == 'all':
        for key in all_keys:
            selected_ids.extend(data[key])
    elif mode == 'keys':
        if not args.keys:
            print("Error: --keys is required when mode is 'keys'")
            exit(1)
        target_keys = [k.strip() for k in args.keys.split(',')]
        for tk in target_keys:
            if tk in data:
                selected_ids.extend(data[tk])
                print(f"Selected key: {tk}")
            else:
                print(f"Warning: Key '{tk}' not found in processed_alphas.json")
    
    if not selected_ids:
        print("No IDs selected.")
        exit(1)
    
    # Remove duplicates
    selected_ids = list(dict.fromkeys(selected_ids))
    print(f"Total unique IDs to process: {len(selected_ids)}")
    
    export_successful_alphas(HISTORY_DIR, OUTPUT_FILE, ids=selected_ids, include_failed=include_failed)
