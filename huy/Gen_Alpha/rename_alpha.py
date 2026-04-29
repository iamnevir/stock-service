import json
import os

def rename_keys(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = {}
    for i, (old_key, value) in enumerate(data.items(), 1):
        new_key = f"quanta_randomizer_{i:03d}"
        new_data[new_key] = value
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Renamed {len(new_data)} keys to quanta in {file_path}")

if __name__ == "__main__":
    target_file = "/home/ubuntu/nevir/huy/Gen_Alpha/quanta_randomizer.json"
    rename_keys(target_file)
