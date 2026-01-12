import pandas as pd


dic = pd.read_pickle("/home/ubuntu/nevir/gen_spot/dic_freqs_alpha_base.pkl")

# Tạo dict mới để lưu kết quả
dic_new = {}

for key, df in dic.items():
    print(f"Processing key: {key}")

    df = df.copy()  # tránh SettingWithCopyWarning

    df['executionT'] = (
        pd.to_datetime(df['day'], format='%Y_%m_%d', errors='coerce')
        + pd.to_timedelta(df['executionTime'].astype(str), errors='coerce')
    )

    dic_new[key] = df

# Lưu file mới
out_path = "/home/ubuntu/nevir/gen_spot/dic_freqs_alpha_base1.pkl"
pd.to_pickle(dic_new, out_path)

print("DONE. Saved to:", out_path)