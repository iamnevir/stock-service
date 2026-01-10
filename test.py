import pickle
import pandas as pd


src_path = "/home/ubuntu/nevir/data/busd.pkl"
dst_path = "/home/ubuntu/nevir/data/busd_last.pkl"


# giả sử busd.pkl là DataFrame
with open(src_path, "rb") as f:
    data = pickle.load(f)

print(data)
# if not isinstance(data, pd.DataFrame):
#     raise ValueError("busd.pkl không phải pandas DataFrame")


# # chỉ giữ index và cột 'last'
# new_df = data[["last"]].copy()


# with open(dst_path, "wb") as f:
#     pickle.dump(new_df, f)


# print(f"Đã lưu file mới tại: {dst_path}")