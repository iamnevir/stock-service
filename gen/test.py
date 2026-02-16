import pandas as pd

def build_executionT_for_dict(dic):
    """
    dic: dict[str, pd.DataFrame]
    return: dict đã có thêm cột executionT
    """
    for k, df in dic.items():
        df = df.copy()

        # Nếu đã có executionT thì bỏ qua
        if "executionT" not in df.columns:
            # Cách 1: nếu executionTime đã là datetime
            if pd.api.types.is_datetime64_any_dtype(df["executionTime"]):
                df["executionT"] = df["executionTime"]

            # Cách 2: executionTime là string HH:MM:SS, day là YYYY_MM_DD
            else:
                df["executionT"] = pd.to_datetime(
                    df["day"].str.replace("_", "-") + " " + df["executionTime"]
                )

        dic[k] = df

    return dic

import pandas as pd

path_in = "/home/ubuntu/nevir/gen/dic_freqs_dollar_bar.pickle"
path_out = "/home/ubuntu/nevir/gen/dic_freqs_dollar_bar_with_executionT.pickle"

dic = pd.read_pickle(path_in)

dic = build_executionT_for_dict(dic)

pd.to_pickle(dic, path_out)

key = list(dic.keys())[1]
print(dic[key][["executionT", "executionTime", "day"]].head())