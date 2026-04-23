import pandas as pd

df = pd.read_pickle("/home/ubuntu/duy/new_strategy/gen1_2/df_fee_1.pkl")

df_filter = df[df['sharpe'] > 2].copy()
# df_filter = df[df['upper'] == 1.0].copy()
print(df_filter)
print(len(df_filter))
print(len(df_filter) / len(df)  * 100)


# print(df_filter[df_filter['sharpe'] > 0])


exit()

    
fn = "/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_2021.pkl"
with open(fn, 'rb') as file:
    df_2021_all = pickle.load(file)
fn = "/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_2022_1.pkl"
with open(fn, 'rb') as file:
    df_2022_1_all = pickle.load(file)
fn = "/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_2022_2.pkl"
with open(fn, 'rb') as file:
    df_2022_2_all = pickle.load(file)
fn = "/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_2023.pkl"
with open(fn, 'rb') as file:
    df_2023_all = pickle.load(file)
    
fn = "/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC.pkl"
with open(fn, 'rb') as file:
    df_2024_all = pickle.load(file)
    
all_dict = {}
for freq in df_2021_all:
    df_2021 = df_2021_all[freq]
    df_2022_1 = df_2022_1_all[freq]
    df_2022_2 = df_2022_2_all[freq]
    df_2023 = df_2023_all[freq]
    df_2024 = df_2024_all[freq]
    
    df_all = pd.concat([df_2021, df_2022_1,df_2022_2,df_2023,df_2024])
    all_dict[freq] = df_all
    
with open("/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_ALL.pkl", 'wb') as file:
    pickle.dump(all_dict, file)

exit()
    
    

    

# print(DIC_FREQS_PS[30][(DIC_FREQS_PS[30]['day'] >= '2022_06_01') & (DIC_FREQS_PS[30]['day'] <= '2022_12_30')])
# print(DIC_FREQS_VN30[30])

# exit()

from datetime import datetime as dt
for freq in DIC_FREQS_VN30:
    df_temp = DIC_FREQS_VN30[freq]
    df_temp['executionT'] = (df_temp['day'] + ' ' + df_temp['executionTime']).map(lambda x: dt.strptime(x, '%Y_%m_%d %H:%M:%S'))
    
    df_ps = DIC_FREQS_PS[freq]
    df_ps = df_ps[df_ps['day']>= '2022_06_01'].copy()
    
    # print(df_temp[df_temp['day'] == '2021_06_11'])
    # print(df_ps[df_ps['day'] == '2021_06_11'])
    
    # df_temp = df_temp[df_temp['day'] == '2021_06_11'].copy()
    # df_ps = df_ps[df_ps['day'] == '2021_06_11'].copy()
    
    
    df_temp['entryPrice'] = df_ps['entryPrice']
    df_temp['exitPrice'] = df_ps['exitPrice']
    df_temp['priceChange'] = df_ps['priceChange']
    
    df_temp = df_temp.dropna(subset=["entryPrice", "exitPrice"], how="all")
    # print(df_temp)
    # exit()

    DIC_FREQS_VN30[freq] = df_temp
# print(DIC_FREQS_VN30[30])
# exit()


with open("/home/ubuntu/duy/new_strategy/gen1_2/df_vn30_OHLC_ps_2022_2.pkl", 'wb') as file:
    pickle.dump(DIC_FREQS_VN30, file)

# import pandas as pd

# df = pd.read_pickle("/home/ubuntu/duy/new_strategy/gen1_2/df_fee_1.pkl")
# df = df[df['sharpe'] >= 1]
# # df = df[df['tvr']]
# print(df)