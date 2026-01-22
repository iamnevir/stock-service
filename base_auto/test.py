log_file = "/home/ubuntu/nevir/base_auto/logs/005 H500_FH500_VN30_FVN30_2022_01_01_2023_01_01_wfa_correlation.log"

import os


print("Path:", repr(log_file))
print("Exists:", os.path.exists(log_file))
print("Is file:", os.path.isfile(log_file))
print("Dir exists:", os.path.exists(os.path.dirname(log_file)))
import glob
print(glob.glob("/home/ubuntu/nevir/base_auto/logs/*005*"))
