import os

# Định nghĩa các hằng số và đường dẫn
BASE_PATH = "Data/DATA_SV/"
HIMA_PATH = os.path.join(BASE_PATH, "Hima")
ERA5_PATH = os.path.join(BASE_PATH, "ERA5")
PRECIP_PATH = os.path.join(BASE_PATH, "Precipitation/Radar")

HIMA_B04B_PATH = os.path.join(HIMA_PATH,"B04B")
HIMA_B05B_PATH = os.path.join(HIMA_PATH,"B05B")
# OUTPUT_PATH = "/kaggle/working/output/"
# os.makedirs(OUTPUT_PATH, exist_ok=True)