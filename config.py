import os

# Định nghĩa các hằng số và đường dẫn
BASE_PATH = "DATA_SV"
HIMA_PATH = os.path.join(BASE_PATH, "Hima")
ERA5_PATH = os.path.join(BASE_PATH, "ERA5")
TEST_PATH = os.path.join(BASE_PATH, "Hima\\B04B\\2019\\04\\01")
PRECIP_PATH = os.path.join(BASE_PATH, "Precipitation/Radar")

CSV_PATH = "csv_data"
CSV_PATH_HIMA = "csv_data/Hima"

HIMA_B04B_PATH = os.path.join(HIMA_PATH,"B04B")
HIMA_B05B_PATH = os.path.join(HIMA_PATH,"B05B")
# OUTPUT_PATH = "/kaggle/working/output/"
# os.makedirs(OUTPUT_PATH, exist_ok=True)

ADM2_PATH = "rainfall_estimates/VNM_ADM2.shp"