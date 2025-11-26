import os
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ========================== ĐƯỜNG DẪN CHÍNH XÁC ==========================
BASE_PATH = r"Data\DATA_SV"  # Dùng raw string hoặc \\ để tránh lỗi \
HIMA_ROOT = os.path.join(BASE_PATH, "Hima")
ERA5_ROOT = os.path.join(BASE_PATH, "ERA5")
AWS_ROOT  = os.path.join(BASE_PATH, "Precipitation", "AWS")

# 14 band cần đọc
BANDS = ['B04B','B05B','B06B','B09B','B10B','B11B','B12B',
         'B14B','B16B','I2B','I4B','IRB','VSB','WVB']

# Mapping để tạo 36 BTdiff
BT_MAPPING = {
    'WVB': 'BT_6.2',   'B09B': 'BT_6.9', 'B10B': 'BT_7.3',
    'B11B': 'BT_8.6',  'B12B': 'BT_9.6', 'IRB':  'BT_10.4',
    'B14B': 'BT_11.2', 'I2B':  'BT_12.4', 'B16B': 'BT_13.3'
}

print("Bắt đầu xử lý dữ liệu - Phiên bản đã sửa hoàn toàn!")

# ========================================================================
# 1. ĐỌC HIMAWARI (từ .tif) - ĐÃ SỬA ĐÚNG TÊN FILE .Z0000_TB
# ========================================================================
print("1/5 Đang đọc Himawari-8 (.tif)...")
all_pixels = []

for band in BANDS:
    pattern = os.path.join(HIMA_ROOT, band, "**", "*.tif")
    tif_files = glob(pattern, recursive=True)
    print(f"   {band}: {len(tif_files)} file")

    for path in tqdm(tif_files, desc=f"   → {band}"):
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)

            valid = (arr > 0) & (~np.isnan(arr)) & (arr != -9999)
            rows, cols = np.where(valid)
            values = arr[valid]

            # LẤY THỜI GIAN ĐÚNG: B04B_20190401.Z0000_TB.tif → 201904010000
            filename = os.path.basename(path)
            # Tách phần sau dấu chấm đầu tiên và trước _TB
            time_part = filename.split('.')[1].split('_')[0]  # Z0000
            dt_str = filename.split('.')[0].split('_')[1] + time_part.lstrip('Z')  # 20190401 + 0000
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")

            temp_df = pd.DataFrame({
                'Datetime': dt,
                'row': rows,
                'col': cols,
                band: values
            })
            all_pixels.append(temp_df)

        except Exception as e:
            print(f"   Lỗi đọc {path}: {e}")

hima_df = pd.concat(all_pixels, ignore_index=True)
print(f"   Tổng pixel Himawari: {len(hima_df):,}")

# ========================================================================
# 2. Pivot + Tạo 36 BTdiff
# ========================================================================
print("2/5 Tạo bảng rộng + 36 BTdiff...")
df_wide = hima_df.pivot_table(
    index=['Datetime', 'row', 'col'],
    columns=hima_df.columns.drop(['Datetime','row','col']).tolist(),
    values=hima_df.columns.drop(['Datetime','row','col']).tolist(),
    aggfunc='first'
).reset_index()
df_wide.columns.name = None
df_wide = df_wide.dropna(subset=BANDS)

# Tạo BT columns
for old, new in BT_MAPPING.items():
    if old in df_wide.columns:
        df_wide[new] = df_wide[old]

bt_cols = list(BT_MAPPING.values())
for c1, c2 in combinations(bt_cols, 2):
    n1, n2 = c1.split('_')[1], c2.split('_')[1]
    df_wide[f"BTdiff_{n1}_{n2}"] = df_wide[c1] - df_wide[c2]

print(f"   Đã tạo {len(list(combinations(bt_cols, 2)))} cột BTdiff")

# ========================================================================
# 3. ĐỌC ERA5 (có giây: 20190401000000)
# ========================================================================
print("3/5 Đang đọc ERA5...")
era5_vars = ['CAPE', 'CIN', 'KX', 'R850', 'TCWV', 'U850', 'V850']  # thêm biến bạn có
era5_dfs = []

for var in era5_vars:
    pattern = os.path.join(ERA5_ROOT, var, "**", "*.csv")
    files = glob(pattern, recursive=True)
    for f in files:
        try:
            df = pd.read_csv(f)
            # Lấy thời gian từ tên file hoặc folder
            basename = os.path.basename(f)
            dt_str = basename.split('_')[-1].split('.')[0]  # 20190401000000
            dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")  # bỏ giây
            df['Datetime'] = dt
            df[var] = df.iloc[:, -1]
            df = df[['Datetime', 'row', 'col', var]]
            era5_dfs.append(df)
        except:
            continue

era5_all = pd.concat(era5_dfs, ignore_index=True)
era5_wide = era5_all.pivot_table(
    index=['Datetime','row','col'],
    columns=era5_all.columns[-1].name,
    values=era5_all.columns[-1],
    aggfunc='first'
).reset_index()
era5_wide.columns.name = None

# ========================================================================
# 4. ĐỌC AWS
# ========================================================================
print("4/5 Đang đọc AWS...")
aws_files = glob(os.path.join(AWS_ROOT, "**", "*.csv"), recursive=True)
aws_dfs = []
for f in aws_files:
    try:
        df = pd.read_csv(f)
        dt_str = os.path.basename(f).split('_')[-1].split('.')[0][:12]
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
        df['Datetime'] = dt
        df = df[['Datetime', 'row', 'col', 'aws']]
        aws_dfs.append(df)
    except:
        continue

aws_df = pd.concat(aws_dfs, ignore_index=True)

# ========================================================================
# 5. GHÉP + XUẤT CSV
# ========================================================================
print("5/5 Ghép dữ liệu và xuất file...")
final = df_wide.merge(era5_wide, on=['Datetime','row','col'], how='inner')
final = final.merge(aws_df, on=['Datetime','row','col'], how='inner')

print(f"\nHOÀN TẤT 100%!")
print(f"   Số mẫu cuối: {len(final):,} | Số cột: {final.shape[1]}")
final.to_csv("final_rainfall_dataset.csv", index=False)
print("ĐÃ XUẤT FILE: final_rainfall_dataset.csv")
print("Bây giờ bạn có thể train model thoải mái!")