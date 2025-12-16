import os
import re
import glob
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from tqdm import tqdm

# =========================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
HIMA_PATH = "DATA_SV/Hima"
ERA5_PATH = "DATA_SV/ERA5"
RADAR_PATH = "DATA_SV/Precipitation/Radar"
SHP_PATH = "gadm41_VNM_shp"  # Thư mục chứa shapefile

# Output
OUTPUT_X = "csv_data/tri_an_thanh_hoa/x_direct.npy"
OUTPUT_Y = "csv_data/tri_an_thanh_hoa/y_direct.npy"

# Features
# selected_features = ['CAPE', 'CIN', 'EWSS', 'IE', 'ISOR',
#                      'PEV', 'R500', 'R850', 'SLHF', 'SLOR',
#                      'SSHF', 'TCLW', 'TCW', 'U250', 'U850',
#                      'V850', 'B05B', 'B09B', 'B10B', 'B12B',
#                      'B14B', 'B16B', 'I2B', 'I4B', 'VSB']
selected_features = [
    "TCW", "I4B", "B04B", "R500", "VSB",
    "R250", "V250", "WVB", "TCLW", "CIN",
    "U250", "B05B", "B06B", "B09B", "V850",
    "PEV", "U850", "KX", "R850", "CAPE"
]


# selected_features = ['B04B', 'B10B', 'B11B', 'B16B', 'IRB',
# 'CAPE', 'R850', 'TCWV', 'U850', 'I2B', 'TCLW', 'TCW']


# =========================================================
# 1. CÁC HÀM TIỆN ÍCH (HELPER)
# =========================================================
def extract_datetime_from_filename(path):
    filename = os.path.basename(path)
    # Kiểu 1: CAPE_20190401000000.tif
    m14 = re.search(r"(\d{14})", filename)
    if m14:
        return pd.to_datetime(m14.group(1), format="%Y%m%d%H%M%S", errors="coerce")

    # Kiểu 2: B04B_20190401.Z0000_TB.tif
    m_date = re.search(r"(\d{8})", filename)
    m_z = re.search(r"Z(\d{4})", filename)
    if m_date:
        date = m_date.group(1)
        if m_z:
            return pd.to_datetime(date + m_z.group(1), format="%Y%m%d%H%M", errors="coerce")
        return pd.to_datetime(date, format="%Y%m%d", errors="coerce")
    return pd.NaT


def get_variable_name(filename):
    # Mapping tên biến đặc biệt cho Radar
    if filename.startswith('Radar') or filename.startswith('2019') or filename.startswith('2020'):
        return 'y'
    return filename.split('_')[0]


def clean_to_minus9999(arr, nodata_val):
    """
    Chuẩn hóa pixel lỗi về -9999

    Quy ước:
    - nodata trong raster
    - giá trị -9999 có sẵn
    - inf / -inf
    - NaN
    => tất cả đều chuyển thành -9999

    KHÔNG xử lý mask tỉnh ở đây
    """
    # nodata do raster khai báo
    if nodata_val is not None:
        arr[arr == nodata_val] = np.nan

    # một số file đã dùng -9999 làm nodata
    arr[arr == -9999] = np.nan

    # loại bỏ giá trị vô hạn
    arr[np.isinf(arr)] = np.nan

    # tất cả NaN -> -9999
    arr[np.isnan(arr)] = -9999.0

    return arr


# =========================================================
# 2. BƯỚC QUAN TRỌNG: TẠO MAP FILE VÀ LỌC TIMESTAMP
# =========================================================
def scan_and_filter_files(folders, required_features):
    print("-> [B1] Scanning files & Indexing...")

    # Map cấu trúc: { timestamp: { 'B04B': 'path/to/file', 'y': 'path/to/file', ... } }
    mega_map = {}

    for folder in folders:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.tif', '.tiff')):
                    fpath = os.path.join(root, f)
                    ts = extract_datetime_from_filename(fpath)
                    if pd.notna(ts):
                        var_name = get_variable_name(f)
                        if ts not in mega_map: mega_map[ts] = {}
                        mega_map[ts][var_name] = fpath

    # Lọc những timestamp có đủ bộ feature
    required_set = set(required_features + ['y'])
    valid_timestamps = []

    for ts, var_dict in mega_map.items():
        existing_vars = set(var_dict.keys())
        if required_set.issubset(existing_vars):
            valid_timestamps.append(ts)

    valid_timestamps.sort()
    print(f"-> Tìm thấy {len(valid_timestamps)} mốc thời gian ĐỦ dữ liệu (Full features + y).")

    return valid_timestamps, mega_map


# =========================================================
# 3. TÍNH TOÁN GEOMETRY & BBOX (CHỈ LÀM 1 LẦN)
# =========================================================
def compute_spatial_metadata(shp_path, sample_tif_path):
    print("-> [B2] Tính toán khung hình học (Bbox) cho ...")

    # Load Shapefile
    vnm_gdf = gpd.read_file(shp_path)
    # Sửa tên tỉnh nếu cần (trong gadm41 thường là 'Nghe An' hoặc 'Thanh Hoa')
    # Ở đây tôi lấy theo code mẫu của bạn là 'Nghe An' (Code gốc bạn ghi load Nghe An nhưng tên hàm là ThanhHoa??)
    # Tôi sẽ assume bạn muốn lấy Thanh Hoa theo tên hàm, hãy sửa lại dòng dưới nếu là Nghe An
    target_province = 'Nghe An'  # <--- KIỂM TRA LẠI TÊN TRONG SHP

    region_gdf = vnm_gdf[vnm_gdf['VARNAME_1'] == target_province]
    if region_gdf.empty:
        # Fallback nếu không tìm thấy, thử tìm Nghe An như code cũ
        region_gdf = vnm_gdf[vnm_gdf['VARNAME_1'] == 'Nghe An']
        print(f"⚠️ Không tìm thấy '{target_province}', đang dùng 'Nghe An'.")

    region_union = region_gdf.geometry.union_all()
    region_crs = region_gdf.crs

    # Load 1 file mẫu để lấy Transform
    with rasterio.open(sample_tif_path) as src:
        transform = src.transform
        src_crs = src.crs
        src_shape = src.shape

    # Reproject Shapefile khớp với Raster
    if region_crs != src_crs:
        region_geom = gpd.GeoSeries([region_union], crs=region_crs).to_crs(src_crs).iloc[0]
    else:
        region_geom = region_union

    # Tính Bounding Box (cắt hình chữ nhật)
    bbox = region_geom.bounds
    min_row, min_col = rasterio.transform.rowcol(transform, bbox[0], bbox[3])
    max_row, max_col = rasterio.transform.rowcol(transform, bbox[2], bbox[1])

    # Kẹp biên (Clip to image bounds)
    min_row, max_row = max(0, min_row), min(src_shape[0], max_row)
    min_col, max_col = max(0, min_col), min(src_shape[1], max_col)

    height = max_row - min_row
    width = max_col - min_col

    print(f"   Crop Window: Rows[{min_row}:{max_row}], Cols[{min_col}:{max_col}]")
    print(f"   Size: {height} x {width}")

    # Tạo Mask nội bộ (để mask những điểm trong hcn nhưng ngoài biên giới tỉnh)
    # Lưu ý: window_transform phải chuẩn cho cửa sổ con
    window_transform = rasterio.windows.transform(
        rasterio.windows.Window(min_col, min_row, width, height),
        transform
    )

    # Mask: False là trong vùng, True là ngoài vùng (theo mặc định rasterio.geometry_mask)
    # Ta muốn 1 là trong vùng, 0 là ngoài vùng để nhân
    mask_binary = geometry_mask(
        [mapping(region_geom)],
        transform=window_transform,
        invert=True,  # Invert=True -> Trong vùng là True
        out_shape=(height, width)
    )

    return (min_row, max_row, min_col, max_col), mask_binary


# =========================================================
# 4. HÀM MAIN: LOAD DIRECT TO NUMPY
# =========================================================
def generate_numpy_dataset():
    # 1. Quét File
    folders = [HIMA_PATH, ERA5_PATH, RADAR_PATH]
    valid_ts, mega_map = scan_and_filter_files(folders, selected_features)

    if not valid_ts:
        print("❌ Không tìm thấy dữ liệu chung nào!")
        return

    # 2. Lấy metadata không gian từ file đầu tiên tìm thấy
    first_ts = valid_ts[0]
    sample_file = mega_map[first_ts]['y']  # Dùng file Radar hoặc feature làm mẫu
    (min_r, max_r, min_c, max_c), region_mask = compute_spatial_metadata(SHP_PATH, sample_file)

    H = max_r - min_r
    W = max_c - min_c
    T = len(valid_ts)

    # Sort feature để đảm bảo thứ tự kênh luôn cố định
    sorted_features = sorted(selected_features)
    C = len(sorted_features)  # Số kênh đầu vào (X)

    print("-" * 40)
    print(f"🚀 KHỞI TẠO TENSOR: Time={T}, C={C}, H={H}, W={W}")
    print("-" * 40)

    # 3. Cấp phát bộ nhớ (RAM)
    # X: (Time, Channels, Height, Width)
    X_data = np.full((T, C, H, W), -1.0, dtype=np.float32)
    # Y: (Time, 1, Height, Width) - Radar
    Y_data = np.full((T, 1, H, W), -1.0, dtype=np.float32)

    # 4. Loop & Fill (Có thể dùng ThreadPool nếu muốn, nhưng loop thường ổn định hơn cho Debug)
    # Dùng tqdm để hiện tiến độ

    for t_idx, ts in enumerate(tqdm(valid_ts, desc="Processing Timestamps")):
        files_at_ts = mega_map[ts]

        # =========================
        # XỬ LÝ INPUT X
        # =========================
        for c_idx, feat_name in enumerate(sorted_features):
            fpath = files_at_ts[feat_name]

            try:
                with rasterio.open(fpath) as src:
                    # 1) Chỉ đọc cửa sổ hình chữ nhật bao Nghệ An
                    window = rasterio.windows.Window(min_c, min_r, W, H)
                    data = src.read(1, window=window).astype(float)

                    # 2) Chuẩn hóa pixel lỗi -> -9999
                    data = clean_to_minus9999(data, src.nodata)

                    # 3) KHÔNG mask theo tỉnh
                    #    (ngoài tỉnh nhưng trong hình chữ nhật vẫn giữ nguyên)
                    X_data[t_idx, c_idx, :, :] = data

            except Exception as e:
                print(f"[X] Error reading {fpath}: {e}")

        # =========================
        # XỬ LÝ LABEL Y (RADAR)
        # =========================
        y_path = files_at_ts['y']

        try:
            with rasterio.open(y_path) as src:
                # 1) Đọc hình chữ nhật bao Nghệ An
                window = rasterio.windows.Window(min_c, min_r, W, H)
                data_y = src.read(1, window=window).astype(float)

                # 2) Chuẩn hóa pixel lỗi -> -9999
                #    (làm TRƯỚC khi mask tỉnh)
                data_y = clean_to_minus9999(data_y, src.nodata)

                # 3) Mask ngoài tỉnh:
                #    - region_mask == True  -> trong tỉnh
                #    - region_mask == False -> ngoài tỉnh
                data_y[~region_mask] = -1.0

                # 4) Gán vào tensor Y
                Y_data[t_idx, 0, :, :] = data_y

        except Exception as e:
            print(f"[Y] Error reading {y_path}: {e}")

    # 5. Lưu File
    print(f"-> Đang lưu file .npy xuống đĩa...")
    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)
    np.save(OUTPUT_X, X_data)
    np.save(OUTPUT_Y, np.squeeze(Y_data, axis=1))  # Y thường để shape (T, H, W)

    print("✅ HOÀN TẤT! Dữ liệu đã được lưu.")
    print(f"   X path: {OUTPUT_X}")
    print(f"   Y path: {OUTPUT_Y}")


# Chạy hàm
if __name__ == "__main__":
    generate_numpy_dataset()
