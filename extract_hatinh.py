import os
import re
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy import ndimage
from functools import partial

# --- Đọc shapefile Việt Nam ---
shp_path = "gadm41_VNM_shp"
vnm_gdf = gpd.read_file(shp_path)

# Lọc geometry Hà Tĩnh
ht_gdf = vnm_gdf[vnm_gdf['VARNAME_1'] == 'Ha Tinh']
ht_crs = ht_gdf.crs.to_string()
ht_union = ht_gdf.geometry.union_all()  # union_all() thay cho unary_union


#

def extract_datetime_from_filename(path):
    filename = os.path.basename(path)

    # 1) Kiểu: CAPE_20190401000000.tif (14 số liên tiếp)
    m14 = re.search(r'(\d{14})', filename)
    if m14:
        return pd.to_datetime(m14.group(1), format='%Y%m%d%H%M%S', errors='coerce')

    # 2) Kiểu: B04B_20190401.Z0000_TB.tif
    # Tách ngày YYYYMMDD
    m_date = re.search(r'(\d{8})', filename)
    # Tách giờ Zhhmm
    m_z = re.search(r'Z(\d{4})', filename)

    if m_date:
        date_str = m_date.group(1)  # YYYYMMDD

        if m_z:
            hhmm = m_z.group(1)  # HHMM
            dt_str = date_str + hhmm  # YYYYMMDDHHMM
            return pd.to_datetime(dt_str, format='%Y%m%d%H%M', errors='coerce')
        else:
            # Không có Z → default 00:00
            return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')

    # Nếu không match gì → trả NaT
    return pd.NaT


# --- Lấy danh sách tất cả file .tif ---
def list_all_files(root):
    tif_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".tif"):
                tif_files.append(os.path.join(dirpath, f))
    return tif_files


def extract_HaTinh_pixels(path, root):
    """
    Extract pixels inside Ha Tinh province from a raster file.
    Nếu raster không có pixel nào trong Hà Tĩnh, vẫn tạo 1 row dummy.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            transform = src.transform
            src_crs = src.crs.to_string() if src.crs else None

        # --- Reproject geometry Hà Tĩnh sang CRS raster nếu cần ---
        if src_crs and src_crs != ht_crs:
            geom = gpd.GeoSeries([ht_union], crs=ht_crs).to_crs(src_crs).iloc[0]
        else:
            geom = ht_union

        # --- Fill nodata bằng nearest neighbor ---
        mask_nan = (data == nodata) | np.isnan(data)
        if mask_nan.any():
            from scipy.interpolate import griddata
            rows_idx, cols_idx = np.indices(data.shape)
            valid_mask = ~mask_nan
            valid_points = np.column_stack([rows_idx[valid_mask], cols_idx[valid_mask]])
            valid_values = data[valid_mask]
            nan_points = np.column_stack([rows_idx[mask_nan], cols_idx[mask_nan]])
            data[mask_nan] = griddata(valid_points, valid_values, nan_points, method='nearest')

        # --- Tạo mask pixel thuộc Hà Tĩnh ---
        mask = geometry_mask([mapping(geom)],
                             invert=True,
                             out_shape=data.shape,
                             transform=transform)

        rows, cols = np.where(mask)
        vals = data[rows, cols]

        # Nếu không có pixel hợp lệ, tạo dummy row trung tâm raster
        if len(vals) == 0:
            center_row = data.shape[0] // 2
            center_col = data.shape[1] // 2
            rows = np.array([center_row])
            cols = np.array([center_col])
            vals = np.array([data[center_row, center_col]])
            # print(f"[{os.path.basename(path)}] Không có pixel Hà Tĩnh, dùng pixel trung tâm", flush=True)

        # --- Lấy lon/lat ---
        lons, lats = rasterio.transform.xy(transform, rows, cols, offset='center')

        # --- Lấy timestamp từ filename ---
        ts = extract_datetime_from_filename(path)
        # filename = os.path.basename(path)
        # ts = pd.NaT
        # match = re.search(r'\d{14}', filename)
        # if match:
        #     datetime_str = match.group()
        #     ts = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S', errors='coerce')

        # --- Variable ---
        rel_path = os.path.relpath(path, root)
        var = rel_path.split(os.sep)[0]

        # --- Tạo DataFrame ---
        df = pd.DataFrame({
            'variable': var,
            'timestamp': ts,
            'row': rows,
            'col': cols,
            'lon': lons,
            'lat': lats,
            'value': vals
        })

        return df

    except Exception as e:
        print(f"Lỗi khi xử lý {path}: {e}", flush=True)
        # Tạo row dummy tối thiểu để vẫn ghi vào CSV
        df = pd.DataFrame({
            'variable': 'unknown',
            'timestamp': pd.NaT,
            'row': [0],
            'col': [0],
            'lon': [np.nan],
            'lat': [np.nan],
            'value': [np.nan]
        })
        return df


# --- Main ---
if __name__ == '__main__':

    # --- Thư mục chứa dữ liệu GeoTIFF và output CSV ---
    out_csv = 'csv_data/HIMA_hatinh.csv'
    # out_csv = 'dungx/IM_hatinh.csv'
    root = 'DATA_SV/Hima'

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    files = list_all_files(root)

    print("Số file tìm thấy:", len(files))

    if os.path.exists(out_csv):
        os.remove(out_csv)

    all_dfs = []

    n_threads = 10  # số luồng, tùy CPU

    # Partial để truyền root vào hàm
    func = partial(extract_HaTinh_pixels, root=root)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(func, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            f = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)
                    # print(f"[{os.path.basename(f)}] Xử lý xong, pixel hợp lệ: {len(df)}", flush=True)
                else:
                    print(f"[{os.path.basename(f)}] Không có pixel hợp lệ", flush=True)
            except Exception as e:
                print(f"Lỗi tổng khi xử lý file {f}: {e}", flush=True)

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(out_csv, index=False)
        print(f"\nHoàn thành flatten dữ liệu Hà Tĩnh! Tổng số pixel: {sum(len(df) for df in all_dfs):,}")
    else:
        print("\nKhông có pixel hợp lệ nào để ghi ra CSV.")


# | Cột         | Ý nghĩa                                                                                   |
# | ----------- | ----------------------------------------------------------------------------------------- |
# | `variable`  | Tên biến (dựa trên thư mục cha của file raster). Ví dụ: `temperature`, `precipitation`…   |
# | `timestamp` | Thời điểm của raster, được trích từ tên file bằng hàm `extract_datetime_from_filename()`. |
# | `row`       | Chỉ số hàng (index row) trong mảng raster 2D `data`.                                      |
# | `col`       | Chỉ số cột (index col) trong mảng raster 2D `data`.                                       |
# | `lon`       | Kinh độ của pixel, tính từ transform của raster (`rasterio.transform.xy`).                |
# | `lat`       | Vĩ độ của pixel, tính từ transform của raster.                                            |
# | `value`     | Giá trị pixel tại vị trí `(row, col)` trong raster.                                       |

# # Hàm extract pixel
# def extract_HaTinh_pixels(path, root):
#     try:
#         with rasterio.open(path) as src:
#             data = src.read(1).astype(float)
#             nodata = src.nodata
#             transform = src.transform
#             src_crs = src.crs.to_string() if src.crs else None
#
#         # Reproject geometry nếu cần
#         if src_crs and src_crs != ht_crs:
#             geom = gpd.GeoSeries([ht_union], crs=ht_crs).to_crs(src_crs).iloc[0]
#         else:
#             geom = ht_union
#
#         # Fill nodata bằng nearest neighbor
#         mask_nan = (data == nodata) | np.isnan(data)
#         if mask_nan.any():
#             coords = np.indices(data.shape)
#             data[mask_nan] = ndimage.map_coordinates(
#                 data, coords[:, mask_nan.ravel()], order=0, mode='nearest'
#             )
#
#         # Tạo lưới pixel
#         rows_all, cols_all = np.meshgrid(np.arange(data.shape[0]),
#                                          np.arange(data.shape[1]),
#                                          indexing='ij')
#         rows_flat = rows_all.ravel()
#         cols_flat = cols_all.ravel()
#
#         # Tạo mask pixel thuộc Hà Tĩnh
#         mask_flat = geometry_mask([mapping(geom)],
#                                   invert=True,
#                                   out_shape=data.shape,
#                                   transform=transform).ravel()
#
#         valid_pixels = mask_flat
#
#         if valid_pixels.sum() == 0:
#             print(f"[{os.path.basename(path)}] Không có pixel Hà Tĩnh")
#             return pd.DataFrame()  # không có pixel hợp lệ
#
#         # Lấy giá trị pixel hợp lệ
#         rows = rows_flat[valid_pixels]
#         cols = cols_flat[valid_pixels]
#         # vals = data.ravel()[valid_pixels]
#         vals = data[mask_flat]  # lấy trực tiếp giá trị hợp lệ
#
#         lons, lats = rasterio.transform.xy(transform, rows, cols, offset='center')
#
#         # Timestamp
#         filename = os.path.basename(path)
#         ts = pd.NaT
#         match = re.search(r'\d{14}', filename)
#         if match:
#             datetime_str = match.group()
#             ts = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S', errors='coerce')
#
#         # Variable
#         rel_path = os.path.relpath(path, root)
#         var = rel_path.split(os.sep)[0]
#
#         # Tạo DataFrame
#         df = pd.DataFrame({
#             'variable': var,
#             'timestamp': ts,
#             'row': rows,
#             'col': cols,
#             'lon': lons,
#             'lat': lats,
#             'value': vals
#         })
#
#         return df
#     except Exception as e:
#         print(f"Lỗi khi xử lý {path}: {e}", flush=True)
#         return pd.DataFrame()
