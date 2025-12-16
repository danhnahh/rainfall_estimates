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


# =========================================================
# 1) LOAD SHAPEFILE HÀ TĨNH
# =========================================================
shp_path = "../gadm41_VNM_shp"
vnm_gdf = gpd.read_file(shp_path)

# Lọc Hà Tĩnh
ht_gdf = vnm_gdf[vnm_gdf['VARNAME_1'] == 'Ha Tinh']

ht_union = ht_gdf.geometry.union_all()
ht_crs = ht_gdf.crs


# =========================================================
# 2) TRÍCH DATETIME TỪ FILENAME
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


# =========================================================
# 3) LIST FILE
# =========================================================
def list_all_files(root):
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith(".tif") or f.endswith(".TIF"):
                out.append(os.path.join(dp, f))
    return out


# =========================================================
# 4) NHANH HƠN: FILL NODATA = NEAREST NEIGHBOR (KHÔNG DÙNG griddata)
# =========================================================
def fast_fill_nodata_nearest(arr):
    mask = np.isnan(arr) | np.isinf(arr)
    if not mask.any():
        return arr

    # Distance transform để tìm pixel valid gần nhất
    dist, (inds_y, inds_x) = ndimage.distance_transform_edt(mask,
        return_indices=True
    )
    return arr[inds_y, inds_x]


# =========================================================
# 5) EXTRACT PIXELS HÀ TĨNH
# =========================================================
def extract_HaTinh_pixels(path, root):

    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            transform = src.transform
            src_crs = src.crs

        # --- chuẩn hóa NODATA ---
        data[data == nodata] = np.nan
        data[data == -9999] = np.nan
        data[np.isinf(data)] = np.nan

        # --- fill nodata bằng nearest ---
        data = fast_fill_nodata_nearest(data)

        # --- reproject Hà Tĩnh sang CRS raster ---
        if src_crs != ht_crs:
            geom = ht_gdf.to_crs(src_crs).geometry.union_all()
        else:
            geom = ht_union

        # --- tạo mask pixel thuộc Hà Tĩnh ---
        mask = geometry_mask(
            [mapping(geom)],
            invert=True,
            out_shape=data.shape,
            transform=transform
        )

        rows, cols = np.where(mask)
        vals = data[rows, cols]

        # --- nếu không có pixel trong Hà Tĩnh ---
        if len(rows) == 0:
            cr = data.shape[0] // 2
            cc = data.shape[1] // 2
            rows, cols = np.array([cr]), np.array([cc])
            vals = np.array([data[cr, cc]])

        # --- chuyển sang lon/lat ---
        lons, lats = rasterio.transform.xy(transform, rows, cols, offset='center')

        # --- timestamp ---
        ts = extract_datetime_from_filename(path)

        # --- variable ---
        rel = os.path.relpath(path, root)
        var = rel.split(os.sep)[0]

        return pd.DataFrame({
            "variable": var,
            "timestamp": ts,
            "row": rows,
            "col": cols,
            "lon": lons,
            "lat": lats,
            "value": vals
        })

    except Exception as e:
        print("ERROR:", path, e)
        return pd.DataFrame({
            "variable": "unknown",
            "timestamp": pd.NaT,
            "row": [0],
            "col": [0],
            "lon": [np.nan],
            "lat": [np.nan],
            "value": [np.nan],
        })


# =========================================================
# 6) MAIN
# =========================================================
if __name__ == "__main__":

    # out_csv = "csv_data/RADAR_hatinh.csv"
    # root = "DATA_SV/Precipitation/Radar"

    # out_csv = "csv_data/HIMA_hatinh.csv"
    # root = "DATA_SV/Hima"

    out_csv = "csv_data/ERA5_hatinh.csv"
    root = "DATA_SV/ERA5"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    files = list_all_files(root)
    print("Tổng file tìm thấy:", len(files))

    if os.path.exists(out_csv):
        os.remove(out_csv)

    func = partial(extract_HaTinh_pixels, root=root)
    results = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(func, f) for f in files]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Process"):
            try:
                df = f.result()
                if df is not None and not df.empty:
                    results.append(df)
            except Exception as e:
                print("Thread error:", e)

    if results:
        final = pd.concat(results, ignore_index=True)
        final.to_csv(out_csv, index=False)
        print("DONE! Tổng pixel =", len(final))
    else:
        print("Không có data.")

