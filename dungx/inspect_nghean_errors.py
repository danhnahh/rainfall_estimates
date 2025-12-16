import os
import glob
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from tqdm import tqdm

# ==========================================
# CẤU HÌNH
# ==========================================
RADAR_FOLDER = "../DATA_SV/Precipitation/Radar"
SHP_PATH = "../gadm41_VNM_shp"
TARGET_PROVINCE = "Nghe An"  # Hoặc 'Thanh Hoa' tùy bạn chỉnh


def inspect_errors_inside_region():
    print(f"-> [1] Đang load Shapefile và lọc tỉnh {TARGET_PROVINCE}...")

    # 1. Load Shapefile
    try:
        vnm_gdf = gpd.read_file(SHP_PATH)
        # Lọc tỉnh
        region_gdf = vnm_gdf[vnm_gdf['VARNAME_1'] == TARGET_PROVINCE]
        if region_gdf.empty:
            # Fallback phòng khi tên không khớp
            region_gdf = vnm_gdf[vnm_gdf['VARNAME_1'].str.contains(TARGET_PROVINCE, case=False, na=False)]

        if region_gdf.empty:
            print(f"❌ Không tìm thấy tỉnh '{TARGET_PROVINCE}' trong Shapefile!")
            return

        # Lấy geometry gốc (chưa project)
        region_geom_origin = region_gdf.geometry.values[0]
        region_crs = region_gdf.crs

    except Exception as e:
        print(f"❌ Lỗi load Shapefile: {e}")
        return

    # 2. Quét file đệ quy
    print(f"-> [2] Đang quét file Radar trong {RADAR_FOLDER}...")
    files = glob.glob(os.path.join(RADAR_FOLDER, "**", "*.tif"), recursive=True) + \
            glob.glob(os.path.join(RADAR_FOLDER, "**", "*.tiff"), recursive=True)

    if not files:
        print("❌ Không tìm thấy file TIF nào.")
        return

    print(f"   Tìm thấy {len(files)} files. Bắt đầu soi lỗi trong vùng nội bộ...")
    print("-" * 90)
    print(f"{'File Name':<35} | {'Inside -Inf':<12} | {'Inside NaN':<10} | {'Inside -9999':<12} | {'Status'}")
    print("-" * 90)

    count_bad_files = 0

    # 3. Duyệt và kiểm tra
    for f_path in tqdm(files, desc="Inspecting"):
        file_name = os.path.basename(f_path)

        try:
            with rasterio.open(f_path) as src:
                # --- A. ĐỒNG BỘ CRS ---
                # Nếu hệ tọa độ khác nhau, ta phải chiếu shapefile theo ảnh vệ tinh
                if region_crs != src.crs:
                    # Tạo GeoDataFrame tạm để to_crs
                    gdf_temp = gpd.GeoDataFrame({'geometry': [region_geom_origin]}, crs=region_crs)
                    gdf_proj = gdf_temp.to_crs(src.crs)
                    geom_proj = [gdf_proj.geometry.values[0]]
                else:
                    geom_proj = [region_geom_origin]

                # --- B. CẮT ĐÚNG HÌNH DÁNG TỈNH (MASKING) ---
                # crop=True: Cắt bỏ phần thừa xung quanh hình chữ nhật luôn
                # out_image: Chỉ chứa dữ liệu trong hình chữ nhật bao quanh
                # out_transform: Transform mới của hình cắt
                out_image, out_transform = mask(src, geom_proj, crop=True, nodata=np.nan)

                # out_image có shape (bands, height, width). Radar chỉ có 1 band.
                data = out_image[0]  # Lấy mảng 2D

                # --- C. LỌC LẤY PIXEL TRONG TỈNH ---
                # Hàm mask() của rasterio mặc định sẽ gán giá trị bên ngoài vùng cắt = nodata (ở đây ta set là NaN)
                # Tuy nhiên, để chắc chắn "trong tỉnh" hay "ngoài tỉnh", ta cần phân biệt:
                # Những điểm bên trong tỉnh sẽ giữ nguyên giá trị gốc.
                # Những điểm bên ngoài tỉnh sẽ bị gán thành NaN (do tham số nodata=np.nan ở trên).

                # Vậy nên: Mọi giá trị khác NaN (và khác nodata gốc) chính là DỮ LIỆU TRONG TỈNH.
                # NHƯNG: Nếu dữ liệu gốc BÊN TRONG tỉnh cũng bị lỗi (là NaN hoặc -inf), ta cần bắt nó.

                # Cách kiểm tra chuẩn nhất:
                # Dữ liệu trả về từ hàm mask() đã biến mọi thứ "ngoài vùng" thành NaN.
                # Vấn đề là dữ liệu gốc lỗi cũng có thể là NaN/Inf.

                # -> Ta cần biết chính xác pixel nào thuộc geometry để chỉ check pixel đó.
                # Dùng rasterio.features.geometry_mask để tạo khuôn

                region_mask = rasterio.features.geometry_mask(
                    geom_proj,
                    out_shape=data.shape,
                    transform=out_transform,
                    invert=True  # True = Bên trong tỉnh
                )

                # Lấy dữ liệu thuần túy nằm trong biên giới tỉnh
                inside_data = data[region_mask]

                # --- D. CHECK LỖI ---
                num_inf = np.isinf(inside_data).sum()
                num_nan = np.isnan(inside_data).sum()
                num_9999 = (inside_data == -9999).sum()

                if num_inf > 0 or num_nan > 0 or num_9999 > 0:
                    count_bad_files += 1
                    print(f"{file_name:<35} | {num_inf:<12} | {num_nan:<10} | {num_9999:<12} | ❌ BAD")

                # Uncomment dòng dưới nếu muốn thấy cả file tốt (sẽ spam màn hình)
                # else:
                #    print(f"{file_name:<35} | 0            | 0          | 0            | ✅ OK")

        except Exception as e:
            print(f"⚠️ Lỗi đọc {file_name}: {e}")

    print("-" * 90)
    print(f"📊 KẾT QUẢ CUỐI CÙNG:")
    print(f"   - Tổng số file quét: {len(files)}")
    print(f"   - Số file có lỗi BÊN TRONG tỉnh {TARGET_PROVINCE}: {count_bad_files}")

    if count_bad_files == 0:
        print(f"\n✅ KHẲNG ĐỊNH: Dữ liệu Radar hoàn toàn sạch bên trong địa phận {TARGET_PROVINCE}.")
        print("   -> Các giá trị lỗi (-inf) chỉ xuất hiện ở vùng rìa/ngoài biển (đã bị loại bỏ).")
    else:
        print(f"\n⚠️ CẢNH BÁO: Có {count_bad_files} file chứa dữ liệu lỗi nằm ngay trong tỉnh!")


if __name__ == "__main__":
    inspect_errors_inside_region()