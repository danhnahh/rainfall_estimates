import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm

# ==========================================
# CẤU HÌNH
# ==========================================
RADAR_FOLDER = "../DATA_SV/Precipitation/Radar"  # Đường dẫn gốc


def check_raw_radar_files_recursive():
    print(f"📂 Đang quét toàn bộ thư mục con trong: {RADAR_FOLDER} ...")

    # Dùng "**" và recursive=True để tìm trong mọi ngóc ngách
    # Tìm cả đuôi .tif và .tiff
    files = glob.glob(os.path.join(RADAR_FOLDER, "**", "*.tif"), recursive=True) + \
            glob.glob(os.path.join(RADAR_FOLDER, "**", "*.tiff"), recursive=True)

    if not files:
        print(f"❌ Không tìm thấy file .tif nào! Hãy kiểm tra lại đường dẫn.")
        return

    print(f"🔍 Tìm thấy tổng cộng {len(files)} files. Đang bắt đầu kiểm tra...")
    print("-" * 85)
    # In tiêu đề cột cho thẳng hàng
    print(f"{'Tên File':<35} | {'NoData':<8} | {'NaN':<8} | {'-9999':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 85)

    total_files_with_error = 0
    count_files_checked = 0

    # Duyệt từng file
    for f_path in tqdm(files, desc="Checking"):
        file_name = os.path.basename(f_path)

        try:
            with rasterio.open(f_path) as src:
                # Đọc dữ liệu thô
                data = src.read(1)
                nodata_val = src.nodata

                # 1. Đếm số lượng NaN
                count_nan = np.isnan(data).sum()

                # 2. Đếm số lượng -9999
                count_9999 = (data == -9999).sum()

                # 3. Lấy Min/Max thực tế của file
                # Dùng nanmin/nanmax để tránh bị NaN làm hỏng kết quả so sánh
                min_val = np.nanmin(data) if data.size > 0 else 0
                max_val = np.nanmax(data) if data.size > 0 else 0

                # ĐIỀU KIỆN IN RA MÀN HÌNH:
                # Chỉ in nếu file có vấn đề (có NaN, có -9999)
                # HOẶC in 10 file đầu tiên để bạn kiểm tra xem nó đọc đúng không
                has_issue = (count_nan > 0) or (count_9999 > 0)

                if has_issue or count_files_checked < 10:
                    status_flag = "⚠️" if has_issue else "✅"
                    print(
                        f"{status_flag} {file_name:<32} | {str(nodata_val):<8} | {count_nan:<8} | {count_9999:<8} | {min_val:<8.2f} | {max_val:<8.2f}")

                    if has_issue:
                        total_files_with_error += 1

                count_files_checked += 1

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_name}: {e}")

    print("-" * 85)
    print("📊 TỔNG KẾT:")
    print(f"   - Tổng số file đã quét: {len(files)}")
    print(f"   - Số file chứa dữ liệu lỗi (NaN hoặc -9999): {total_files_with_error}")

    if total_files_with_error == 0:
        print("\n✅ NHẬN XÉT: Dữ liệu Raw rất sạch, không có NaN hay -9999.")
        print("   -> Nếu Min = 0, tức là 'không mưa' được gán bằng 0.")
        print("   -> Nếu Min là số âm khác (vd -32768), cần sửa code load data để xử lý giá trị này.")


if __name__ == "__main__":
    check_raw_radar_files_recursive()