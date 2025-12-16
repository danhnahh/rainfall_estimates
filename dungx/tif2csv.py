import rasterio
import numpy as np
import pandas as pd
import os
from dungx.config import CSV_PATH_HIMA
from rasterio.windows import from_bounds

def crop_tif(input_tif, left, bottom, right, top):
    """
    Cắt vùng từ file ảnh .tif dựa trên bounding box tọa độ địa lý.

    Parameters
    ----------
    input_tif : str
        Đường dẫn file .tif gốc muốn cắt.
    left, bottom, right, top : float
        Bounding box dưới dạng (lon_min, lat_min, lon_max, lat_max).
        - left   : kinh độ nhỏ nhất
        - bottom : vĩ độ nhỏ nhất
        - right  : kinh độ lớn nhất
        - top    : vĩ độ lớn nhất

    Returns
    -------
    data : numpy.ndarray
        Mảng dữ liệu ảnh sau khi đã được cắt theo bounding box.
    transform : Affine
        Ma trận biến đổi (geo-transform) mới tương ứng với vùng cắt.
    """

    # Mở file .tif để đọc
    with rasterio.open(input_tif) as src:

        # 1) Tạo cửa sổ (window) dựa trên bounding box bằng hệ tọa độ của ảnh
        #    Window quy định pixel nào thuộc vùng ảnh cần cắt
        window = from_bounds(left, bottom, right, top, src.transform)

        # 2) Đọc dữ liệu của vùng cắt (band 1)
        data = src.read(1, window=window)

        # 3) Tạo transform mới cho vùng cắt
        #    Transform này giúp giữ đúng tọa độ cho ảnh vùng nhỏ
        transform = src.window_transform(window)

    return data, transform


def extract_prefix(path):
    """
    Hàm lấy ra phần 'B04B_20190401' từ đường dẫn file .tif.
    Ví dụ:
    path = "DATA_SV/Hima/B04B/2019/04/01/B04B_20190401.Z0000_TB.tif"
    Kết quả trả về: "B04B_20190401"
    """

    # Lấy phần tên file từ đường dẫn (vd: "B04B_20190401.Z0000_TB.tif")
    filename = os.path.basename(path)

    # Cắt chuỗi tại dấu '.' đầu tiên → lấy phần trước đó
    # "B04B_20190401.Z0000_TB.tif" → "B04B_20190401"
    prefix = filename.split(".")[0]

    return prefix

def process_data_to_csv(data_array, filename, csv_path, rows=16, cols=34):
    """
    Nhận một mảng NumPy, resize (nếu cần), và lưu thành file CSV dạng:
    tên_file, row, col, value

    rows, cols = kích thước cuối cùng mong muốn.
    """

    # 1. Resize dữ liệu thành ma trận (Dùng data_array.shape nếu không muốn resize)
    # Lưu ý: Việc resize cố định (16x34) có thể làm mất ý nghĩa tọa độ địa lý.
    # Nếu bạn muốn giữ nguyên kích thước cắt: rows, cols = data_array.shape
    resized = np.resize(data_array, (rows, cols))

    # 2. Tạo list lưu dữ liệu dạng bảng (Tương tự code cũ)
    records = []
    for r in range(rows):
        for c in range(cols):
            # Lấy giá trị từ mảng đã resize
            records.append([filename, r, c, resized[r, c]])

    # 3. Convert list → DataFrame
    df = pd.DataFrame(records, columns=["file_name", "row", "col", "value"])

    # 4. Ghi ra CSV
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu CSV tại: {csv_path}")


def test_data_to_csv():
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(CSV_PATH_HIMA, exist_ok=True)

    # File test
    file_test = "../DATA_SV/Hima/B04B/2019/04/01/B04B_20190401.Z0000_TB.tif"
    numpy_tif = crop_tif(file_test,105.13, 17.95, 106.5, 18.6)
    data_numpy = numpy_tif[0]

    # Tạo tên file CSV đầy đủ
    csv_file = os.path.join(CSV_PATH_HIMA, extract_prefix(file_test) + ".csv")

    # Gọi hàm để tạo CSV
    process_data_to_csv(data_numpy, file_test, csv_file)

test_data_to_csv()