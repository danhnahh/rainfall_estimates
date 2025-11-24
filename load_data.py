import os
import rasterio
from config import BASE_PATH, HIMA_PATH, HIMA_B04B_PATH, HIMA_B05B_PATH
import time
from collections import defaultdict


def read_tif_folder(folder_path, limit=None):
    """
    Đọc toàn bộ ảnh .tif trong folder (kể cả thư mục con)
    limit = số file tối đa muốn đọc (optional)
    """

    results = []
    count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tif"):
                filepath = os.path.join(root, file)

                try:
                    with rasterio.open(filepath) as src:
                        data = src.read(1)

                        info = {
                            "file": filepath,  # Đường dẫn file ảnh
                            "shape": data.shape,  # (H, W) - kích thước ảnh (height, width)
                            "dtype": src.dtypes[0],  # Kiểu dữ liệu pixel: float32/int16...
                            "bands": src.count,  # Số lượng band trong file (Himawari thường = 1)
                            "crs": src.crs,  # Hệ tọa độ (Coordinate Reference System), thường là WGS84 (EPSG:4326)
                            "res": src.res,  # Độ phân giải pixel (size theo độ), vd: (0.04°, 0.04°)
                        }

                    results.append(info)
                    count += 1

                    if limit and count >= limit:
                        return results

                except Exception as e:
                    print(f"⚠️ Không đọc được file: {filepath}")
                    print("Lỗi:", e)

    return results


def load_data(path):
    datas = []
    results2 = read_tif_folder(path)  # đọc tất cả các dữ liệu
    datas = group_data_by_date(results2)  # nhóm dữ liệu cùng ngày // trong 1 ngày gồm các dict theo giờ

    return datas


import os
import re
from collections import defaultdict


def group_data_by_date(dict_list):
    """
    Nhóm các dict có cùng ngày (YYYYMMDD) dựa vào tên file.

    Params:
        dict_list: list các dict, mỗi dict phải có key 'file' chứa đường dẫn file.
    Returns:
        list các list dict, mỗi list chứa các dict cùng ngày.
    """
    # Khởi tạo defaultdict để nhóm dữ liệu, khóa là ngày tháng YYYYMMDD
    groups = defaultdict(list)

    # Biểu thức chính quy tìm kiếm 8 chữ số liên tiếp (YYYYMMDD)
    # Đây là mẫu ngày tháng phổ biến trong các tên file dữ liệu vệ tinh
    DATE_PATTERN = re.compile(r'(\d{8})')

    for d in dict_list:
        # lấy giá trị của khóa 'file' trong dictionary
        filepath = d.get('file', '')
        if not filepath:
            continue  # Bỏ qua nếu không có key 'file' hoặc giá trị rỗng

        # Lấy tên file cuối cùng trong đường dẫn (ví dụ: WVB_20201026.Z0200_TB.tif)
        filename = os.path.basename(filepath)

        # Tìm kiếm ngày tháng trong tên file
        match = DATE_PATTERN.search(filename)

        # 1. Trích xuất khóa ngày tháng (YYYYMMDD)
        if match:
            # Lấy chuỗi 8 chữ số được tìm thấy (ví dụ: '20201026')
            date_key = match.group(1)
        else:
            # Nếu không tìm thấy ngày tháng, gán một khóa riêng biệt (ví dụ: 'UNKNOWN')
            date_key = "UNKNOWN_DATE"

            # 2. Thêm dict vào nhóm tương ứng
        groups[date_key].append(d)

    # 3. Loại bỏ nhóm 'UNKNOWN_DATE' nếu bạn chỉ muốn các nhóm có ngày
    if "UNKNOWN_DATE" in groups:
        del groups["UNKNOWN_DATE"]

    # Trả về danh sách các list dict (mỗi list là một nhóm)
    return list(groups.values())


def test():
    # # Ví dụ: đọc thử 3 file đầu tiên trong BASE_PATH
    #
    # ############################
    # # tất cả dữ liệu trong hima chưa gom nhóm
    #
    start_time = time.time()
    # results = read_tif_folder(BASE_PATH)
    results = read_tif_folder(HIMA_PATH)
    end_time = time.time()

    # # results2 = read_tif_folder(HIMA_B05B_PATH, limit=3)
    # # i = 0
    # # for r in results:
    # #     i += 1
    # #     print(i, '. ', r)
    # ###########################
    #

    data_date = group_data_by_date(results)

    # dem = 0
    # for r in data_date:
    #     for k in r:
    #         print(k['file'])
    #         dem += 1
    #     print('số lần chụp trong 1 ngày: ', dem/14)
    #     print('------------------------------')
    #     dem = 0
    #
    # print(f'thời gian chạy của hàm read_tif_folder : {(end_time - start_time):.4f} giây')

    for r in data_date:
        for k in r:
        #     print(k['file'])
        # print('---------------------\n')
            print(k)
        print('---------------------\n')

    # print(data_date)

    # print(load_data(HIMA_PATH))


test()
