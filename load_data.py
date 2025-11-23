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
    datas = group_data_by_date(results2) # nhóm dữ liệu cùng ngày // trong 1 ngày gồm các dict theo giờ

    return datas


def group_data_by_date(dict_list):
    """
    Nhóm các dict có cùng ngày (dựa vào phần 'B04B_YYYYMMDD' trong tên file)
    Params:
        dict_list: list các dict, mỗi dict phải có key 'file'
    Returns:
        list các list dict, mỗi list chứa các dict cùng ngày
    """
    groups = defaultdict(list)

    for d in dict_list:
        # Lấy tên file cuối cùng trong đường dẫn
        filename = os.path.basename(d['file'])
        # Lấy phần trước dấu chấm, ví dụ: 'B04B_20190401'
        date_key = filename.split('.')[0]
        # Thêm dict vào nhóm tương ứng
        groups[date_key].append(d)

    # Trả về danh sách các nhóm
    return list(groups.values())


def test():
    # # Ví dụ: đọc thử 3 file đầu tiên trong BASE_PATH
    #
    # ############################
    # # tất cả dữ liệu trong hima chưa gom nhóm
    #
    # results = read_tif_folder(HIMA_B04B_PATH)
    # # results2 = read_tif_folder(HIMA_B05B_PATH, limit=3)
    # # i = 0
    # # for r in results:
    # #     i += 1
    # #     print(i, '. ', r)
    # ###########################
    #
    # data_date = group_data_by_date(results)
    #
    # dem = 0
    # for r in data_date:
    #     for k in r:
    #         print(k['file'])
    #         dem += 1
    #     print('số lần chụp trong 1 ngày: ', dem)
    #     print('------------------------------')
    #     dem = 0
    # # print(data_date)

    print(load_data(HIMA_PATH))


test()
