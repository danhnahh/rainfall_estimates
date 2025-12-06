import numpy as np
from scipy.interpolate import griddata


def preprocess_data(data, data_type):
    """
    Tiền xử lý dữ liệu raster:
    - Thay thế giá trị nan / inf / -9999 bằng nội suy nearest
    - Radar: ép không âm
    - ERA5 / Himawari: chuẩn hóa Min-Max về [0, 1]
    """

    if data is None:
        return None

    # --- 1. Xử lý giá trị không hợp lệ ---
    invalid_mask = np.isinf(data) | np.isnan(data) | (data == -9999)

    if invalid_mask.any():
        x, y = np.indices(data.shape)

        # pixel hợp lệ
        valid_points = np.column_stack((x[~invalid_mask], y[~invalid_mask]))
        valid_values = data[~invalid_mask]

        # pixel không hợp lệ
        invalid_points = np.column_stack((x[invalid_mask], y[invalid_mask]))

        if len(valid_values) > 0:
            # Nội suy nearest cho pixel không hợp lệ
            interpolated = griddata(valid_points, valid_values, invalid_points, method='nearest')
            data[invalid_mask] = interpolated
        else:
            # Nếu toàn bộ ảnh lỗi → gán 0 hết
            data = np.zeros_like(data)

    # --- 2. Xử lý theo loại dữ liệu ---
    if data_type == "Radar":
        # Radar không có giá trị âm → ép về >=0
        data = np.maximum(data, 0)

    else:
        # Min-max scaling cho ERA5 & Himawari
        data_min, data_max = np.min(data), np.max(data)

        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

    return data

def normalization():
    XhatinhPath = 'csv_data/x_hatinh.npy'
    yhatinhPath = 'csv_data/y_hatinh_new.npy'

    xHatinh = np.load(XhatinhPath)
    yhatinh= np.load(yhatinhPath)

    xHatinh = preprocess_data(xHatinh,'')
    yhatinh = preprocess_data(yhatinh,'Radar')

    print(xHatinh.shape)
    print(yhatinh.shape)

    print(np.min(xHatinh),'   ',np.max(xHatinh))
    print(np.min(yhatinh),'   ',np.max(yhatinh))

    np.save(XhatinhPath,xHatinh)
    np.save(yhatinhPath,yhatinh)


normalization()