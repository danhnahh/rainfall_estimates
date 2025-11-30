import glob
import pandas as pd
import numpy as np
from scripts.h2py import ignores


def common_timestamps(list_of_lists):
    """
    Trả về list các timestamp xuất hiện trong tất cả các list đầu vào.

    Args:
        list_of_lists: list các list, mỗi list chứa các timestamp

    Returns:
        list các timestamp xuất hiện trong tất cả các list, theo thứ tự sorted
    """
    if not list_of_lists:
        return []
    common_set = set(list_of_lists[0])  # Chuyển list đầu tiên thành set làm cơ sở
    for lst in list_of_lists[1:]:  # Lấy giao với các list tiếp theo
        common_set &= set(lst)  # intersection
    return sorted(common_set)


def create_tensor_form_csv(list_path):
    list_df = []
    for r in list_path:
        df = pd.read_csv(r)
        df["variable"] = df["variable"].astype(str)  # ép kiểu string
        list_df.append(df)

    list_timestamp_series = [df['timestamp'] for df in list_df]

    timestamp = common_timestamps(list_timestamp_series)

    list_df = [df[df['timestamp'].isin(timestamp)] for df in list_df]
    if len(list_path) > 1:
        list_df = [df[~df['variable'].isin(['2019', '2020'])] for df in list_df]

    # Lấy danh sách timestamp & variable theo thứ tự
    unique_vars = sorted(pd.concat([df["variable"] for df in list_df]).unique())
    unique_times = sorted(pd.concat([df["timestamp"] for df in list_df]).unique())

    # Map timestamp và variable về chỉ số
    time_to_index = {t: i for i, t in enumerate(unique_times)}
    var_to_index = {v: i for i, v in enumerate(unique_vars)}

    # Max row/col
    all_rows = pd.concat([df['row'] for df in list_df])
    all_cols = pd.concat([df['col'] for df in list_df])
    min_row, max_row = all_rows.min(), all_rows.max()
    min_col, max_col = all_cols.min(), all_cols.max()

    # Tạo array 4 chiều: time × variable × row × col
    a = np.zeros((len(unique_times),
                  len(unique_vars),
                  max_row - min_row + 1,
                  max_col - min_col + 1))

    # Đổ dữ liệu
    for df in list_df:
        for _, r in df.iterrows():
            t = time_to_index[r["timestamp"]]
            v = var_to_index[r["variable"]]
            a[t, v, r["row"] - min_row, r["col"] - min_col] = r["value"]

    return np.array(a)


def test2():
    create_tensor_form_csv()


def test1():
    # Load y trước
    list_file = ['csv_data/HIMA_hatinh.csv',
                 'csv_data/ERA5_hatinh.csv',
                 'csv_data/RADAR_hatinh.csv']

    y = create_tensor_form_csv(['csv_data/RADAR_hatinh.csv'])
    y_2019 = y[:, 0, :, :]
    y_2020 = y[:, 1, :, :]

    x = create_tensor_form_csv(list_file)
    # Giả sử x, y đã có
    np.save("csv_data/y_hatinh.npy", y)
    np.save("csv_data/x_hatinh.npy", x)


if __name__ == '__main__':
    test1()
    # test2()

# def load_data(path, keep_timestamps=None):
#     """
#     Nếu keep_timestamps không None → chỉ giữ các timestamp trong list đó.
#     """
#     files = glob.glob(path)
#     dfs = [pd.read_csv(f, sep=",") for f in files]
#     df_all = pd.concat(dfs, ignore_index=True)
#
#     # Lọc timestamp nếu cần
#     if keep_timestamps is not None:
#         df_all = df_all[df_all["timestamp"].isin(keep_timestamps)]
#
#     # Lấy danh sách timestamp & variable theo thứ tự
#     unique_times = sorted(df_all["timestamp"].unique())
#     unique_vars = sorted(df_all["variable"].unique())
#
#     # Map timestamp và variable về chỉ số
#     time_to_index = {t: i for i, t in enumerate(unique_times)}
#     var_to_index = {v: i for i, v in enumerate(unique_vars)}
#
#     # Max row/col
#     max_row = df_all["row"].max()
#     max_col = df_all["col"].max()
#
#     min_row = df_all["row"].min()
#     min_col = df_all["col"].min()
#
#     # Tạo array 4 chiều: time × variable × row × col
#     a = np.zeros((len(unique_times),
#                   len(unique_vars),
#                   max_row - min_row + 1,
#                   max_col - min_col + 1))
#
#     # Đổ dữ liệu
#     for _, r in df_all.iterrows():
#         t = time_to_index[r["timestamp"]]
#         v = var_to_index[r["variable"]]
#         a[t, v, r["row"] - min_row, r["col"] - min_col] = r["value"]
#
#     return np.array(a), unique_times
