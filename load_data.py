import glob
import pandas as pd
import numpy as np

def load_data(path, keep_timestamps=None):
    """
    Nếu keep_timestamps không None → chỉ giữ các timestamp trong list đó.
    """
    files = glob.glob(path)
    dfs = [pd.read_csv(f, sep=",") for f in files]
    df_all = pd.concat(dfs, ignore_index=True)

    # Lọc timestamp nếu cần
    if keep_timestamps is not None:
        df_all = df_all[df_all["timestamp"].isin(keep_timestamps)]

    # Lấy danh sách timestamp & variable theo thứ tự
    unique_times = sorted(df_all["timestamp"].unique())
    unique_vars = sorted(df_all["variable"].unique())

    # Map timestamp và variable về chỉ số
    time_to_index = {t: i for i, t in enumerate(unique_times)}
    var_to_index = {v: i for i, v in enumerate(unique_vars)}

    # Max row/col
    max_row = df_all["row"].max()
    max_col = df_all["col"].max()

    # Tạo array 4 chiều: time × variable × row × col
    a = np.zeros((len(unique_times),
                  len(unique_vars),
                  max_row + 1,
                  max_col + 1))

    # Đổ dữ liệu
    for _, r in df_all.iterrows():
        t = time_to_index[r["timestamp"]]
        v = var_to_index[r["variable"]]
        a[t, v, r["row"], r["col"]] = r["value"]

    return np.array(a), unique_times


if __name__ == '__main__':
    # Load y trước
    y, y_times = load_data('csv_data/RADAR_hatinh.csv')
    y_2019 = y[:, 0, :, :]
    y_2020 = y[:, 1, :, :]

    # Load x_IMERG_E, chỉ giữ các timestamp có trong y
    x_IMERG_E, _ = load_data('csv_data/IMERG_E_hatinh.csv', keep_timestamps=y_times)
    x_radar, _ = load_data('csv_data/RADAR_hatinh.csv', keep_timestamps=y_times)
    x = np.concatenate([x_IMERG_E, x_radar], axis=1)

    # Nếu muốn kiểm tra đồng bộ
    print("y timestamps:", y_times[:10])

    # Giả sử x, y đã có
    np.save("csv_data/y_hatinh.npy", y)
    np.save("csv_data/x_hatinh.npy", x)

