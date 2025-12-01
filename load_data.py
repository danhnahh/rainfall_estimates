import pandas as pd
import numpy as np
from tqdm import tqdm


def create_x_y_from_csv(list_path):
    """
    Tạo tensor X và Y từ danh sách CSV
    - Band '2019' hoặc '2020' → 'y' (target)
    - Các band còn lại → X (input)
    - Chỉ giữ timestamp đầy đủ band
    - Row/Col tạo hình chữ nhật từ min → max
    """

    # 1) Đọc CSV và gộp
    dfs = []
    print("[B1] Đọc CSV...")
    for p in tqdm(list_path, desc="Đọc file CSV"):
        df = pd.read_csv(p)
        df["variable"] = df["variable"].astype(str)
        df.loc[df["variable"].isin(['2019','2020']), "variable"] = 'y'
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # 2) Lấy unique timestamp và band
    all_timestamps = sorted(df_all["timestamp"].unique())
    all_bands = sorted(df_all["variable"].unique())

    # 3) Min/Max row/col → tạo hình chữ nhật
    min_row, max_row = df_all["row"].min(), df_all["row"].max()
    min_col, max_col = df_all["col"].min(), df_all["col"].max()
    n_row = max_row - min_row + 1
    n_col = max_col - min_col + 1

    # 4) Chỉ giữ timestamp đầy đủ band
    print("[B2] Lọc timestamp đầy đủ band...")
    ts_valid = []
    for t in tqdm(all_timestamps, desc="Kiểm tra timestamp"):
        sub = df_all[df_all["timestamp"] == t]["variable"].unique()
        if set(sub) == set(all_bands):
            ts_valid.append(t)

    # 5) Tạo tensor
    tensor = np.zeros((len(ts_valid), len(all_bands), n_row, n_col), dtype=float)
    t_to_idx = {t: i for i, t in enumerate(ts_valid)}
    b_to_idx = {b: i for i, b in enumerate(all_bands)}

    # 6) Tối ưu đổ dữ liệu bằng vectorization
    print("[B3] Đổ dữ liệu vào Tensor...")
    df_valid = df_all[df_all["timestamp"].isin(ts_valid)].copy()
    df_valid["t_idx"] = df_valid["timestamp"].map(t_to_idx)
    df_valid["b_idx"] = df_valid["variable"].map(b_to_idx)
    df_valid["r_idx"] = df_valid["row"] - min_row
    df_valid["c_idx"] = df_valid["col"] - min_col

    # Sử dụng tqdm để hiển thị tiến độ theo chunks
    for start in tqdm(range(0, len(df_valid), 10000), desc="Đổ dữ liệu (chunks)"):
        chunk = df_valid.iloc[start:start+10000]
        tensor[chunk["t_idx"].values,
               chunk["b_idx"].values,
               chunk["r_idx"].values,
               chunk["c_idx"].values] = chunk["value"].values

    # 7) Tách band 'y' ra làm target
    y_idx = b_to_idx['y']
    y = tensor[:, y_idx, :, :]          # shape = (timestamp, row, col)
    x_indices = [i for i, b in enumerate(all_bands) if b != 'y']
    x = tensor[:, x_indices, :, :]      # shape = (timestamp, band_except_y, row, col)

    return x, y, ts_valid, [b for b in all_bands if b != 'y'], (min_row, max_row), (min_col, max_col)


if __name__ == "__main__":
    list_file = [
        'csv_data/HIMA_hatinh.csv',
        'csv_data/ERA5_hatinh.csv',
        'csv_data/RADAR_hatinh.csv'
    ]

    x, y, timestamps, x_bands, row_range, col_range = create_x_y_from_csv(list_file)

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    print("[B4] Lưu tensor...")
    np.save("csv_data/x_hatinh.npy", x)
    np.save("csv_data/y_hatinh.npy", y)
