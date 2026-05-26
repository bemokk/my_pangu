# import numpy as np
#
# a = np.load(r"E:\pyCharmProject\pangu\model_input\2025-08-01-00-00\input_surface.npy")  # shape (4, 721, 1440)
# b = np.load(r"E:\pyCharmProject\pangu\model_input\2025-08-01-00-00\input_surface.npy")
#
# # 展平成一维
# a_flat = a.reshape(-1)
# b_flat = b.reshape(-1)
#
# corr = np.corrcoef(a_flat, b_flat)[0, 1]
# print("Correlation:", corr)
import numpy as np
import pandas as pd
import os

path_a = r"E:\pyCharmProject\pangu\gdas\nc\processed\2018102300\input_upper.npy"
path_b = r"/model_input/single_time_point/era5\2018-10-23-00-00\input_upper.npy"
out_dir = r"E:\pyCharmProject\pangu\compare_output"  # 输出目录（会创建）
os.makedirs(out_dir, exist_ok=True)

a = np.load(path_a)
b = np.load(path_b)

if a.shape != b.shape:
    raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")

if a.ndim < 2:
    raise ValueError("数组维度 < 2，无法进行 2D 比较")

# -------- 核心思想 --------
# 把 (..., H, W) 统一展平成 (N, H, W)
H, W = a.shape[-2:]
lead_shape = a.shape[:-2]          # 例如 (5, 13)
num_fields = np.prod(lead_shape)   # 5 * 13 = 65

a_2d = a.reshape(num_fields, H, W)
b_2d = b.reshape(num_fields, H, W)

print(f"展开为 {num_fields} 个 2D 矩阵（每个 {H}×{W}）")

# 用于把线性 index 反解为原始 (var, level)
def unravel_index(idx, shape):
    return np.unravel_index(idx, shape)

rows = []

for k in range(num_fields):
    a_slice = a_2d[k].astype(float)
    b_slice = b_2d[k].astype(float)
    diff = b_slice - a_slice

    mask = np.isfinite(a_slice) & np.isfinite(b_slice)
    valid = int(mask.sum())

    orig_idx = unravel_index(k, lead_shape)  # 如 (var, level)

    if valid == 0:
        stats = dict(
            field=k,
            orig_index=str(orig_idx),
            valid_count=0,
            mean_diff=np.nan,
            mae=np.nan,
            rmse=np.nan,
            std_diff=np.nan,
            corr=np.nan,
        )
    else:
        d = diff[mask]
        a_v = a_slice[mask]
        b_v = b_slice[mask]

        corr = (
            np.corrcoef(a_v, b_v)[0, 1]
            if a_v.size > 1 and np.std(a_v) > 0 and np.std(b_v) > 0
            else np.nan
        )

        stats = dict(
            field=k,
            orig_index=str(orig_idx),
            valid_count=valid,
            mean_diff=float(d.mean()),
            mae=float(np.abs(d).mean()),
            rmse=float(np.sqrt(np.mean(d**2))),
            std_diff=float(d.std()),
            corr=float(corr),
        )

        # # 可选：保存每个差值场
        # np.save(
        #     os.path.join(out_dir, f"diff_field_{k:03d}.npy"),
        #     diff
        # )

    rows.append(stats)

    print(
        f"[{k:02d}] idx={orig_idx} "
        f"RMSE={stats['rmse']:.3g} "
        f"corr={stats['corr']:.3g}"
    )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, "surface_diff_summary.csv"), index=False)
print("完成，结果已保存")
