# #!/usr/bin/env python3
# """
# copy_q_to_upper.py
#
# 将 b.nc 中的 q 变量拷贝到 upper.nc（支持 valid_time -> time 的映射）。
# 依赖: netCDF4, numpy
# """
#
# import sys
# import numpy as np
# from netCDF4 import Dataset
# import netCDF4 as nc
#
# # ========== 用户配置 ==========
# b_path = PROJECT_ROOT / "model_input" / "2018-09-11-00-00" / "upper.nc"  # 源文件（包含 q）
# upper_path = "nc/处理后/upper.nc"  # 目标 upper.nc
# varname = "q"                 # 源变量名（b.nc 中）
# overwrite_if_exists = True    # 如果 upper.nc 中已有 q，是否覆盖
# # ==============================
#
# # 维度名映射（源名 -> 目标名），常见映射在此添加
# DIM_MAP = {
#     "valid_time": "time",
#     # 如果有其他映射需要可在此加入，例如 "t":"time"
# }
#
# def map_dim_name(src_name):
#     return DIM_MAP.get(src_name, src_name)
#
# def main():
#     # 打开文件
#     src = Dataset(b_path, "r")
#     dst = Dataset(upper_path, "r+")
#
#     # 检查 q 是否存在于 b.nc
#     if varname not in src.variables:
#         src.close()
#         dst.close()
#         raise RuntimeError(f"{b_path} 中未找到变量 '{varname}'。")
#
#     src_var = src.variables[varname]
#     src_dims = tuple(src_var.dimensions)  # e.g. ('valid_time','pressure_level','latitude','longitude')
#     src_shape = tuple(src_var.shape)
#
#     # 映射源维度名到目标维度名
#     mapped_dims = tuple(map_dim_name(d) for d in src_dims)
#
#     # 检查目标文件是否包含这些维度（使用映射后的名字）
#     for d_src, d_dst, s in zip(src_dims, mapped_dims, src_shape):
#         if d_dst not in dst.dimensions:
#             src.close()
#             dst.close()
#             raise RuntimeError(f"目标文件 {upper_path} 中缺少维度 '{d_dst}'（源维度名为 '{d_src}'）。")
#         dst_len = dst.dimensions[d_dst].size
#         if dst_len != s:
#             src.close()
#             dst.close()
#             raise RuntimeError(
#                 f"维度长度不匹配：源 '{d_src}' 长度 {s}，目标 '{d_dst}' 长度 {dst_len}。"
#             )
#
#     # 准备目标变量（使用映射后的维度顺序）
#     if varname in dst.variables:
#         dst_var = dst.variables[varname]
#         # 目标已有变量时，确认其维度名顺序与我们希望的一致（或者形状一致）
#         dst_dims_existing = tuple(dst_var.dimensions)
#         if dst_var.shape != src_shape:
#             # 形状不一致 -> 报错
#             src.close()
#             dst.close()
#             raise RuntimeError(f"目标文件中已存在变量 '{varname}'，但形状 {dst_var.shape} 与源 {src_shape} 不匹配。")
#         if not overwrite_if_exists:
#             print(f"{upper_path} 中已存在变量 '{varname}' 且设置为不覆盖，退出。")
#             src.close()
#             dst.close()
#             return
#         print(f"目标文件中已存在变量 '{varname}'，将覆盖写入（overwrite_if_exists=True）。")
#         # 覆盖时尝试复制源属性到目标（覆盖）
#         try:
#             for attr in src_var.ncattrs():
#                 setattr(dst_var, attr, getattr(src_var, attr))
#         except Exception:
#             pass
#     else:
#         # 创建新变量，使用 float32 写入
#         dtype = 'f4'
#         dst_dims_tuple = mapped_dims  # (time, pressure_level, latitude, longitude)
#         dst_var = dst.createVariable(varname, dtype, dst_dims_tuple, zlib=True, complevel=4,
#                                      fill_value=np.float32(np.nan))
#         # 复制属性
#         for attr in src_var.ncattrs():
#             try:
#                 setattr(dst_var, attr, getattr(src_var, attr))
#             except Exception:
#                 pass
#         print(f"在 {upper_path} 中创建变量 '{varname}'（dtype=float32，dims={dst_dims_tuple}）。")
#
#     # 确定 src_var 和 dst_var 中 time/plev/lat/lon 在各自维度列表的位置，方便分片读取/写入
#     # 目标 dst_var.dimensions 可能与 mapped_dims 相同或不同顺序；我们按实际的维度顺序写入。
#     src_dim_list = list(src_dims)
#     dst_dim_list = list(dst_var.dimensions)
#
#     # 要写入的主要四个维度名称（映射后统一用目标维度名）
#     wanted = ['time', 'pressure_level', 'latitude', 'longitude']
#
#     # 在源中找到对应索引（使用原始源维度名）
#     # Note: 在 src 中，time 可能叫 valid_time；但 src_dim_list 保存原名，我们用它来索引 src_var
#     # 所以我们寻找在 src_dim_list 中对应于每个 wanted 的项（通过反向映射）
#     # 构造从 wanted -> src_dim_name 的映射：找出 src_dim_name s.t. map_dim_name(src_dim_name) == wanted_name
#     wanted_to_src_dim = {}
#     for w in wanted:
#         match = None
#         for src_d in src_dim_list:
#             if map_dim_name(src_d) == w:
#                 match = src_d
#                 break
#         if match is None:
#             src.close()
#             dst.close()
#             raise RuntimeError(f"在源变量维度中找不到可映射到 '{w}' 的维度（源维度列表: {src_dim_list}）")
#         wanted_to_src_dim[w] = match
#
#     # 得到在 src_var 中的索引位置
#     src_idx = {w: src_dim_list.index(w_src) for w, w_src in wanted_to_src_dim.items()}
#
#     # 在 dst_var 中得到索引位置（注意 dst 已经使用目标维度名）
#     dst_idx = {}
#     for w in wanted:
#         if w not in dst_dim_list:
#             src.close()
#             dst.close()
#             raise RuntimeError(f"目标变量维度中找不到 '{w}'（目标维度列表: {dst_dim_list}）")
#         dst_idx[w] = dst_dim_list.index(w)
#
#     # 获取维度长度
#     ntime = src_shape[src_idx['time']]
#     nplev = src_shape[src_idx['pressure_level']]
#     nlat = src_shape[src_idx['latitude']]
#     nlon = src_shape[src_idx['longitude']]
#     print(f"准备写入: ntime={ntime}, nplev={nplev}, nlat={nlat}, nlon={nlon}")
#
#     # 逐切片拷贝，按 time, pressure_level 循环
#     # 构造 src_index_template 和 dst_index_template（tuple），用 slice(None) 填充
#     src_ndim = src_var.ndim
#     dst_ndim = dst_var.ndim
#
#     # src template: list of slice(None) len src_ndim
#     src_template = [slice(None)] * src_ndim
#     dst_template = [slice(None)] * dst_ndim
#
#     for t in range(ntime):
#         for p in range(nplev):
#             # 设置 src 索引：在 src_var 对应 time/plev 的轴上设置具体索引
#             src_template_local = list(src_template)
#             src_template_local[src_idx['time']] = t
#             src_template_local[src_idx['pressure_level']] = p
#             # 读取 2D 切片 (lat, lon)
#             data_slice = src_var[tuple(src_template_local)].astype(np.float32)  # shape (nlat, nlon)
#
#             # 设置 dst 索引：在 dst_var 对应 time/plev 的轴上设置具体索引
#             dst_template_local = list(dst_template)
#             dst_template_local[dst_idx['time']] = t
#             dst_template_local[dst_idx['pressure_level']] = p
#             # 保证写入位置的维度数量正确（其他轴为 slice(None)）
#             # 写入
#             dst_var[tuple(dst_template_local)] = data_slice
#
#         # 可选进度提示（每个时间写完）
#         print(f"写入完成 time index {t+1}/{ntime}")
#
#     # flush & close
#     dst.sync()
#     src.close()
#     dst.close()
#     print("完成：已将 q 从 b.nc 写入到 upper.nc。")
#
#     upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
#     with nc.Dataset(PROJECT_ROOT / "gdas" / "nc" / "处理后" / "upper.nc") as nc_file:
#         upper_data[0] = nc_file.variables['z'][:].astype(np.float32)
#         upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
#         upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
#         upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
#         upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
#     np.save('../nc/处理后/2018091100/input_upper.npy', upper_data)
#     print("完成：已生成input_upper.npy")
#
#
# if __name__ == "__main__":
#     try:
#         main()
#         # Convert the upper air data to npy
#
#     except Exception as e:
#         print("出错：", e)
#         sys.exit(1)
#
# utils/upper_utils.py
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc

# 维度名映射
DIM_MAP = {
    "valid_time": "time",
}

def map_dim_name(src_name):
    return DIM_MAP.get(src_name, src_name)


def copy_q_to_upper(
    src_path,
    dst_path,
    varname="q",
    overwrite_if_exists=True,
    verbose=True,
):
    """
    将 src_path 中的 q 变量拷贝到 dst_path (upper.nc)

    Parameters
    ----------
    src_path : str
        含 q 的源 nc 文件（如 b.nc）
    dst_path : str
        目标 upper.nc
    varname : str
        变量名，默认 "q"
    overwrite_if_exists : bool
        目标已存在时是否覆盖
    verbose : bool
        是否打印进度
    """

    src = Dataset(src_path, "r")
    dst = Dataset(dst_path, "r+")

    if varname not in src.variables:
        raise RuntimeError(f"{src_path} 中未找到变量 '{varname}'")

    src_var = src.variables[varname]
    src_dims = tuple(src_var.dimensions)
    src_shape = src_var.shape

    mapped_dims = tuple(map_dim_name(d) for d in src_dims)

    # 检查维度
    for d_src, d_dst, s in zip(src_dims, mapped_dims, src_shape):
        if d_dst not in dst.dimensions:
            raise RuntimeError(f"目标缺少维度 '{d_dst}'")
        if dst.dimensions[d_dst].size != s:
            raise RuntimeError(
                f"维度长度不匹配：{d_src}={s}, {d_dst}={dst.dimensions[d_dst].size}"
            )

    # 创建 / 获取变量
    if varname in dst.variables:
        dst_var = dst.variables[varname]
        if dst_var.shape != src_shape:
            raise RuntimeError("目标变量已存在但形状不一致")
        if not overwrite_if_exists:
            if verbose:
                print(f"{varname} 已存在，未覆盖")
            src.close()
            dst.close()
            return
    else:
        dst_var = dst.createVariable(
            varname,
            "f4",
            mapped_dims,
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

    # 复制属性
    for attr in src_var.ncattrs():
        try:
            setattr(dst_var, attr, getattr(src_var, attr))
        except Exception:
            pass

    # 维度索引
    src_dim_list = list(src_dims)
    dst_dim_list = list(dst_var.dimensions)

    wanted = ["time", "pressure_level", "latitude", "longitude"]

    def find_src_dim(w):
        for d in src_dim_list:
            if map_dim_name(d) == w:
                return d
        raise RuntimeError(f"源中找不到维度 {w}")

    src_idx = {w: src_dim_list.index(find_src_dim(w)) for w in wanted}
    dst_idx = {w: dst_dim_list.index(w) for w in wanted}

    ntime = src_shape[src_idx["time"]]
    nplev = src_shape[src_idx["pressure_level"]]

    # 拷贝数据
    for t in range(ntime):
        for p in range(nplev):
            src_sel = [slice(None)] * src_var.ndim
            dst_sel = [slice(None)] * dst_var.ndim

            src_sel[src_idx["time"]] = t
            src_sel[src_idx["pressure_level"]] = p
            dst_sel[dst_idx["time"]] = t
            dst_sel[dst_idx["pressure_level"]] = p

            data = src_var[tuple(src_sel)].astype(np.float32)
            dst_var[tuple(dst_sel)] = data

        if verbose:
            print(f"time {t+1}/{ntime} 写入完成")

    dst.sync()
    src.close()
    dst.close()

    if verbose:
        print("q 已成功写入 upper.nc")
