import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def crop_to_china_sea(ds, lat_max=42, lon_min=103, lat_min=13, lon_max=130):
    """
    智能裁切函数：将数据集裁切到中国海及周边区域。
    """
    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    if (lats.max() <= lat_max + 1) and (lats.min() >= lat_min - 1) and \
            (lons.max() <= lon_max + 1) and (lons.min() >= lon_min - 1):
        print("  --> 数据已是目标区域范围，跳过裁切步骤。")
        return ds

    print(f"  --> 正在裁切数据至范围: N:{lat_max}, S:{lat_min}, W:{lon_min}, E:{lon_max} ...")

    if lats[0] > lats[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_slice = slice(lon_min, lon_max)

    ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    return ds_cropped


def plot_spatial_diff_map(gdas_nc_path, era5_nc_path, output_image_path, title_text, vmin=-5.0, vmax=5.0):
    """
    读取、裁切、比对并绘制空间差异热力图。
    """
    print(f"\n======================================")
    print(f"加载 GDAS: {gdas_nc_path}")
    ds_gdas = xr.open_dataset(gdas_nc_path)
    print(f"加载 ERA5: {era5_nc_path}")
    ds_era5 = xr.open_dataset(era5_nc_path)

    # 1. 裁切
    ds_gdas = crop_to_china_sea(ds_gdas)
    ds_era5 = crop_to_china_sea(ds_era5)

    # 2. 提取变量
    possible_t2m_names = ['t2m', 'temperature_2m', 'T2M', '2t']
    var_gdas = next((var for var in possible_t2m_names if var in ds_gdas.variables), None)
    if not var_gdas:
        raise ValueError(f"❌ GDAS 缺失气温变量: {list(ds_gdas.data_vars.keys())}")

    var_era5 = next((var for var in possible_t2m_names if var in ds_era5.variables), None)
    if not var_era5:
        raise ValueError(f"❌ ERA5 缺失气温变量: {list(ds_era5.data_vars.keys())}")

    print(f"  --> 匹配到气温变量: GDAS用了 '{var_gdas}', ERA5用了 '{var_era5}'")

    # 注意：这里给 GDAS 也加上了 [0]，防止出现时间维度不匹配
    t2m_gdas = ds_gdas[var_gdas][0] if 'time' in ds_gdas[var_gdas].dims else ds_gdas[var_gdas]
    t2m_era5 = ds_era5[var_era5][0] if 'time' in ds_era5[var_era5].dims else ds_era5[var_era5]
    t2m_era5 = t2m_era5[0]
    print(f"\n[数据检查 - GDAS]")
    print(f"  - 变量维度: {t2m_gdas.dims}, 形状: {t2m_gdas.shape}")
    print(f"  - 数值范围: [{float(t2m_gdas.min()):.2f}, {float(t2m_gdas.max()):.2f}]")

    print(f"\n[数据检查 - ERA5]")
    print(f"  - 变量维度: {t2m_era5.dims}, 形状: {t2m_era5.shape}")
    print(f"  - 数值范围: [{float(t2m_era5.min()):.2f}, {float(t2m_era5.max()):.2f}]")

    # 动态获取坐标名称以防报错
    lat_name_g = 'latitude' if 'latitude' in t2m_gdas.coords else 'lat'
    lon_name_g = 'longitude' if 'longitude' in t2m_gdas.coords else 'lon'
    lat_name_e = 'latitude' if 'latitude' in t2m_era5.coords else 'lat'
    lon_name_e = 'longitude' if 'longitude' in t2m_era5.coords else 'lon'

    # ================= 核心修改区：网格对齐 =================
    if not t2m_gdas.coords[lat_name_g].equals(t2m_era5.coords[lat_name_e]) or \
            not t2m_gdas.coords[lon_name_g].equals(t2m_era5.coords[lon_name_e]):
        print("\n⚠️ 警告: GDAS 与 ERA5 的经纬度网格不完全一致！")
        print("  --> 正在使用双线性插值 (Bilinear Interpolation) 将 ERA5 对齐至 GDAS 网格...")

        # 强制将 ERA5 的网格插值成和 GDAS 一模一样
        t2m_era5 = t2m_era5.interp_like(t2m_gdas, method='linear')

        print("  ✅ 插值完成！现在两者的 Shape 完全一致。")
    else:
        print("\n✅ 经纬度网格完美匹配。")
    # ========================================================

    # 3. 计算差异 (GDAS - ERA5)
    diff = t2m_gdas - t2m_era5

    if 'time' in diff.dims:
        diff = diff.squeeze('time')

    # 绘图设置
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(9, 8), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([103, 130, 13, 42], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', color='black', linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.6)

    levels = np.linspace(vmin, vmax, 21)

    plot = diff.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        levels=levels,
        extend='both',
        add_colorbar=False,
        add_labels=False
    )

    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', shrink=0.75, pad=0.08)
    cbar.set_label('Temperature Difference (K)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}

    if title_text:
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)

    plt.savefig(output_image_path, format='png', bbox_inches='tight')
    print(f"\n✅ 成功保存图片至: {output_image_path}")
    plt.close()


if __name__ == "__main__":
    V_MIN = -5.0
    V_MAX = 5.0

    GDAS_T24_PATH = PROJECT_ROOT / "model_output" / "gdas" / "2018-07-01-00-00" / "24" / "output_surface_2018-07-02-00-00.nc"
    ERA5_T24_PATH = PROJECT_ROOT / "src" / "trash" / "nc_data" / "era5" / "2018-07-01-00-00" / "24" / "surface.nc"
    OUT_B_PATH = PROJECT_ROOT / "src" / "figure" / "old" / "Fig3_3c_T24_Diff.png"

    plot_spatial_diff_map(
        gdas_nc_path=GDAS_T24_PATH,
        era5_nc_path=ERA5_T24_PATH,
        output_image_path=OUT_B_PATH,  # 修复了原来路径里的 '...'
        title_text='Time: 2018-07-2 00:00 UTC (T+24h)',
        vmin=V_MIN,
        vmax=V_MAX
    )



# import xarray as xr
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np
#
#
# def plot_spatial_diff_map(gdas_nc_path, era5_nc_path, output_image_path, title_text, vmin=-4.0, vmax=4.0):
#     """
#     读取 GDAS 和 ERA5 的 surface.nc 文件，计算 2米气温 (t2m) 的差异，并绘制带有海岸线的空间热力图。
#
#     参数:
#         gdas_nc_path: GDAS 数据 .nc 文件路径
#         era5_nc_path: ERA5 数据 .nc 文件路径
#         output_image_path: 保存图片的输出路径
#         title_text: 图表顶部的标题 (如果您要后期加文字，这里可以留空)
#         vmin, vmax: Colorbar 的固定量程，保证多张图颜色标准一致
#     """
#     print(f"正在加载数据...\nGDAS: {gdas_nc_path}\nERA5: {era5_nc_path}")
#
#     # 1. 读取 NetCDF 数据
#     ds_gdas = xr.open_dataset(gdas_nc_path)
#     ds_era5 = xr.open_dataset(era5_nc_path)
#
#     # 2. 提取 2米气温 (通常变量名为 't2m'，部分数据集可能为 'temperature_2m' 或 'T2M')
#     # 盘古模型的标准输入变量名通常是 t2m
#     var_name_gdas = 'temperature_2m'
#     var_name = 't2m'
#     t2m_gdas = ds_gdas[var_name_gdas][0]
#     t2m_era5 = ds_era5[var_name][0]
#
#     # 3. 计算差异 (GDAS - ERA5)
#     # xarray 会自动对齐经纬度坐标进行相减
#     diff = t2m_gdas - t2m_era5
#
#     # 如果数据有多余的时间维度 (time=1)，压缩掉它
#     if 'time' in diff.dims:
#         diff = diff.squeeze('time')
#
#     # ================= 绘图设置 =================
#     # 设置高水平学术期刊风格字体
#     plt.rcParams['font.family'] = 'Arial'
#
#     # 创建画板，指定地图投影为 PlateCarree (等距圆柱投影)
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#
#     # 4. 设置地图范围 (聚焦中国东部沿海及西北太平洋，海陆交界最明显)
#     # [西经, 东经, 南纬, 北纬] -> 您可以根据需要调整这个框
#     ax.set_extent([105, 135, 15, 45], crs=ccrs.PlateCarree())
#
#     # 5. 添加地图底图特征
#     # 添加海岸线，粗细设置为 1.2，颜色为黑色
#     ax.coastlines(resolution='50m', color='black', linewidth=1.2)
#     # 添加国界线 (可选)
#     ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.6)
#
#     # 6. 绘制差异等值线填充图 (Contourf)
#     # 使用 RdBu_r 色标 (红-白-蓝的翻转版，红色代表 GDAS 偏高，蓝色代表偏低)
#     levels = np.linspace(vmin, vmax, 21)  # 划分 21 个颜色层级使过渡平滑
#
#     # Xarray 的内置画图方法，结合 cartopy 非常方便
#     plot = diff.plot.contourf(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap='RdBu_r',  # 发散型色标
#         levels=levels,  # 锁定色标范围
#         extend='both',  # 超出范围的值显示为箭头
#         add_colorbar=False,  # 我们手动加 colorbar 以更好地控制位置
#         add_labels=False,
#     )
#
#     # 7. 添加和美化 Colorbar
#     # shrink 缩小比例，pad 是与主图的间距
#     cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', shrink=0.7, pad=0.08)
#     cbar.set_label('Temperature Difference (K)', fontsize=14, fontweight='bold')
#     cbar.ax.tick_params(labelsize=12)
#
#     # 8. 添加经纬度网格和刻度标签
#     gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False  # 关闭顶部的纬度标签
#     gl.right_labels = False  # 关闭右侧的经度标签
#     gl.xlabel_style = {'size': 12}
#     gl.ylabel_style = {'size': 12}
#
#     # 设置标题
#     if title_text:
#         ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
#
#     # 9. 保存出图
#     plt.savefig(output_image_path, format='png', bbox_inches='tight')
#     print(f"成功保存图片至: {output_image_path}\n")
#     plt.close()
#
#
# # ================= 运行区域 =================
#
# if __name__ == "__main__":
#     # 统一设定 Colorbar 的绝对范围
#     # 这样图 (b) 和图 (c) 的颜色才具有对比意义！
#     V_MIN = -5.0
#     V_MAX = 5.0
#
#     # --- 绘制图 3-3 (b) : 初始时刻 (T=0) 的域偏移 ---
#     # 根据您提供的路径
#     # GDAS_T0_PATH = PROJECT_ROOT / "model_input" / "single_time_point" / "gdas" / "2018-07-01-00-00" / "surface.nc"
#     # ERA5_T0_PATH = PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / "2018-07-01-00-00" / "surface.nc"
#     # OUT_B_PATH = PROJECT_ROOT / "src" / "figure" / "Fig3_3b_T0_Diff.png"
#     #
#     # plot_spatial_diff_map(
#     #     gdas_nc_path=GDAS_T0_PATH,
#     #     era5_nc_path=ERA5_T0_PATH,
#     #     output_image_path=OUT_B_PATH,
#     #     title_text='Time: 2018-07-01 00:00',
#     #     vmin=V_MIN,
#     #     vmax=V_MAX
#     # )
#
#     # --- 绘制图 3-3 (c) : T+6h 时刻的流形修复 ---
#     # 要生成图 c，您只需要准备 T+6h 的预测输出文件和对应的真值文件
#     # 取消下面代码的注释并修改路径即可：
#
#
#     GDAS_T6_PATH = PROJECT_ROOT / "model_output" / "gdas" / "2018-07-01-00-00" / "6" / "output_surface_2018-07-01-06-00.nc"
#     ERA5_T6_PATH = PROJECT_ROOT / "src" / "nc_data" / "era5" / "2018-07-01-00-00" / "6" / "surface.nc"
#     OUT_C_PATH = PROJECT_ROOT / "src" / "figure" / "Fig3_3c_T6_Diff.png"
#
#     plot_spatial_diff_map(
#         gdas_nc_path=GDAS_T6_PATH,
#         era5_nc_path=ERA5_T6_PATH,
#         output_image_path=OUT_C_PATH,
#         title_text='Time: 2018-07-01 06:00',
#         vmin=V_MIN,
#         vmax=V_MAX
#     )
