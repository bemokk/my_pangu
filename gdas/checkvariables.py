# # import pygrib
# #
# # # 打开文件
# # grb_file = pygrib.open(r'E:\pyCharmProject\pangu\gdas\grib2\fnl\gdas1.fnl0p25.2018102300.f00.grib2')
# #
# # # 打印文件基本信息
# # print(f"文件中共有 {grb_file.messages} 条记录")
# #
# # # 查看每条记录的具体信息
# # print("\n详细信息：")
# # for i, grb in enumerate(grb_file, 1):
# #     print(f"第{i}条记录：")
# #     print(f"  变量名: {grb.name}")
# #     print(f"  参数ID: {grb.parameterNumber}")
# #     print(f"  数据类型: {grb.typeOfLevel}")
# #     print(f"  层级: {grb.level}")
# #     print(f"  时间: {grb.validDate}")
# #     print(f"  short: {grb.validDate}")
# #     print("-" * 30)
# #
# # grb_file.close()
#
#
# #265 266 260 351
#
#
#
# import pygrib
#
# grbs = pygrib.open(r"E:\pyCharmProject\pangu\model_input\multiple_time_point\era5\q_2018-07-01-05-10-15-20-25-30.nc")
#
# pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']
# pressure_levels = [int(i) for i in pressure_levels]
# # print(pressure_levels)
#
# # for i, m in enumerate(grbs):
# #     if m.shortName in ("10u","10v","2t","prmsl"):
# #         print(
# #             i,
# #             m.shortName,
# #             m.typeOfLevel,
# #             m.level,
# #             m.name
# #         )
#
# j = 0
#
# for i, m in enumerate(grbs):
#     if m.shortName in ("gh", "q", "t","u","v"):
#         if m.level in pressure_levels and m.typeOfLevel == 'isobaricInhPa':
#             print(
#                 i,
#                 m.shortName,
#                 m.typeOfLevel,
#                 m.level,
#                 m.name
#             )
#             j+=1
#             #满足条件的写入upper.nc，按照
#
# print(j)
import netCDF4 as nc

nc_file = r"E:\pyCharmProject\pangu\model_input\multiple_time_point\era5\q_2018-07-01-05-10-15-20-25-30.nc"  # 替换为你的文件路径

with nc.Dataset(nc_file) as f:
    # 查看所有维度的名称和大小
    print("文件维度信息：")
    for dim_name, dim_obj in f.dimensions.items():
        print(f"{dim_name}: {len(dim_obj)}")

    # 打印具体的时间变量值（通常是数字，需要结合 units 属性转换）
    if 'valid_time' in f.variables:
        print("\n时间变量的原始值：", f.variables['valid_time'][:])
        print("时间变量的单位：", f.variables['valid_time'].units)