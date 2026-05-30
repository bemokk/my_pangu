import os
import sys
import subprocess
from datetime import datetime, timedelta

# Python 解释器，使用当前 conda 环境中的 python
python_exe = sys.executable

# GRIB2 文件所在目录
grib_dir = r"E:\PyCharm_WorkSpace\pangu\gdas\grib2\fnl"

# 两个转换脚本路径
surface_script = r"E:\PyCharm_WorkSpace\pangu\gdas\select_surface_variables.py"
upper_script = r"E:\PyCharm_WorkSpace\pangu\gdas\select_upper_variables.py"

# 日期范围：2025年7月1日 至 2025年7月30日
start_date = datetime(2025, 7, 31)
end_date = datetime(2025, 7, 31)

current_date = start_date

success_files = []
failed_files = []
missing_files = []

while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")

    grib_file = os.path.join(
        grib_dir,
        f"gdas1.fnl0p25.{date_str}00.f00.grib2"
    )

    print("=" * 80)
    print(f"正在处理：{date_str}00")
    print(f"GRIB 文件：{grib_file}")

    if not os.path.exists(grib_file):
        print(f"文件不存在，跳过：{grib_file}")
        missing_files.append(grib_file)
        current_date += timedelta(days=1)
        continue

    try:
        print("\n开始转换 surface.nc 和 input_surface.npy ...")
        subprocess.run(
            [
                python_exe,
                surface_script,
                "--grib_file",
                grib_file
            ],
            check=True
        )

        print("\n开始转换 upper.nc 和 input_upper.npy ...")
        subprocess.run(
            [
                python_exe,
                upper_script,
                "--grib_file",
                grib_file
            ],
            check=True
        )

        print(f"\n{date_str}00 转换完成")
        success_files.append(grib_file)

    except subprocess.CalledProcessError as e:
        print(f"\n{date_str}00 转换失败")
        print(f"失败脚本返回码：{e.returncode}")
        failed_files.append(grib_file)

    current_date += timedelta(days=1)

print("\n" + "=" * 80)
print("批量转换结束")
print(f"成功数量：{len(success_files)}")
print(f"失败数量：{len(failed_files)}")
print(f"缺失数量：{len(missing_files)}")

if failed_files:
    print("\n转换失败的文件：")
    for f in failed_files:
        print(f)

if missing_files:
    print("\n缺失的文件：")
    for f in missing_files:
        print(f)