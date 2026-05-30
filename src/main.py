from infer import *
from datetime import datetime, timedelta

# 数据类型
data_type = "era5"  # era5 或 gdas

# 预报时效
time_step = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72]

# 如果预测从当天开始，time_diff = 0
# 如果想整体后移，例如从 24h 开始，则 time_diff = 24
time_diff = 120

time_step = [ts + time_diff for ts in time_step]

# 推理日期范围：
start_time = datetime(year=2025, month=6, day=27, hour=0, minute=0)
end_time = datetime(year=2025, month=7, day=26, hour=0, minute=0)

current_time = start_time

success_list = []
failed_list = []

while current_time <= end_time:
    print("=" * 80)
    print(f"开始推理：{current_time.strftime('%Y-%m-%d %H:%M')}")

    try:
        for ts in time_step:
            print(f"正在推理：{current_time.strftime('%Y-%m-%d %H:%M')}  预报时效: {ts} h")
            infer(current_time, ts, data_type)

        print(f"完成推理：{current_time.strftime('%Y-%m-%d %H:%M')}")
        success_list.append(current_time.strftime("%Y-%m-%d %H:%M"))

    except Exception as e:
        print(f"推理失败：{current_time.strftime('%Y-%m-%d %H:%M')}")
        print("错误信息：", e)
        failed_list.append(current_time.strftime("%Y-%m-%d %H:%M"))

    current_time += timedelta(days=1)

print("\n" + "=" * 80)
print("全部推理结束")
print(f"成功数量：{len(success_list)}")
print(f"失败数量：{len(failed_list)}")

if failed_list:
    print("\n失败日期：")
    for item in failed_list:
        print(item)


















# from infer import *
# from datetime import datetime, timedelta
#
# pred_start_time = datetime(
#     year=2025,
#     month=7,
#     day=2,
#     hour=0,
#     minute=0)
#
# data_type = "gdas"  # 填入era5或gdas
#
# time_step = [1, 3, 6, 12, 24, 48, 72]
#
# #预测五天后，5*24=120, 预测从当天开始，就填0
# time_diff = 0
#
# for i in range(len(time_step)):
#     time_step[i]= time_step[i] + time_diff
#
#
# for ts in time_step:
#     infer(pred_start_time, ts, data_type)
#
