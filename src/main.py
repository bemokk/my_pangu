from infer import *
from datetime import datetime, timedelta

pred_start_time = datetime(
    year=2018,
    month=7,
    day=25,
    hour=0,
    minute=0)

data_type = "era5"  # 填入era5或gdas

time_step = [1, 3, 6, 12, 24, 48, 72]

#预测五天后，5*24=120, 预测从当天开始，就填0
time_diff = 120

for i in range(len(time_step)):
    time_step[i]= time_step[i] + time_diff


for ts in time_step:
    infer(pred_start_time, ts, data_type)

