"""执行盘古天气模型 GPU 推理。运行前只需修改“用户配置区”。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import inference  # noqa: E402


# =============================================================================
# 用户配置区：只修改这一段
# =============================================================================

# 输入数据类型只能填写 "gdas" 或 "era5"，必须与 download.py 下载的数据一致。
DATA_TYPE = "era5"

# 推理起止时刻，必须与 model_input/single_time_point/ 下的目录名对应。
# 只预测一个初始时刻时，让 START_TIME 和 END_TIME 完全相同。
START_TIME = datetime(2026, 6, 20, 0, 0)
END_TIME = datetime(2026, 6, 20, 0, 0)

# 需要预测的未来时效，单位为小时；每一项必须是正整数。
# FORECAST_HOURS = [3, 12, 72]
FORECAST_HOURS = [3, 6, 9]
# FORECAST_HOURS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36,
#                   39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72]

# 批量初始场之间的时间间隔，单位为小时；每天一个初始场填写 24。
INITIAL_TIME_INTERVAL_HOURS = 24

# True：某个任务失败后立即停止；False：记录失败并继续后面的任务。
STOP_ON_ERROR = True

# =============================================================================
# 用户配置区结束：下面的代码通常不需要修改
# 本程序固定使用 GPU，不提供 CPU/GPU 切换选项。
# =============================================================================


def validate_config() -> None:
    if DATA_TYPE not in {"era5", "gdas"}:
        raise ValueError('DATA_TYPE 只能填写 "era5" 或 "gdas"')
    if not isinstance(START_TIME, datetime) or not isinstance(END_TIME, datetime):
        raise TypeError("START_TIME 和 END_TIME 必须使用 datetime(年, 月, 日, 时, 分)")
    if END_TIME < START_TIME:
        raise ValueError("END_TIME 不能早于 START_TIME")
    if (
        not isinstance(INITIAL_TIME_INTERVAL_HOURS, int)
        or INITIAL_TIME_INTERVAL_HOURS < 1
    ):
        raise ValueError("INITIAL_TIME_INTERVAL_HOURS 必须是正整数")
    if not isinstance(FORECAST_HOURS, (list, tuple)) or not FORECAST_HOURS:
        raise ValueError("FORECAST_HOURS 必须是非空列表，例如 [3, 12, 72]")
    if any(not isinstance(hour, int) or hour < 1 for hour in FORECAST_HOURS):
        raise ValueError("FORECAST_HOURS 中的每一项都必须是正整数")
    if not isinstance(STOP_ON_ERROR, bool):
        raise TypeError("STOP_ON_ERROR 只能填写 True 或 False，不要加引号")


def main() -> int:
    validate_config()
    inference.use_GPU = True

    current = START_TIME
    task_count = 0
    failures: list[tuple[datetime, int, str]] = []
    forecast_hours = list(dict.fromkeys(FORECAST_HOURS))

    while current <= END_TIME:
        for forecast_hour in forecast_hours:
            task_count += 1
            print(
                f"\n[开始] {DATA_TYPE} 初始时刻={current:%Y-%m-%d %H:%M} "
                f"预报时效={forecast_hour}h"
            )
            try:
                inference.infer(current, forecast_hour, DATA_TYPE)
            except Exception as exc:
                failures.append((current, forecast_hour, str(exc)))
                print(f"[失败] {exc}")
                if STOP_ON_ERROR:
                    raise
        current += timedelta(hours=INITIAL_TIME_INTERVAL_HOURS)

    print(f"\n全部结束：任务数={task_count}，失败数={len(failures)}")
    for init_time, forecast_hour, message in failures:
        print(f"  {init_time:%Y-%m-%d %H:%M} T+{forecast_hour}h：{message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
