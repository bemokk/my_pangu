"""下载 ERA5/GDAS，并转换为盘古模型输入。运行前只需修改“用户配置区”。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import era5, gdas  # noqa: E402


# =============================================================================
# 用户配置区：只修改这一段
# =============================================================================

# 数据源只能填写 "gdas" 或 "era5"，注意保留英文引号。
DATA_TYPE = "era5"

# 下载起止时刻，格式为 datetime(年, 月, 日, 时, 分)。
# 只下载一个时刻时，让 START_TIME 和 END_TIME 完全相同。
# GDAS 支持每天 00、06、12、18 UTC，分必须填 0。
START_TIME = datetime(2026, 5, 1, 0, 0)
END_TIME = datetime(2026, 5, 20, 0, 0)

# 相邻下载时次之间的小时数。
# GDAS 通常填 6（逐时次）或 24（每天同一时次），必须是 6 的倍数。
# ERA5 可按需求填写，例如 1、6、12、24。
TIME_INTERVAL_HOURS = 24

# =============================================================================
# 用户配置区结束：下面的代码通常不需要修改
# =============================================================================


def validate_config() -> None:
    if DATA_TYPE not in {"era5", "gdas"}:
        raise ValueError('DATA_TYPE 只能填写 "era5" 或 "gdas"')
    if not isinstance(START_TIME, datetime) or not isinstance(END_TIME, datetime):
        raise TypeError("START_TIME 和 END_TIME 必须使用 datetime(年, 月, 日, 时, 分)")
    if END_TIME < START_TIME:
        raise ValueError("END_TIME 不能早于 START_TIME")
    if not isinstance(TIME_INTERVAL_HOURS, int) or TIME_INTERVAL_HOURS < 1:
        raise ValueError("TIME_INTERVAL_HOURS 必须是正整数")
    if DATA_TYPE == "gdas":
        for name, value in [("START_TIME", START_TIME), ("END_TIME", END_TIME)]:
            if value.hour not in gdas.SUPPORTED_CYCLES:
                raise ValueError(
                    f"{name}：GDAS 时只能填写 0、6、12、18，当前为 {value.hour}"
                )
            if value.minute != 0 or value.second != 0 or value.microsecond != 0:
                raise ValueError(f"{name}：GDAS 时次的分和秒必须为 0")
        if TIME_INTERVAL_HOURS % 6 != 0:
            raise ValueError("GDAS 的 TIME_INTERVAL_HOURS 必须是 6 的正整数倍")


def main() -> int:
    validate_config()

    try:
        if DATA_TYPE == "era5":
            current = START_TIME
            prepared: list[Path] = []
            while current <= END_TIME:
                prepared.append(era5.download_and_prepare(current))
                current += timedelta(hours=TIME_INTERVAL_HOURS)
        else:
            prepared = gdas.download_and_prepare(
                START_TIME,
                END_TIME,
                interval_hours=TIME_INTERVAL_HOURS,
            )
    except gdas.GDASCycleUnavailable as exc:
        print("\n[退出] 请求的 GDAS 时次当前不可用。")
        print(exc)
        print("请稍后重试，或把 START_TIME/END_TIME 改为已经发布的较早时次。")
        return 2

    print(f"\n已准备 {len(prepared)} 个初始时刻：")
    for path in prepared:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
