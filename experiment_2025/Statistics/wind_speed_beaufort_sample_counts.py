"""Export effective sample counts by observed Beaufort wind-force group."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STATS_ROOT = Path(__file__).resolve().parent
DATA_ROOT = STATS_ROOT / "data"
INPUT_CSV = DATA_ROOT / "wind_speed_metrics_by_beaufort_2025.csv"
LEADS = (24, 48, 72, 96, 120)
GROUPS = ("<=2", "3", "4", "5", "6", "7", ">=8")
EXPERIMENTS = ("E0", "G0", "E120", "G120")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    frame = pd.read_csv(INPUT_CSV)
    frame = frame[frame["lead_hour"].isin(LEADS) & frame["beaufort_group"].isin(GROUPS)].copy()
    frame["beaufort_group"] = pd.Categorical(frame["beaufort_group"], categories=GROUPS, ordered=True)
    frame = frame.sort_values(["lead_hour", "beaufort_group"])

    count_columns = [f"n_{name}" for name in EXPERIMENTS]
    if not (frame[count_columns].nunique(axis=1) == 1).all():
        raise ValueError("The effective sample counts differ across experiments.")

    result = frame[["lead_hour", "beaufort_group", "n_E0"]].rename(columns={"n_E0": "n"})
    wide = result.pivot(index="lead_hour", columns="beaufort_group", values="n").reindex(
        index=LEADS, columns=GROUPS
    )
    wide.index.name = "lead_hour"
    wide.columns.name = None
    wide = wide.astype("Int64")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(DATA_ROOT / "wind_speed_beaufort_sample_counts_2025.csv", encoding="utf-8-sig")

    lines = [
        "### 不同蒲福风级的有效样本量",
        "",
        "| 时效/h | <=2 | 3 | 4 | 5 | 6 | 7 | >=8 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lead, row in wide.iterrows():
        values = " | ".join(str(int(row[group])) for group in GROUPS)
        lines.append(f"| {int(lead)} | {values} |")
    (DATA_ROOT / "wind_speed_beaufort_sample_counts_2025.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    summary = {
        "source": str(INPUT_CSV),
        "leads": list(LEADS),
        "beaufort_groups": list(GROUPS),
        "experiments": list(EXPERIMENTS),
        "counts_equal_across_experiments": True,
        "table": wide.reset_index().to_dict(orient="records"),
    }
    (DATA_ROOT / "wind_speed_beaufort_sample_counts_2025_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
