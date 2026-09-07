#!/usr/bin/env python3
"""Question 1: 2014 load characteristic analysis.

Reads:
  project_materials/Area1_Load.csv
  project_materials/Area2_Load.csv

Generates:
  project_materials/analysis/q1/Q1_ANALYSIS.md
  project_materials/analysis/q1/Q1_daily_metrics_2014.csv
  project_materials/analysis/q1/Q1_summary_statistics.csv
  project_materials/analysis/q1/Q1_load_duration_points.csv
  project_materials/analysis/q1/figures/*.svg
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]  # project_materials/
OUT_DIR = ROOT / "analysis" / "q1"
FIG_DIR = OUT_DIR / "figures"


def read_load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dates, values = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 97:
            raise ValueError(f"{path.name}: expected 97 columns, got {len(header)}")
        for row in reader:
            if not row:
                continue
            if len(row) != 97:
                raise ValueError(f"{path.name}: malformed row at {row[0] if row else '?'}")
            dates.append(row[0])
            values.append([float(v) for v in row[1:]])
    return np.asarray(dates), np.asarray(values, dtype=float)


def select_year(dates: np.ndarray, values: np.ndarray, year: str) -> tuple[np.ndarray, np.ndarray]:
    mask = np.char.startswith(dates, year)
    return dates[mask], values[mask]


def metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    daily_max = values.max(axis=1)
    daily_min = values.min(axis=1)
    daily_mean = values.mean(axis=1)
    return {
        "daily_max": daily_max,
        "daily_min": daily_min,
        "peak_valley": daily_max - daily_min,
        "load_factor": daily_mean / daily_max,
        "daily_mean": daily_mean,
    }


def describe(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "cv": float(x.std(ddof=1) / x.mean()),
        "min": float(x.min()),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(x.max()),
    }


def lag_metrics(values: np.ndarray, lag_days: int) -> tuple[float, float]:
    actual = values[lag_days:].ravel()
    lagged = values[:-lag_days].ravel()
    corr = float(np.corrcoef(actual, lagged)[0, 1])
    mape = float(np.mean(np.abs((actual - lagged) / actual)) * 100)
    return corr, mape


def ldc_value(ldc: np.ndarray, p: float) -> float:
    idx = min(len(ldc) - 1, int(round((p / 100) * (len(ldc) - 1))))
    return float(ldc[idx])


def save_hist(a1: np.ndarray, a2: np.ndarray, xlabel: str, path: Path, percent: bool = False) -> None:
    if percent:
        a1, a2 = a1 * 100, a2 * 100
    plt.figure(figsize=(8, 5))
    plt.hist(a1, bins=30, alpha=0.55, label="Area 1")
    plt.hist(a2, bins=30, alpha=0.55, label="Area 2")
    plt.xlabel(xlabel)
    plt.ylabel("Number of days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def generate_report(
    d1: np.ndarray, x1: np.ndarray, m1: dict[str, np.ndarray],
    d2: np.ndarray, x2: np.ndarray, m2: dict[str, np.ndarray],
    ldc1: np.ndarray, ldc2: np.ndarray,
) -> str:
    s1 = {k: describe(v) for k, v in m1.items()}
    s2 = {k: describe(v) for k, v in m2.items()}
    lag1_a1, lag1_mape_a1 = lag_metrics(x1, 1)
    lag1_a2, lag1_mape_a2 = lag_metrics(x2, 1)
    lag7_a1, lag7_mape_a1 = lag_metrics(x1, 7)
    lag7_a2, lag7_mape_a2 = lag_metrics(x2, 7)

    min_peak_date_a1 = d1[np.argmin(m1["daily_max"])]
    min_peak_date_a2 = d2[np.argmin(m2["daily_max"])]
    max_peak_date_a1 = d1[np.argmax(m1["daily_max"])]

    return f"""# Question 1 — 2014 年两地区负荷特征分析

## 1. 数据与指标口径

- 数据：`Area1_Load.csv`、`Area2_Load.csv`
- 时间：2014-01-01 至 2014-12-31
- 每日 96 个 15 min 采样点，共 365 × 96 = 35,040 个样本/地区
- 单位：MW
- 2014 年两个地区均未发现缺失值或非正负荷值
- 日负荷率 = 日平均负荷 / 日最高负荷

## 2. 总体负荷水平

| 指标 | Area 1 | Area 2 |
|---|---:|---:|
| 年平均负荷（MW） | {fmt(float(x1.mean()))} | {fmt(float(x2.mean()))} |
| 年最大负荷（MW） | {fmt(float(x1.max()))} | {fmt(float(x2.max()))} |
| 年最小负荷（MW） | {fmt(float(x1.min()))} | {fmt(float(x2.min()))} |
| 年平均负荷 / 年峰值 | {pct(float(x1.mean()/x1.max()))} | {pct(float(x2.mean()/x2.max()))} |

Area 2 的年平均负荷与年峰值均高于 Area 1，说明其总体负荷规模更大。

## 3. 四项日负荷指标分布

### 3.1 日最高负荷

| 统计量 | Area 1（MW） | Area 2（MW） |
|---|---:|---:|
| 均值 | {fmt(s1['daily_max']['mean'])} | {fmt(s2['daily_max']['mean'])} |
| 标准差 | {fmt(s1['daily_max']['std'])} | {fmt(s2['daily_max']['std'])} |
| CV | {pct(s1['daily_max']['cv'])} | {pct(s2['daily_max']['cv'])} |
| 中位数 | {fmt(s1['daily_max']['median'])} | {fmt(s2['daily_max']['median'])} |
| 最小值 | {fmt(s1['daily_max']['min'])} | {fmt(s2['daily_max']['min'])} |
| 最大值 | {fmt(s1['daily_max']['max'])} | {fmt(s2['daily_max']['max'])} |

Area 2 的日峰值 CV 更低，日峰值水平相对更稳定。两地区最大日峰值均出现在
**{max_peak_date_a1[:4]}-{max_peak_date_a1[4:6]}-{max_peak_date_a1[6:]}**；
最低日峰值分别出现在 **{min_peak_date_a1[:4]}-{min_peak_date_a1[4:6]}-{min_peak_date_a1[6:]}**
和 **{min_peak_date_a2[:4]}-{min_peak_date_a2[4:6]}-{min_peak_date_a2[6:]}**。

![Daily maximum distribution](figures/daily_max_distribution.svg)

### 3.2 日最低负荷

| 统计量 | Area 1（MW） | Area 2（MW） |
|---|---:|---:|
| 均值 | {fmt(s1['daily_min']['mean'])} | {fmt(s2['daily_min']['mean'])} |
| 标准差 | {fmt(s1['daily_min']['std'])} | {fmt(s2['daily_min']['std'])} |
| CV | {pct(s1['daily_min']['cv'])} | {pct(s2['daily_min']['cv'])} |
| 中位数 | {fmt(s1['daily_min']['median'])} | {fmt(s2['daily_min']['median'])} |
| 最小值 | {fmt(s1['daily_min']['min'])} | {fmt(s2['daily_min']['min'])} |
| 最大值 | {fmt(s1['daily_min']['max'])} | {fmt(s2['daily_min']['max'])} |

Area 2 的日最低负荷水平更高，但最低负荷的相对波动略大。

![Daily minimum distribution](figures/daily_min_distribution.svg)

### 3.3 日峰谷差

| 统计量 | Area 1（MW） | Area 2（MW） |
|---|---:|---:|
| 均值 | {fmt(s1['peak_valley']['mean'])} | {fmt(s2['peak_valley']['mean'])} |
| 标准差 | {fmt(s1['peak_valley']['std'])} | {fmt(s2['peak_valley']['std'])} |
| CV | {pct(s1['peak_valley']['cv'])} | {pct(s2['peak_valley']['cv'])} |
| 中位数 | {fmt(s1['peak_valley']['median'])} | {fmt(s2['peak_valley']['median'])} |
| 最小值 | {fmt(s1['peak_valley']['min'])} | {fmt(s2['peak_valley']['min'])} |
| 最大值 | {fmt(s1['peak_valley']['max'])} | {fmt(s2['peak_valley']['max'])} |

Area 2 的平均峰谷差更大，但其峰谷差 CV 明显更低：
Area 1 为 {pct(s1['peak_valley']['cv'])}，Area 2 为 {pct(s2['peak_valley']['cv'])}。
这说明 Area 2 的日内峰谷幅度虽大，但跨日期更稳定。

![Peak-valley distribution](figures/peak_valley_distribution.svg)

### 3.4 日负荷率

| 统计量 | Area 1 | Area 2 |
|---|---:|---:|
| 均值 | {pct(s1['load_factor']['mean'])} | {pct(s2['load_factor']['mean'])} |
| 标准差 | {pct(s1['load_factor']['std'])} | {pct(s2['load_factor']['std'])} |
| 中位数 | {pct(s1['load_factor']['median'])} | {pct(s2['load_factor']['median'])} |
| 最小值 | {pct(s1['load_factor']['min'])} | {pct(s2['load_factor']['min'])} |
| 最大值 | {pct(s1['load_factor']['max'])} | {pct(s2['load_factor']['max'])} |

Area 2 平均日负荷率略高，全天负荷相对日峰值更饱满。

![Load-factor distribution](figures/load_factor_distribution.svg)

## 4. 负荷持续曲线

负荷持续曲线将全年 35,040 个负荷点按从高到低排序，横轴为“负荷被超过的时间比例”。

![Load duration curve](figures/load_duration_curve.svg)

为了排除两个地区绝对负荷规模不同的影响，同时绘制峰值归一化曲线：

![Normalized load duration curve](figures/load_duration_curve_normalized.svg)

| 被超过时间比例 | Area 1（MW） | Area 2（MW） |
|---:|---:|---:|
| 1% | {fmt(ldc_value(ldc1,1))} | {fmt(ldc_value(ldc2,1))} |
| 5% | {fmt(ldc_value(ldc1,5))} | {fmt(ldc_value(ldc2,5))} |
| 10% | {fmt(ldc_value(ldc1,10))} | {fmt(ldc_value(ldc2,10))} |
| 25% | {fmt(ldc_value(ldc1,25))} | {fmt(ldc_value(ldc2,25))} |
| 50% | {fmt(ldc_value(ldc1,50))} | {fmt(ldc_value(ldc2,50))} |
| 75% | {fmt(ldc_value(ldc1,75))} | {fmt(ldc_value(ldc2,75))} |
| 90% | {fmt(ldc_value(ldc1,90))} | {fmt(ldc_value(ldc2,90))} |
| 95% | {fmt(ldc_value(ldc1,95))} | {fmt(ldc_value(ldc2,95))} |
| 99% | {fmt(ldc_value(ldc1,99))} | {fmt(ldc_value(ldc2,99))} |

Area 2 在绝对负荷上整体更高。归一化后，两地区主体区间较接近，但 Area 1 的极低负荷尾部下降更深，
说明特殊日期造成的极端低负荷状态更突出。

## 5. 两地区主要差异

**Area 1**
- 负荷规模较低；
- 日最高负荷相对波动更大；
- 日峰谷差跨日期变化明显更大；
- 负荷持续曲线低负荷尾部更深；
- 特殊日期对负荷水平影响更强。

**Area 2**
- 总体负荷规模更高；
- 日最高负荷相对更稳定；
- 峰谷差虽然更大，但峰谷结构更稳定；
- 平均日负荷率略高；
- 整体日内形态更规则。

## 6. 初步可预测性判断

**初步判断：Area 2 更可能获得更高的短期负荷预测精度。**

核心依据是 Area 2 的日峰值 CV 较低、峰谷差 CV 明显较低，并且极端低负荷尾部没有 Area 1 那么突出。

作为辅助验证，计算简单的历史滞后规律：

| 辅助指标 | Area 1 | Area 2 |
|---|---:|---:|
| 相邻日同一时刻相关系数 | {lag1_a1:.4f} | {lag1_a2:.4f} |
| 前 1 日直接预测 MAPE | {lag1_mape_a1:.2f}% | {lag1_mape_a2:.2f}% |
| 7 日滞后同一时刻相关系数 | {lag7_a1:.4f} | {lag7_a2:.4f} |
| 前 7 日直接预测 MAPE | {lag7_mape_a1:.2f}% | {lag7_mape_a2:.2f}% |

Area 2 的滞后相关性更高，简单历史持续法误差也更低，因此进一步支持其规律性更强。
该结论仍属于第 1 问阶段的初步判断，后续 Question 4/5 会用严格时间序列回测验证。

## 7. 输出文件

- `Q1_ANALYSIS.md`
- `Q1_daily_metrics_2014.csv`
- `Q1_summary_statistics.csv`
- `Q1_load_duration_points.csv`
- `q1_load_analysis.py`
- `figures/daily_max_distribution.svg`
- `figures/daily_min_distribution.svg`
- `figures/peak_valley_distribution.svg`
- `figures/load_factor_distribution.svg`
- `figures/load_duration_curve.svg`
- `figures/load_duration_curve_normalized.svg`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    d1_all, x1_all = read_load_csv(ROOT / "Area1_Load.csv")
    d2_all, x2_all = read_load_csv(ROOT / "Area2_Load.csv")
    d1, x1 = select_year(d1_all, x1_all, "2014")
    d2, x2 = select_year(d2_all, x2_all, "2014")

    if len(d1) != 365 or len(d2) != 365:
        raise ValueError("Expected 365 rows for 2014 in both areas.")
    if np.isnan(x1).any() or np.isnan(x2).any():
        raise ValueError("Missing load values detected in 2014.")
    if (x1 <= 0).any() or (x2 <= 0).any():
        raise ValueError("Non-positive load values detected in 2014.")

    m1, m2 = metrics(x1), metrics(x2)

    with (OUT_DIR / "Q1_daily_metrics_2014.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "date",
            "Area1_daily_max_MW", "Area1_daily_min_MW", "Area1_peak_valley_MW",
            "Area1_load_factor", "Area1_daily_mean_MW",
            "Area2_daily_max_MW", "Area2_daily_min_MW", "Area2_peak_valley_MW",
            "Area2_load_factor", "Area2_daily_mean_MW",
        ])
        for i, date in enumerate(d1):
            w.writerow([
                date,
                f"{m1['daily_max'][i]:.6f}", f"{m1['daily_min'][i]:.6f}",
                f"{m1['peak_valley'][i]:.6f}", f"{m1['load_factor'][i]:.8f}",
                f"{m1['daily_mean'][i]:.6f}",
                f"{m2['daily_max'][i]:.6f}", f"{m2['daily_min'][i]:.6f}",
                f"{m2['peak_valley'][i]:.6f}", f"{m2['load_factor'][i]:.8f}",
                f"{m2['daily_mean'][i]:.6f}",
            ])

    stat_order = ["mean", "std", "cv", "min", "p25", "median", "p75", "max"]
    with (OUT_DIR / "Q1_summary_statistics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "metric", *stat_order])
        for area, mm in [("Area1", m1), ("Area2", m2)]:
            for name, arr in mm.items():
                s = describe(arr)
                w.writerow([area, name, *[f"{s[k]:.8f}" for k in stat_order]])

    ldc1, ldc2 = np.sort(x1.ravel())[::-1], np.sort(x2.ravel())[::-1]
    points = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    with (OUT_DIR / "Q1_load_duration_points.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "duration_exceeded_percent", "Area1_load_MW", "Area2_load_MW",
            "Area1_pct_of_peak", "Area2_pct_of_peak",
        ])
        for p in points:
            a1, a2 = ldc_value(ldc1, p), ldc_value(ldc2, p)
            w.writerow([p, f"{a1:.6f}", f"{a2:.6f}",
                        f"{a1/ldc1[0]:.8f}", f"{a2/ldc2[0]:.8f}"])

    save_hist(m1["daily_max"], m2["daily_max"], "Daily maximum load (MW)",
              FIG_DIR / "daily_max_distribution.svg")
    save_hist(m1["daily_min"], m2["daily_min"], "Daily minimum load (MW)",
              FIG_DIR / "daily_min_distribution.svg")
    save_hist(m1["peak_valley"], m2["peak_valley"], "Daily peak-valley difference (MW)",
              FIG_DIR / "peak_valley_distribution.svg")
    save_hist(m1["load_factor"], m2["load_factor"], "Daily load factor (%)",
              FIG_DIR / "load_factor_distribution.svg", percent=True)

    idx = np.linspace(0, len(ldc1) - 1, 600).astype(int)
    duration = idx / (len(ldc1) - 1) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(duration, ldc1[idx], label="Area 1")
    plt.plot(duration, ldc2[idx], label="Area 2")
    plt.xlabel("Duration exceeded (%)")
    plt.ylabel("Load (MW)")
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "load_duration_curve.svg", format="svg")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(duration, ldc1[idx] / ldc1[0] * 100, label="Area 1")
    plt.plot(duration, ldc2[idx] / ldc2[0] * 100, label="Area 2")
    plt.xlabel("Duration exceeded (%)")
    plt.ylabel("Load as % of annual peak")
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "load_duration_curve_normalized.svg", format="svg")
    plt.close()

    report = generate_report(d1, x1, m1, d2, x2, m2, ldc1, ldc2)
    (OUT_DIR / "Q1_ANALYSIS.md").write_text(report, encoding="utf-8")

    print(f"Question 1 completed: {OUT_DIR}")


if __name__ == "__main__":
    main()
