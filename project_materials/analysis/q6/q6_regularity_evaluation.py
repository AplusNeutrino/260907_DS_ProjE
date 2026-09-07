#!/usr/bin/env python3
"""Question 6: comprehensive comparison of load regularity between Area 1 and Area 2.

The script combines already-verified evidence from Q2/Q4/Q5 with additional
leakage-safe structural diagnostics computed directly from the historical load data.
It does not use any Jan 4-10 2015 actual load values.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "q6"
FIG = OUT / "figures"
SLOTS = 96


def parse_date(x: str) -> datetime:
    x = x.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    raise ValueError(x)


def read_load(area: str):
    path = ROOT / f"{area}_Load.csv"
    dates, values = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if len(header) != 97:
            raise ValueError(f"{path.name}: expected 97 columns")
        for row in r:
            if not row:
                continue
            dates.append(parse_date(row[0]))
            values.append([float(x) for x in row[1:]])
    return dates, np.asarray(values, dtype=float)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(pred - actual) / actual) * 100)


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(pred - actual)))


def structural_metrics(area: str):
    dates, values = read_load(area)
    idx14 = np.array([i for i, d in enumerate(dates) if d.year == 2014], dtype=int)
    v14 = values[idx14]
    d14 = [dates[i] for i in idx14]

    # Same-time daily and weekly persistence, restricted to 2014 pairs.
    lag1_corr = corr(v14[1:], v14[:-1])
    lag1_mape = mape(v14[1:], v14[:-1])
    lag7_corr = corr(v14[7:], v14[:-7])
    lag7_mape = mape(v14[7:], v14[:-7])

    # Daily-shape stability after removing each day's absolute level.
    norm = v14 / v14.mean(axis=1, keepdims=True)
    shape_slot_sd = norm.std(axis=0, ddof=1)
    mean_shape_sd = float(shape_slot_sd.mean())
    p90_shape_sd = float(np.percentile(shape_slot_sd, 90))

    # Day-to-day change in normalized profile; lower means smoother evolution.
    profile_change = np.sqrt(np.mean((norm[1:] - norm[:-1]) ** 2, axis=1))
    mean_profile_change = float(profile_change.mean())
    p90_profile_change = float(np.percentile(profile_change, 90))

    # Leakage-safe 2014 weekday/slot template: template is fitted only on 2009-2013.
    train_idx = np.array([i for i, d in enumerate(dates) if d.year <= 2013], dtype=int)
    by_weekday = defaultdict(list)
    for i in train_idx:
        by_weekday[dates[i].weekday()].append(values[i])
    templates = {w: np.median(np.asarray(rows), axis=0) for w, rows in by_weekday.items()}
    pred_template = np.vstack([templates[d.weekday()] for d in d14])
    template_mae = mae(v14, pred_template)
    template_rmse = rmse(v14, pred_template)
    template_mape = mape(v14, pred_template)

    # A more local leakage-safe template: 2013 same weekday/slot median only.
    train13_idx = np.array([i for i, d in enumerate(dates) if d.year == 2013], dtype=int)
    by_weekday13 = defaultdict(list)
    for i in train13_idx:
        by_weekday13[dates[i].weekday()].append(values[i])
    templates13 = {w: np.median(np.asarray(rows), axis=0) for w, rows in by_weekday13.items()}
    pred_template13 = np.vstack([templates13[d.weekday()] for d in d14])
    template13_mape = mape(v14, pred_template13)

    return {
        "area": area,
        "lag1_corr": lag1_corr,
        "lag1_mape": lag1_mape,
        "lag7_corr": lag7_corr,
        "lag7_mape": lag7_mape,
        "mean_shape_sd": mean_shape_sd,
        "p90_shape_sd": p90_shape_sd,
        "mean_profile_change": mean_profile_change,
        "p90_profile_change": p90_profile_change,
        "template_mae": template_mae,
        "template_rmse": template_rmse,
        "template_mape": template_mape,
        "template13_mape": template13_mape,
    }


def read_dict_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prior_evidence():
    # Q2: use the best silhouette score actually selected for each area.
    q2 = read_dict_csv(ROOT / "analysis" / "q2" / "Q2_silhouette_scores.csv")
    silhouettes = {}
    for area in ("Area1", "Area2"):
        rows = [r for r in q2 if r.get("area") == area]
        if not rows:
            raise ValueError(f"No Q2 silhouette rows for {area}")
        # tolerate column naming variations
        score_key = "silhouette" if "silhouette" in rows[0] else "silhouette_score"
        best = max(rows, key=lambda r: float(r[score_key]))
        silhouettes[area] = {"k": int(float(best["k"])), "score": float(best[score_key])}

    q4 = read_dict_csv(ROOT / "analysis" / "q4" / "Q4_model_summary.csv")
    q4best = {}
    for area in ("Area1", "Area2"):
        rows = [r for r in q4 if r["area"] == area]
        best = min(rows, key=lambda r: float(r["mean_MAPE_pct"]))
        q4best[area] = {
            "model": best["model"],
            "mape": float(best["mean_MAPE_pct"]),
            "rmse": float(best["mean_RMSE_MW"]),
            "fold_sd": float(best["fold_MAPE_sd"]),
        }

    q5 = read_dict_csv(ROOT / "analysis" / "q5" / "Q5_model_summary.csv")
    q5best, q5ctrl = {}, {}
    for area in ("Area1", "Area2"):
        rows = [r for r in q5 if r["area"] == area]
        weather_rows = [r for r in rows if r["model"] != "no_weather"]
        best = min(weather_rows, key=lambda r: float(r["mean_MAPE_pct"]))
        ctrl = next(r for r in rows if r["model"] == "no_weather")
        bm = float(best["mean_MAPE_pct"])
        cm = float(ctrl["mean_MAPE_pct"])
        q5best[area] = {
            "model": best["model"], "label": best.get("label", best["model"]),
            "mape": bm, "rmse": float(best["mean_RMSE_MW"]),
            "fold_sd": float(best["fold_MAPE_sd"]),
            "improvement": (cm - bm) / cm * 100,
        }
        q5ctrl[area] = {"mape": cm}
    return silhouettes, q4best, q5best, q5ctrl


def write_structural_csv(rows):
    keys = list(rows[0].keys())
    with (OUT / "Q6_structural_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def write_scorecard(struct, sil, q4, q5):
    # Direction-aware evidence. This is not a black-box composite score: each row is explicit.
    evidence = [
        ("1-day same-time correlation", "higher", "lag1_corr"),
        ("1-day persistence MAPE", "lower", "lag1_mape"),
        ("7-day same-time correlation", "higher", "lag7_corr"),
        ("7-day persistence MAPE", "lower", "lag7_mape"),
        ("normalized daily-shape dispersion", "lower", "mean_shape_sd"),
        ("day-to-day normalized-profile change", "lower", "mean_profile_change"),
        ("2009-2013 weekday-template -> 2014 MAPE", "lower", "template_mape"),
        ("Q2 best silhouette", "higher", None),
        ("Q4 best rolling MAPE", "lower", None),
        ("Q4 fold MAPE SD", "lower", None),
        ("Q5 best weather rolling MAPE", "lower", None),
        ("Q5 weather relative improvement", "higher", None),
    ]
    a = {r["area"]: r for r in struct}
    out = []
    for name, direction, key in evidence:
        if key:
            v1, v2 = float(a["Area1"][key]), float(a["Area2"][key])
        elif name == "Q2 best silhouette":
            v1, v2 = sil["Area1"]["score"], sil["Area2"]["score"]
        elif name == "Q4 best rolling MAPE":
            v1, v2 = q4["Area1"]["mape"], q4["Area2"]["mape"]
        elif name == "Q4 fold MAPE SD":
            v1, v2 = q4["Area1"]["fold_sd"], q4["Area2"]["fold_sd"]
        elif name == "Q5 best weather rolling MAPE":
            v1, v2 = q5["Area1"]["mape"], q5["Area2"]["mape"]
        elif name == "Q5 weather relative improvement":
            v1, v2 = q5["Area1"]["improvement"], q5["Area2"]["improvement"]
        else:
            raise AssertionError(name)
        winner = "Area1" if ((v1 > v2) if direction == "higher" else (v1 < v2)) else "Area2"
        out.append({"evidence": name, "preferred_direction": direction, "Area1": v1, "Area2": v2, "winner": winner})

    with (OUT / "Q6_evidence_scorecard.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    return out


def plot_evidence(struct, sil, q4, q5):
    a = {r["area"]: r for r in struct}

    labels = ["Lag 1d corr", "Lag 7d corr", "Q2 silhouette"]
    x1 = [a["Area1"]["lag1_corr"], a["Area1"]["lag7_corr"], sil["Area1"]["score"]]
    x2 = [a["Area2"]["lag1_corr"], a["Area2"]["lag7_corr"], sil["Area2"]["score"]]
    x = np.arange(len(labels)); width = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width/2, x1, width, label="Area 1")
    plt.bar(x + width/2, x2, width, label="Area 2")
    plt.xticks(x, labels); plt.ylabel("Correlation / silhouette")
    plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "regularity_correlation_evidence.svg", format="svg"); plt.close()

    labels = ["Lag 1d", "Lag 7d", "Weekday template", "Q4 HGB", "Q5 weather HGB"]
    m1 = [a["Area1"]["lag1_mape"], a["Area1"]["lag7_mape"], a["Area1"]["template_mape"], q4["Area1"]["mape"], q5["Area1"]["mape"]]
    m2 = [a["Area2"]["lag1_mape"], a["Area2"]["lag7_mape"], a["Area2"]["template_mape"], q4["Area2"]["mape"], q5["Area2"]["mape"]]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, m1, width, label="Area 1")
    plt.bar(x + width/2, m2, width, label="Area 2")
    plt.xticks(x, labels, rotation=16); plt.ylabel("MAPE (%) — lower is better")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "regularity_prediction_evidence.svg", format="svg"); plt.close()

    labels = ["Mean shape SD", "Mean profile change"]
    s1 = [a["Area1"]["mean_shape_sd"], a["Area1"]["mean_profile_change"]]
    s2 = [a["Area2"]["mean_shape_sd"], a["Area2"]["mean_profile_change"]]
    x = np.arange(len(labels))
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x - width/2, s1, width, label="Area 1")
    plt.bar(x + width/2, s2, width, label="Area 2")
    plt.xticks(x, labels); plt.ylabel("Normalized dispersion — lower is better")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "regularity_shape_stability.svg", format="svg"); plt.close()


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"


def build_report(struct, sil, q4, q5, scorecard):
    a = {r["area"]: r for r in struct}
    wins = defaultdict(int)
    for r in scorecard: wins[r["winner"]] += 1

    report = f'''# Question 6 — 两地区负荷规律性综合评价

## 1. 评价结论

综合 Q1–Q5 与本题新增的独立规律性指标，**Area 2 的底层负荷结构整体更规则、更稳定**；但需要保留一个重要限定：在 Q4/Q5 的 HGB 七日回测中，Area 1 的平均 MAPE 略低。因此，结论不是“Area 2 在所有预测模型上误差都更小”，而是：

- **Area 2：周期结构更清晰、简单历史规律更强、跨回测稳定性更好，天气信息带来的相对增益更大。**
- **Area 1：结构性扰动更强，但经过非线性机器学习建模后，当前 Q4/Q5 回测的平均 MAPE 略优。**

在本题列出的 12 项方向明确的证据中，Area 2 在 **{wins['Area2']}** 项占优，Area 1 在 **{wins['Area1']}** 项占优。这个计数只用于展示证据方向，不作为人为加权的“总分”。

## 2. Q1–Q5 已有证据汇总

### 2.1 Q1：负荷波动与滞后规律

2014 年 Area 2 的日峰值 CV 和峰谷差 CV 更低；Q1 已经提示其日内结构跨日期更稳定。本题重新从原始负荷计算的同一时刻滞后规律如下：

| 指标 | Area 1 | Area 2 | 更规则 |
|---|---:|---:|---|
| 1 日滞后相关系数 | {f3(a['Area1']['lag1_corr'])} | {f3(a['Area2']['lag1_corr'])} | {'Area 1' if a['Area1']['lag1_corr']>a['Area2']['lag1_corr'] else 'Area 2'} |
| 1 日 persistence MAPE | {f2(a['Area1']['lag1_mape'])}% | {f2(a['Area2']['lag1_mape'])}% | {'Area 1' if a['Area1']['lag1_mape']<a['Area2']['lag1_mape'] else 'Area 2'} |
| 7 日滞后相关系数 | {f3(a['Area1']['lag7_corr'])} | {f3(a['Area2']['lag7_corr'])} | {'Area 1' if a['Area1']['lag7_corr']>a['Area2']['lag7_corr'] else 'Area 2'} |
| 7 日 persistence MAPE | {f2(a['Area1']['lag7_mape'])}% | {f2(a['Area2']['lag7_mape'])}% | {'Area 1' if a['Area1']['lag7_mape']<a['Area2']['lag7_mape'] else 'Area 2'} |

![Correlation evidence](figures/regularity_correlation_evidence.svg)

简单 persistence 不需要复杂拟合，因此它是判断“原始规律性”的直接证据。Area 2 的同一时刻相关性更高、直接历史复制误差更低，说明其周期重复性更强。

### 2.2 Q2：日负荷状态可分性

最佳 KMeans 聚类均为 `k=3`，但最佳轮廓系数为：

- Area 1：**{sil['Area1']['score']:.4f}**
- Area 2：**{sil['Area2']['score']:.4f}**

Area 2 明显更高，说明其低负荷、正常和高温高负荷三种日状态类内更紧、类间更清晰。这是“规律性更容易被离散状态描述”的证据。

### 2.3 Q4：无天气七日预测

严格 rolling-origin 回测的最优无天气模型均为 `{q4['Area1']['model']}`：

| 指标 | Area 1 | Area 2 |
|---|---:|---:|
| 平均 MAPE | {q4['Area1']['mape']:.2f}% | {q4['Area2']['mape']:.2f}% |
| 平均 RMSE | {q4['Area1']['rmse']:.2f} MW | {q4['Area2']['rmse']:.2f} MW |
| fold MAPE SD | {q4['Area1']['fold_sd']:.2f} pp | {q4['Area2']['fold_sd']:.2f} pp |

Area 1 的平均 MAPE 略低，但 Area 2 的 fold 波动显著更小（{q4['Area2']['fold_sd']:.2f} vs {q4['Area1']['fold_sd']:.2f} pp）。因此 Area 2 的预测表现更**稳定**，Area 1 的平均精度在当前 HGB 规格下略好。

### 2.4 Q5：天气增强七日预测

两地区最佳天气模型均为 `mean_temp_quad_hr`：

| 指标 | Area 1 | Area 2 |
|---|---:|---:|
| 最佳天气模型 MAPE | {q5['Area1']['mape']:.2f}% | {q5['Area2']['mape']:.2f}% |
| fold MAPE SD | {q5['Area1']['fold_sd']:.2f} pp | {q5['Area2']['fold_sd']:.2f} pp |
| 相对无天气控制组改善 | {q5['Area1']['improvement']:.2f}% | {q5['Area2']['improvement']:.2f}% |

Area 1 的绝对 MAPE 仍略低，但 Area 2 从天气获得的相对改善更大。这与 Q3 中 Area 2 对平均温度的关系更强一致，说明其负荷变化更能由可观测外生因素解释。

## 3. 本题新增的“其他证据”

### 3.1 归一化日负荷形态稳定性

为排除两个地区绝对负荷规模差异，把每天 96 点除以当日日均负荷，再观察同一 slot 跨日离散程度。

| 指标 | Area 1 | Area 2 | 含义 |
|---|---:|---:|---|
| 96 个 slot 的平均标准差 | {a['Area1']['mean_shape_sd']:.4f} | {a['Area2']['mean_shape_sd']:.4f} | 越低越稳定 |
| slot 标准差 P90 | {a['Area1']['p90_shape_sd']:.4f} | {a['Area2']['p90_shape_sd']:.4f} | 越低越稳定 |
| 相邻日归一化曲线 RMS 变化 | {a['Area1']['mean_profile_change']:.4f} | {a['Area2']['mean_profile_change']:.4f} | 越低越稳定 |
| 上述变化 P90 | {a['Area1']['p90_profile_change']:.4f} | {a['Area2']['p90_profile_change']:.4f} | 越低越稳定 |

![Shape stability](figures/regularity_shape_stability.svg)

这组指标直接考察“曲线形状是否稳定”，而不是只看负荷水平。它与 Q1 的峰谷差 CV、Q2 的聚类可分性互为补充。

### 3.2 完全样本外的 weekday/slot 模板检验

再建立一个极简且无信息泄漏的基准：只使用 **2009–2013** 历史数据，按星期几和 15 分钟 slot 计算中位数模板，然后一次性预测整个 2014 年。它没有用 2014 的真实负荷拟合任何参数。

| 指标 | Area 1 | Area 2 |
|---|---:|---:|
| MAE | {a['Area1']['template_mae']:.2f} MW | {a['Area2']['template_mae']:.2f} MW |
| RMSE | {a['Area1']['template_rmse']:.2f} MW | {a['Area2']['template_rmse']:.2f} MW |
| MAPE | {a['Area1']['template_mape']:.2f}% | {a['Area2']['template_mape']:.2f}% |
| 仅用 2013 模板预测 2014 MAPE | {a['Area1']['template13_mape']:.2f}% | {a['Area2']['template13_mape']:.2f}% |

![Prediction evidence](figures/regularity_prediction_evidence.svg)

这个测试的意义在于：如果一个地区的负荷日历规律更稳定，那么极简单的 weekday/slot 历史模板就应当表现更好。它避免用复杂模型能力掩盖底层规律性的差异。

## 4. 为什么“规律性更强”与“当前 HGB 平均 MAPE 更低”可能不是同一地区

两者并不矛盾：

1. **规律性**描述数据生成过程的重复性、稳定性和可解释性；
2. **模型平均误差**还取决于模型结构、训练窗口、异常日分布和模型对复杂非线性的适配程度；
3. Area 1 的模式虽然更容易出现特殊扰动，但 HGB 可能更有效地拟合这些非线性，因此在当前 13 个回测周上取得略低的平均 MAPE；
4. Area 2 的优势更集中在简单基准、聚类结构和跨 fold 稳定性，这些指标更直接反映底层规律。

因此，不能只凭 Q4/Q5 单一平均 MAPE 就推翻 Q1/Q2 的结构证据，也不能因为 Area 2 结构更规则就声称它在所有模型上都必然更准。

## 5. 综合评价

**综合结论：Area 2 的整体负荷规律性优于 Area 1。**

主要理由是：

- 同一时刻日/周滞后相关性更强，简单 persistence 误差更低；
- 日负荷聚类轮廓系数更高，典型状态更清晰；
- Q4 的 rolling-origin 误差跨 fold 更稳定；
- Q5 中天气变量带来的相对改善更大，说明负荷更容易被可观测气象因素解释；
- 本题新增的归一化曲线稳定性与样本外 weekday/slot 模板提供了独立佐证。

与此同时，**Area 1 在当前 Q4/Q5 HGB 回测中的平均 MAPE 略低**，应作为模型层面的反例和限定条件明确保留。若目标是“哪个地区的数据生成规律更稳定”，选择 Area 2；若目标是“当前已实现的 HGB 模型在这组回测上的平均误差谁更低”，则是 Area 1。

## 6. 防泄漏与解释边界

- 新增结构指标仅使用 2014 年及以前的历史数据；
- weekday/slot 样本外检验只用 2009–2013 拟合、2014 验证；
- 不使用 2015-01-04 至 10 的任何真实负荷；
- Q4/Q5 证据直接读取已完成的 rolling-origin 结果，不重新用未来数据选择模型；
- 证据计数不等价于统计显著性检验，也不是人为构造的综合评分。

## 7. 输出文件

- `Q6_ANALYSIS.md`
- `Q6_structural_metrics.csv`
- `Q6_evidence_scorecard.csv`
- `q6_regularity_evaluation.py`
- `figures/regularity_correlation_evidence.svg`
- `figures/regularity_prediction_evidence.svg`
- `figures/regularity_shape_stability.svg`
'''
    (OUT / "Q6_ANALYSIS.md").write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    struct = [structural_metrics("Area1"), structural_metrics("Area2")]
    sil, q4, q5, _ = prior_evidence()
    write_structural_csv(struct)
    score = write_scorecard(struct, sil, q4, q5)
    plot_evidence(struct, sil, q4, q5)
    build_report(struct, sil, q4, q5, score)
    print("Q6 complete")
    for r in struct:
        print(r)
    print("scorecard wins:", {a: sum(x['winner']==a for x in score) for a in ('Area1','Area2')})


if __name__ == "__main__":
    main()
