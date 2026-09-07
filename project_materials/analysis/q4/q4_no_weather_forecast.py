#!/usr/bin/env python3
"""Question 4: 7-day 15-minute load forecast without weather.

Strict leakage rule: every feature for a forecast point uses calendar information and
loads at least 7 days old. Thus the full 7-day horizon can be predicted directly
without using unknown Jan 4-10 actual loads or recursive predictions.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "q4"
FIG = OUT / "figures"
SLOTS = 96
LAGS = np.array([7, 14, 21, 28, 35, 42, 49, 56], dtype=int)
TARGET_START = datetime(2015, 1, 4)


def parse_date(x: str) -> datetime:
    x = x.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported date format: {x!r}")


def read_load(path: Path):
    dates, vals = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if len(header) != 97:
            raise ValueError(f"{path.name}: expected 97 columns, got {len(header)}")
        for row in r:
            if not row:
                continue
            dates.append(parse_date(row[0]))
            vals.append([float(x) for x in row[1:]])
    return dates, np.asarray(vals, dtype=float), header


def feature_from_lags(d: datetime, slot: int, lagvals: np.ndarray) -> np.ndarray:
    doy = d.timetuple().tm_yday
    return np.array([
        d.month, d.weekday(), 1.0 if d.weekday() >= 5 else 0.0,
        np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
        np.sin(2*np.pi*slot/SLOTS), np.cos(2*np.pi*slot/SLOTS),
        *lagvals.tolist(),
        float(lagvals[:4].mean()), float(np.median(lagvals[:4])),
        float(lagvals[:4].std()), float(lagvals.mean()),
    ], dtype=float)


def make_training(dates, values, cutoff_idx: int, lookback_days: int = 4*365):
    start = max(int(LAGS.max()), cutoff_idx - lookback_days)
    X, y = [], []
    for i in range(start, cutoff_idx):
        d = dates[i]
        for s in range(SLOTS):
            lagvals = np.array([values[i-k, s] for k in LAGS])
            X.append(feature_from_lags(d, s, lagvals))
            y.append(values[i, s])
    return np.asarray(X), np.asarray(y)


def fit_ml(dates, values, cutoff_idx: int):
    X, y = make_training(dates, values, cutoff_idx)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.07, max_iter=180,
        max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
        random_state=42,
    )
    model.fit(X, y)
    return model


def baseline_predict(values: np.ndarray, idx: int, method: str) -> np.ndarray:
    if method == "lag7":
        return values[idx-7:idx]
    weeks = np.stack([values[idx-k:idx-k+7] for k in [7,14,21,28]], axis=0)
    if method == "avg4":
        return weeks.mean(axis=0)
    if method == "weighted4":
        w = np.array([0.4,0.3,0.2,0.1])[:,None,None]
        return (weeks*w).sum(axis=0)
    raise ValueError(method)


def forecast_date(dates, idx: int, h: int) -> datetime:
    if idx < len(dates):
        return dates[idx] + timedelta(days=h)
    return TARGET_START + timedelta(days=h)


def ml_predict(model, dates, values, idx: int) -> np.ndarray:
    rows = []
    for h in range(7):
        d = forecast_date(dates, idx, h)
        for s in range(SLOTS):
            lagvals = np.array([values[idx+h-k, s] for k in LAGS])
            rows.append(feature_from_lags(d, s, lagvals))
    return model.predict(np.asarray(rows)).reshape(7, SLOTS)


def metrics(actual, pred):
    err = pred-actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err)/actual)*100),
        "Bias": float(np.mean(err)),
    }


def daily_mape(actual, pred):
    return np.mean(np.abs(pred-actual)/actual, axis=1)*100


def choose_backtest_indices(dates):
    index = {d:i for i,d in enumerate(dates)}
    out=[]
    target=datetime(2014,1,5)
    while target <= datetime(2014,12,7):
        if target in index and index[target]+7 <= len(dates):
            out.append(index[target])
        target += timedelta(days=28)
    return out


def run_area(area: str, values: np.ndarray, dates, header):
    back_idxs=choose_backtest_indices(dates)
    methods=["lag7","avg4","weighted4","hgb_fixed_lags"]
    records=[]
    preds_by_method={m:[] for m in methods}
    for fold, idx in enumerate(back_idxs,1):
        actual=values[idx:idx+7]
        for m in methods[:3]:
            p=baseline_predict(values, idx, m)
            mm=metrics(actual,p)
            records.append([area,fold,dates[idx].strftime("%Y-%m-%d"),m,*[mm[k] for k in ["MAE","RMSE","MAPE","Bias"]]])
            preds_by_method[m].append((actual,p))
        model=fit_ml(dates,values,idx)
        p=ml_predict(model,dates,values,idx)
        mm=metrics(actual,p)
        records.append([area,fold,dates[idx].strftime("%Y-%m-%d"),"hgb_fixed_lags",*[mm[k] for k in ["MAE","RMSE","MAPE","Bias"]]])
        preds_by_method["hgb_fixed_lags"].append((actual,p))

    summary=[]
    for m in methods:
        rs=[r for r in records if r[3]==m]
        summary.append([area,m,*[float(np.mean([r[j] for r in rs])) for j in range(4,8)],float(np.std([r[6] for r in rs],ddof=1))])
    best=min(summary,key=lambda x:x[4])

    idx=len(dates)
    candidate_final={m:baseline_predict(values,idx,m) for m in methods[:3]}
    final_model=fit_ml(dates,values,idx)
    candidate_final["hgb_fixed_lags"]=ml_predict(final_model,dates,values,idx)
    final=candidate_final[best[1]]

    out_path=OUT/f"Q4_{area}_Load.csv"
    with out_path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(header)
        for h in range(7):
            # Preserve source submission date style (YYYYMMDD).
            w.writerow([(TARGET_START+timedelta(days=h)).strftime("%Y%m%d"), *[f"{v:.4f}" for v in final[h]]])

    with (OUT/f"Q4_{area}_candidate_forecasts.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["date","slot","method","forecast_MW"])
        for m,p in candidate_final.items():
            for h in range(7):
                for s in range(SLOTS):
                    w.writerow([(TARGET_START+timedelta(days=h)).strftime("%Y-%m-%d"),s,m,f"{p[h,s]:.4f}"])

    x=np.arange(7*SLOTS)/4
    plt.figure(figsize=(11,5)); plt.plot(x,final.ravel(),lw=1.1)
    plt.xlabel("Hours from 2015-01-04 00:00"); plt.ylabel("Forecast load (MW)")
    plt.tight_layout(); plt.savefig(FIG/f"{area.lower()}_q4_forecast.svg",format="svg"); plt.close()

    labels=[s[1] for s in summary]; mapes=[s[4] for s in summary]
    plt.figure(figsize=(8,4.5)); plt.bar(labels,mapes); plt.ylabel("Mean 7-day backtest MAPE (%)")
    plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(FIG/f"{area.lower()}_model_backtest.svg",format="svg"); plt.close()

    dm=[]
    for actual,p in preds_by_method[best[1]]: dm.extend(daily_mape(actual,p).tolist())
    return records,summary,best,final,np.asarray(dm)


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    results={}; all_records=[]; all_summary=[]
    for area in ["Area1","Area2"]:
        dates,values,header=read_load(ROOT/f"{area}_Load.csv")
        if dates[-1] != datetime(2015,1,3): raise ValueError(f"{area}: last date is {dates[-1]}")
        rec,summ,best,final,dm=run_area(area,values,dates,header)
        all_records += rec; all_summary += summ
        results[area]=(best,final,dm)

    with (OUT/"Q4_backtest_folds.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["area","fold","origin","model","MAE_MW","RMSE_MW","MAPE_pct","Bias_MW"]); w.writerows(all_records)
    with (OUT/"Q4_model_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["area","model","mean_MAE_MW","mean_RMSE_MW","mean_MAPE_pct","mean_Bias_MW","fold_MAPE_sd"]); w.writerows(all_summary)

    def row(area):
        best,final,dm=results[area]
        return {"model":best[1],"mae":best[2],"rmse":best[3],"mape":best[4],"bias":best[5],"sd":best[6],
                "p10":float(np.percentile(dm,10)),"p50":float(np.percentile(dm,50)),"p90":float(np.percentile(dm,90)),
                "fmean":float(final.mean()),"fmax":float(final.max()),"fmin":float(final.min())}
    a1=row("Area1"); a2=row("Area2")

    def model_table(area):
        rs=[r for r in all_summary if r[0]==area]
        lines=["| Model | Mean MAE (MW) | Mean RMSE (MW) | Mean MAPE | Fold MAPE SD |","|---|---:|---:|---:|---:|"]
        for r in rs: lines.append(f"| {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f}% | {r[6]:.2f} pp |")
        return "\n".join(lines)

    report=f'''# Question 4 — 不计气象因素的短期负荷预测

## 1. 预测目标与约束

- 预测区间：2015-01-04 至 2015-01-10，共 7 天。
- 每天 96 个 15 min 点，每个地区共 672 个预测值。
- 本题完全不使用气象数据。
- 已知负荷截止 2015-01-03；预测期真实负荷不可见。
- 严格防止时间泄漏：所有机器学习负荷滞后均为 7、14、21、28、35、42、49、56 天，不使用未来真实 `lag_1d`，因此整个 7 天 horizon 可直接预测而不递归引用未知真实值。

## 2. 候选模型

1. `lag7`：上周同日同一时刻。
2. `avg4`：过去四周同一 weekday/slot 的等权平均。
3. `weighted4`：过去四周按 0.4/0.3/0.2/0.1 加权。
4. `hgb_fixed_lags`：HistGradientBoostingRegressor；特征包括月份、星期、周末、年周期、15 min slot 周期编码，以及 7–56 天固定周滞后及其统计量。

最终模型不是凭全样本拟合优度选择，而由历史 rolling-origin 7-day backtest 的平均 MAPE 决定。

## 3. 时间序列回测设计

在 2014 年设置 13 个约每 4 周一次的星期日起点，每次只用该起点之前可获得的负荷训练/构造特征，然后预测后续完整 7 天。没有随机 train/test split。

### Area 1

{model_table('Area1')}

**选择：`{a1['model']}`**。

![Area1 backtest](figures/area1_model_backtest.svg)

历史被选模型的单日 MAPE 分布：P10={a1['p10']:.2f}%，中位数={a1['p50']:.2f}%，P90={a1['p90']:.2f}%。

### Area 2

{model_table('Area2')}

**选择：`{a2['model']}`**。

![Area2 backtest](figures/area2_model_backtest.svg)

历史被选模型的单日 MAPE 分布：P10={a2['p10']:.2f}%，中位数={a2['p50']:.2f}%，P90={a2['p90']:.2f}%。

## 4. 2015-01-04 至 2015-01-10 最终预测

### Area 1

- 预测均值：{a1['fmean']:.2f} MW
- 预测最大值：{a1['fmax']:.2f} MW
- 预测最小值：{a1['fmin']:.2f} MW

![Area1 final forecast](figures/area1_q4_forecast.svg)

正式提交文件：`Q4_Area1_Load.csv`。

### Area 2

- 预测均值：{a2['fmean']:.2f} MW
- 预测最大值：{a2['fmax']:.2f} MW
- 预测最小值：{a2['fmin']:.2f} MW

![Area2 final forecast](figures/area2_q4_forecast.svg)

正式提交文件：`Q4_Area2_Load.csv`。

## 5. 在不知道实际负荷时的准确度推断

准确度只能依据严格历史外推表现推断，不能宣称 2015-01-04 至 10 的真实误差已经知道。

- Area 1 被选模型的 2014 rolling 7-day 平均 MAPE 为 **{a1['mape']:.2f}%**，fold 间 MAPE 标准差 {a1['sd']:.2f} 个百分点。
- Area 2 被选模型的对应平均 MAPE 为 **{a2['mape']:.2f}%**，fold 间标准差 {a2['sd']:.2f} 个百分点。
- 因此，在未来一周没有出现历史中未覆盖的结构突变时，2015 目标周的误差量级可参考上述 backtest；P10/P50/P90 单日 MAPE 给出了比单一均值更保守的历史经验区间。

从相对规律性看，历史回测 MAPE 更低、跨 fold 波动更小的地区应被认为预测置信度更高。该判断将在 Q5 使用天气后进一步比较，并在 Q6 综合确认。

## 6. 防泄漏说明

本题机器学习模型不使用 Q2 中由未来真实负荷聚类得到的标签，因为这会泄漏目标期信息；也不使用任何 2015-01-04 之后真实负荷。所有用于目标周的固定周滞后均落在 2015-01-03 或更早。

## 7. 输出文件

- `Q4_ANALYSIS.md`
- `Q4_Area1_Load.csv`
- `Q4_Area2_Load.csv`
- `Q4_Area1_candidate_forecasts.csv`
- `Q4_Area2_candidate_forecasts.csv`
- `Q4_backtest_folds.csv`
- `Q4_model_summary.csv`
- `q4_no_weather_forecast.py`
- `figures/area1_model_backtest.svg`
- `figures/area2_model_backtest.svg`
- `figures/area1_q4_forecast.svg`
- `figures/area2_q4_forecast.svg`
'''
    (OUT/"Q4_ANALYSIS.md").write_text(report,encoding="utf-8")

if __name__ == "__main__": main()
