#!/usr/bin/env python3
"""Question 5: 7-day 15-minute load forecast with weather.

Leakage control:
- load features use only fixed weekly lags >= 7 days, so the complete 7-day horizon
  is forecast directly without unknown target-week loads or recursive predictions;
- weather for the forecast horizon is treated as exogenous information available to
  the forecaster, matching the question's "consider weather" setting;
- backtests use observed weather for each historical forecast week as a proxy for
  available weather forecasts, so reported gains are conditional on weather being
  known accurately and do not include weather-forecast error.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "q5"
FIG = OUT / "figures"
SLOTS = 96
LAGS = np.array([7, 14, 21, 28, 35, 42, 49, 56], dtype=int)
TARGET_START = datetime(2015, 1, 4)
WEATHER_START = datetime(2012, 1, 1)


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


def fnum(x: str) -> float:
    x = x.strip()
    return np.nan if x == "" else float(x)


def read_weather(path: Path, area: str):
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if len(header) != 6:
            raise ValueError(f"{path.name}: expected 6 columns, got {len(header)}")
        for row in r:
            if not row:
                continue
            d = parse_date(row[0])
            v = np.array([fnum(x) for x in row[1:]], dtype=float)
            # Preserve Q3's data-quality decisions without modifying source data.
            # [max temp, min temp, mean temp, humidity, rainfall]
            if area == "Area1" and d in {datetime(2012,3,21), datetime(2012,3,27)}:
                v[0] = np.nan
            if area == "Area2" and d in {datetime(2012,10,8), datetime(2012,10,9), datetime(2012,10,10), datetime(2012,10,11)}:
                v[1] = np.nan
            out[d] = v
    return out


def weather_vec(w: np.ndarray, spec: str) -> np.ndarray:
    tmax, tmin, tmean, hum, rain = w.tolist()
    if spec == "no_weather":
        return np.array([], dtype=float)
    if spec == "mean_temp_quad":
        return np.array([tmean, tmean*tmean], dtype=float)
    if spec == "mean_temp_quad_hr":
        return np.array([tmean, tmean*tmean, hum, rain], dtype=float)
    if spec == "all_weather":
        return np.array([
            tmax, tmin, tmean,
            tmax*tmax if np.isfinite(tmax) else np.nan,
            tmin*tmin if np.isfinite(tmin) else np.nan,
            tmean*tmean if np.isfinite(tmean) else np.nan,
            hum, rain,
        ], dtype=float)
    raise ValueError(spec)


def base_feature(d: datetime, slot: int, lagvals: np.ndarray) -> np.ndarray:
    doy = d.timetuple().tm_yday
    return np.array([
        d.month, d.weekday(), 1.0 if d.weekday() >= 5 else 0.0,
        np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
        np.sin(2*np.pi*slot/SLOTS), np.cos(2*np.pi*slot/SLOTS),
        *lagvals.tolist(),
        float(lagvals[:4].mean()), float(np.median(lagvals[:4])),
        float(lagvals[:4].std()), float(lagvals.mean()),
    ], dtype=float)


def feature(d: datetime, slot: int, lagvals: np.ndarray, weather: dict, spec: str) -> np.ndarray:
    x = base_feature(d, slot, lagvals)
    if spec == "no_weather":
        return x
    if d not in weather:
        raise ValueError(f"Weather unavailable for {d:%Y-%m-%d}")
    return np.concatenate([x, weather_vec(weather[d], spec)])


def make_training(dates, values, weather, cutoff_idx: int, spec: str):
    # Controlled A/B: every candidate starts at 2012-01-01 so weather models and
    # no-weather control have the same training period.
    date_to_idx = {d:i for i,d in enumerate(dates)}
    start = max(int(LAGS.max()), date_to_idx[WEATHER_START])
    X, y = [], []
    for i in range(start, cutoff_idx):
        d = dates[i]
        if spec != "no_weather" and d not in weather:
            continue
        for s in range(SLOTS):
            lagvals = np.array([values[i-k, s] for k in LAGS])
            X.append(feature(d, s, lagvals, weather, spec))
            y.append(values[i, s])
    return np.asarray(X), np.asarray(y)


def fit_model(dates, values, weather, cutoff_idx: int, spec: str):
    X, y = make_training(dates, values, weather, cutoff_idx, spec)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.07, max_iter=180,
        max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
        random_state=42,
    )
    model.fit(X, y)
    return model


def forecast_date(dates, idx: int, h: int) -> datetime:
    if idx < len(dates):
        return dates[idx] + timedelta(days=h)
    return TARGET_START + timedelta(days=h)


def model_predict(model, dates, values, weather, idx: int, spec: str) -> np.ndarray:
    rows = []
    for h in range(7):
        d = forecast_date(dates, idx, h)
        for s in range(SLOTS):
            lagvals = np.array([values[idx+h-k, s] for k in LAGS])
            rows.append(feature(d, s, lagvals, weather, spec))
    return model.predict(np.asarray(rows)).reshape(7, SLOTS)


def metrics(actual, pred):
    err = pred - actual
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
    out = []
    target = datetime(2014,1,5)
    while target <= datetime(2014,12,7):
        if target in index and index[target]+7 <= len(dates):
            out.append(index[target])
        target += timedelta(days=28)
    return out


def run_area(area: str, dates, values, header, weather):
    specs = ["no_weather", "mean_temp_quad", "mean_temp_quad_hr", "all_weather"]
    labels = {
        "no_weather":"HGB no weather (controlled)",
        "mean_temp_quad":"HGB + mean T + T²",
        "mean_temp_quad_hr":"HGB + mean T + T² + RH + rain",
        "all_weather":"HGB + all weather",
    }
    back_idxs = choose_backtest_indices(dates)
    records, daily_errors = [], {s:[] for s in specs}

    for fold, idx in enumerate(back_idxs, 1):
        actual = values[idx:idx+7]
        for spec in specs:
            model = fit_model(dates, values, weather, idx, spec)
            pred = model_predict(model, dates, values, weather, idx, spec)
            mm = metrics(actual, pred)
            records.append([area,fold,dates[idx].strftime("%Y-%m-%d"),spec,
                            mm["MAE"],mm["RMSE"],mm["MAPE"],mm["Bias"]])
            daily_errors[spec].extend(daily_mape(actual,pred).tolist())

    summary=[]
    for spec in specs:
        rs=[r for r in records if r[3]==spec]
        summary.append([
            area,spec,labels[spec],
            float(np.mean([r[4] for r in rs])),
            float(np.mean([r[5] for r in rs])),
            float(np.mean([r[6] for r in rs])),
            float(np.mean([r[7] for r in rs])),
            float(np.std([r[6] for r in rs],ddof=1)),
        ])

    # Q5 must use weather, so choose best weather-enhanced candidate, not the control.
    best = min([r for r in summary if r[1] != "no_weather"], key=lambda x:x[5])
    control = [r for r in summary if r[1] == "no_weather"][0]

    idx = len(dates)
    candidates={}
    for spec in specs:
        model=fit_model(dates,values,weather,idx,spec)
        candidates[spec]=model_predict(model,dates,values,weather,idx,spec)
    final=candidates[best[1]]

    out_path=OUT/f"Q5_{area}_Load.csv"
    with out_path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for h in range(7):
            w.writerow([(TARGET_START+timedelta(days=h)).strftime("%Y%m%d"),
                        *[f"{v:.4f}" for v in final[h]]])

    with (OUT/f"Q5_{area}_candidate_forecasts.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["date","slot","model","forecast_MW"])
        for spec,p in candidates.items():
            for h in range(7):
                for s in range(SLOTS):
                    w.writerow([(TARGET_START+timedelta(days=h)).strftime("%Y-%m-%d"),s,spec,f"{p[h,s]:.4f}"])

    # Final forecast plot.
    x=np.arange(7*SLOTS)/4
    plt.figure(figsize=(11,5)); plt.plot(x,final.ravel(),lw=1.1)
    plt.xlabel("Hours from 2015-01-04 00:00"); plt.ylabel("Forecast load (MW)")
    plt.tight_layout(); plt.savefig(FIG/f"{area.lower()}_q5_forecast.svg",format="svg"); plt.close()

    # Controlled backtest comparison.
    plt.figure(figsize=(8.5,4.7))
    plt.bar([r[2] for r in summary],[r[5] for r in summary])
    plt.ylabel("Mean 7-day backtest MAPE (%)"); plt.xticks(rotation=18,ha="right")
    plt.tight_layout(); plt.savefig(FIG/f"{area.lower()}_weather_model_backtest.svg",format="svg"); plt.close()

    # Target-week weather figure (daily mean temperature).
    ds=[TARGET_START+timedelta(days=h) for h in range(7)]
    tm=[weather[d][2] for d in ds]
    plt.figure(figsize=(8,4.2)); plt.plot([d.strftime("%m-%d") for d in ds],tm,marker="o")
    plt.xlabel("Date"); plt.ylabel("Mean temperature (°C)")
    plt.tight_layout(); plt.savefig(FIG/f"{area.lower()}_target_week_weather.svg",format="svg"); plt.close()

    return records,summary,best,control,final,np.asarray(daily_errors[best[1]])


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    all_records=[]; all_summary=[]; results={}; weather_rows=[]

    for area in ["Area1","Area2"]:
        dates,values,header=read_load(ROOT/f"{area}_Load.csv")
        weather=read_weather(ROOT/f"{area}_Weather.csv",area)
        if dates[-1] != datetime(2015,1,3):
            raise ValueError(f"{area}: last load date is {dates[-1]}")
        for h in range(7):
            d=TARGET_START+timedelta(days=h)
            if d not in weather:
                raise ValueError(f"{area}: missing target weather {d:%Y-%m-%d}")
            w=weather[d]
            weather_rows.append([area,d.strftime("%Y-%m-%d"),*w.tolist()])
        rec,summ,best,control,final,dm=run_area(area,dates,values,header,weather)
        all_records += rec; all_summary += summ
        results[area]=(best,control,final,dm)

    with (OUT/"Q5_backtest_folds.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["area","fold","origin","model","MAE_MW","RMSE_MW","MAPE_pct","Bias_MW"]); w.writerows(all_records)
    with (OUT/"Q5_model_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["area","model","label","mean_MAE_MW","mean_RMSE_MW","mean_MAPE_pct","mean_Bias_MW","fold_MAPE_sd"]); w.writerows(all_summary)
    with (OUT/"Q5_target_week_weather.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["area","date","max_temp_C","min_temp_C","mean_temp_C","relative_humidity_pct","rainfall"]); w.writerows(weather_rows)

    def pack(area):
        best,control,final,dm=results[area]
        imp=(control[5]-best[5])/control[5]*100
        return {
            "model":best[1],"label":best[2],"mae":best[3],"rmse":best[4],"mape":best[5],"bias":best[6],"sd":best[7],
            "control_mape":control[5],"improvement":imp,
            "p10":float(np.percentile(dm,10)),"p50":float(np.percentile(dm,50)),"p90":float(np.percentile(dm,90)),
            "fmean":float(final.mean()),"fmax":float(final.max()),"fmin":float(final.min()),
        }
    a1=pack("Area1"); a2=pack("Area2")

    def table(area):
        rs=[r for r in all_summary if r[0]==area]
        lines=["| Model | Mean MAE (MW) | Mean RMSE (MW) | Mean MAPE | Fold MAPE SD |","|---|---:|---:|---:|---:|"]
        for r in rs:
            lines.append(f"| {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f}% | {r[7]:.2f} pp |")
        return "\n".join(lines)

    def weather_table(area):
        rs=[r for r in weather_rows if r[0]==area]
        lines=["| Date | Tmax °C | Tmin °C | Tmean °C | RH % | Rain |","|---|---:|---:|---:|---:|---:|"]
        for r in rs:
            lines.append(f"| {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} | {r[6]:.2f} |")
        return "\n".join(lines)

    report=f'''# Question 5 — 计及气象因素的短期负荷预测

## 1. 预测目标与信息边界

- 目标区间：2015-01-04 至 2015-01-10，共 7 天 × 96 个 15 min 点/地区。
- 已知负荷截止 2015-01-03。
- 本题使用提供的目标周气象数据作为外生变量。
- 为与 Q3 保持一致，重点检验平均温度及其二次项，同时把湿度、降雨和全部天气变量作为候选增强项。
- 严格防止负荷泄漏：所有负荷滞后均为 7、14、21、28、35、42、49、56 天；不会使用目标周真实负荷，也不使用未来真实 `lag_1d`。
- 历史回测使用每个验证周的观测天气模拟“天气预报可用”的场景，因此回测衡量的是**在天气信息准确可得时**的负荷模型性能；真实业务中还会叠加天气预报误差。

## 2. 与 Q3 的衔接

Q3 表明：

- 温度变量对负荷的解释力显著高于湿度和降雨；
- 温度—负荷关系存在非线性，`T + T²` 优于纯线性；
- Area 2 从平均温度获得的增益尤其明显。

因此 Q5 不盲目把所有气象变量全部塞入模型，而是在严格时间回测中比较四种 HGB 规格：

1. `no_weather`：控制组，仅日历 + 固定周滞后；
2. `mean_temp_quad`：加入平均温度和平均温度²；
3. `mean_temp_quad_hr`：再加入相对湿度和降雨；
4. `all_weather`：加入最高/最低/平均温度及平方项、湿度、降雨。

为公平比较，四个候选均使用共同训练起点 2012-01-01。

## 3. 数据质量与缺失策略

不修改源 CSV。沿用 Q3 已识别的明显气温录入异常，将其在模型特征中置为缺失值；`HistGradientBoostingRegressor` 原生支持缺失值分支，因此不人为猜测或插值异常气温。目标周气象数据完整。

## 4. Rolling-origin 7-day 回测

与 Q4 相同，在 2014 年使用 13 个约每四周一次的星期日起点。每个 fold 只使用预测起点之前的负荷训练，再使用该验证周天气预测后续完整七天。

### Area 1

{table('Area1')}

**最佳天气模型：{a1['label']} (`{a1['model']}`)**。

- 最佳天气模型平均 MAPE：**{a1['mape']:.2f}%**。
- 同训练窗口无天气控制组 MAPE：**{a1['control_mape']:.2f}%**。
- 相对 MAPE 改善：**{a1['improvement']:.2f}%**。
- 单日 MAPE：P10={a1['p10']:.2f}%，中位数={a1['p50']:.2f}%，P90={a1['p90']:.2f}%。

![Area1 weather model backtest](figures/area1_weather_model_backtest.svg)

### Area 2

{table('Area2')}

**最佳天气模型：{a2['label']} (`{a2['model']}`)**。

- 最佳天气模型平均 MAPE：**{a2['mape']:.2f}%**。
- 同训练窗口无天气控制组 MAPE：**{a2['control_mape']:.2f}%**。
- 相对 MAPE 改善：**{a2['improvement']:.2f}%**。
- 单日 MAPE：P10={a2['p10']:.2f}%，中位数={a2['p50']:.2f}%，P90={a2['p90']:.2f}%。

![Area2 weather model backtest](figures/area2_weather_model_backtest.svg)

## 5. 目标周气象条件

### Area 1

{weather_table('Area1')}

![Area1 target weather](figures/area1_target_week_weather.svg)

### Area 2

{weather_table('Area2')}

![Area2 target weather](figures/area2_target_week_weather.svg)

## 6. 2015-01-04 至 2015-01-10 最终预测

### Area 1

- 最终模型：`{a1['model']}`
- 预测均值：{a1['fmean']:.2f} MW
- 预测最大值：{a1['fmax']:.2f} MW
- 预测最小值：{a1['fmin']:.2f} MW

![Area1 Q5 forecast](figures/area1_q5_forecast.svg)

### Area 2

- 最终模型：`{a2['model']}`
- 预测均值：{a2['fmean']:.2f} MW
- 预测最大值：{a2['fmax']:.2f} MW
- 预测最小值：{a2['fmin']:.2f} MW

![Area2 Q5 forecast](figures/area2_q5_forecast.svg)

## 7. 天气是否值得加入

判断标准不是全样本 R²，而是相同历史外推任务下天气模型相对无天气控制组的 MAPE 改善。如果增益为正且跨 fold 稳定，说明天气提供了日历和历史负荷之外的额外信息；如果湿度/降雨规格没有继续改善，则不应为了变量数量而保留它们。

这一结果与 Q3 的变量筛选形成闭环：Q3 用日尺度回归确定哪些天气变量具有稳定解释/验证价值，Q5 再在真正的 15 分钟七日预测任务中检验其实际预测贡献。

## 8. 与 Q4 的关系

Q4 与 Q5 的最终模型训练窗口并不完全相同：Q4 可使用更早的纯负荷历史，而 Q5 的天气增强模型受天气数据从 2012 年开始的约束。因此，最严格的“天气净增益”应以本题的共同训练窗口 A/B 控制结果为准，而不是简单拿 Q4 的最终 backtest 数字相减。

Q4 仍作为“没有任何天气信息时”的正式预测基线；Q5 给出“天气信息可用时”的正式预测结果。

## 9. 提交与输出

分析目录生成：

- `Q5_ANALYSIS.md`
- `Q5_Area1_Load.csv`
- `Q5_Area2_Load.csv`
- `Q5_Area1_candidate_forecasts.csv`
- `Q5_Area2_candidate_forecasts.csv`
- `Q5_backtest_folds.csv`
- `Q5_model_summary.csv`
- `Q5_target_week_weather.csv`
- `q5_weather_forecast.py`
- `figures/area1_weather_model_backtest.svg`
- `figures/area2_weather_model_backtest.svg`
- `figures/area1_q5_forecast.svg`
- `figures/area2_q5_forecast.svg`
- `figures/area1_target_week_weather.svg`
- `figures/area2_target_week_weather.svg`

正式 XLSX 提交文件将在结果验证后从上述七日预测表无损转换，以保持与题目附件命名要求一致。
'''
    (OUT/"Q5_ANALYSIS.md").write_text(report,encoding="utf-8")


if __name__ == "__main__":
    main()
