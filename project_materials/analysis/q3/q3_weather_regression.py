from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "project_materials"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEATHER = ["最高温度", "最低温度", "平均温度", "相对湿度", "降雨量"]
TARGETS = ["load_max", "load_min", "load_mean"]
TARGET_LABELS = {
    "load_max": "Daily maximum load",
    "load_min": "Daily minimum load",
    "load_mean": "Daily mean load",
}
FACTOR_EN = {
    "最高温度": "Max temperature",
    "最低温度": "Min temperature",
    "平均温度": "Mean temperature",
    "相对湿度": "Relative humidity",
    "降雨量": "Rainfall",
}


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def mape(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((y - pred) / y)) * 100)


def md_table(df, floatfmt=".3f"):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_daily(area):
    load = pd.read_csv(DATA_DIR / f"{area}_Load.csv")
    weather = pd.read_csv(DATA_DIR / f"{area}_Weather.csv")

    load["date"] = pd.to_datetime(load["YMD"].astype(str), format="%Y%m%d")
    weather["date"] = pd.to_datetime(weather["日期"].astype(str), format="%Y%m%d")

    load = load[(load["date"] >= "2012-01-01") & (load["date"] <= "2014-12-31")].copy()
    weather = weather[(weather["date"] >= "2012-01-01") & (weather["date"] <= "2014-12-31")].copy()

    value_cols = [c for c in load.columns if c.startswith("T")]
    vals = load[value_cols].astype(float)
    daily = pd.DataFrame({
        "date": load["date"].values,
        "load_max": vals.max(axis=1).values,
        "load_min": vals.min(axis=1).values,
        "load_mean": vals.mean(axis=1).values,
    })

    merged = daily.merge(
        weather[["date"] + WEATHER],
        on="date",
        how="left",
        validate="one_to_one",
    )
    return merged


def clean_weather(df, area):
    d = df.copy()
    quality = []

    bad_max = d["最高温度"] > 45
    if bad_max.any():
        for dt, v in d.loc[bad_max, ["date", "最高温度"]].itertuples(index=False):
            quality.append([area, dt.date().isoformat(), "最高温度", v, "Excluded: obvious high-temperature transcription outlier"])
        d.loc[bad_max, "最高温度"] = np.nan

    bad_min = d["最低温度"] < -5
    if bad_min.any():
        for dt, v in d.loc[bad_min, ["date", "最低温度"]].itertuples(index=False):
            quality.append([area, dt.date().isoformat(), "最低温度", v, "Excluded: obvious low-temperature transcription outlier"])
        d.loc[bad_min, "最低温度"] = np.nan

    for factor in WEATHER:
        raw_missing = df[factor].isna()
        for dt in df.loc[raw_missing, "date"]:
            quality.append([area, dt.date().isoformat(), factor, np.nan, "Source missing value; omitted from relevant regression"])

    return d, quality


def correlation_results(df, area):
    corr = df[TARGETS + WEATHER].corr().loc[TARGETS, WEATHER]
    rows = []
    for t in TARGETS:
        for f in WEATHER:
            rows.append([area, t, f, corr.loc[t, f]])
    return pd.DataFrame(rows, columns=["area", "target", "factor", "pearson_r"])


def univariate_results(df, area):
    rows = []
    for target in TARGETS:
        for factor in WEATHER:
            sub = df[["date", target, factor]].dropna().copy()
            X = sub[[factor]]
            y = sub[target]

            model = LinearRegression().fit(X, y)
            pred = model.predict(X)
            rows.append([
                area, target, factor, "linear_full",
                len(sub), np.nan, len(sub),
                model.intercept_, model.coef_[0],
                r2_score(y, pred), rmse(y, pred),
                mean_absolute_error(y, pred), mape(y, pred),
            ])

            train = sub["date"] < "2014-01-01"
            test = ~train
            lr = LinearRegression().fit(sub.loc[train, [factor]], sub.loc[train, target])
            p = lr.predict(sub.loc[test, [factor]])
            rows.append([
                area, target, factor, "linear_2014_test",
                int(train.sum()), int(test.sum()), len(sub),
                lr.intercept_, lr.coef_[0],
                r2_score(sub.loc[test, target], p), rmse(sub.loc[test, target], p),
                mean_absolute_error(sub.loc[test, target], p), mape(sub.loc[test, target], p),
            ])

            if "温度" in factor:
                poly = make_pipeline(
                    PolynomialFeatures(degree=2, include_bias=False),
                    LinearRegression(),
                )
                poly.fit(sub.loc[train, [factor]], sub.loc[train, target])
                pq = poly.predict(sub.loc[test, [factor]])
                rows.append([
                    area, target, factor, "quadratic_2014_test",
                    int(train.sum()), int(test.sum()), len(sub),
                    np.nan, np.nan,
                    r2_score(sub.loc[test, target], pq), rmse(sub.loc[test, target], pq),
                    mean_absolute_error(sub.loc[test, target], pq), mape(sub.loc[test, target], pq),
                ])

    return pd.DataFrame(rows, columns=[
        "area", "target", "factor", "model",
        "n_train", "n_test", "n_total",
        "intercept", "slope",
        "R2", "RMSE", "MAE", "MAPE",
    ])


def calendar_features(sub):
    date = sub["date"]
    X = pd.DataFrame(index=sub.index)
    X["trend"] = (date - pd.Timestamp("2012-01-01")).dt.days.astype(float)
    doy = date.dt.dayofyear
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    dow = pd.get_dummies(date.dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    month = pd.get_dummies(date.dt.month, prefix="mon", drop_first=True, dtype=float)
    dow.index = X.index
    month.index = X.index
    return pd.concat([X, dow, month], axis=1)


def validation_results(df, area):
    rows = []
    candidates = [
        ("calendar", []),
        ("最高温度", ["最高温度"]),
        ("最低温度", ["最低温度"]),
        ("平均温度", ["平均温度"]),
        ("相对湿度", ["相对湿度"]),
        ("降雨量", ["降雨量"]),
        ("temp_mean+humidity+rain", ["平均温度", "相对湿度", "降雨量"]),
        ("all_weather", WEATHER),
    ]

    for target in TARGETS:
        for model_name, factors in candidates:
            sub = df[["date", target] + factors].dropna().copy()
            X = calendar_features(sub)
            for f in factors:
                X[f] = sub[f].astype(float)
                if "温度" in f:
                    X[f"{f}^2"] = sub[f].astype(float) ** 2

            train = sub["date"] < "2014-01-01"
            test = ~train
            model = LinearRegression().fit(X.loc[train], sub.loc[train, target])
            pred = model.predict(X.loc[test])
            rows.append([
                area, target, model_name, int(train.sum()), int(test.sum()),
                r2_score(sub.loc[test, target], pred),
                rmse(sub.loc[test, target], pred),
                mean_absolute_error(sub.loc[test, target], pred),
                mape(sub.loc[test, target], pred),
            ])

    return pd.DataFrame(rows, columns=[
        "area", "target", "model", "n_train", "n_test", "R2", "RMSE", "MAE", "MAPE"
    ])


def plot_corr(df, area):
    corr = df[TARGETS + WEATHER].corr().loc[TARGETS, WEATHER]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(WEATHER)), [FACTOR_EN[x] for x in WEATHER], rotation=25, ha="right")
    ax.set_yticks(range(len(TARGETS)), [TARGET_LABELS[x] for x in TARGETS])
    for i in range(len(TARGETS)):
        for j in range(len(WEATHER)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Weather factor")
    ax.set_ylabel("Load metric")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{area.lower()}_weather_correlation.svg")
    plt.close(fig)


def plot_full_r2(univ, area):
    temp = univ[(univ["area"] == area) & (univ["model"] == "linear_full") & (univ["target"] == "load_mean")].copy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(temp))
    ax.bar(x, temp["R2"].values)
    ax.set_xticks(x, [FACTOR_EN[f] for f in temp["factor"]], rotation=25, ha="right")
    ax.set_ylabel("R²")
    ax.set_xlabel("Weather factor")
    ax.set_ylim(0, max(0.65, temp["R2"].max() * 1.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{area.lower()}_loadmean_univariate_r2.svg")
    plt.close(fig)


def plot_validation(valid, area):
    show_models = ["calendar", "平均温度", "temp_mean+humidity+rain", "all_weather"]
    sub = valid[(valid["area"] == area) & (valid["model"].isin(show_models))].copy()
    labels = {
        "calendar": "Calendar only",
        "平均温度": "Calendar + mean temp²",
        "temp_mean+humidity+rain": "Calendar + mean temp² + RH + rain",
        "all_weather": "Calendar + all weather",
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.18
    targets_order = TARGETS
    x = np.arange(len(targets_order))
    for idx, model in enumerate(show_models):
        vals = [
            sub[(sub["target"] == t) & (sub["model"] == model)]["RMSE"].iloc[0]
            for t in targets_order
        ]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=labels[model])
    ax.set_xticks(x, [TARGET_LABELS[t] for t in targets_order])
    ax.set_ylabel("2014 holdout RMSE (MW)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{area.lower()}_validation_rmse.svg")
    plt.close(fig)


def plot_temp_response(df, area):
    sub = df[["date", "load_mean", "平均温度"]].dropna().copy()
    train = sub["date"] < "2014-01-01"
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )
    model.fit(sub.loc[train, ["平均温度"]], sub.loc[train, "load_mean"])
    grid = np.linspace(sub["平均温度"].min(), sub["平均温度"].max(), 200)
    pred = model.predict(pd.DataFrame({"平均温度": grid}))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(sub["平均温度"], sub["load_mean"], s=8, alpha=0.25)
    ax.plot(grid, pred, linewidth=2)
    ax.set_xlabel("Mean temperature (°C)")
    ax.set_ylabel("Daily mean load (MW)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{area.lower()}_mean_temperature_response.svg")
    plt.close(fig)


def compact_corr_table(corr_df, area):
    p = corr_df[corr_df["area"] == area].pivot(index="target", columns="factor", values="pearson_r")
    p = p.loc[TARGETS, WEATHER].reset_index()
    p["target"] = p["target"].map(TARGET_LABELS)
    p.columns = ["Load metric"] + WEATHER
    return p


def compact_r2_table(univ, area):
    p = univ[(univ["area"] == area) & (univ["model"] == "linear_full")].pivot(
        index="target", columns="factor", values="R2"
    )
    p = p.loc[TARGETS, WEATHER].reset_index()
    p["target"] = p["target"].map(TARGET_LABELS)
    p.columns = ["Load metric"] + WEATHER
    return p


def compact_quad_table(univ, area):
    p = univ[(univ["area"] == area) & (univ["model"] == "quadratic_2014_test")].pivot(
        index="target", columns="factor", values="R2"
    )
    p = p.loc[TARGETS, ["最高温度", "最低温度", "平均温度"]].reset_index()
    p["target"] = p["target"].map(TARGET_LABELS)
    p.columns = ["Load metric", "最高温度", "最低温度", "平均温度"]
    return p


def compact_validation_table(valid, area):
    keep = ["calendar", "最高温度", "最低温度", "平均温度", "相对湿度", "降雨量",
            "temp_mean+humidity+rain", "all_weather"]
    rows = []
    for target in TARGETS:
        b = valid[(valid["area"] == area) & (valid["target"] == target) & (valid["model"] == "calendar")].iloc[0]
        for model in keep:
            r = valid[(valid["area"] == area) & (valid["target"] == target) & (valid["model"] == model)].iloc[0]
            rows.append([
                TARGET_LABELS[target], model, r["RMSE"], r["MAPE"],
                (b["RMSE"] - r["RMSE"]) / b["RMSE"] * 100,
            ])
    return pd.DataFrame(rows, columns=["Load metric", "Model", "RMSE (MW)", "MAPE (%)", "RMSE improvement vs calendar (%)"])


def main():
    all_quality = []
    data = {}
    for area in ["Area1", "Area2"]:
        raw = load_daily(area)
        clean, quality = clean_weather(raw, area)
        data[area] = clean
        all_quality.extend(quality)

    quality_df = pd.DataFrame(all_quality, columns=["area", "date", "factor", "raw_value", "treatment"])
    corr = pd.concat([correlation_results(data[a], a) for a in data], ignore_index=True)
    univ = pd.concat([univariate_results(data[a], a) for a in data], ignore_index=True)
    valid = pd.concat([validation_results(data[a], a) for a in data], ignore_index=True)

    quality_df.to_csv(OUT_DIR / "Q3_data_quality.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT_DIR / "Q3_correlations.csv", index=False, encoding="utf-8-sig")
    univ.to_csv(OUT_DIR / "Q3_univariate_regression.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(OUT_DIR / "Q3_validation_metrics.csv", index=False, encoding="utf-8-sig")

    for area in data:
        plot_corr(data[area], area)
        plot_full_r2(univ, area)
        plot_validation(valid, area)
        plot_temp_response(data[area], area)

    temp_corr = {}
    for area, d in data.items():
        temp_corr[area] = d[["最高温度", "最低温度", "平均温度"]].corr()

    lines = []
    lines.append("# Question 3 — 气象因素与日负荷回归分析")
    lines.append("")
    lines.append("## 1. 分析目标与数据")
    lines.append("")
    lines.append("- 时间范围：2012-01-01 至 2014-12-31，共 1,096 天/地区。")
    lines.append("- 因变量：日最高负荷、日最低负荷、日平均负荷。")
    lines.append("- 气象变量：日最高温度、日最低温度、日平均温度、相对湿度、降雨量。")
    lines.append("- 负荷由每天 96 个 15 分钟采样点汇总得到。")
    lines.append("- 为避免只用拟合优度判断变量价值，除全样本回归外，还采用时间顺序验证：2012–2013 训练、2014 独立测试。")
    lines.append("")
    lines.append("## 2. 数据质量处理")
    lines.append("")
    lines.append("不修改原始 CSV，仅在受影响变量的回归中排除明显录入异常或缺失值。")
    if len(quality_df):
        qshow = quality_df.copy()
        qshow["raw_value"] = qshow["raw_value"].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        lines.append("")
        lines.append(md_table(qshow))
    lines.append("")
    lines.append("其中 Area 1 的两个 `最高温度=54.9°C`、Area 2 的四个明显异常负最低温度，以及源文件已有的少量缺失值均未被人工猜测替换；这可以避免错误插值影响变量选择。")
    lines.append("")
    lines.append("## 3. 相关性与单变量线性回归")
    for area in ["Area1", "Area2"]:
        lines.append("")
        lines.append(f"### {area}")
        lines.append("")
        lines.append("Pearson 相关系数：")
        lines.append("")
        lines.append(md_table(compact_corr_table(corr, area)))
        lines.append("")
        lines.append("全样本单变量线性回归 R²：")
        lines.append("")
        lines.append(md_table(compact_r2_table(univ, area)))
        lines.append("")
        lines.append(f"![{area} weather correlation](figures/{area.lower()}_weather_correlation.svg)")
        lines.append("")
        lines.append(f"![{area} univariate R2](figures/{area.lower()}_loadmean_univariate_r2.svg)")
    lines.append("")
    lines.append("总体上，三个温度变量与负荷的线性关系明显强于湿度和降雨。Area 1 的温度变量可解释约 34%–46% 的全样本负荷变化；Area 2 约为 46%–57%。湿度与降雨的单变量 R² 仅约 0.5%–2%，单独使用时解释力很弱。")
    lines.append("")
    lines.append("## 4. 温度非线性与 2014 独立验证")
    lines.append("")
    lines.append("温度与用电需求并不一定严格线性，因此对三个温度变量额外测试二次回归 `load ~ T + T²`，仍严格使用 2012–2013 拟合并在 2014 测试。")
    for area in ["Area1", "Area2"]:
        lines.append("")
        lines.append(f"### {area}：二次温度模型 2014 测试 R²")
        lines.append("")
        lines.append(md_table(compact_quad_table(univ, area)))
        lines.append("")
        lines.append(f"![{area} mean-temperature response](figures/{area.lower()}_mean_temperature_response.svg)")
    lines.append("")
    lines.append("二次项普遍改善 2014 外推表现，说明温度—负荷关系存在明显非线性。Area 2 的提升尤其明显，因此 Question 5 中不应只使用单一线性温度系数。")
    lines.append("")
    lines.append("## 5. 气象因素是否能在时间规律之外继续降低误差")
    lines.append("")
    lines.append("为了避免把季节性误认为气象因果关系，再建立一个简单的日历基准（趋势、年周期、月份、星期），并逐项加入天气变量。温度以 `T + T²` 形式加入。所有指标仍为 2014 独立测试结果。")
    for area in ["Area1", "Area2"]:
        lines.append("")
        lines.append(f"### {area}")
        lines.append("")
        vt = compact_validation_table(valid, area)
        vt["Model"] = vt["Model"].replace({
            "calendar": "Calendar only",
            "最高温度": "Calendar + max temperature²",
            "最低温度": "Calendar + min temperature²",
            "平均温度": "Calendar + mean temperature²",
            "相对湿度": "Calendar + humidity",
            "降雨量": "Calendar + rainfall",
            "temp_mean+humidity+rain": "Calendar + mean temperature² + humidity + rainfall",
            "all_weather": "Calendar + all weather",
        })
        lines.append(md_table(vt, floatfmt=".2f"))
        lines.append("")
        lines.append(f"![{area} validation RMSE](figures/{area.lower()}_validation_rmse.svg)")
    lines.append("")
    lines.append("### 5.1 Area 1")
    lines.append("")
    lines.append("- 日历基准 RMSE：日最高约 1192.7 MW、日平均约 940.6 MW、日最低约 704.8 MW。")
    lines.append("- 加入平均温度二次项后，RMSE 分别降至约 1148.2、887.0、640.0 MW。")
    lines.append("- 最低温度的表现与平均温度非常接近，在日最高/日平均负荷上略优；但平均温度数据更稳定、缺失/异常更少，也更适合作为统一主变量。")
    lines.append("- 湿度和降雨单独加入几乎没有稳定改善。")
    lines.append("")
    lines.append("### 5.2 Area 2")
    lines.append("")
    lines.append("- 日历基准 RMSE：日最高约 1040.9 MW、日平均约 889.8 MW、日最低约 742.2 MW。")
    lines.append("- 加入平均温度二次项后，RMSE 分别降至约 893.5、718.7、585.1 MW，改善明显。")
    lines.append("- 平均温度在三个目标上均是最稳定的单一气象变量。")
    lines.append("- 湿度和降雨单独作用仍弱；与平均温度组合后，仅对部分目标有很小的额外改善。")
    lines.append("")
    lines.append("## 6. 多重共线性与变量选择")
    lines.append("")
    lines.append("三个温度指标彼此高度相关：")
    for area in ["Area1", "Area2"]:
        tc = temp_corr[area].reset_index()
        tc.columns = ["Variable", "最高温度", "最低温度", "平均温度"]
        lines.append("")
        lines.append(f"### {area}")
        lines.append(md_table(tc))
    lines.append("")
    lines.append("因此不建议在普通线性回归中同时无约束地放入最高、最低、平均温度并直接解释系数；系数会因共线性变得不稳定。若后续机器学习模型使用全部温度变量，应依靠时间序列回测、正则化或树模型控制这一问题。")
    lines.append("")
    lines.append("## 7. 推荐用于提高负荷预测精度的气象因素")
    lines.append("")
    lines.append("**首选：日平均温度。**")
    lines.append("")
    lines.append("理由：")
    lines.append("")
    lines.append("1. 对两个地区、三个负荷目标都保持较高相关性和单变量 R²；")
    lines.append("2. 在 2014 独立测试中，加入平均温度二次项能稳定降低日历基准 RMSE；")
    lines.append("3. Area 2 的改善尤其显著，说明温度对其负荷具有较强预测价值；")
    lines.append("4. 与最高/最低温度相比，平均温度在本数据中没有明显异常录入问题，工程上更稳健。")
    lines.append("")
    lines.append("**次选：最低温度（Area 1 可重点测试）以及相对湿度。**")
    lines.append("")
    lines.append("- Area 1 的最低温度在部分 2014 验证指标上与平均温度持平或略优，可作为候选补充变量。")
    lines.append("- 相对湿度单独解释力很弱，但可能与高温共同作用；Question 5 中可作为交互/辅助变量通过回测决定是否保留。")
    lines.append("- 降雨量的单变量预测价值最低之一，不建议作为核心特征；可以保留为候选变量，但必须以回测是否改善为准。")
    lines.append("")
    lines.append("## 8. 对 Question 5 的建模建议")
    lines.append("")
    lines.append("后续天气增强预测优先采用以下特征层级：")
    lines.append("")
    lines.append("1. 基础：历史负荷滞后 + 星期/月/节假日等日历特征；")
    lines.append("2. 必选天气：平均温度及非线性项（如 `T²`，或树模型自动学习非线性）；")
    lines.append("3. 候选天气：最高温度、最低温度、相对湿度、降雨量；")
    lines.append("4. 是否保留候选变量只根据滚动时间序列回测决定，不能依据训练集拟合优度。")
    lines.append("")
    lines.append("## 9. 输出文件")
    lines.append("")
    lines.append("- `Q3_ANALYSIS.md`")
    lines.append("- `Q3_data_quality.csv`")
    lines.append("- `Q3_correlations.csv`")
    lines.append("- `Q3_univariate_regression.csv`")
    lines.append("- `Q3_validation_metrics.csv`")
    lines.append("- `q3_weather_regression.py`")
    lines.append("- `figures/area1_weather_correlation.svg`")
    lines.append("- `figures/area2_weather_correlation.svg`")
    lines.append("- `figures/area1_loadmean_univariate_r2.svg`")
    lines.append("- `figures/area2_loadmean_univariate_r2.svg`")
    lines.append("- `figures/area1_validation_rmse.svg`")
    lines.append("- `figures/area2_validation_rmse.svg`")
    lines.append("- `figures/area1_mean_temperature_response.svg`")
    lines.append("- `figures/area2_mean_temperature_response.svg`")

    (OUT_DIR / "Q3_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
