from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]  # project_materials
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_RANGE = range(2, 9)

def load_area(area):
    path = ROOT / f"Area{area}_Load.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["YMD"].astype(str), format="%Y%m%d")
    df = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2014-12-31")].copy()
    time_cols = [c for c in df.columns if c.startswith("T")]
    X = df[time_cols].to_numpy(dtype=float)
    return df, time_cols, X

def load_weather(area):
    path = ROOT / f"Area{area}_Weather.csv"
    w = pd.read_csv(path)
    w["date"] = pd.to_datetime(w["日期"].astype(int).astype(str), format="%Y%m%d")
    return w[(w["date"] >= "2014-01-01") & (w["date"] <= "2014-12-31")].copy()

def evaluate_k(Xs):
    rows = []
    labels_by_k = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = km.fit_predict(Xs)
        rows.append({"k": k, "silhouette": silhouette_score(Xs, labels), "inertia": km.inertia_})
        labels_by_k[k] = labels
    return pd.DataFrame(rows), labels_by_k

def relabel_by_mean(labels, daily_mean):
    tmp = pd.DataFrame({"raw": labels, "mean": daily_mean})
    order = tmp.groupby("raw")["mean"].mean().sort_values().index.tolist()
    mapping = {raw: i + 1 for i, raw in enumerate(order)}
    return np.array([mapping[x] for x in labels])

def time_to_hour(tcol):
    hh = int(tcol[1:3])
    mm = int(tcol[3:5])
    return hh + mm / 60

def analyze_area(area):
    df, time_cols, X = load_area(area)
    weather = load_weather(area)

    # Standardize each clock-time feature across days. This preserves daily
    # level/shape differences while preventing high-magnitude times dominating.
    Xs = StandardScaler().fit_transform(X)
    validity, labels_by_k = evaluate_k(Xs)
    best_k = int(validity.loc[validity["silhouette"].idxmax(), "k"])
    labels = relabel_by_mean(labels_by_k[best_k], X.mean(axis=1))

    out = pd.DataFrame({
        "date": df["date"].values,
        "area": area,
        "cluster": labels,
        "daily_mean": X.mean(axis=1),
        "daily_max": X.max(axis=1),
        "daily_min": X.min(axis=1),
    })
    out["peak_valley"] = out["daily_max"] - out["daily_min"]
    out["load_factor"] = out["daily_mean"] / out["daily_max"]
    out["weekday"] = pd.to_datetime(out["date"]).dt.day_name()
    out["is_weekend"] = pd.to_datetime(out["date"]).dt.weekday >= 5
    out["month"] = pd.to_datetime(out["date"]).dt.month
    out = out.merge(
        weather[["date", "最高温度", "最低温度", "平均温度", "相对湿度", "降雨量"]],
        on="date", how="left"
    )

    summary = out.groupby("cluster").agg(
        n_days=("date", "size"),
        share=("date", lambda x: len(x) / len(out)),
        mean_load_MW=("daily_mean", "mean"),
        mean_peak_MW=("daily_max", "mean"),
        mean_valley_MW=("daily_min", "mean"),
        mean_peak_valley_MW=("peak_valley", "mean"),
        mean_load_factor=("load_factor", "mean"),
        weekend_share=("is_weekend", "mean"),
        mean_temp_C=("平均温度", "mean"),
        mean_max_temp_C=("最高温度", "mean"),
        mean_humidity_pct=("相对湿度", "mean"),
        mean_rainfall=("降雨量", "mean"),
    ).reset_index()
    summary.insert(0, "area", area)

    month_dist = pd.crosstab(out["cluster"], out["month"]).reset_index()
    month_dist.insert(0, "area", area)

    profile_rows = []
    for c in sorted(np.unique(labels)):
        mean_curve = X[labels == c].mean(axis=0)
        for t, val in zip(time_cols, mean_curve):
            profile_rows.append({
                "area": area, "cluster": int(c), "time": t,
                "hour": time_to_hour(t), "mean_load_MW": float(val)
            })
    profiles = pd.DataFrame(profile_rows)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(Xs)
    pca_df = pd.DataFrame({
        "area": area,
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "cluster": labels,
        "PC1": pts[:, 0],
        "PC2": pts[:, 1],
    })
    pca_var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(validity["k"], validity["silhouette"], marker="o")
    ax.axvline(best_k, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette coefficient")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"area{area}_silhouette.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in sorted(profiles["cluster"].unique()):
        d = profiles[profiles["cluster"] == c]
        ax.plot(d["hour"], d["mean_load_MW"], label=f"Cluster {c}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Load (MW)")
    ax.set_xlim(0, 23.75)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"area{area}_cluster_profiles.svg", format="svg")
    plt.close(fig)

    month_counts = pd.crosstab(out["month"], out["cluster"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    month_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of days")
    ax.legend(title="Cluster")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"area{area}_cluster_months.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(pts[m, 0], pts[m, 1], s=18, alpha=0.75, label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({pca_var[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca_var[1] * 100:.1f}% var.)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"area{area}_pca_clusters.svg", format="svg")
    plt.close(fig)

    return validity.assign(area=area), out, summary, month_dist, profiles, pca_df, best_k

def write_report(all_validity, summary_all, best_ks):
    def pct(x): return f"{x * 100:.1f}%"
    lines = [
        "# Question 2 — KMeans 日负荷曲线聚类分析", "",
        "## 1. 方法", "",
        "- 分析期间：2014-01-01 至 2014-12-31，每个地区 365 天。",
        "- 每天使用 96 个 15 分钟负荷点作为一个样本，因此原始特征维度为 96。",
        "- 聚类前对每个 15 分钟特征在全年 365 天之间进行 `StandardScaler` 标准化。",
        "  这样保留日负荷水平与曲线形态差异，同时避免某些绝对负荷较高的时刻在欧氏距离中占据过大权重。",
        "- 对 `k = 2...8` 分别运行 KMeans，`random_state=42`，`n_init=50`。",
        "- 使用 Silhouette coefficient 选择最优 k。数值越接近 1，类内越紧密、类间越分离。",
        "- 聚类标签本身没有自然顺序，因此最终按各簇平均日负荷从低到高重新编号为 Cluster 1、2、3。",
        "- 气象数据只用于解释聚类，不参与本题 KMeans 拟合。", "",
        "## 2. 聚类有效性", ""
    ]
    for area in [1, 2]:
        v = all_validity[all_validity["area"] == area].sort_values("k")
        lines += [f"### Area {area}", "", "| k | Silhouette |", "|---:|---:|"]
        for _, r in v.iterrows():
            lines.append(f"| {int(r['k'])} | {r['silhouette']:.4f} |")
        lines += ["", f"**最优 k = {best_ks[area]}**。", "", f"![Area {area} silhouette](figures/area{area}_silhouette.svg)", ""]
    lines += [
        "两个地区的轮廓系数均在 **k=3** 时达到最高，因此后续统一采用 3 类，便于比较两地区的负荷类型。", "",
        "## 3. 聚类特征", ""
    ]
    for area in [1, 2]:
        s = summary_all[summary_all["area"] == area].sort_values("cluster")
        lines += [
            f"### Area {area}", "",
            "| Cluster | 天数 | 占比 | 日均负荷 MW | 日峰值 MW | 日谷值 MW | 峰谷差 MW | 日负荷率 | 周末占比 | 平均温度 °C | 日最高温 °C |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ]
        for _, r in s.iterrows():
            lines.append(
                f"| {int(r['cluster'])} | {int(r['n_days'])} | {pct(r['share'])} | "
                f"{r['mean_load_MW']:.2f} | {r['mean_peak_MW']:.2f} | {r['mean_valley_MW']:.2f} | "
                f"{r['mean_peak_valley_MW']:.2f} | {pct(r['mean_load_factor'])} | "
                f"{pct(r['weekend_share'])} | {r['mean_temp_C']:.2f} | {r['mean_max_temp_C']:.2f} |"
            )
        lines += [
            "", f"![Area {area} mean cluster profiles](figures/area{area}_cluster_profiles.svg)", "",
            f"![Area {area} cluster by month](figures/area{area}_cluster_months.svg)", "",
            f"![Area {area} PCA view](figures/area{area}_pca_clusters.svg)", ""
        ]
    lines += [
        "## 4. 语义解释与分类标签", "",
        "### Cluster 1：低负荷 / 节假日型曲线簇", "",
        "- 两地区 Cluster 1 都是全年平均负荷最低的一组。",
        "- 日期高度集中在 1 月下旬至 2 月上旬，同时也出现少量其他低负荷日期。",
        "- 周末比例并不足以单独解释该簇，因此更合适的名称是‘低负荷/节假日型’，而不是简单称为‘周末型’。",
        "- 其特征是全天负荷整体下移，代表特殊休息日、长假或其他低需求状态。", "",
        "### Cluster 2：正常 / 常规负荷曲线簇", "",
        "- 这是两个地区规模最大的类别，约占全年六成。",
        "- 主要覆盖冬春、秋冬的普通日期，负荷水平居中。",
        "- 可作为常规工作日与一般周末混合情况下的主导基准型曲线。", "",
        "### Cluster 3：高温 / 夏季高负荷曲线簇", "",
        "- 两地区 Cluster 3 几乎都集中在 5–10 月，6–9 月最密集。",
        "- 平均温度和日最高温度显著高于 Cluster 1/2。",
        "- 日均负荷、日峰值和日谷值均明显上升，符合高温期空调负荷增加的预期。",
        "- 因此可合理标记为‘高温/夏季高负荷型’。", "",
        "## 5. 两地区比较", "",
        "- Area 2 的最佳三类轮廓系数高于 Area 1，说明 Area 2 的三个日负荷状态分离得更清晰。",
        "- Area 2 的高负荷簇持续天数略多，并且在夏季月份分布更连续。",
        "- Area 1 的低负荷簇天数略多，特殊低负荷日期对全年结构的扰动更明显。",
        "- 这与 Question 1 中 Area 2 具有更强规律性、较低简单预测误差的初步判断一致。", "",
        "## 6. 在后续预测中的使用方法", "",
        "本题生成的 Cluster 标签可作为 Question 4/5 的候选分类特征，但必须避免数据泄漏：", "",
        "1. 对历史训练日期，可以直接使用已经得到的聚类标签。",
        "2. 对未来预测日期，不能使用未来真实负荷再进行聚类后把标签作为输入。",
        "3. 正确做法是根据可提前获知的变量（月份、星期、节假日、天气等）训练一个‘日类型分类器’或建立规则，将未来日期映射到 Cluster 1/2/3。",
        "4. 另一种做法是将聚类只用于解释和分组建模，而不直接作为未来未知特征。", "",
        "## 7. 输出文件", "",
        "- `Q2_ANALYSIS.md`",
        "- `Q2_cluster_labels_2014.csv`",
        "- `Q2_cluster_summary.csv`",
        "- `Q2_silhouette_scores.csv`",
        "- `Q2_cluster_profiles.csv`",
        "- `Q2_month_distribution.csv`",
        "- `Q2_pca_points.csv`",
        "- `q2_kmeans_analysis.py`",
        "- `figures/area1_silhouette.svg`, `figures/area2_silhouette.svg`",
        "- `figures/area1_cluster_profiles.svg`, `figures/area2_cluster_profiles.svg`",
        "- `figures/area1_cluster_months.svg`, `figures/area2_cluster_months.svg`",
        "- `figures/area1_pca_clusters.svg`, `figures/area2_pca_clusters.svg`", ""
    ]
    (OUT / "Q2_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    valids = []
    labels = []
    sums = []
    months = []
    profiles = []
    pcas = []
    best = {}
    for area in [1, 2]:
        v, l, s, m, p, pc, k = analyze_area(area)
        valids.append(v)
        labels.append(l)
        sums.append(s)
        months.append(m)
        profiles.append(p)
        pcas.append(pc)
        best[area] = k

    all_validity = pd.concat(valids, ignore_index=True)
    labels_all = pd.concat(labels, ignore_index=True)
    summary_all = pd.concat(sums, ignore_index=True)
    month_all = pd.concat(months, ignore_index=True)
    profiles_all = pd.concat(profiles, ignore_index=True)
    pca_all = pd.concat(pcas, ignore_index=True)

    all_validity[["area", "k", "silhouette", "inertia"]].to_csv(OUT / "Q2_silhouette_scores.csv", index=False)
    labels_all.to_csv(OUT / "Q2_cluster_labels_2014.csv", index=False, date_format="%Y-%m-%d")
    summary_all.to_csv(OUT / "Q2_cluster_summary.csv", index=False)
    month_all.to_csv(OUT / "Q2_month_distribution.csv", index=False)
    profiles_all.to_csv(OUT / "Q2_cluster_profiles.csv", index=False)
    pca_all.to_csv(OUT / "Q2_pca_points.csv", index=False)
    write_report(all_validity, summary_all, best)

if __name__ == "__main__":
    main()
