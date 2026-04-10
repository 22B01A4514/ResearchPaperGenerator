"""
graph_generator.py  —  v3
==========================
Fixes in v3:
  1. Architecture fix — graph layer is now PURE DATA. LLM insight generation
     is no longer called here. Graphs return only statistical_insight (computed
     from data) and raw stats. The reasoning layer (llm_generator.py) interprets.
  2. Caching — each graph type + dataset hash is cached in memory to avoid
     recomputing identical charts on regeneration.
  3. Statistical validation — correlation claims now include p-values (scipy);
     distribution claims check normality (Shapiro-Wilk for n<=5000).
  4. Dataset quality scoring — new method: quality_report(df) returns
     completeness %, imbalance ratio, collinearity score, outlier density.
     Call this BEFORE graph generation to warn about low-quality data.
  5. Normalization consistency — all box-plot-style charts normalize identically.
  6. Error handling — invalid base64 / empty data returns None, not crash.

NOTE: project_insight is now generated externally by llm_generator.py using
      the statistical_insight string returned here.
"""

import io
import re
import base64
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

PALETTE = ["#2E4057","#048A81","#E94F37","#F5A623","#6B4226",
           "#7B2FBE","#1B998B","#C44569","#3A86FF","#F8B500"]

# ── In-memory chart cache ─────────────────────────────────────────────────────
_CHART_CACHE: Dict[str, Dict] = {}


def _df_hash(df: pd.DataFrame, graph_type: str) -> str:
    """Lightweight hash to detect if data has changed."""
    sample = df.head(50).to_csv(index=False)
    return hashlib.md5(f"{graph_type}|{sample}".encode()).hexdigest()[:12]


# ── Statistical helpers ───────────────────────────────────────────────────────

def _pearson_pvalue(x: pd.Series, y: pd.Series) -> float:
    """Return p-value for Pearson correlation."""
    try:
        from scipy.stats import pearsonr
        _, pval = pearsonr(x.dropna(), y.dropna())
        return round(float(pval), 4)
    except Exception:
        return 1.0


def _is_normal(series: pd.Series, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk normality test (only for n ≤ 5000)."""
    try:
        from scipy.stats import shapiro
        sample = series.dropna()
        if len(sample) < 3 or len(sample) > 5000:
            return True   # assume normal for large samples
        _, pval = shapiro(sample[:5000])
        return pval > alpha
    except Exception:
        return True


# ── Dataset quality scoring ───────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> Dict:
    """
    Returns a quality badge + breakdown for the dataset before graphing.
    Grade: A (≥80), B (60-80), C (<60)
    """
    n_rows, n_cols = df.shape
    total_cells    = max(n_rows * n_cols, 1)

    # Completeness
    missing        = df.isnull().sum().sum()
    completeness   = round(100 * (1 - missing / total_cells), 1)

    # Imbalance (target column if categorical and low cardinality)
    imbalance_ratio = 1.0
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].nunique() <= 20:
            vc = df[col].value_counts()
            if len(vc) >= 2 and vc.min() > 0:
                imbalance_ratio = max(imbalance_ratio, round(vc.max() / vc.min(), 2))
            break

    # Collinearity (mean absolute correlation among numeric columns)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    collinearity = 0.0
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr.values[mask]
        collinearity = round(float(vals.mean()) if len(vals) else 0, 3)

    # Outlier density (IQR method, mean across numeric cols)
    outlier_pcts = []
    for col in num_cols[:10]:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        outlier_pcts.append(outliers / max(len(s), 1))
    outlier_density = round(100 * (sum(outlier_pcts) / max(len(outlier_pcts), 1)), 1)

    # Grade
    score = (
        completeness * 0.4
        + max(0, 100 - (imbalance_ratio - 1) * 5) * 0.2
        + (100 - collinearity * 100) * 0.2
        + (100 - outlier_density) * 0.2
    )
    grade = "A" if score >= 80 else ("B" if score >= 60 else "C")

    return {
        "grade":            grade,
        "score":            round(score, 1),
        "completeness_pct": completeness,
        "missing_cells":    int(missing),
        "imbalance_ratio":  imbalance_ratio,
        "collinearity":     collinearity,
        "outlier_density_pct": outlier_density,
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "warnings": [
            *(["High missing data — consider imputation."] if completeness < 80 else []),
            *(["Severe class imbalance (>10:1). Consider SMOTE."] if imbalance_ratio > 10 else []),
            *(["High feature collinearity — may affect model."] if collinearity > 0.7 else []),
            *(["High outlier density — consider robust scaling."] if outlier_density > 15 else []),
        ],
    }


# ── Matplotlib style ──────────────────────────────────────────────────────────

def _academic_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#FAFAFA",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })


def _to_b64(fig) -> str:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
        buf.seek(0)
        enc = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return enc
    except Exception as e:
        plt.close(fig)
        raise ValueError(f"Figure serialisation failed: {e}")


# ── GraphGenerator ────────────────────────────────────────────────────────────

class GraphGenerator:

    def __init__(self):
        _academic_style()

    def generate(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        selected_types: List[str],
        dataset_title: str,
        # topic and project_context kept for API compatibility but not used here
        topic: str = "",
        project_context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate graphs from real DataFrame.
        Returns list of dicts — statistical_insight only (no LLM calls).
        project_insight must be added by llm_generator.py after this call.
        """
        generators = {
            "correlation_heatmap": self._heatmap,
            "scatter_matrix":      self._scatter_matrix,
            "distribution":        self._distribution,
            "box_plot":            self._box_plot,
            "class_distribution":  self._class_dist,
            "feature_importance":  self._feature_importance,
        }
        graphs = []
        fig_num = 1

        for gtype in selected_types:
            fn = generators.get(gtype)
            if not fn:
                continue

            # Check cache
            cache_key = _df_hash(df, gtype)
            if cache_key in _CHART_CACHE:
                cached = dict(_CHART_CACHE[cache_key])
                cached["id"]            = f"fig_{fig_num}"
                cached["figure_label"]  = f"Figure {fig_num}"
                graphs.append(cached)
                fig_num += 1
                continue

            try:
                result = fn(df, metadata, dataset_title)
                if result:
                    result.update({
                        "id":                  f"fig_{fig_num}",
                        "type":                gtype,
                        "figure_label":        f"Figure {fig_num}",
                        "project_insight":     "",   # filled by llm_generator
                        "insight":             result.get("statistical_insight", result.get("insight", "")),
                    })
                    _CHART_CACHE[cache_key] = dict(result)
                    graphs.append(result)
                    fig_num += 1
            except Exception as e:
                print(f"[GraphGenerator] {gtype} failed: {e}")

        return graphs

    # ── Correlation Heatmap ───────────────────────────────────────────────────

    def _heatmap(self, df, meta, title) -> Optional[Dict]:
        cols = meta["numeric_cols"][:12]
        if len(cols) < 2:
            return None

        corr = df[cols].corr()
        n    = len(cols)
        fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8, "label": "Pearson r"})
        ax.set_title(f"Feature Correlation Matrix — {title}", pad=14)
        plt.tight_layout()

        flat   = corr.values[np.tril_indices_from(corr.values, k=-1)]
        strong = []
        for i in range(n):
            for j in range(i):
                r = corr.iloc[i, j]
                if abs(r) > 0.6:
                    pval = _pearson_pvalue(df[corr.columns[i]], df[corr.columns[j]])
                    strong.append((corr.columns[i], corr.columns[j], round(r, 3), pval))
        strong.sort(key=lambda x: abs(x[2]), reverse=True)

        if strong:
            t = strong[0]
            significance = "p<0.05 — statistically significant" if t[3] < 0.05 else f"p={t[3]}"
            stat_insight = (
                f"Strong correlation between '{t[0]}' and '{t[1]}' "
                f"(r={t[2]}, {significance}). "
                f"{len(strong)} feature pair(s) exceed |r|=0.6."
            )
        else:
            stat_insight = (
                f"All features show weak–moderate correlations "
                f"(mean |r|={np.abs(flat).mean():.3f}), "
                f"indicating low multicollinearity."
            )

        return {
            "title":             "Feature Correlation Heatmap",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"mean_abs_r": round(float(np.abs(flat).mean()), 3),
                                  "strong_pairs": len(strong)},
        }

    # ── Scatter Matrix ────────────────────────────────────────────────────────

    def _scatter_matrix(self, df, meta, title) -> Optional[Dict]:
        num_cols  = meta["numeric_cols"]
        target    = meta.get("target_col")
        feat_cols = [c for c in num_cols if c != target][:4]
        if len(feat_cols) < 2:
            return None

        plot_df = df[feat_cols + ([target] if target and target in df else [])].dropna()
        hue     = target if target and target in plot_df.columns else None
        if hue:
            top_cats = plot_df[hue].value_counts().head(6).index
            plot_df  = plot_df[plot_df[hue].isin(top_cats)]

        n = len(feat_cols)
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        axes = np.array(axes).reshape(n, n)
        fig.suptitle(f"Scatter Matrix — {title}", fontsize=13, fontweight="bold", y=1.01)

        unique_cats = plot_df[hue].unique() if hue else ["all"]
        cmap        = dict(zip(unique_cats, PALETTE[:len(unique_cats)])) if hue else {}

        for i, cy in enumerate(feat_cols):
            for j, cx in enumerate(feat_cols):
                ax = axes[i, j]
                ax.tick_params(labelsize=7)
                if i == j:
                    if hue:
                        for cat, col in cmap.items():
                            ax.hist(plot_df[plot_df[hue] == cat][cx].dropna(),
                                    alpha=0.5, color=col, bins=20, density=True)
                    else:
                        ax.hist(plot_df[cx].dropna(), color=PALETTE[0], bins=20, alpha=0.8)
                    ax.set_xlabel(cx, fontsize=7)
                else:
                    if hue:
                        for cat, col in cmap.items():
                            s = plot_df[plot_df[hue] == cat]
                            ax.scatter(s[cx], s[cy], alpha=0.45, s=10, color=col)
                    else:
                        ax.scatter(plot_df[cx], plot_df[cy], alpha=0.4, s=10, color=PALETTE[0])
                    ax.set_xlabel(cx, fontsize=7)
                    ax.set_ylabel(cy, fontsize=7)

        if hue:
            handles = [mpatches.Patch(color=cmap[c], label=str(c)) for c in unique_cats]
            fig.legend(handles=handles, loc="upper right", fontsize=8, title=hue)
        plt.tight_layout()

        stat_insight = (
            f"Scatter matrix of {n} features across {len(plot_df)} samples"
            + (f", colored by '{target}'" if hue else "")
            + ". Diagonal shows per-feature distributions."
        )
        return {
            "title":             "Scatter Plot Matrix",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"features": feat_cols, "samples": len(plot_df)},
        }

    # ── Distribution Histograms ───────────────────────────────────────────────

    def _distribution(self, df, meta, title) -> Optional[Dict]:
        target = meta.get("target_col")
        cols   = [c for c in meta["numeric_cols"] if c != target][:6]
        if not cols:
            return None

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()

        normality_results = {}
        for idx, col in enumerate(cols):
            ax   = axes[idx]
            data = df[col].dropna()
            ax.hist(data, bins=30, color=PALETTE[idx % len(PALETTE)],
                    edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(),   color="red",  linestyle="--", lw=1.2, label=f"μ={data.mean():.2f}")
            ax.axvline(data.median(), color="blue", linestyle=":",  lw=1.2, label=f"M={data.median():.2f}")
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("Value"); ax.set_ylabel("Count")
            ax.legend(fontsize=7)
            normality_results[col] = _is_normal(data)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Feature Distributions — {title}", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        skews = {c: round(float(df[c].dropna().skew()), 3) for c in cols}
        ms    = max(skews, key=lambda k: abs(skews[k]))
        non_normal = [c for c, n_ok in normality_results.items() if not n_ok]
        note  = (f" Shapiro-Wilk test: {', '.join(non_normal)} show non-normal distributions."
                 if non_normal else "")

        stat_insight = (
            f"Distributions of {n} numerical features. "
            f"'{ms.replace('_', ' ')}' is most skewed (skew={skews[ms]}).{note}"
        )
        return {
            "title":             "Feature Distribution Histograms",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"skewness": skews, "non_normal_cols": non_normal},
        }

    # ── Box Plots ─────────────────────────────────────────────────────────────

    def _box_plot(self, df, meta, title) -> Optional[Dict]:
        target = meta.get("target_col")
        cols   = [c for c in meta["numeric_cols"] if c != target][:8]
        if not cols:
            return None

        # Consistent min-max normalization across all columns
        norm = df[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.1), 5))
        bp = ax.boxplot(
            [norm[c].dropna().values for c in cols],
            labels=[c.replace("_", "\n") for c in cols],
            patch_artist=True,
            medianprops={"color": "red", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.7)

        ax.set_title(f"Feature Box Plots (Min-Max Normalized) — {title}", fontweight="bold")
        ax.set_ylabel("Normalized Value [0–1]")
        plt.tight_layout()

        outliers = {}
        for c in cols:
            d      = df[c].dropna()
            q1, q3 = d.quantile(0.25), d.quantile(0.75)
            iqr    = q3 - q1
            outliers[c] = int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum())
        worst = max(outliers, key=outliers.get)

        stat_insight = (
            f"Box plots (min-max normalized) across {len(cols)} features. "
            f"'{worst.replace('_', ' ')}' has the most outliers ({outliers[worst]}), "
            f"requiring robust scaling or outlier removal before modelling."
        )
        return {
            "title":             "Feature Box Plots",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"outlier_counts": outliers},
        }

    # ── Class Distribution ────────────────────────────────────────────────────

    def _class_dist(self, df, meta, title) -> Optional[Dict]:
        target = meta.get("target_col")
        if not target:
            if meta["categorical_cols"]:
                target = meta["categorical_cols"][0]
            else:
                return None

        counts = df[target].value_counts().head(10)
        if counts.empty:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        bars = ax1.bar(range(len(counts)), counts.values,
                       color=PALETTE[:len(counts)], edgecolor="white")
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels([str(c) for c in counts.index], rotation=30, ha="right")
        ax1.set_title(f"Class Counts: {target}")
        ax1.set_ylabel("Count"); ax1.set_xlabel(target.replace("_", " ").title())
        for bar, val in zip(bars, counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, str(val), ha="center", va="bottom", fontsize=9)

        ax2.pie(counts.values, labels=[str(c) for c in counts.index],
                colors=PALETTE[:len(counts)], autopct="%1.1f%%",
                startangle=90, pctdistance=0.85)
        ax2.set_title("Class Proportions")

        fig.suptitle(f"Target Distribution — {title}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        ratio = (round(counts.max() / counts.min(), 2) if counts.min() > 0 else float("inf"))
        bal   = ("Imbalanced — consider SMOTE or class weighting." if ratio > 3
                 else "Reasonably balanced.")

        stat_insight = (
            f"'{target}' contains {len(counts)} classes. "
            f"Majority: '{counts.index[0]}' ({counts.max()} samples), "
            f"minority: '{counts.index[-1]}' ({counts.min()} samples). "
            f"Imbalance ratio: {ratio}. {bal}"
        )
        return {
            "title":             "Class Distribution",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"counts": counts.to_dict(), "imbalance_ratio": ratio},
        }

    # ── Feature Importance ────────────────────────────────────────────────────

    def _feature_importance(self, df, meta, title) -> Optional[Dict]:
        target    = meta.get("target_col")
        feat_cols = [c for c in meta["numeric_cols"] if c != target][:12]
        if not feat_cols:
            return None

        importance = None
        method     = "Variance-Based"

        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            X = df[feat_cols].fillna(df[feat_cols].mean())
            y = df[target].dropna() if target else None

            if y is not None and len(X) == len(y):
                if y.dtype == object or y.nunique() < 20:
                    le    = LabelEncoder()
                    y_enc = le.fit_transform(y.astype(str))
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                else:
                    y_enc = y.values
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X.fillna(X.mean()), y_enc)
                importance = model.feature_importances_
                method     = "Random Forest"
        except Exception:
            pass

        if importance is None:
            var        = df[feat_cols].var()
            importance = (var / var.sum()).values
            method     = "Variance-Based"

        imp_df = pd.DataFrame(
            {"feature": feat_cols, "importance": importance}
        ).sort_values("importance")

        fig, ax = plt.subplots(figsize=(8, max(4, len(feat_cols) * 0.55)))
        bars = ax.barh(
            imp_df["feature"], imp_df["importance"],
            color=[PALETTE[i % len(PALETTE)] for i in range(len(imp_df))],
            edgecolor="white", height=0.65,
        )
        ax.set_title(f"Feature Importance ({method}) — {title}", fontweight="bold")
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, imp_df["importance"]):
            ax.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()

        top       = imp_df.iloc[-1]["feature"]
        top_score = round(imp_df.iloc[-1]["importance"], 4)
        bot       = imp_df.iloc[0]["feature"]

        stat_insight = (
            f"{method} importance: '{top.replace('_', ' ')}' is most predictive "
            f"(score={top_score}); '{bot.replace('_', ' ')}' contributes least. "
            f"Top features should be prioritised during model selection."
        )
        return {
            "title":             f"Feature Importance ({method})",
            "data":              _to_b64(fig),
            "statistical_insight": stat_insight,
            "insight":           stat_insight,
            "stats":             {"method": method, "top_feature": top,
                                  "importances": dict(
                                      zip(imp_df["feature"], imp_df["importance"].round(4)))},
        }

    # ── LLM context builder ───────────────────────────────────────────────────

    def build_llm_context(
        self,
        graphs: List[Dict],
        dataset_title: str,
        topic: str = "",
        project_context: Optional[Dict] = None,
    ) -> str:
        """
        Returns a string injected into the Results section LLM prompt.
        Contains only statistical findings — project insights added later.
        """
        if not graphs:
            return ""

        proj_title = (project_context or {}).get("title", topic) or dataset_title
        lines = [
            f"DATA SOURCE: {dataset_title} (real dataset, not synthetic)",
            f"RESEARCH PROJECT: {proj_title}",
            "",
            "The following visualizations were generated from actual data.",
            "Reference each figure as [FIGURE_N] where the finding is relevant.",
            "",
        ]
        for g in graphs:
            lines.append(f"{g['figure_label']} — {g['title']}")
            lines.append(f"  Statistical finding: {g.get('statistical_insight', g.get('insight', ''))}")
            if g.get("stats"):
                stats_str = ", ".join(
                    f"{k}={v}" for k, v in list(g["stats"].items())[:3]
                    if not isinstance(v, (dict, list))
                )
                if stats_str:
                    lines.append(f"  Key stats: {stats_str}")
            lines.append("")

        return "\n".join(lines)