"""Draw the merged four-panel Fig. 2 for the Nature Cities manuscript.

Panels
------
a  Observed recovery-outcome composition by tract-income group.
b  Covariate-adjusted rebuilding probability across continuous tract income.
c  Upgrade share among rebuilt buildings by tract-income group.
d  Fire-level upgrade share vs tract income for fires with >=20 rebuilds.

The data selection and models reproduce the existing Fig. 2 and Fig. 3
analyses; this script changes presentation and combines the two figures.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from patsy import build_design_matrices
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "论文修订工作区" / "图表修订预览"
MEDIA_DIR = (
    ROOT
    / "论文修订工作区"
    / "主文工作稿"
    / "InsuranceRetreat_main_round1_media"
    / "media"
)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = PREVIEW_DIR / "fig2_combined_income_v1.png"
PDF_OUT = PREVIEW_DIR / "fig2_combined_income_v1.pdf"
MANUSCRIPT_OUT = MEDIA_DIR / "image2.png"

OBSERVATION_END_YEAR = 2026
MIN_YEARS_SINCE_FIRE = 4
INCOME_CUTS = (60_000, 100_000)
INCOME_LOWER_K = 30
MIN_FIRE_REBUILT = 20

TRAJECTORY_PRIORITY = [
    "qwen3.6-35b-a3b-thinking",
    "qwen3.6-35b-a3b",
    "claude-sonnet-4-6",
    "opus-4-7",
    "sonnet-4-6",
    "qwen3-vl-8b-multi",
    "qwen3-vl-8b",
]
REBUILT = {"rebuilt_equal", "rebuilt_improved", "rebuilt_degraded"}
CLASSIFIABLE = [
    "empty_lot",
    "rebuilt_degraded",
    "rebuilt_equal",
    "rebuilt_improved",
]
TIER_ORDER = ["Low", "Middle", "High"]

# Muted, print-safe palette shared across all four panels.
COLORS = {
    "empty_lot": "#C8CDD2",
    "rebuilt_degraded": "#C99946",
    "rebuilt_equal": "#86ACBE",
    "rebuilt_improved": "#2D708E",
    "curve": "#205E7F",
    "curve_band": "#BED5E2",
    "grid": "#D7DEE2",
    "axis": "#24313A",
    "muted": "#4F5E68",
    "threshold": "#A7B3BA",
}
GROUP_COLORS = ["#B1CAD5", "#6E98AB", "#2D708E"]
GROUP_BACKGROUNDS = ["#F7F8F8", "#F1F5F6", "#EAF2F5"]

LABELS = {
    "empty_lot": "Empty lot",
    "rebuilt_degraded": "Rebuilt degraded",
    "rebuilt_equal": "Rebuilt equal",
    "rebuilt_improved": "Rebuilt improved",
}

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 7.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_analysis_data() -> pd.DataFrame:
    buildings = pd.read_parquet(ROOT / "data/processed/table_A_buildings.parquet")
    buildings["crash_id"] = buildings["crash_id"].astype(str)
    buildings["event_date"] = pd.to_datetime(buildings["event_date"], errors="coerce")
    buildings["years_since_fire"] = (
        OBSERVATION_END_YEAR - buildings["event_date"].dt.year
    )

    judgments = pd.read_csv(ROOT / "data/processed/vlm_judgments.csv")
    judgments["crash_id"] = judgments["crash_id"].astype(str)
    rank = {name: position for position, name in enumerate(TRAJECTORY_PRIORITY)}
    judgments["rank"] = judgments["judged_by"].map(rank)
    judgments = (
        judgments.dropna(subset=["rank"])
        .sort_values("rank")
        .drop_duplicates("crash_id", keep="first")
    )

    data = buildings.merge(
        judgments[["crash_id", "trajectory"]], on="crash_id", how="inner"
    )
    data["rebuilt_any"] = data["trajectory"].isin(REBUILT).astype(int)
    data["improved"] = (data["trajectory"] == "rebuilt_improved").astype(int)
    data = data[
        (~data["trajectory"].isin(["uncertain", "obscured"]))
        & (data["years_since_fire"] >= MIN_YEARS_SINCE_FIRE)
    ].dropna(subset=["pre_tract_med_HH_income"])

    data = data.copy()
    data["income_group"] = pd.cut(
        data["pre_tract_med_HH_income"],
        [-np.inf, INCOME_CUTS[0], INCOME_CUTS[1], np.inf],
        labels=TIER_ORDER,
        right=False,
    )
    return data


def fit_rebuilding_curve(data: pd.DataFrame):
    model_data = data.dropna(
        subset=[
            "pre_pct_white_NH",
            "pre_pct_Hispanic",
            "pre_pct_below_poverty",
            "pre_pct_owner_occupied",
            "assessed_value_usd",
            "mtbs_severity_code",
        ]
    ).copy()
    model_data = model_data[model_data["assessed_value_usd"] > 0]
    model_data["log_inc"] = np.log10(model_data["pre_tract_med_HH_income"])
    model_data["log_value"] = np.log10(model_data["assessed_value_usd"])
    model_data["mtbs_sev"] = model_data["mtbs_severity_code"].astype(float)
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "log_inc",
            "log_value",
            "mtbs_sev",
            "pre_pct_white_NH",
            "pre_pct_Hispanic",
            "pre_pct_below_poverty",
            "pre_pct_owner_occupied",
        ]
    )

    model = smf.logit(
        "rebuilt_any ~ log_inc + pre_pct_white_NH + pre_pct_Hispanic + "
        "pre_pct_below_poverty + pre_pct_owner_occupied + log_value + mtbs_sev",
        data=model_data,
    ).fit(disp=False)

    observed_income_k = model_data["pre_tract_med_HH_income"] / 1_000
    axis_limits_k = (
        INCOME_LOWER_K,
        max(110, float(np.ceil(observed_income_k.max() / 10) * 10)),
    )
    curve_limits_k = (
        max(axis_limits_k[0], observed_income_k.min()),
        observed_income_k.max(),
    )
    income_grid_log = np.linspace(
        np.log10(curve_limits_k[0] * 1_000),
        np.log10(curve_limits_k[1] * 1_000),
        180,
    )
    medians = model_data[
        [
            "pre_pct_white_NH",
            "pre_pct_Hispanic",
            "pre_pct_below_poverty",
            "pre_pct_owner_occupied",
            "log_value",
            "mtbs_sev",
        ]
    ].median()
    prediction_data = pd.DataFrame(
        {
            "log_inc": income_grid_log,
            "pre_pct_white_NH": medians["pre_pct_white_NH"],
            "pre_pct_Hispanic": medians["pre_pct_Hispanic"],
            "pre_pct_below_poverty": medians["pre_pct_below_poverty"],
            "pre_pct_owner_occupied": medians["pre_pct_owner_occupied"],
            "log_value": medians["log_value"],
            "mtbs_sev": medians["mtbs_sev"],
        }
    )
    design = np.asarray(
        build_design_matrices([model.model.data.design_info], prediction_data)[0]
    )
    linear = design @ model.params.to_numpy()
    linear_se = np.sqrt(
        np.einsum("ij,jk,ik->i", design, model.cov_params().to_numpy(), design)
    )

    logistic = lambda values: 1 / (1 + np.exp(-values))
    predicted = logistic(linear)
    lower = logistic(linear - 1.96 * linear_se)
    upper = logistic(linear + 1.96 * linear_se)
    income_grid_k = (10**income_grid_log) / 1_000
    return model_data, income_grid_k, predicted, lower, upper, axis_limits_k


def add_group_backgrounds(
    ax,
    upper_y: float = 100,
    label_positions: tuple[float, float, float] = (43, 77, 130),
    x_limits: tuple[float, float] = (30, 170),
) -> None:
    ranges = [
        (x_limits[0], 60),
        (60, 100),
        (100, x_limits[1]),
    ]
    for (left, right), color in zip(ranges, GROUP_BACKGROUNDS):
        ax.axvspan(left, right, color=color, zorder=-5)
    for cut in (60, 100):
        ax.axvline(
            cut,
            color=COLORS["threshold"],
            linestyle=(0, (2.3, 3.2)),
            linewidth=0.9,
            zorder=1,
        )
    group_labels = ("Lower  <60", "Middle  60–100", "Higher  >100")
    for x, label in zip(label_positions, group_labels):
        ax.text(
            x,
            upper_y - 3.5,
            label,
            ha="center",
            va="top",
            fontsize=6.8,
            color=COLORS["muted"],
            zorder=6,
        )


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["axis"], length=3.5, width=0.75)
    ax.grid(
        axis=grid_axis,
        color=COLORS["grid"],
        linewidth=0.6,
        linestyle="-",
        zorder=-4,
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLORS["axis"],
    )


def draw_panel_a(ax, data: pd.DataFrame) -> None:
    table = pd.crosstab(data["income_group"], data["trajectory"]).reindex(
        index=TIER_ORDER, columns=CLASSIFIABLE, fill_value=0
    )
    shares = table.div(table.sum(axis=1), axis=0) * 100
    x = np.arange(len(TIER_ORDER))
    bottom = np.zeros(len(TIER_ORDER))

    for trajectory in CLASSIFIABLE:
        values = shares[trajectory].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=COLORS[trajectory],
            edgecolor="white",
            linewidth=0.9,
            label=LABELS[trajectory],
            zorder=3,
        )
        for i, value in enumerate(values):
            if value >= 8:
                text_color = "white" if trajectory == "rebuilt_improved" else "#26323A"
                ax.text(
                    x[i],
                    bottom[i] + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=text_color,
                    fontweight="normal",
                    zorder=4,
                )
        bottom += values

    ax.set_title("Observed building-level recovery outcomes", loc="left", pad=8)
    ax.set_ylabel("Share of destroyed buildings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Lower income\n<60", "Middle income\n60 to 100", "Higher income\n>100"])
    ax.set_xlabel("Pre-fire census-tract income group\n(thousand USD)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axis(ax)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_order = (0, 2, 1, 3)
    ax.legend(
        [legend_handles[i] for i in legend_order],
        [legend_labels[i] for i in legend_order],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.14),
        frameon=False,
        columnspacing=1.05,
        handlelength=1.25,
        handleheight=0.75,
        borderaxespad=0,
        labelspacing=0.45,
        fontsize=6.6,
    )
    panel_label(ax, "a")


def draw_panel_b(
    ax, model_data, income_k, predicted, lower, upper, axis_limits_k
) -> None:
    higher_label_x = float(np.sqrt(100 * axis_limits_k[1]))
    add_group_backgrounds(
        ax,
        label_positions=(43, 77, higher_label_x),
        x_limits=axis_limits_k,
    )

    # A low-profile rug shows the observed model support without assigning a
    # second y-scale or competing with the rebuilding-probability curve.
    observed_income_k = model_data["pre_tract_med_HH_income"].to_numpy() / 1_000
    observed_income_k = observed_income_k[
        (observed_income_k >= axis_limits_k[0])
        & (observed_income_k <= axis_limits_k[1])
    ]
    ax.scatter(
        observed_income_k,
        np.full_like(observed_income_k, 1.0),
        marker="|",
        s=20,
        linewidth=0.45,
        color=COLORS["rebuilt_equal"],
        alpha=0.14,
        zorder=3,
        rasterized=True,
    )
    ax.text(
        36.0,
        3.1,
        "Observed incomes",
        fontsize=6.4,
        color=COLORS["muted"],
        va="center",
        zorder=6,
        bbox={
            "facecolor": GROUP_BACKGROUNDS[0],
            "edgecolor": "none",
            "alpha": 0.92,
            "pad": 0.8,
        },
    )

    ax.fill_between(
        income_k,
        lower * 100,
        upper * 100,
        color=COLORS["curve_band"],
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        income_k,
        predicted * 100,
        color=COLORS["curve"],
        linewidth=2.25,
        zorder=4,
    )
    for cut_k in (60, 100):
        y = np.interp(cut_k, income_k, predicted * 100)
        ax.scatter(
            [cut_k],
            [y],
            s=24,
            facecolor=COLORS["curve"],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    estimate_key = Line2D(
        [0], [0], color=COLORS["curve"], linewidth=2.25,
        label="Adjusted estimate",
    )
    interval_key = Patch(
        facecolor=COLORS["curve_band"], edgecolor="none",
        label="95% confidence interval",
    )
    curve_legend = ax.legend(
        handles=[estimate_key, interval_key],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.88),
        frameon=True,
        fontsize=6.4,
        handlelength=1.45,
        handletextpad=0.6,
        labelspacing=0.45,
        borderaxespad=0,
        borderpad=0.35,
    )
    curve_legend.get_frame().set_facecolor("white")
    curve_legend.get_frame().set_edgecolor("none")
    curve_legend.get_frame().set_alpha(0.88)

    ax.set_title("Adjusted rebuilding probability", loc="left", pad=8)
    ax.set_xscale("log")
    ax.set_xlim(*axis_limits_k)
    ticks = [
        tick for tick in (30, 50, 75, 100, 150, 200, 250)
        if axis_limits_k[0] <= tick <= axis_limits_k[1]
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Pre-fire census-tract income\n(thousand USD, log scale)")
    ax.set_ylabel("Adjusted rebuilding probability (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axis(ax)
    panel_label(ax, "b")


def draw_panel_c(ax, data: pd.DataFrame) -> None:
    rebuilt = data[data["rebuilt_any"] == 1].copy()
    summary = rebuilt.groupby("income_group", observed=False).agg(
        share=("improved", "mean"),
        n=("improved", "count"),
    ).reindex(TIER_ORDER)
    summary["ci"] = 1.96 * np.sqrt(
        summary["share"] * (1 - summary["share"]) / summary["n"]
    )

    x = np.arange(len(TIER_ORDER))
    values = summary["share"].to_numpy() * 100
    errors = summary["ci"].to_numpy() * 100
    ax.bar(
        x,
        values,
        yerr=errors,
        width=0.60,
        color=GROUP_COLORS,
        edgecolor="white",
        linewidth=0.8,
        error_kw={
            "ecolor": COLORS["axis"],
            "elinewidth": 0.9,
            "capsize": 3.0,
            "capthick": 0.9,
        },
        zorder=3,
    )
    for i, (value, n) in enumerate(zip(values, summary["n"])):
        ax.text(
            i,
            value + errors[i] + 2.2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=COLORS["axis"],
        )
        sample_color = COLORS["axis"] if i == 0 else "white"
        ax.text(
            i,
            3.0,
            f"n = {int(n):,}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=sample_color,
            zorder=5,
        )

    ax.set_title("Upgrade share among rebuilt buildings", loc="left", pad=8)
    ax.set_ylabel("Rebuilt buildings classified as upgraded (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Lower income\n<60", "Middle income\n60 to 100", "Higher income\n>100"])
    ax.set_xlabel("Pre-fire census-tract income group\n(thousand USD)")
    ax.set_ylim(0, 72)
    ax.set_yticks(np.arange(0, 71, 10))
    style_axis(ax)
    panel_label(ax, "c")


def draw_panel_d(ax, data: pd.DataFrame) -> None:
    rebuilt = data[data["rebuilt_any"] == 1].copy()
    fire = rebuilt.groupby("incident").agg(
        share=("improved", "mean"),
        income=("pre_tract_med_HH_income", "median"),
        n=("improved", "count"),
    )
    fire = fire[fire["n"] >= MIN_FIRE_REBUILT].sort_values("income")
    fire["income_k"] = fire["income"] / 1_000
    fire["share_pct"] = fire["share"] * 100

    add_group_backgrounds(
        ax,
        upper_y=80,
        label_positions=(50, 80, 130),
        x_limits=(40, 160),
    )
    cmap = LinearSegmentedColormap.from_list("income_blue", GROUP_COLORS)
    norm = Normalize(vmin=40, vmax=160)
    # Compress the very uneven fire-level sample sizes so that Tubbs remains
    # visibly larger without overwhelming the smaller qualifying fires.
    marker_area = lambda n: 42 + 34 * np.log10(1 + np.asarray(n) / MIN_FIRE_REBUILT)
    sizes = marker_area(fire["n"].to_numpy())
    ax.scatter(
        fire["income_k"],
        fire["share_pct"],
        s=sizes,
        c=fire["income_k"],
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.96,
        zorder=4,
    )

    offsets = {
        "Caldor": (0, 12),
        "Carr": (10, 3),
        "Nuns": (-7, 13),
        "Tubbs": (0, -16),
        "Thomas": (-8, 14),
        "Atlas": (0, -16),
        "Woolsey": (-15, -10),
    }
    for name, row in fire.iterrows():
        dx, dy = offsets.get(name, (4, 6))
        ax.annotate(
            name,
            (row["income_k"], row["share_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["axis"],
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.5,
            },
        )

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(marker_area(n)),
            markerfacecolor=GROUP_COLORS[1],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"n = {n:,}",
        )
        for n in (20, 100, 1_000)
    ]
    ax.legend(
        handles=size_handles,
        title="Rebuilt buildings",
        loc="lower right",
        frameon=False,
        fontsize=6.2,
        title_fontsize=6.4,
        borderaxespad=0.5,
        handletextpad=0.5,
        labelspacing=0.5,
    )

    ax.set_title("Fire-level upgrade share and tract income", loc="left", pad=8)
    ax.set_xlabel("Fire-level census-tract income (thousand USD)")
    ax.set_ylabel("Rebuilt buildings classified as upgraded (%)")
    ax.set_xlim(40, 160)
    ax.set_xticks([50, 75, 100, 125, 150])
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    style_axis(ax)
    panel_label(ax, "d")


def main() -> None:
    data = load_analysis_data()
    (
        model_data,
        income_k,
        predicted,
        lower,
        upper,
        axis_limits_k,
    ) = fit_rebuilding_curve(data)

    fig = plt.figure(figsize=(190 / 25.4, 155 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.22],
        height_ratios=[1.0, 0.92],
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.875,
        wspace=0.31,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_panel_a(ax_a, data)
    draw_panel_b(
        ax_b,
        model_data,
        income_k,
        predicted,
        lower,
        upper,
        axis_limits_k,
    )
    draw_panel_c(ax_c, data)
    draw_panel_d(ax_d, data)

    fig.savefig(PNG_OUT, dpi=300, facecolor="white")
    fig.savefig(PDF_OUT, facecolor="white")
    fig.savefig(MANUSCRIPT_OUT, dpi=300, facecolor="white")
    plt.close(fig)

    # Store opaque RGB PNGs so manuscript and preview applications render the
    # white page consistently.
    for png_path in (PNG_OUT, MANUSCRIPT_OUT):
        with Image.open(png_path) as png_image:
            rgb_image = png_image.convert("RGB")
        rgb_image.save(png_path, dpi=(300, 300))

    print(f"Analysis sample: {len(data):,}")
    print(f"Curve model sample: {len(model_data):,}")
    print(f"PNG written: {PNG_OUT.name}")
    print(f"PDF written: {PDF_OUT.name}")
    print(f"Manuscript image written: {MANUSCRIPT_OUT.name}")


if __name__ == "__main__":
    main()
