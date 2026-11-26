"""Redraw the final main-text figure with the manuscript's shared palette.

The data selection and calculations follow ``build_fig9_displacement.py``.
Only figure styling, labels, and layout are changed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"C:\program1\jiejing_20260704\jiejing\crash_response_inequality")
MEDIA = (
    ROOT
    / "论文修订工作区"
    / "主文工作稿"
    / "InsuranceRetreat_main_round1_media"
    / "media"
)
PREVIEW = ROOT / "论文修订工作区" / "图表修订预览"

OUT_PNG = PREVIEW / "fig5_prices_income_unified_v1.png"
OUT_PDF = PREVIEW / "fig5_prices_income_unified_v1.pdf"
MANUSCRIPT_PNG = MEDIA / "image8.png"

CM = 1 / 2.54

# Shared visual language used in the revised main-text figures.
AXIS = "#24313A"
MUTED = "#65727B"
GRID = "#D7DEE2"
SLATE = "#4F5E68"
EMPTY = "#C8CDD2"
OCHRE = "#C99946"
MID_BLUE = "#6E98AB"
EQUAL_BLUE = "#86ACBE"
DARK_BLUE = "#2D708E"

TRAJ_ORDER = ["empty_lot", "rebuilt_equal", "rebuilt_improved"]
TRAJ_LABELS = {
    "empty_lot": "Empty\nlot",
    "rebuilt_equal": "Rebuilt\nequal",
    "rebuilt_improved": "Rebuilt\nimproved",
}
TRAJ_COLORS = {
    "empty_lot": EMPTY,
    "rebuilt_equal": EQUAL_BLUE,
    "rebuilt_improved": DARK_BLUE,
}
TRAJ_EDGES = {
    "empty_lot": "#879199",
    "rebuilt_equal": "#527F94",
    "rebuilt_improved": "#245E77",
}
TRAJ_MARKERS = {
    "empty_lot": "o",
    "rebuilt_equal": "s",
    "rebuilt_improved": "^",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.4,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "axes.labelcolor": AXIS,
        "text.color": AXIS,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the source figure's analysis frame and price subset."""
    df = pd.read_parquet(ROOT / "data/processed/analysis/analysis_frame.parquet")
    vlm = pd.read_csv(ROOT / "data/processed/vlm_judgments.csv")
    rank = {
        "qwen3.6-35b-a3b-thinking": 0,
        "qwen3.6-35b-a3b": 1,
        "opus-4-7": 2,
        "claude": 3,
        "qwen-vl-8b": 4,
    }
    vlm["_r"] = vlm["judged_by"].map(rank).fillna(99)
    vlm = vlm.sort_values("_r").drop_duplicates("crash_id", keep="first")
    vlm["trajectory"] = vlm["trajectory"].replace({"uncertain": "obscured"})
    df = df.merge(vlm[["crash_id", "trajectory"]], on="crash_id", how="left")
    df = df[df["trajectory"] != "obscured"].copy()

    sold = df[df["ownership_class"] == "sold_post_fire"].copy()
    for column in ["last_sale_price", "prior_sale_price"]:
        sold[column] = pd.to_numeric(sold[column], errors="coerce")
    sold = sold[(sold["last_sale_price"] > 0) & (sold["prior_sale_price"] > 0)].copy()
    sold["price_change_pct"] = (
        sold["last_sale_price"] / sold["prior_sale_price"] - 1
    ) * 100
    return df, sold


def clean_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
    """Apply the restrained open-frame styling shared across figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", direction="out", length=3.0, width=0.7, pad=2.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.45)


def panel_heading(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.14,
        1.12,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=AXIS,
        clip_on=False,
    )
    ax.text(
        0,
        1.12,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        color=AXIS,
        clip_on=False,
    )


def draw_panel_a(ax: plt.Axes, sold: pd.DataFrame) -> None:
    values = sold["price_change_pct"].to_numpy()
    bins = np.concatenate(
        [
            np.linspace(-100, 0, 6)[:-1],
            np.geomspace(1, 2000, 30),
        ]
    )
    ax.hist(
        values,
        bins=bins,
        color=MID_BLUE,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.96,
    )
    median = float(np.nanmedian(values))
    ax.axvline(0, color=SLATE, linestyle=(0, (3, 2)), linewidth=0.75, zorder=3)
    ax.axvline(median, color=OCHRE, linewidth=1.6, zorder=4)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([-100, 0, 10, 100, 1000])
    ax.set_xticklabels(["−100", "0", "10", "100", "1,000"])
    ax.set_xlabel("Post-fire sale-price change (%)", labelpad=5)
    ax.set_ylabel("Changed-ownership lots", labelpad=5)
    ax.text(
        median / 1.12,
        0.93,
        f"Median +{median:.0f}%",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=OCHRE,
    )
    clean_axes(ax)
    panel_heading(ax, "a", "Post-fire transaction-price change")


def draw_panel_b(ax: plt.Axes, sold: pd.DataFrame) -> None:
    grouped = sold.groupby("trajectory").agg(
        n=("price_change_pct", "count"),
        median=("price_change_pct", "median"),
        q1=("price_change_pct", lambda s: s.quantile(0.25)),
        q3=("price_change_pct", lambda s: s.quantile(0.75)),
    )
    grouped = grouped.reindex(TRAJ_ORDER).dropna()

    x = np.arange(len(grouped))
    for xi, trajectory in zip(x, grouped.index):
        row = grouped.loc[trajectory]
        color = TRAJ_COLORS[trajectory]
        edge = TRAJ_EDGES[trajectory]
        marker = TRAJ_MARKERS[trajectory]
        ax.errorbar(
            xi,
            row["median"],
            yerr=[[row["median"] - row["q1"]], [row["q3"] - row["median"]]],
            fmt="none",
            ecolor=edge,
            elinewidth=1.25,
            capsize=4.0,
            capthick=1.25,
            zorder=3,
        )
        ax.scatter(
            xi,
            row["median"],
            s=48,
            marker=marker,
            facecolor=color,
            edgecolor=edge,
            linewidth=0.9,
            zorder=4,
        )

    label_y = float(grouped["q3"].max() + 42)
    for xi, n in zip(x, grouped["n"]):
        ax.text(
            xi,
            label_y,
            f"n = {int(n)}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=AXIS,
        )

    ax.axhline(0, color=SLATE, linestyle=(0, (3, 2)), linewidth=0.75, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TRAJ_LABELS[t] for t in grouped.index])
    ax.set_ylabel("Post-fire sale-price change (%)", labelpad=5)
    ax.set_ylim(-105, label_y + 55)
    ax.text(
        0,
        1.02,
        "Median and interquartile range",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
    )
    clean_axes(ax)
    panel_heading(ax, "b", "Price change by recovery outcome")


def draw_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    values = df.dropna(subset=["pre_tract_med_HH_income", "tract_med_HH_income"])
    pre = values["pre_tract_med_HH_income"] / 1000
    post = values["tract_med_HH_income"] / 1000
    limit = float(max(pre.max(), post.max()) * 1.02)

    ax.scatter(
        pre,
        post,
        s=7,
        alpha=0.13,
        color=DARK_BLUE,
        edgecolor="none",
        rasterized=True,
        zorder=3,
    )
    ax.plot(
        [0, limit],
        [0, limit],
        color=SLATE,
        linestyle=(0, (4, 3)),
        linewidth=0.85,
        zorder=2,
    )
    ticks = [0, 50, 100, 150, 200, 250]
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pre-fire census-tract income\n(thousand USD)", labelpad=5)
    ax.set_ylabel("Post-fire census-tract income\n(thousand USD)", labelpad=5)
    ax.text(
        0.70,
        0.64,
        "No change",
        transform=ax.transAxes,
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    clean_axes(ax, xgrid=True)
    panel_heading(ax, "c", "Pre- and post-fire tract income")


def main() -> None:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    df, sold = load_data()

    fig = plt.figure(figsize=(18.3 * CM, 7.0 * CM), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.08, 0.96, 1.08],
        left=0.066,
        right=0.992,
        bottom=0.24,
        top=0.82,
        wspace=0.42,
    )
    draw_panel_a(fig.add_subplot(grid[0, 0]), sold)
    draw_panel_b(fig.add_subplot(grid[0, 1]), sold)
    draw_panel_c(fig.add_subplot(grid[0, 2]), df)

    fig.savefig(OUT_PNG, dpi=400)
    fig.savefig(OUT_PDF)
    fig.savefig(MANUSCRIPT_PNG, dpi=400)
    plt.close(fig)

    # Opaque RGB PNGs render consistently in Word and the Codex desktop app.
    for path in [OUT_PNG, MANUSCRIPT_PNG]:
        with Image.open(path) as image:
            image.convert("RGB").save(path, dpi=(400, 400))

    grouped = sold.groupby("trajectory")["price_change_pct"].count()
    print(f"price sample: n={len(sold)}, median={sold['price_change_pct'].median():.3f}%")
    print(
        "panel-b counts: "
        + ", ".join(f"{key}={int(grouped[key])}" for key in TRAJ_ORDER)
    )
    print(f"wrote preview/{OUT_PNG.name}")
    print(f"wrote preview/{OUT_PDF.name}")
    print(f"updated manuscript media/{MANUSCRIPT_PNG.name}")


if __name__ == "__main__":
    main()
