"""Combine the former Figs. 4 and 5 into a compact, Nature-style figure.

Panel a is designed at approximately one journal-column width and preserves
the three existing Street View examples. Panel b restyles the existing
multi-year vegetation analysis without changing its sample, bins or values.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

PNG_OUT = PREVIEW_DIR / "fig3_combined_vegetation_v1.png"
PDF_OUT = PREVIEW_DIR / "fig3_combined_vegetation_v1.pdf"
PANEL_A_OUT = PREVIEW_DIR / "fig3a_segmentation_one_column.png"
PANEL_A_PDF_OUT = PREVIEW_DIR / "fig3a_segmentation_one_column.pdf"
MANUSCRIPT_OUT = MEDIA_DIR / "image4.png"

# Shared manuscript palette. Income-group colors match the merged Fig. 2.
COLORS = {
    "empty": "#C8CDD2",
    "ochre": "#C99946",
    "blue_light": "#B1CAD5",
    "blue_mid": "#6E98AB",
    "blue_dark": "#2D708E",
    "teal": "#3B8272",
    "slate": "#4F5E68",
    "axis": "#24313A",
    "grid": "#D7DEE2",
    "post_bg": "#F1F5F6",
}

TIER_COLORS = {
    "Low": COLORS["blue_light"],
    "Middle": COLORS["blue_mid"],
    "High": COLORS["blue_dark"],
}
TIER_MARKERS = {"Low": "o", "Middle": "s", "High": "^"}

# Existing condensed SegFormer palette -> harmonized manuscript palette.
SEGMENTATION_COLORS = {
    "Vegetation": COLORS["teal"],
    "Building": COLORS["ochre"],
    "Road": COLORS["empty"],
    "Sky": "#86ACBE",
    "Other": COLORS["slate"],
}
OLD_SEGMENTATION_RGB = {
    "Vegetation": (60, 180, 60),
    "Building": (200, 40, 40),
    "Road": (140, 140, 140),
    "Sky": (110, 180, 230),
    "Other": (30, 30, 30),
}

# These are the same three examples displayed in the manuscript's former Fig. 4.
EXAMPLES = [
    (
        "233583",
        "pre_2013-07-01_4hSG6EMjDAolpqab5GLJQg_183_seg.png",
        "post_2019-04-01_rLHCy9pVznj4zqhAVQZw1w_178_seg.png",
    ),
    (
        "237692",
        "pre_2013-07-01_rh-OQOwnDmJLoTBezwhomQ_173_seg.png",
        "post_2019-03-01_FFiJ5Wh0ynEBzS4XxyqAHw_171_seg.png",
    ),
    (
        "240479",
        "pre_2013-07-01_eO7PkqpkaEttG6JEGzojRg_132_seg.png",
        "post_2019-03-01_Q1_8Y9HgCfRxwSLdC5jpMQ_128_seg.png",
    ),
]

fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.15,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def split_source_image(path: Path) -> tuple[Image.Image, Image.Image]:
    source = Image.open(path).convert("RGB")
    width, height = source.size
    third = (width - 4) // 3
    original = source.crop((0, 0, third, height))
    segmentation = source.crop((third + 2, 0, 2 * third + 2, height))
    return original, recolor_segmentation(segmentation)


def recolor_segmentation(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGB")).copy()
    for label, old_rgb in OLD_SEGMENTATION_RGB.items():
        mask = np.all(pixels == np.asarray(old_rgb), axis=2)
        pixels[mask] = np.asarray(hex_to_rgb(SEGMENTATION_COLORS[label]))
    return Image.fromarray(pixels, mode="RGB")


def load_examples():
    root = ROOT / "data" / "seg_visuals"
    rows = []
    for crash_id, pre_name, post_name in EXAMPLES:
        pre_original, pre_segmentation = split_source_image(
            root / crash_id / pre_name
        )
        post_original, post_segmentation = split_source_image(
            root / crash_id / post_name
        )
        rows.append(
            (
                pre_original,
                pre_segmentation,
                post_original,
                post_segmentation,
            )
        )
    return rows


def load_vegetation_summary() -> dict[str, pd.DataFrame]:
    vegetation = pd.read_parquet(
        ROOT / "data" / "processed" / "all_vegetation_segformer.parquet"
    )
    vegetation["crash_id"] = vegetation["crash_id"].astype(str)
    vegetation["pano_date"] = pd.to_datetime(vegetation["pano_date"])

    buildings = pd.read_parquet(
        ROOT / "data" / "processed" / "table_A_buildings.parquet"
    )
    buildings["crash_id"] = buildings["crash_id"].astype(str)
    buildings["event_date"] = pd.to_datetime(
        buildings["event_date"], errors="coerce"
    )
    buildings = buildings.dropna(
        subset=["pre_tract_med_HH_income", "event_date"]
    ).copy()

    thresholds = buildings["pre_tract_med_HH_income"].quantile([1 / 3, 2 / 3])

    def income_group(value: float) -> str:
        if value < thresholds.iloc[0]:
            return "Low"
        if value < thresholds.iloc[1]:
            return "Middle"
        return "High"

    buildings["tier"] = buildings["pre_tract_med_HH_income"].apply(income_group)
    merged = vegetation.merge(
        buildings[["crash_id", "event_date", "tier"]],
        on="crash_id",
        how="inner",
    )
    merged["years_since"] = (
        merged["pano_date"] - merged["event_date"]
    ).dt.days / 365.25
    merged = merged[merged["years_since"].between(-10, 6)].copy()
    merged["yr_bin"] = np.floor(merged["years_since"]).astype(int)

    baseline = (
        merged[merged["yr_bin"] < 0].groupby("tier")["veg_ratio"].mean()
    )
    merged["deviation"] = merged.apply(
        lambda row: row["veg_ratio"] - baseline[row["tier"]], axis=1
    )

    summaries = {}
    for tier in ("Low", "Middle", "High"):
        group = (
            merged[merged["tier"] == tier]
            .groupby("yr_bin")["deviation"]
            .agg(["mean", "sem", "count"])
        )
        summaries[tier] = group[group["count"] >= 20]
    return summaries


def style_plot_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["axis"], length=3.2, width=0.75)
    ax.grid(
        axis="y",
        color=COLORS["grid"],
        linewidth=0.55,
        linestyle="-",
        zorder=-4,
    )


def draw_segmentation_panel(fig, parent_spec, rows):
    nested = parent_spec.subgridspec(
        4,
        5,
        height_ratios=[1, 1, 1, 0.20],
        width_ratios=[1, 1, 0.20, 1, 1],
        wspace=0.025,
        hspace=0.10,
    )
    axes = []
    columns = (0, 1, 3, 4)
    titles = (
        "Original",
        "Segmentation",
        "Original",
        "Segmentation",
    )
    for row_index, images in enumerate(rows):
        row_axes = []
        for column_index, (grid_column, image) in enumerate(zip(columns, images)):
            ax = fig.add_subplot(nested[row_index, grid_column])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#859099")
                spine.set_linewidth(0.45)
            if row_index == 0:
                ax.set_title(
                    titles[column_index],
                    fontsize=5.8,
                    color=COLORS["axis"],
                    pad=3.0,
                    fontweight="normal",
                )
            row_axes.append(ax)
        axes.append(row_axes)

    # Pair headings plus the deliberately wider central gap implement Holly's
    # request that each original/segmentation pair read as a unit.
    axes[0][0].text(
        1.02,
        1.23,
        "Pre-fire",
        transform=axes[0][0].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLORS["axis"],
    )
    axes[0][2].text(
        1.02,
        1.23,
        "Post-fire",
        transform=axes[0][2].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLORS["axis"],
    )

    legend_axis = fig.add_subplot(nested[3, :])
    legend_axis.axis("off")
    handles = [
        mpatches.Patch(
            facecolor=SEGMENTATION_COLORS[label], edgecolor="none", label=label
        )
        for label in ("Vegetation", "Sky", "Building", "Other", "Road")
    ]
    legend_axis.legend(
        handles=handles,
        ncol=3,
        loc="center",
        frameon=False,
        columnspacing=0.9,
        handlelength=1.25,
        handleheight=0.75,
        handletextpad=0.45,
        labelspacing=0.35,
        fontsize=5.8,
    )

    axes[0][0].text(
        -0.22,
        1.43,
        "a",
        transform=axes[0][0].transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["axis"],
        va="bottom",
    )
    axes[0][0].text(
        0.0,
        1.43,
        "Pre- and post-fire street-view segmentation",
        transform=axes[0][0].transAxes,
        fontsize=8.2,
        color=COLORS["axis"],
        va="bottom",
    )
    return axes


def draw_vegetation_panel(ax, summaries) -> None:
    ax.axvspan(0, 5.2, color=COLORS["post_bg"], zorder=-6)
    ax.axhline(0, color="#8B969D", linewidth=0.65, zorder=1)
    ax.axvline(
        0,
        color=COLORS["axis"],
        linestyle=(0, (3.0, 2.6)),
        linewidth=0.8,
        zorder=2,
    )

    for tier in ("Low", "Middle", "High"):
        group = summaries[tier]
        ax.errorbar(
            group.index,
            group["mean"],
            yerr=1.96 * group["sem"],
            fmt=TIER_MARKERS[tier] + "-",
            markersize=3.5,
            markerfacecolor=TIER_COLORS[tier],
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.25,
            color=TIER_COLORS[tier],
            ecolor=TIER_COLORS[tier],
            elinewidth=0.65,
            capsize=1.7,
            capthick=0.65,
            zorder=4,
        )

        last_x = float(group.index.max())
        last_y = float(group.loc[group.index.max(), "mean"])
        ax.text(
            last_x + 0.24,
            last_y,
            {"Low": "Lower income", "Middle": "Middle income", "High": "Higher income"}[tier],
            fontsize=6.5,
            color=COLORS["axis"],
            ha="left",
            va="center",
        )

    ax.text(
        -5.1,
        0.126,
        "Pre-fire",
        ha="center",
        va="top",
        fontsize=6.4,
        color=COLORS["slate"],
    )
    ax.text(
        2.15,
        0.126,
        "Post-fire",
        ha="center",
        va="top",
        fontsize=6.4,
        color=COLORS["slate"],
    )
    ax.text(
        0.14,
        0.105,
        "Wildfire",
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["axis"],
    )

    ax.set_xlim(-10, 5.25)
    ax.set_ylim(-0.44, 0.14)
    ax.set_xticks(np.arange(-10, 5, 2))
    ax.set_yticks(np.arange(-0.4, 0.11, 0.1))
    ax.set_xlabel("Years since wildfire")
    ax.set_ylabel("Vegetation change from pre-fire baseline")
    ax.set_title(
        "Street-level vegetation change",
        loc="left",
        pad=8,
        fontsize=8.2,
        color=COLORS["axis"],
    )
    ax.text(
        -0.13,
        1.04,
        "b",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["axis"],
        va="bottom",
    )
    style_plot_axis(ax)


def save_opaque_rgb(fig, png_path: Path, pdf_path: Path | None = None) -> None:
    fig.savefig(png_path, dpi=300, facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, facecolor="white")
    with Image.open(png_path) as image:
        rgb_image = image.convert("RGB")
    rgb_image.save(png_path, dpi=(300, 300))


def main() -> None:
    rows = load_examples()
    summaries = load_vegetation_summary()

    fig = plt.figure(figsize=(180 / 25.4, 88 / 25.4))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.10],
        left=0.035,
        right=0.985,
        bottom=0.15,
        top=0.88,
        wspace=0.23,
    )
    draw_segmentation_panel(fig, outer[0, 0], rows)
    vegetation_axis = fig.add_subplot(outer[0, 1])
    draw_vegetation_panel(vegetation_axis, summaries)

    save_opaque_rgb(fig, PNG_OUT, PDF_OUT)
    save_opaque_rgb(fig, MANUSCRIPT_OUT)
    plt.close(fig)

    # Optional one-column export of the former Fig. 4 content.
    one_column = plt.figure(figsize=(88 / 25.4, 78 / 25.4))
    one_spec = one_column.add_gridspec(
        1, 1, left=0.06, right=0.98, bottom=0.08, top=0.90
    )
    draw_segmentation_panel(one_column, one_spec[0, 0], rows)
    save_opaque_rgb(one_column, PANEL_A_OUT, PANEL_A_PDF_OUT)
    plt.close(one_column)

    print("Vegetation series retained from the former Fig. 5 analysis:")
    for tier, summary in summaries.items():
        print(
            f"  {tier}: years {summary.index.min()} to {summary.index.max()}, "
            f"n bins={len(summary)}"
        )
    print(f"PNG written: {PNG_OUT.name}")
    print(f"PDF written: {PDF_OUT.name}")
    print(f"One-column panel written: {PANEL_A_OUT.name}")


if __name__ == "__main__":
    main()
