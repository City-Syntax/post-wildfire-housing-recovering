"""Plotting functions used by render_supplementary.py."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
TRAJ_COLOR = {'empty_lot': '#C8CDD2', 'rebuilt_degraded': '#C99946', 'rebuilt_equal': '#86ACBE', 'rebuilt_improved': '#2D708E', 'obscured': '#E6E8EA'}
TRAJ_LABEL = {'empty_lot': 'Empty lot', 'rebuilt_degraded': 'Rebuilt degraded', 'rebuilt_equal': 'Rebuilt equal', 'rebuilt_improved': 'Rebuilt improved', 'obscured': 'Obscured (excluded)'}
TRAJ_ORDER = ['empty_lot', 'rebuilt_degraded', 'rebuilt_equal', 'rebuilt_improved', 'obscured']

def fig_s2(df):
    """Stacked-bar grid for the 20 largest fires (by working-sample n)."""
    g = df.groupby('incident').agg(n=('crash_id', 'count'), year=('event_date', lambda x: pd.to_datetime(x).dt.year.iloc[0]), state=('state_fips', 'first')).reset_index()
    top20 = g.sort_values('n', ascending=False).head(20)
    keep = set(top20['incident'])
    df_top = df[df['incident'].isin(keep)]
    fr = df_top.groupby(['incident', 'traj']).size().unstack(fill_value=0).reindex(columns=TRAJ_ORDER, fill_value=0).div(df_top.groupby('incident').size(), axis=0)
    fr = fr.merge(top20[['incident', 'n', 'year']], left_index=True, right_on='incident')
    fr = fr.sort_values('n', ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    y = np.arange(len(fr))
    left = np.zeros(len(fr))
    for traj in TRAJ_ORDER:
        ax.barh(y, fr[traj], left=left, color=TRAJ_COLOR[traj], edgecolor='white', linewidth=0.3, label=TRAJ_LABEL[traj])
        left += fr[traj].values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['incident']} ({r['year']:.0f})  n={int(r['n']):,}" for _, r in fr.iterrows()], fontsize=7.2)
    ax.set_xlabel('Share of buildings', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f'{int(x * 100)}%' for x in np.arange(0, 1.01, 0.2)])
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.02), frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / 'figS2_per_fire_trajectory.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figS2_per_fire_trajectory.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  S2 ok')

def fig_s4(df):
    """ZIP-level map of fire-affected California ZIPs coloured by FAIR Plan
    share + sized by destroyed-building count. Falls back to non-FHSZ map
    if shapefile unavailable."""
    import geopandas as gpd
    SHP = ROOT / 'data/raw/shapefiles/cb_2022_us_county_500k.shp'
    counties = gpd.read_file(SHP)
    ca = counties[counties['STATE_NAME'] == 'California'].to_crs(4326)
    outline = ca.dissolve()
    ca_df = df[df['state_fips'] == '06'].copy()
    ag = ca_df.dropna(subset=['zip5']).groupby('zip5').agg(lat=('lat', 'median'), lon=('lon', 'median'), n=('crash_id', 'count'), fair=('fair_pct_2022', 'first'), nonr=('nonrenew_rate_2020_2023', 'first')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    cmap_f = LinearSegmentedColormap.from_list('f', ['#f4f1e8', '#f4d35e', '#ee964b', '#c0392b'])
    cmap_n = LinearSegmentedColormap.from_list('n', ['#e8f4f8', '#a3c4dc', '#5d8aa8', '#2c4a6b'])
    for ax, (col, lab, cmap, vmax) in zip(axes, [('fair', 'FAIR Plan share, 2022 (%)', cmap_f, 50), ('nonr', 'Cumulative non-renewal rate, 2020–2023 (%)', cmap_n, 30)]):
        ax.set_facecolor('#e1eef4')
        ax.set_xlim(-125, -114)
        ax.set_ylim(32.3, 42.2)
        outline.plot(ax=ax, facecolor='#f7f3ec', edgecolor='#3a3a3a', linewidth=1.0)
        ca.plot(ax=ax, facecolor='none', edgecolor='#cfc6b8', linewidth=0.4)
        sub = ag.dropna(subset=[col])
        sizes = np.clip(np.sqrt(sub['n']) * 6, 18, 360)
        sc = ax.scatter(sub['lon'], sub['lat'], s=sizes, c=sub[col] * 100, cmap=cmap, vmin=0, vmax=vmax, edgecolor='white', linewidth=0.6, alpha=0.92, zorder=8)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.5)
        cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / 'figS4_zip_insurance_map.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figS4_zip_insurance_map.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  S4 ok')

def fig_s6():
    """Per-building scatter: SegFormer vegetation ratio vs HSV green ratio."""
    seg = pd.read_parquet(PROC / 'all_vegetation_segformer_per_building.parquet')
    hsv = pd.read_parquet(PROC / 'all_vegetation_hsv_per_building.parquet')
    print('  seg cols:', list(seg.columns)[:8])
    print('  hsv cols:', list(hsv.columns)[:8])
    seg_col = next((c for c in seg.columns if 'veg' in c.lower() and 'pre' in c.lower()), None)
    hsv_col = next((c for c in hsv.columns if 'green' in c.lower() and 'pre' in c.lower()), None)
    if seg_col is None or hsv_col is None:
        seg_col = [c for c in seg.columns if c not in ('crash_id',)][0]
        hsv_col = [c for c in hsv.columns if c not in ('crash_id',)][0]
    m = seg[['crash_id', seg_col]].merge(hsv[['crash_id', hsv_col]], on='crash_id', how='inner').dropna()
    if len(m) > 5000:
        m = m.sample(5000, random_state=0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(m[seg_col], m[hsv_col], s=4, alpha=0.24, color='#2D708E', edgecolor='none')
    if len(m) > 5:
        b1, b0 = np.polyfit(m[seg_col], m[hsv_col], 1)
        xs = np.linspace(m[seg_col].min(), m[seg_col].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color='#205E7F', lw=1.3)
        r = np.corrcoef(m[seg_col], m[hsv_col])[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {r:.2f}\nn = {len(m):,}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none'))
    ax.set_xlabel('Pre-fire vegetation fraction from SegFormer', fontsize=9)
    ax.set_ylabel('Pre-fire vegetation fraction from HSV thresholding', fontsize=9)
    ax.plot([0, 1], [0, 1], color='#89959C', lw=0.8, ls='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / 'figS6_hsv_vs_segformer.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figS6_hsv_vs_segformer.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  S6 ok')
