"""Restore the current composite layouts using original numerical definitions."""
from pathlib import Path
import json, sys, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from common import load, logit, POLICY, prediction, write_json, save
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

def figure4(stage, out):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 0.9, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    d = load()
    d = d[d.state_fips.eq('06')].copy()
    labels = ['Low', 'Middle', 'High']
    colors = ['#c7522a', '#e0a82e', '#1f6f8b']
    fig = plt.figure(figsize=(4859 / 300, 2429.01 / 300), facecolor='white')
    lefts = [0.052, 0.247, 0.442]
    columns = ['nonrenew_rate_2020_2023', 'fair_pct_2022', 'denial_rate']
    limits = [(0, 0.25), (0, 45), (0, 0.3)]
    top_titles = ['Non-renewal rates by income', 'FAIR Plan shares by income', 'Mortgage denial rates by income']
    bottom_titles = ['Rebuilding by non-renewal rate', 'Rebuilding by FAIR Plan share', 'Rebuilding by mortgage denial']
    ylabels = ['Non-renewal rate (ZIP, 2020–23)', 'FAIR Plan share of ZIP, 2022 (%)', 'HMDA mortgage denial rate (tract)']
    xlabels = ['Non-renewal rate (ZIP)', 'FAIR Plan share of ZIP (%)', 'HMDA mortgage denial rate (tract)']
    curve_rows = []
    for j, col in enumerate(columns):
        ax = fig.add_axes([lefts[j], 0.565, 0.141, 0.375])
        values = [d.loc[d.income_group.eq(k), col].dropna().to_numpy() for k in labels]
        bodies = ax.violinplot(values, positions=np.arange(3), widths=0.78, showextrema=False)
        for i, (body, val) in enumerate(zip(bodies['bodies'], values)):
            body.set_facecolor(colors[i])
            body.set_edgecolor(colors[i])
            body.set_alpha(0.55)
            a, b = np.percentile(val, [25, 75])
            median = np.median(val)
            ax.plot([i, i], [a, b], color='black', lw=2)
            ax.plot([i - 0.18, i + 0.18], [median, median], color='white', lw=2)
        ax.set_xticks(range(3), ['Lower\nincome', 'Middle\nincome', 'Higher\nincome'])
        ax.set(ylabel=ylabels[j], ylim=(0, 0.28) if j == 0 else limits[j])
        ax.set_title(top_titles[j], loc='left', pad=10)
        ax.text(-0.24, 1.055, 'abc'[j], transform=ax.transAxes, fontsize=18, fontweight='bold')
        ax = fig.add_axes([lefts[j], 0.065, 0.141, 0.375])
        for k, color in zip(labels, colors):
            u = d[d.income_group.eq(k)].dropna(subset=[col, 'rebuilt'])
            x = u[col].astype(float)
            qlo, qhi = np.percentile(x, [2, 98])
            xx = np.linspace(max(limits[j][0], qlo), min(limits[j][1], qhi), 200)
            model = Logit(u.rebuilt.astype(int), add_constant(x)).fit(disp=False)
            pr, lo, hi = prediction(model, add_constant(pd.DataFrame({col: xx})), critical=1.645)
            ax.plot(xx, pr, color=color, lw=1.7)
            ax.fill_between(xx, lo, hi, color=color, alpha=0.15)
            curve_rows.extend(({'variable': col, 'group': k, 'x': a, 'p': b, 'low90': c, 'high90': e, 'N': len(u)} for a, b, c, e in zip(xx, pr, lo, hi)))
        ax.set(xlim=(0, 0.25) if j == 2 else limits[j], ylim=(0, 1), xlabel=xlabels[j], ylabel='Rebuild probability')
        ax.set_title(bottom_titles[j], loc='left', pad=10)
        ax.text(-0.24, 1.055, 'def'[j], transform=ax.transAxes, fontsize=18, fontweight='bold')
        handles = [Line2D([], [], color=c, lw=2, label=t) for c, t in zip(colors, ['Lower (<USD 60,000)', 'Middle (USD 60,000–100,000)', 'Higher (≥USD 100,000)'])]
        ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8)
    table, model, u, mu, sd = logit(d, POLICY, cluster=True, scale_all=True)
    t = table.set_index('variable')
    ax = fig.add_axes([0.695, 0.065, 0.282, 0.875])
    ys = np.arange(8)[::-1]
    pretty = ['Non-renewal\nrate (ZIP)', 'FAIR Plan\nexposure (ZIP)', 'HMDA\ndenial rate', 'Insurance\nproxy (tract)', 'FEMA IHP\nfunding', 'Justice40\ndisadvantaged', 'log(tract\nincome)', 'Burn severity\n(MTBS)']
    for y, col in zip(ys, POLICY):
        r = t.loc[col]
        color = colors[0] if r.p < 0.05 else '#888888'
        ax.errorbar(r.OR, y, xerr=[[r.OR - r.CI_low], [r.CI_high - r.OR]], fmt='o', color=color, markersize=8, capsize=4, lw=1.6)
        ax.text(r.CI_high * 1.15, y, f'{r.OR:.2f}' + ('***' if r.p < 0.001 else '**' if r.p < 0.01 else '*' if r.p < 0.05 else ''), va='center', fontweight='bold', fontsize=11)
    ax.axvline(1, color='black', ls='--', lw=0.8)
    ax.set_yticks(ys, pretty)
    ax.set_xscale('log')
    ax.set_xlim(0.025, 11)
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3], labels=['0.05', '0.1', '0.3', '1', '3'])
    ax.set_xlabel('Odds ratio (per 1 SD)')
    ax.set_title('Joint-model associations with rebuilding', loc='left', pad=12)
    ax.text(-0.22, 1.025, 'g', transform=ax.transAxes, fontsize=18, fontweight='bold')
    ax.axhline(1.5, ls=':', color='#bbbbbb', lw=0.6)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / ('Figure4.' + ext), dpi=300, bbox_inches=None, facecolor='white')
    plt.close(fig)
    save(table, out / 'Figure4_joint_model_data.csv')
    save(pd.DataFrame(curve_rows), out / 'Figure4_curve_data.csv')
    write_json({'mode': 'Newly fitted and redrawn from frozen numerical sample; original eight-predictor joint model and original curve support/90% bands retained. Layout reconstructed from supplied scripts and current artwork.', 'N_joint': len(u)}, out / 'Figure4_render_provenance.json')
