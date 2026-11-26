from pathlib import Path
import sys, json
WORK = Path(__file__).resolve().parent
ANALYSIS = WORK.parent.parent.parent / '补充分析_20261029'
sys.path.insert(0, str(ANALYSIS / '_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

d = pd.read_csv(WORK / 'structure_type_model_results.csv')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .65, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.15, 3.55), facecolor='white')
ax = fig.add_axes([.29, .25, .43, .61])
labels = ['All classifiable buildings', 'Single-family only', 'Mobile and motor homes excluded']
display = ['All classifiable\nbuildings', 'Single-family\nonly', 'Mobile and motor\nhomes excluded']
ys = [2.6, 1.3, 0.0]
colors = {'Not added': '#86ACBE', 'Cubic spline': '#2D708E'}
markers = {'Not added': 'o', 'Cubic spline': 'D'}
for k, cohort in enumerate(labels):
    for label, offset in [('Not added', .19), ('Cubic spline', -.19)]:
        row = d[d.cohort.eq(cohort) & d.time_label.eq(label)].iloc[0]
        y = ys[k] + offset
        ax.errorbar(row.OR, y, xerr=[[row.OR - row.CI_low], [row.CI_high - row.OR]], fmt=markers[label], color=colors[label], ecolor=colors[label], elinewidth=1.15, capsize=2.5, markersize=5, markeredgewidth=.7, zorder=3)
        ax.text(1.075, y, f'{row.OR:.2f} [{row.CI_low:.2f}, {row.CI_high:.2f}]', transform=transforms.blended_transform_factory(ax.transAxes, ax.transData), va='center', fontsize=8, color='#24313A')
    ax.text(-.09, ys[k], display[k] + f'\nn = {int(row.N):,}', transform=transforms.blended_transform_factory(ax.transAxes, ax.transData), va='center', ha='right', fontsize=8.3, linespacing=1.25, color='#24313A')
for y in [.65, 1.95]:
    ax.axhline(y, color='#E4E8EB', lw=.55, zorder=0)
ax.axvline(1, ls=(0, (4, 3)), color='#7B8790', lw=.75, zorder=1)
ax.set_xscale('log')
ax.set_xlim(.8, 8.5)
ax.set_ylim(-.65, 3.2)
ax.set_yticks([])
ax.set_xticks([1, 2, 4, 8], labels=['1', '2', '4', '8'])
ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(axis='x', length=3, width=.65, colors='#24313A')
ax.set_xlabel('Adjusted odds ratio for rebuilding', labelpad=6, color='#24313A')
ax.text(1.075, 1.04, 'OR [95% CI]', transform=ax.transAxes, ha='left', fontsize=8, fontweight='bold', color='#24313A')
handles = [plt.Line2D([], [], marker=markers[k], color=colors[k], lw=1.1, markersize=4.5, label=label) for k, label in [('Not added', 'Original seven covariates'), ('Cubic spline', '+ actual observation time')]]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .017), ncol=2, frameon=False, fontsize=8, columnspacing=2.0, handlelength=1.5)
for suffix in ['png', 'pdf', 'svg']:
    fig.savefig(WORK / ('SupplementaryFigure5_updated.' + suffix), dpi=600, facecolor='white')
plt.close(fig)
print(json.dumps({'figure': str(WORK / 'SupplementaryFigure5_updated.png'), 'rows': len(d), 'positive_CI_lower_bounds': bool(d.CI_low.gt(1).all())}))
