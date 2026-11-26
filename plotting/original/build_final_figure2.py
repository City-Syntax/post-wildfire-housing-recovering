from pathlib import Path
import ast, sys, json, hashlib
WORK = Path(__file__).resolve().parent
REV = WORK.parent.parent
PROJECT = REV.parents[2]
ANALYSIS = REV.parent / '补充分析_20261029'
sys.path.insert(0, str(ANALYSIS / '_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import build_design_matrices
OUT = Path('C:\\program1\\jiejing_20260704\\jiejing\\crash_response_inequality\\论文与意见\\5.NC第一轮返修的\\返修草拟_20261029\\已确认修改_系统稿768406_20261105_v15\\图2_R3-06_逐火灾观测点')
OUT.mkdir(exist_ok=True)
SOURCE = PROJECT / 'scripts/draw_fig2_combined_income.py'
FROZEN = ANALYSIS / 'outputs/frozen_original_numerical_sample.parquet'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
frozen_hash = sha(FROZEN)
drawing_source = SOURCE.read_text(encoding='utf-8').replace('Middle 60–100', 'Middle 60–<100').replace('Middle  60–100', 'Middle  60–<100').replace('Higher  >100', 'Higher  ≥100')
module = ast.parse(drawing_source)
constants = {'OBSERVATION_END_YEAR', 'MIN_YEARS_SINCE_FIRE', 'INCOME_CUTS', 'INCOME_LOWER_K', 'MIN_FIRE_REBUILT', 'TRAJECTORY_PRIORITY', 'REBUILT', 'CLASSIFIABLE', 'TIER_ORDER', 'COLORS', 'GROUP_COLORS', 'GROUP_BACKGROUNDS', 'LABELS'}
for node in module.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id in constants):
        globals()[node.targets[0].id] = ast.literal_eval(node.value)

class FormulaMatrixCompatibility(ast.NodeTransformer):

    def visit_Assign(self, node):
        if any((isinstance(t, ast.Name) and t.id == 'design' for t in node.targets)):
            node.value = ast.parse('prediction_data.assign(Intercept=1.0).loc[:, model.model.exog_names].to_numpy(dtype=float)', mode='eval').body
        return self.generic_visit(node)
functions = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
functions = FormulaMatrixCompatibility().visit(functions)
exec(compile(ast.fix_missing_locations(functions), 'original_figure_functions', 'exec'), globals())
fm._load_fontmanager(try_read_cache=False)
plt.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'font.size': 7.5, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.75, 'lines.linewidth': 1.2, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
frame = pd.read_parquet(FROZEN)
frame['rebuilt_any'] = frame.trajectory.isin(REBUILT).astype(int)
frame['improved'] = frame.trajectory.eq('rebuilt_improved').astype(int)
data = frame[frame.trajectory.isin(CLASSIFIABLE) & (2026 - frame.fire_year).ge(4) & frame.pre_tract_med_HH_income.notna()].copy()
data['income_group'] = pd.cut(data.pre_tract_med_HH_income, [-np.inf, 60000, 100000, np.inf], labels=TIER_ORDER, right=False)
assert len(data) == 4310
model_data, income_k, predicted, lower, upper, original_xlim = fit_rebuilding_curve(data)
assert len(model_data) == 4121
fires = model_data.groupby('event_id', as_index=False).agg(incident=('incident', 'first'), event_date=('event_date', 'first'), n=('crash_id', 'size'), rebuilt_n=('rebuilt_any', 'sum'), income_median=('pre_tract_med_HH_income', 'median'))
fires['observed_rebuilt_pct'] = 100 * fires.rebuilt_n / fires.n
fires['income_k'] = fires.income_median / 1000
assert len(fires) == 32 and fires.n.sum() == len(model_data)
reference = pd.read_csv(REV / '_work/sixteen_changes_20261105/next_R3_06_Figure2b_same_model_sample_fire_points.csv')
check = fires.merge(reference, on='event_id', suffixes=('', '_reference'), validate='one_to_one')
for col in ['n', 'rebuilt_n', 'income_median', 'observed_rebuilt_pct']:
    assert np.allclose(check[col], check[col + '_reference'])
fires.sort_values('n', ascending=False).to_csv(OUT / 'Figure2b_fire_observations.csv', index=False, encoding='utf-8-sig')
fires[fires.n.ge(50)].sort_values('n', ascending=False).to_csv(OUT / 'Figure2b_labeled_events.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'income_thousand_USD': income_k, 'adjusted_probability': predicted, 'lower95': lower, 'upper95': upper}).to_csv(OUT / 'Figure2b_adjusted_curve.csv', index=False, encoding='utf-8-sig')
POINT_COLOR = '#7998A6'

def point_size(n, scale=1):
    return (10 + 3.5 * np.sqrt(np.asarray(n))) * scale ** 2

def draw_preview_b(ax, scale=1):
    xlim = (25, original_xlim[1])
    add_group_backgrounds(ax, label_positions=(40, 77, float(np.sqrt(100 * xlim[1]))), x_limits=xlim)
    ax.fill_between(income_k, lower * 100, upper * 100, color=COLORS['curve_band'], alpha=0.78, linewidth=0, zorder=2)
    ax.plot(income_k, predicted * 100, color=COLORS['curve'], linewidth=2.25 * scale, zorder=4)
    ordered = fires.sort_values('n', ascending=False)
    ax.scatter(ordered.income_k, ordered.observed_rebuilt_pct, s=point_size(ordered.n, scale), facecolor=POINT_COLOR, edgecolor='white', linewidth=0.65 * scale, alpha=0.88, zorder=5, clip_on=False)
    offsets = {'Tubbs': (-10, 0, 'right'), 'Caldor': (7, 0, 'left'), 'Woolsey': (0, -10.5, 'center'), 'Thomas': (0, 10, 'center'), 'Nuns': (0, 9, 'center'), 'Carr': (-6, 0, 'right'), 'Atlas': (7, 0, 'left'), 'Valley': (-6, 0, 'right')}
    for _, row in fires[fires.n.ge(50)].iterrows():
        marker_radius_pt = float(np.sqrt(point_size(row.n, scale))) / 2
        arrow = {'arrowstyle': '-|>', 'color': '#587381', 'lw': 0.45 * scale, 'mutation_scale': 4.7 * scale, 'shrinkA': 0.4 * scale, 'shrinkB': marker_radius_pt + 0.55 * scale, 'connectionstyle': 'arc3,rad=0'}
        if row.incident == 'Camp':
            arrow['connectionstyle'] = 'angle,angleA=0,angleB=90,rad=0'
            label = ax.annotate('Camp', (row.income_k, row.observed_rebuilt_pct), xytext=(43.4, 2.7), xycoords='data', textcoords='data', ha='right', va='center', fontsize=5.9 * scale, color=COLORS['axis'], annotation_clip=False, arrowprops=arrow, zorder=8)
        else:
            dx, dy, ha = offsets[row.incident]
            label = ax.annotate(row.incident, (row.income_k, row.observed_rebuilt_pct), xytext=(dx * scale, dy * scale), textcoords='offset points', ha=ha, va='center', fontsize=6.2 * scale, color=COLORS['axis'], arrowprops=arrow, zorder=8)
        label.set_gid('fire_label_' + row.incident)
        label.set_path_effects([pe.Stroke(linewidth=1.2 * scale, foreground='white', alpha=0.85), pe.Normal()])
        assert np.allclose(label.xy, [row.income_k, row.observed_rebuilt_pct])
    handles = [Line2D([0], [0], color=COLORS['curve'], linewidth=2.25 * scale, label='Adjusted estimate'), Patch(facecolor=COLORS['curve_band'], edgecolor='none', label='95% confidence interval'), Line2D([0], [0], marker='o', linestyle='none', markersize=4.5 * scale, markerfacecolor=POINT_COLOR, markeredgecolor='white', markeredgewidth=0.6 * scale, label='Observed fire-level proportion')]
    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.905), frameon=True, fontsize=6.15 * scale, handlelength=1.45, handletextpad=0.55, labelspacing=0.45, borderaxespad=0, borderpad=0.3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_alpha(0.92)
    sizekey = ax.inset_axes([0.56, 0.72, 0.355, 0.19], zorder=10)
    sizekey.set_facecolor('white')
    for spine in sizekey.spines.values():
        spine.set_edgecolor('#D9E1E5')
        spine.set_linewidth(0.45)
    sizekey.set_xticks([])
    sizekey.set_yticks([])
    sizekey.set_xlim(0, 1)
    sizekey.set_ylim(0, 1)
    sizekey.text(0.5, 0.875, 'Buildings per fire', transform=sizekey.transAxes, ha='center', va='center', fontsize=6.1 * scale, color=COLORS['muted'])
    for x, n in zip([0.16, 0.5, 0.84], [10, 100, 1000]):
        sizekey.scatter([x], [0.455], s=point_size(n, scale), transform=sizekey.transAxes, facecolor=POINT_COLOR, edgecolor='white', linewidth=0.65 * scale, alpha=0.88, zorder=9)
        sizekey.text(x, 0.11, f'{n:,}', transform=sizekey.transAxes, ha='center', va='center', fontsize=5.9 * scale, color=COLORS['muted'])
    ax.set_title('Observed and adjusted rebuilding', loc='left', pad=8 * scale, fontsize=9 * scale)
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ticks = [t for t in [30, 50, 75, 100, 150, 200, 250] if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7 * scale)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Pre-fire census-tract income\n(thousand USD, log scale)', fontsize=8 * scale)
    ax.set_ylabel('Rebuilding (%)', fontsize=8 * scale)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=7 * scale)
    ax.tick_params(axis='x', pad=6 * scale)
    style_axis(ax)
    panel_label(ax, 'b')
fig = plt.figure(figsize=(190 / 25.4, 155 / 25.4))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.22], height_ratios=[1, 0.92], left=0.09, right=0.985, bottom=0.1, top=0.875, wspace=0.31, hspace=0.48)
axes = [fig.add_subplot(grid[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
draw_panel_a(axes[0], data)
axes[0].set_title('Observed recovery outcomes', loc='left', pad=8)
axes[0].set_xticklabels(['Low\n<60', 'Middle\n60 to <100', 'High\n≥100'])
axes[0].get_legend().remove()
legend_order = ['rebuilt_equal', 'rebuilt_improved', 'empty_lot', 'rebuilt_degraded']
axes[0].legend(handles=[Patch(facecolor=COLORS[k], edgecolor='white', label=LABELS[k]) for k in legend_order], ncol=4, loc='lower left', bbox_to_anchor=(-0.15, 1.1), frameon=False, fontsize=6.0, handlelength=1.1, handletextpad=0.45, columnspacing=0.7, borderaxespad=0)
draw_preview_b(axes[1])
draw_panel_c(axes[2], data)
axes[2].set_title('Upgraded share among rebuilt buildings', loc='left', pad=8)
axes[2].set_xticklabels(['Low\n<60', 'Middle\n60 to <100', 'High\n≥100'])
draw_panel_d(axes[3], data)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / ('Figure2_updated.' + ext), dpi=300, facecolor='white')
plt.close(fig)
fig, ax = plt.subplots(figsize=(5.4, 4.3))
fig.subplots_adjust(left=0.16, right=0.975, bottom=0.18, top=0.85)
draw_preview_b(ax, scale=1.3)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / ('Figure2b_detail.' + ext), dpi=300, facecolor='white')
plt.close(fig)
