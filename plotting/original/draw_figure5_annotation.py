"""R3-07 only: reproduce the existing Fig. 5 and explain panel a's bin width.

No thresholds, bin edges, samples, price pairings or outcome calculations change.
Panels b and c call the existing plotting functions without alterations.
"""
from pathlib import Path
import sys, ast, json, hashlib

WORK = Path(__file__).resolve().parent
REV = WORK.parent.parent
PROJECT = REV.parents[2]
sys.path.append(str(REV.parent / '补充分析_20261029/_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

OUT = REV / '已确认修改_20261104_v05'
FIGS = OUT / '图5a柱宽说明'
FIGS.mkdir(parents=True, exist_ok=True)
source = PROJECT / 'scripts/draw_fig5_unified_palette.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
# Do not run the source main() or write to its legacy figure destinations.
tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef))]
env = {'__file__': str(source), '__name__': 'figure5_source_readonly'}
exec(compile(tree, str(source), 'exec'), env)
df, sold = env['load_data']()
assert len(sold) == 710
assert abs(sold.price_change_pct.median() - 138.3849765258216) < 1e-9
assert sold.groupby('trajectory').size().reindex(env['TRAJ_ORDER']).tolist() == [325, 203, 177]

plt.rcParams.update({'font.family':'Arial', 'font.size':7,
    'axes.labelsize':7.2, 'axes.linewidth':.7,
    'xtick.labelsize':6.6, 'ytick.labelsize':6.6,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'savefig.facecolor':'white'})

def heading(ax, letter, title):
    # The submitted figure uses panel letters without descriptive headings.
    ax.figure.text({'a':.012,'b':.335,'c':.686}[letter], .992, letter,
            ha='left', va='top',
            fontsize=8, fontweight='bold', color=env['AXIS'], clip_on=False)

env['panel_heading'] = heading

def annotate_bin_width(ax):
    # Bracket follows the actual bin [-20, 1], using data coordinates.
    ax.annotate('', xy=(1,43), xytext=(-20,43),
                arrowprops={'arrowstyle':'|-|','color':env['AXIS'],
                            'linewidth':.65,'shrinkA':0,'shrinkB':0,
                            'mutation_scale':3}, zorder=7)
    # The midpoint on a symmetric-log scale is about -9.1, not (-20+1)/2.
    transform = ax.xaxis.get_transform()
    midpoint = transform.inverted().transform((transform.transform(-20)+transform.transform(1))/2)
    ax.text(float(midpoint), 46, 'Price-change\ninterval',
            ha='center', va='bottom', fontsize=6.2,
            linespacing=1.1, color=env['AXIS'], zorder=8,
            bbox={'facecolor':'white','edgecolor':'none','pad':.3})

def save(fig, stem):
    for extension in ['png','pdf','svg']:
        fig.savefig(FIGS / (stem+'.'+extension), dpi=600, facecolor='white')

# Match the existing figure's aspect ratio, panel positions and restrained palette.
width_mm = 180
height_mm = width_mm * 624 / 1787
fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), facecolor='white')
axes = [fig.add_axes([.062,.200,.265,.673]),
        fig.add_axes([.410,.194,.231,.660]),
        fig.add_axes([.760,.194,.230,.660])]
env['draw_panel_a'](axes[0], sold)
env['draw_panel_b'](axes[1], sold)
env['draw_panel_c'](axes[2], df)
annotate_bin_width(axes[0])
save(fig, 'Fig5_仅补充a柱宽标注')
fig.savefig(WORK/'figure_preview.png', dpi=250, facecolor='white')
plt.close(fig)

# Standalone panel is convenient for discussing the single reviewer question.
fig = plt.figure(figsize=(88/25.4,72/25.4), facecolor='white')
ax = fig.add_axes([.18,.19,.78,.69])
env['draw_panel_a'](ax,sold)
annotate_bin_width(ax)
save(fig,'Fig5a_柱宽标注')
plt.close(fig)

bins = np.r_[np.linspace(-100,0,6)[:-1],np.geomspace(1,2000,30)]
counts,_ = np.histogram(sold.price_change_pct,bins=bins)
pd.DataFrame({'bin_left_percent':bins[:-1],'bin_right_percent':bins[1:],
              'number_of_lots':counts}).to_csv(FIGS/'Fig5a_原分箱与计数.csv',index=False,encoding='utf-8-sig')
assert counts.sum() == 710
audit={'scope':'R3-07 Fig. 5a bar-width annotation only',
       'source_script':str(source),'source_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'n':len(sold),'median_price_change_percent':float(sold.price_change_pct.median()),
       'b_n':[325,203,177], 'bin_edges_percent':bins.tolist(),'bin_counts':counts.tolist(),
       'annotation':'Bracket over the original [-20, 1] percent bin; label Price-change interval',
       'width_mm':width_mm,'height_mm':height_mm,
       'other_figures_changed':False,'date_pairing_changed':False,
       'note':'The separate date-pairing issue is unresolved and excluded from this narrow annotation update.'}
(WORK/'figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))
