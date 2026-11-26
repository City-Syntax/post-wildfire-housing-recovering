"""Run the supplied original plotting routines in an isolated output workspace."""
from pathlib import Path
import ast,argparse,json,shutil,subprocess,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'plotting/original'
sys.path.insert(0,str(ROOT/'code'))
from common import DATA,write_json

def settings():
    p=ROOT/'local_assets.json'
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else {}

def workspace():
    stage=ROOT/'.render_workspace';stage.mkdir(exist_ok=True)
    files={'building_inventory.parquet':'table_A_buildings.parquet','legacy_analysis_frame.parquet':'analysis/analysis_frame.parquet','label_records.csv':'vlm_judgments.csv','all_vegetation_segformer.parquet':'all_vegetation_segformer.parquet'}
    for src,dst in files.items():
        out=stage/'data/processed'/dst;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DATA/src,out)
    (stage/'scripts').mkdir(exist_ok=True)
    for src in ORIGINAL.glob('draw_fig*.py'):shutil.copyfile(src,stage/'scripts'/src.name)
    return stage

def load_source(name,stage,overrides=None):
    source=ORIGINAL/name;tree=ast.parse(source.read_text(encoding='utf8'))
    # Original functions, constants and style calls are retained. Legacy mkdir
    # expressions and the script's main guard are not executed on import.
    keep=[]
    for n in tree.body:
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign,ast.FunctionDef,ast.ClassDef)):keep.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
            call=ast.unparse(n.value.func)
            if call.endswith(('rcParams.update','filterwarnings','_load_fontmanager')):keep.append(n)
    env={'__file__':str(stage/'scripts'/name),'__name__':'original_plot_functions'}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(source),'exec'),env)
    env['ROOT']=stage
    env.update(overrides or {})
    return env

def figure3(stage,out):
    cfg=settings();e=load_source('draw_fig3_combined_vegetation.py',stage)
    if not cfg.get('image_asset_root'):
        summaries=e['load_vegetation_summary']()
        plt.rcParams['savefig.bbox']=None
        fig,ax=plt.subplots(figsize=(5.2,4.1));e['draw_vegetation_panel'](ax,summaries);ax.set_title('',loc='left')
        for text in list(ax.texts):
            if text.get_text()=='b':text.remove()
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/('Figure3_vegetation.'+ext),dpi=300,facecolor='white',bbox_inches=None)
        plt.close(fig)
        return
    assets=Path(cfg['image_asset_root'])
    for ident,pre,post in e['EXAMPLES']:
        for name in [pre,post]:
            dst=stage/'data/seg_visuals'/ident/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(assets/'data/seg_visuals'/ident/name,dst)
    for k,name in [('PNG_OUT','Figure3.png'),('PDF_OUT','Figure3.pdf'),('MANUSCRIPT_OUT','Figure3.png'),('PANEL_A_OUT','Figure3a.png'),('PANEL_A_PDF_OUT','Figure3a.pdf')]:e[k]=out/name
    original_seg=e['draw_segmentation_panel'];original_veg=e['draw_vegetation_panel']
    def segmentation(fig,spec,rows):
        axes=original_seg(fig,spec,rows)
        for text in list(axes[0][0].texts):
            if text.get_text()=='Pre- and post-fire street-view segmentation':text.remove()
        legend_axis=fig.axes[-1]
        import matplotlib.patches as patches
        handles=[patches.Patch(facecolor=e['SEGMENTATION_COLORS'][k],edgecolor='none',label=k) for k in ['Vegetation','Building','Road','Sky','Other']]
        legend_axis.get_legend().remove();legend_axis.legend(handles=handles,ncol=5,loc='center',frameon=False,columnspacing=.8,handlelength=1.2,fontsize=5.8)
        return axes
    def vegetation(ax,summaries):
        original_veg(ax,summaries);ax.set_title('',loc='left')
        for text in ax.texts:
            replacements={'Lower income':'Low','Middle income':'Middle','Higher income':'High'}
            if text.get_text() in replacements:text.set_text(replacements[text.get_text()])
    rows=e['load_examples']();summaries=e['load_vegetation_summary']()
    plt.rcParams['savefig.bbox']=None
    fig=plt.figure(figsize=(2000/300,891/300),facecolor='white')
    spec=fig.add_gridspec(1,1,left=.035,right=.455,bottom=.062,top=.844)
    axes=segmentation(fig,spec[0,0],rows)
    for text in list(axes[0][0].texts):
        if text.get_text()=='a':text.remove()
    ax=fig.add_axes([.551,.12,.437,.79]);vegetation(ax,summaries)
    for text in list(ax.texts):
        if text.get_text()=='b':text.remove()
    for x,label in [(.014,'a'),(.500,'b')]:fig.text(x,.958,label,fontsize=10.5,fontweight='bold',color=e['COLORS']['axis'])
    for ext in ['png','pdf']:fig.savefig(out/('Figure3.'+ext),dpi=300,facecolor='white',bbox_inches=None)
    plt.close(fig)

def figure5(stage,out):
    import numpy as np
    e=load_source('draw_fig5_unified_palette.py',stage)
    data,sold=e['load_data']();assert len(sold)==710
    assert abs(sold.price_change_pct.median()-138.3849765258216)<1e-9
    bins=np.r_[np.linspace(-100,0,6)[:-1],np.geomspace(1,2000,30)];hist,_=np.histogram(sold.price_change_pct,bins=bins)
    original_bins=__import__('pandas').read_csv(ROOT/'reference/original_plotting_data/Figure5_histogram_bins.csv')
    assert hist.tolist()==original_bins.number_of_lots.tolist() and int(hist.sum())==710
    src=ast.parse((ORIGINAL/'draw_figure5_annotation.py').read_text(encoding='utf8'))
    env={'env':e,'np':np,'plt':plt}
    exec(compile(ast.Module(body=[x for x in src.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'original_figure5_annotation','exec'),env)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.2,'axes.linewidth':.7,'xtick.labelsize':6.6,'ytick.labelsize':6.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    e['panel_heading']=env['heading']
    width_mm=180;height_mm=width_mm*624/1787
    fig=plt.figure(figsize=(width_mm/25.4,height_mm/25.4),facecolor='white')
    axs=[fig.add_axes([.062,.200,.265,.673]),fig.add_axes([.410,.194,.231,.660]),fig.add_axes([.760,.194,.230,.660])]
    e['draw_panel_a'](axs[0],sold);e['draw_panel_b'](axs[1],sold);e['draw_panel_c'](axs[2],data);env['annotate_bin_width'](axs[0])
    axs[0].set_xlabel('Recorded sale-price change (%)',labelpad=5);axs[0].set_ylabel('Building-linked records',labelpad=5);axs[1].set_ylabel('Recorded sale-price change (%)',labelpad=5)
    for ext in ['png','pdf','svg']:fig.savefig(out/('Figure5.'+ext),dpi=600,facecolor='white')
    plt.close(fig)

def figure2(stage,out):
    source=ORIGINAL/'build_final_figure2.py';tree=ast.parse(source.read_text(encoding='utf8'))
    reference=ROOT/'artwork/original/Figure2.png'
    overrides={'WORK':out,'REV':stage/'revision','PROJECT':stage,'ANALYSIS':stage/'analysis','OUT':out,'SOURCE':stage/'scripts/draw_fig2_combined_income.py','FROZEN':DATA/'frozen_sample.parquet','original_figure':reference}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            key=n.targets[0].id
            if key in overrides:n.value=ast.parse('Path('+repr(str(overrides[key]))+')',mode='eval').body
            if key=='reference':n.value=ast.parse('pd.read_csv('+repr(str(ROOT/'reference/original_plotting_data/Figure2b_fire_observations.csv'))+')',mode='eval').body
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__file__':str(source),'__name__':'approved_figure2_renderer'})
    # Apply the author's final SVG-only label positioning changes to the newly
    # calculated drawing, without loading the archived final SVG as a template.
    shutil.copyfile(out/'Figure2_updated.svg',out/'12_Figure2_完整图.svg')
    tree=ast.parse((ORIGINAL/'refine_figure2.py').read_text(encoding='utf8'))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='citation_tools')]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORK' for t in n.targets):n.value=ast.parse('Path('+repr(str(out))+')',mode='eval').body
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree),'original_figure2_final_svg_refinement','exec'),{'B':out,'__file__':str(ORIGINAL/'refine_figure2.py'),'__name__':'original_svg_refinement'})
    shutil.copyfile(out/'23_Figure2_分图位置精调_20261122.svg',out/'Figure2.svg')
    from finish_layout import align_figure2
    align_figure2(out/'Figure2.svg',ROOT/'artwork/original/Figure2_layout.svg',out/'Figure2_layout_audit.json')
    render_svg(out/'Figure2.svg',out/'Figure2.png',600,4388,3525)

def render_svg(src,dst,dpi,width=None,height=None):
    cfg=settings();node=cfg.get('node_executable','node')
    command=[node,str(ROOT/'plotting/render_svg.cjs'),str(src),str(dst),str(dpi)]
    if width is not None:command.extend([str(width),str(height)])
    subprocess.run(command,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('figure',choices=['1','2','3','4','5','all']);args=parser.parse_args()
    if args.figure=='1':
        print('Figure 1 redraw omitted; Result 1 numerical outputs are retained.');return
    stage=workspace();out=stage/'main_figures';out.mkdir(parents=True,exist_ok=True)
    for n in (['2','3','4','5'] if args.figure=='all' else [args.figure]):
        print('Rendering from supplied source: Figure',n,flush=True)
        plt.rcdefaults()
        if n in ['1','4']:
            import restored_compositions
            getattr(restored_compositions,'figure'+n)(stage,out)
        else:globals()['figure'+n](stage,out)
        from finish_layout import publish
        stem='Figure3_vegetation' if n=='3' and not settings().get('image_asset_root') else 'Figure'+n
        publish(out,ROOT/'outputs/figures',[stem])
        print('Saved outputs/figures/'+stem+'.png',flush=True)

if __name__=='__main__':main()
