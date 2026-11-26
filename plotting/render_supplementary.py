"""Recompute and draw supplementary numerical figures in their original styles."""
from pathlib import Path
import ast,argparse,json,sys,zipfile,shutil,contextlib,io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'plotting'))
from common import load,DATA,recovery_event_counts,save
from render_original import workspace,settings

def frame():
    f=load().reset_index(drop=True);m=pd.read_parquet(DATA/'building_inventory.parquet');m['crash_id']=m.crash_id.astype(str)
    f=f.merge(m[['crash_id','lat','lon']],on='crash_id',how='left',validate='one_to_one');f['traj']=f.trajectory
    return f

def original_supp_functions(stage,out):
    tree=ast.parse((ROOT/'plotting/original/build_supp_figs.py').read_text(encoding='utf8'))
    keep=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('TRAJ_'):keep.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in ['fig_s2','fig_s4','fig_s6']:
            if n.name=='fig_s4':
                class Adapt(ast.NodeTransformer):
                    def visit_Assign(self,node):
                        if any(isinstance(t,ast.Name) and t.id=='SHP' for t in node.targets):node.value=ast.parse('Path('+repr(str(stage/'geography/cb_2022_us_county_500k.shp'))+')',mode='eval').body
                        return self.generic_visit(node)
                    def visit_BinOp(self,node):
                        if ast.unparse(node)=='sub[col] * 100':return ast.parse("sub[col] if col == 'fair' else sub[col] * 100",mode='eval').body
                        return self.generic_visit(node)
                n=Adapt().visit(n)
            class ExportBounds(ast.NodeTransformer):
                def visit_Call(self,node):
                    node=self.generic_visit(node)
                    if isinstance(node.func,ast.Attribute) and node.func.attr=='savefig':
                        node.args.insert(0,node.func.value);node.func=ast.Name(id='save_original_bounds',ctx=ast.Load())
                    return node
            keep.append(ExportBounds().visit(n))
    def save_original_bounds(fig,path,**kwargs):
        from finish_layout import save_reference_canvas
        n={'figS2_per_fire_trajectory':1,'figS4_zip_insurance_map':2,'figS6_hsv_vs_segformer':6}[Path(path).stem]
        dims=json.loads((ROOT/'reference/figure_dimensions.json').read_text())[f'SupplementaryFigure{n}']
        save_reference_canvas(fig,path,dims,**kwargs)
    e={'pd':pd,'np':np,'plt':plt,'Path':Path,'LinearSegmentedColormap':LinearSegmentedColormap,'ROOT':stage,'PROC':DATA,'OUT':out,'save_original_bounds':save_original_bounds}
    # The original S6 function uses the long-form file names already present in DATA.
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),'original_supplementary_functions','exec'),e)
    return e

def geometry(stage):
    dest=stage/'geography';dest.mkdir(exist_ok=True)
    for p in (ROOT/'data/geography').glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                f=(dest/i.filename).resolve();assert f.is_relative_to(dest.resolve())
                if not i.is_dir() and not f.exists():f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(i))

def run1(out):
    e=original_supp_functions(workspace(),out)
    counts=recovery_event_counts(load());save(counts,ROOT/'outputs/supplementary/Figure_S1_per_fire_counts.csv')
    expected=json.loads((ROOT/'reference/supplementary_figure1_events.json').read_text(encoding='utf8'))
    assert [(x['incident'],int(x['fire_year']),int(x['n'])) for x in counts.to_dict('records')]==[tuple(x) for x in expected['events']]
    display=counts.sort_values(['n','incident'],ascending=[True,False])
    dims=json.loads((ROOT/'reference/figure_dimensions.json').read_text())['SupplementaryFigure1']
    fig=plt.figure(figsize=((dims[0]+.01)/300,(dims[1]+.01)/300),facecolor='white');ax=fig.add_axes([.241,.075,.739,.865])
    ys=np.arange(len(display));left=np.zeros(len(display))
    for trajectory in e['TRAJ_ORDER']:
        values=display[trajectory].to_numpy()/display.n.to_numpy()
        ax.barh(ys,values,left=left,color=e['TRAJ_COLOR'][trajectory],edgecolor='white',linewidth=.3,label=e['TRAJ_LABEL'][trajectory]);left+=values
    ax.set_yticks(ys,[f'{row.incident} ({row.fire_year})  n = {row.n:,}' for row in display.itertuples()],fontsize=7.2)
    ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2),labels=[f'{int(x*100)}%' for x in np.arange(0,1.01,.2)])
    ax.set_xlabel('Share of buildings',fontsize=9)
    ax.legend(fontsize=7.5,loc='lower left',bbox_to_anchor=(.02,1.022),frameon=False,ncol=5,handlelength=1.8,columnspacing=1.2,borderaxespad=0)
    for extension in ['png','pdf','svg']:fig.savefig(out/('SupplementaryFigure1.'+extension),dpi=300,bbox_inches=None,facecolor='white')
    plt.close(fig)

def run3(stage,out):
    from scipy.stats import gaussian_kde
    master=pd.read_parquet(DATA/'building_inventory.parquet');master['event_date']=pd.to_datetime(master.event_date);candidate=master[master.state_fips.eq('06')&master.damage_level.str.lower().eq('destroyed')&master.event_date.dt.year.between(2014,2025)].copy()
    sample=load().reset_index(drop=True);sample=sample[sample.state_fips.eq('06')].merge(candidate[['crash_id','year_built']],on='crash_id',how='left',validate='one_to_one');assert len(candidate)==47653 and len(sample)==4614
    working=out/'S3_work';working.mkdir(exist_ok=True)
    dims=json.loads((ROOT/'reference/figure_dimensions.json').read_text())['SupplementaryFigure3']
    Image.new('RGB',tuple(dims),'white').save(working/'original_SupplementaryFigure3.png')
    src=(ROOT/'plotting/original/prepare_coverage.py').read_text(encoding='utf8');start=src.index('plt.rcParams.update(');end=src.index("pd.DataFrame(stats).to_csv",start);script=src[start:end].replace('MTBS burn severity (1=low … 4=high)','MTBS burn severity')
    exec(compile(script,'original_supplementary3_drawing','exec'),{'plt':plt,'np':np,'pd':pd,'Image':Image,'gaussian_kde':gaussian_kde,'candidate':candidate,'sample':sample,'WORK':working,'ASSETS':working})
    for ext in ['png','pdf','svg']:shutil.copy2(working/('SupplementaryFigure3_统一口径.'+ext),out/('SupplementaryFigure3.'+ext))

def run5(out):
    from supplementary import structure_and_time
    with contextlib.redirect_stdout(io.StringIO()):structure_and_time()
    df=pd.read_csv(ROOT/'outputs/supplementary/Table_S11_structure_models_full.csv');df['cohort']=df.Sample;df['time_label']=df['Actual-time term'];df.to_csv(out/'structure_type_model_results.csv',index=False)
    tree=ast.parse((ROOT/'plotting/original/draw_supplementary5.py').read_text(encoding='utf8'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORK' for t in n.targets):n.value=ast.parse('Path('+repr(str(out))+')',mode='eval').body
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree),'original_supplementary5_drawing','exec'),{'__file__':str(ROOT/'plotting/original/draw_supplementary5.py')})
    for ext in ['png','pdf','svg']:shutil.copy2(out/('SupplementaryFigure5_updated.'+ext),out/('SupplementaryFigure5.'+ext))
    from finish_layout import align_vector_text
    from render_original import render_svg
    align_vector_text(out/'SupplementaryFigure5.svg',ROOT/'artwork/original/SupplementaryFigure5_layout.svg',ROOT/'logs/SupplementaryFigure5_layout_audit.json')
    render_svg(out/'SupplementaryFigure5.svg',out/'SupplementaryFigure5.png',600,4290,2130)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('figure',choices=['1','2','3','5','6','all']);args=parser.parse_args();stage=workspace();out=stage/'supplementary_figures';out.mkdir(parents=True,exist_ok=True)
    for n in (['1','2','3','5','6'] if args.figure=='all' else [args.figure]):
        print('Supplementary Figure',n,flush=True);plt.rcdefaults();plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
        e=original_supp_functions(stage,out)
        if n=='1':run1(out);stem=None
        elif n=='2':
            geometry(stage)
            with contextlib.redirect_stdout(io.StringIO()):e['fig_s4'](frame())
            stem='figS4_zip_insurance_map'
        elif n=='3':run3(stage,out);stem=None
        elif n=='5':run5(out);stem=None
        else:
            with contextlib.redirect_stdout(io.StringIO()):e['fig_s6']()
            stem='figS6_hsv_vs_segformer'
        if stem:
            for ext in ['png','pdf']:shutil.copy2(out/(stem+'.'+ext),out/(f'SupplementaryFigure{n}.'+ext))
        from finish_layout import publish
        publish(out,ROOT/'outputs/figures',[f'SupplementaryFigure{n}'])
        print(f'Saved outputs/figures/SupplementaryFigure{n}.png',flush=True)

if __name__=='__main__':main()
