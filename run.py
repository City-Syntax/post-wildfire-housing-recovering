"""Run the building-level recovery analyses and generate selected outputs."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
LABELS={'1':'Result 1: recovery summary','2':'Result 2: income and rebuilding','3':'Result 3: Table 1 models','4':'Result 4: insurance and finance','5':'Figure 5: transactions and neighborhood income','vegetation':'Figure 3: vegetation trajectories','supplementary':'Supplementary analyses'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=['all','1','2','3','4','5','vegetation','supplementary'])
    parser.add_argument('--item',help='Select a supplementary table, figure or note, e.g. S14, FigureS6 or Note9.')
    parser.add_argument('--figures',action='store_true',help='Generate the corresponding figures as well as numerical outputs.')
    args=parser.parse_args()
    from verify_inputs import verify
    verify()
    (ROOT/'logs').mkdir(exist_ok=True)
    if args.target=='all':
        jobs=[(k,[str(ROOT/'run.py'),k]) for k in LABELS]
        if args.figures:jobs.extend([('main_figures',[str(ROOT/'plotting/render_original.py'),'all']),('supplementary_figures',[str(ROOT/'plotting/render_supplementary.py'),'all'])])
        runs=[]
        for name,command in jobs:
            started=time.time()
            with (ROOT/'logs'/f'run_{name}.log').open('w',encoding='utf8') as stream:
                result=subprocess.run([sys.executable]+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            runs.append({'target':name,'returncode':result.returncode,'seconds':round(time.time()-started,2)})
            print(('Completed: ' if result.returncode==0 else 'Failed: ')+LABELS.get(name,name.replace('_',' ')),flush=True)
        success=all(x['returncode']==0 for x in runs)
        if success:
            from verify_results import verify_results
            verify_results()
        report={'python':platform.python_version(),'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','statsmodels','matplotlib','pyarrow']},'runs':runs,'completed':success}
        (ROOT/'logs/execution_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        if success:
            print('Results: outputs/\nTables: outputs/tables/\nFigures: outputs/figures/',flush=True)
        else:print('See logs/run_<target>.log for the failed command.',flush=True)
        return 0 if success else 1
    if args.target.isdigit():
        import main_results
        getattr(main_results,'result'+args.target)()
    elif args.target=='vegetation':
        from supplementary import vegetation
        vegetation()
    else:
        import supplementary as si
        groups={'structure_and_time':['S11','S14','FigureS5'],'composition_windows_coverage':['S10','S12','S13','FigureS1','FigureS3'],'insurer_models':['S15','S16','Note9'],'old_sensitivity_models':['S3','S6'],'descriptive_tables':['S2','S4','S5','S8','S9','FigureS2'],'vegetation':['FigureS6']}
        if args.item=='S1':
            import main_results
            main_results.result1()
        elif args.item:
            found=[name for name,items in groups.items() if args.item in items]
            if not found:parser.error('Unknown supplementary item: '+args.item)
            getattr(si,found[0])()
        else:si.supplementary()
    if args.figures and args.target in ['2','4','5','vegetation']:
        number='3' if args.target=='vegetation' else args.target
        subprocess.run([sys.executable,str(ROOT/'plotting/render_original.py'),number],check=True)
    elif args.figures and args.target=='supplementary' and (not args.item or args.item.startswith('FigureS')):
        subprocess.run([sys.executable,str(ROOT/'plotting/render_supplementary.py'),args.item.removeprefix('FigureS') if args.item else 'all'],check=True)
    print('Completed: '+LABELS[args.target],flush=True)
    print('Numerical outputs: outputs/'+{'1':'01_recovery_Figure1','2':'02_income_Figure2','3':'03_models_Table1','4':'04_insurance_Figure4','5':'05_transactions_Figure5','vegetation':'02b_vegetation_Figure3','supplementary':'supplementary'}[args.target],flush=True)
    if args.target=='3':print('Formatted table: outputs/tables/Table1.csv',flush=True)
    if args.figures:print('Figures: outputs/figures/',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
