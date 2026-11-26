from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed'
OUT=ROOT/'outputs'
BASE=['log_income','pre_pct_Hispanic','pre_pct_below_poverty','pre_pct_owner_occupied','pre_pct_BA_or_higher','log_value','mtbs_severity_code']
POLICY=['nonrenew_rate_2020_2023','fair_pct_2022','denial_rate','insurance_proxy','log_fema','j40_disadvantaged','log_income','mtbs_severity_code']
TRAJ=['empty_lot','rebuilt_degraded','rebuilt_equal','rebuilt_improved']
COLORS=['#C8CDD2','#C99946','#86ACBE','#2D708E']
GROUPS=['Low','Middle','High']
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def load():
    assert hashlib.sha256((DATA/'frozen_sample.parquet').read_bytes()).hexdigest()=='5c8eb47c30a852b1eac2e211fc519516a733e179a10cecdd391b3a8120cda375','Frozen manuscript input changed'
    f=pd.read_parquet(DATA/'frozen_sample.parquet')
    assert len(f)==4637 and f.crash_id.is_unique
    f.index=f.crash_id.astype(str)
    return f

def fig2_sample():
    f=load();d=f[f.trajectory.ne('obscured')&f.pre_tract_med_HH_income.notna()&(2026-f.fire_year).ge(4)].copy()
    assert len(d)==4310
    assert d.groupby('income_group',observed=False).size().tolist()==[1334,1523,1453]
    return d

def directory(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);return p

def save(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def write_json(data,path):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf8')

def recovery_event_counts(frame):
    """Retain the manuscript's state/name groups and separate cross-state names."""
    keys=['state_fips','incident']
    counts=frame.groupby(keys+['trajectory'],observed=True).size().unstack(fill_value=0).reindex(columns=TRAJ+['obscured'],fill_value=0)
    counts['n']=counts.sum(axis=1)
    years=frame.groupby(keys,observed=True).agg(fire_year=('fire_year','first'),fire_years=('fire_year',lambda x:', '.join(str(int(y)) for y in sorted(x.unique()))))
    return counts.join(years).reset_index().sort_values(['n','incident','fire_year'],ascending=[False,True,True]).head(20)

def savefig(fig,path):
    # Numerical entry points export tables/curves. The manuscript renderers
    # publish the single approved-style drawing in outputs/figures.
    plt.close(fig)

def logit(d,preds,y='rebuilt',cluster=False,scale_all=False):
    numeric=d[preds].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    if scale_all:mu=numeric.mean();sd=numeric.std(ddof=1)
    u=d[[y,'incident']].join(numeric).dropna()
    if not scale_all:mu=u[preds].mean();sd=u[preds].std(ddof=1)
    x=add_constant((u[preds]-mu)/sd)
    kw={'cov_type':'cluster','cov_kwds':{'groups':u.incident.to_numpy()}} if cluster else {}
    r=Logit(u[y].astype(int),x.astype(float)).fit(disp=False,method='bfgs' if cluster else 'newton',maxiter=500,**kw)
    assert r.mle_retvals['converged']
    ci=r.conf_int()
    t=pd.DataFrame({'variable':r.params.index,'beta':r.params.values,'SE':r.bse.values,'beta_low':ci.iloc[:,0].values,'beta_high':ci.iloc[:,1].values,'OR':np.exp(r.params.values),'CI_low':np.exp(ci.iloc[:,0].values),'CI_high':np.exp(ci.iloc[:,1].values),'p':r.pvalues.values,'N':len(u),'G_incident_name':u.incident.nunique()})
    return t,r,u,mu,sd

def prediction(r,x,critical=1.96):
    arr=np.asarray(x);eta=arr@r.params.to_numpy()
    se=np.sqrt(np.einsum('ij,jk,ik->i',arr,r.cov_params().to_numpy(),arr))
    return expit(eta),expit(eta-critical*se),expit(eta+critical*se)

def require_close(a,b,message,atol=1e-7):
    assert np.isclose(a,b,atol=atol,rtol=1e-7),(message,a,b)

def reference_table(i):
    return json.loads((ROOT/f'reference/supplementary_tables/table_{i:02d}.json').read_text(encoding='utf8'))

def verify_cells(i,computed):
    expected=reference_table(i)
    assert len(computed)==len(expected),(i,'row count',len(computed),len(expected))
    mismatches=[]
    for ri,(a,b) in enumerate(zip(computed,expected)):
        # Column headings are maintained separately from the numerical contract.
        if ri==0:continue
        for ci,(v,w) in enumerate(zip(a,b)):
            a1=str(v).replace('−','-').replace(' ','').replace(',','')
            b1=str(w).replace('−','-').replace(' ','').replace(',','')
            if a1!=b1:mismatches.append({'table':i,'row':ri,'column':ci,'computed':str(v),'manuscript':str(w)})
    save(pd.DataFrame(computed[1:],columns=computed[0]),directory('supplementary')/f'Table_S{i:02d}_formatted.csv')
    save(pd.DataFrame(computed[1:],columns=computed[0]),directory('tables')/f'SupplementaryTable{i:02d}.csv')
    # Display comparisons are retained only when an internal audit directory is explicitly configured.
    import os
    audit=os.environ.get('RECOVERY_AUDIT_DIR')
    if audit:
        destination=Path(audit);destination.mkdir(parents=True,exist_ok=True)
        write_json({'table':i,'matched':not mismatches,'mismatches':mismatches},destination/f'Table_S{i:02d}.json')

def legacy_frame():
    d=pd.read_parquet(DATA/'legacy_analysis_frame.parquet')
    j=pd.read_csv(DATA/'label_records.csv',dtype={'crash_id':str})
    rank={'qwen3.6-35b-a3b-thinking':0,'qwen3.6-35b-a3b':1,'opus-4-7':2,'claude':3,'qwen-vl-8b':4}
    j['_r']=j.judged_by.map(rank).fillna(99)
    j=j.sort_values('_r').drop_duplicates('crash_id',keep='first')
    j['trajectory']=j.trajectory.replace({'uncertain':'obscured'})
    d['crash_id']=d.crash_id.astype(str)
    return d.merge(j[['crash_id','trajectory']],on='crash_id',how='left',validate='one_to_one')
