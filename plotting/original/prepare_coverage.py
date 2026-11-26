from pathlib import Path
import sys,json,hashlib
WORK=Path(__file__).resolve().parent;REV=WORK.parent.parent;PROJECT=REV.parents[2]
ANALYSIS=PROJECT/'论文与意见/5.NC第一轮返修的/补充分析_20261029'
sys.path.insert(0,str(ANALYSIS/'_deps'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image

ASSETS=WORK/'assets';ASSETS.mkdir(exist_ok=True)
inputs=[PROJECT/'data/processed/table_A_buildings.parquet',ANALYSIS/'outputs/frozen_original_numerical_sample.parquet']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in inputs}
master=pd.read_parquet(inputs[0]);master['state_fips']=master.state_fips.astype('string').str.zfill(2)
master['event_date']=pd.to_datetime(master.event_date,errors='coerce')
candidate=master[(master.state_fips=='06') & master.damage_level.astype(str).str.lower().eq('destroyed') & master.event_date.dt.year.between(2014,2025)].copy()
sample=pd.read_parquet(inputs[1]);sample=sample[sample.state_fips=='06'].copy();sample['event_date']=pd.to_datetime(sample.event_date)
assert len(candidate)==candidate.crash_id.astype(str).nunique()==47653
assert len(sample)==sample.crash_id.astype(str).nunique()==4614
assert set(sample.crash_id.astype(str)).issubset(set(candidate.crash_id.astype(str)))
sample=sample.merge(candidate[['crash_id','year_built']],on='crash_id',how='left',validate='one_to_one')
candidate['year']=candidate.event_date.dt.year;sample['year']=sample.event_date.dt.year
counts=candidate.groupby('incident').agg(candidate_n=('crash_id','size'),years=('year',lambda x:', '.join(str(v) for v in sorted(x.unique()))))
retained=sample.groupby('incident').size();counts['retained_n']=retained.reindex(counts.index,fill_value=0)
counts['coverage_percent']=100*counts.retained_n/counts.candidate_n
assert counts.candidate_n.sum()==47653 and counts.retained_n.sum()==4614
counts.reset_index().to_csv(ASSETS/'全部候选事件名称组_覆盖率.csv',index=False,encoding='utf-8-sig')
positive=counts[counts.retained_n.gt(0)].sort_values(['retained_n','candidate_n'],ascending=False)
assert len(positive)==52
rows=[]
for incident,r in positive.iterrows():rows.append([incident,r.years,f'{int(r.candidate_n):,}',f'{int(r.retained_n):,}',f'{r.coverage_percent:.2f}'])
other=counts[counts.retained_n.eq(0)]
rows.append([f'Other candidate incident-name groups (n = {len(other)})','2014–2025',f'{int(other.candidate_n.sum()):,}','0','0.00'])
rows.append(['Total California candidate inventory','2014–2025','47,653','4,614',f'{100*4614/47653:.2f}'])
(ASSETS/'覆盖表_Word行.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(rows,columns=['Incident-name group','Year(s)','Candidate buildings','Retained buildings','Coverage (%)']).to_csv(ASSETS/'补充表10_覆盖率.csv',index=False,encoding='utf-8-sig')
# Retain a date-resolved audit alongside the manuscript's existing incident-name grouping.
event=candidate.groupby(['incident','event_date']).size().rename('candidate_n').to_frame()
event['retained_n']=sample.groupby(['incident','event_date']).size().reindex(event.index,fill_value=0)
event['coverage_percent']=100*event.retained_n/event.candidate_n
event.reset_index().to_csv(ASSETS/'事件名称与日期_覆盖率核查.csv',index=False,encoding='utf-8-sig')

plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42,'ps.fonttype':42})
width,height=Image.open(WORK/'original_SupplementaryFigure3.png').size
fig,axes=plt.subplots(1,3,figsize=(11,11*height/width))
colors={'Sample':'#2D708E','Inventory':'#C8CDD2'}
stats=[]
specs=[('pre_tract_med_HH_income','Pre-fire tract household income (thousand USD)',(40000,200000)),
 ('mtbs_severity_code','MTBS burn severity (1=low … 4=high)',(.5,4.5)),
 ('year_built','Year built (pre-fire structure)',(1900,2025))]
for ax,(col,label,limits) in zip(axes,specs):
 for name,frame in [('Inventory',candidate),('Sample',sample)]:
  v=pd.to_numeric(frame[col],errors='coerce')
  if col=='pre_tract_med_HH_income':v=v[v.gt(0)]
  if col=='year_built':v=v[v.ge(1700)&v.le(frame.event_date.dt.year)]
  if col=='mtbs_severity_code':v=v[v.isin([1,2,3,4])]
  available=v.dropna();shown=available[available.between(*limits)]
  stats.append({'variable':col,'population':name,'population_n':len(frame),'available_n':len(available),'displayed_n':len(shown),'display_limits':limits})
  if col=='pre_tract_med_HH_income':
   x=np.linspace(*limits,220);y=gaussian_kde(shown.to_numpy(),bw_method=.5)(x)
   ax.fill_between(x,y,color=colors[name],alpha=.45);ax.plot(x,y,color=colors[name],lw=1.2)
  else:
   bins=np.arange(.5,5.5,1) if col=='mtbs_severity_code' else np.linspace(1900,2025,26)
   ax.hist(shown,bins=bins,density=True,color=colors[name],alpha=.55,edgecolor='white',linewidth=.35)
 ax.set_xlim(limits);ax.set_xlabel(label);ax.set_ylabel('Density')
 if col=='pre_tract_med_HH_income':ax.set_xticks([50000,100000,150000,200000],['50','100','150','200'])
 if col=='mtbs_severity_code':ax.set_xticks([1,2,3,4])
 if col=='year_built':ax.set_xticks([1900,1920,1940,1960,1980,2000,2020])
 a,b=stats[-2],stats[-1]
 ax.text(.02,.98,f'Available n: {b["available_n"]:,} / {a["available_n"]:,}\nDisplayed n: {b["displayed_n"]:,} / {a["displayed_n"]:,}',
  transform=ax.transAxes,ha='left',va='top',fontsize=6.8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1.5})
 ax.margins(y=.25)
 ax.set_ylim(bottom=0)
from matplotlib.patches import Patch
fig.legend([Patch(facecolor=colors['Sample'],edgecolor=colors['Sample'],alpha=.55),Patch(facecolor=colors['Inventory'],edgecolor=colors['Inventory'],alpha=.55)],
 ['California analytical sample (n = 4,614)','California DINS candidate inventory (n = 47,653)'],loc='upper center',ncol=2,frameon=False,fontsize=8.6,bbox_to_anchor=(.5,1.015))
fig.subplots_adjust(left=.055,right=.995,bottom=.22,top=.79,wspace=.34)
for ext in ['png','pdf','svg']:fig.savefig(ASSETS/('SupplementaryFigure3_统一口径.'+ext),dpi=350,facecolor='white')
plt.close(fig)
pd.DataFrame(stats).to_csv(ASSETS/'补充图3_有效样本数.csv',index=False,encoding='utf-8-sig')
metadata={'scope':{'state':'California','damage_category':'Destroyed','event_years':[2014,2025]},'candidate_n':47653,'retained_n':4614,'coverage_percent':100*4614/47653,
 'incident_name_grouping':'Manuscript incident-name grouping is retained; contributing years are listed. A separate date-resolved CSV is supplied for auditing.',
 'positive_incident_name_groups':52,'other_candidate_incident_name_groups':len(other),'figure_panel_counts':stats,'input_sha256':before}
(WORK/'coverage_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
for p in inputs:assert sha(p)==before[str(p)]
print(json.dumps({k:metadata[k] for k in ['candidate_n','retained_n','coverage_percent','positive_incident_name_groups','other_candidate_incident_name_groups']},ensure_ascii=False))
