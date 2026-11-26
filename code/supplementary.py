from common import *
import archived_fit as af
from scipy.stats import t as tdist

def fitted(d,name,preds=BASE,target='log_income',**kw):
    q=af.fit(d,name,preds,target=target,**kw)
    assert q is not None and q[0]['converged'],(name,af.failures)
    return q

def vegetation():
    p=directory('02b_vegetation_Figure3');v=pd.read_parquet(DATA/'all_vegetation_segformer.parquet');m=pd.read_parquet(DATA/'building_inventory.parquet')
    v['crash_id']=v.crash_id.astype(str);m['crash_id']=m.crash_id.astype(str)
    v['pano_date']=pd.to_datetime(v.pano_date);m['event_date']=pd.to_datetime(m.event_date,errors='coerce')
    m=m.dropna(subset=['pre_tract_med_HH_income','event_date']).copy();cuts=m.pre_tract_med_HH_income.quantile([1/3,2/3])
    m['tier']=np.where(m.pre_tract_med_HH_income.lt(cuts.iloc[0]),'Low',np.where(m.pre_tract_med_HH_income.lt(cuts.iloc[1]),'Middle','High'))
    d=v.merge(m[['crash_id','event_date','tier']],on='crash_id',how='inner');d['years_since']=(d.pano_date-d.event_date).dt.days/365.25;d=d[d.years_since.between(-10,6)].copy();d['yr_bin']=np.floor(d.years_since).astype(int)
    baseline=d[d.yr_bin.lt(0)].groupby('tier').veg_ratio.mean();d['deviation']=d.veg_ratio-d.tier.map(baseline)
    summary=d.groupby(['tier','yr_bin']).deviation.agg(['mean','sem','count']).reset_index();summary=summary[summary['count'].ge(20)];summary['CI_low']=summary['mean']-1.96*summary['sem'];summary['CI_high']=summary['mean']+1.96*summary['sem'];save(summary,p/'Figure3_b_vegetation_trajectory.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    for k,c in zip(GROUPS,COLORS[1:]):
        q=summary[summary.tier.eq(k)];ax.errorbar(q.yr_bin,q['mean'],yerr=1.96*q['sem'],label=k,color=c,capsize=2)
    ax.axvline(0,ls='--',color='k');ax.axhline(0,color='gray',lw=.5);ax.set(xlabel='Years since fire',ylabel='Vegetation fraction relative to pre-fire baseline');ax.legend();savefig(fig,p/'Figure3_b_numerical_review')
    high4=float(summary.loc[summary.tier.eq('High')&summary.yr_bin.eq(4),'mean'].iloc[0]);assert round(high4,2)==-.12,high4
    write_json({'income_tertile_cutpoints_usd':cuts.tolist(),'baseline_vegetation_fraction':baseline.to_dict(),'high_income_year4_deviation':high4,'scope':'Recomputed from per-image segmentation measurements; image segmentation inference and photographic panel 3a are upstream steps. Original source uses inventory income tertiles for this figure.'},p/'key_results.json')
    seg=pd.read_parquet(DATA/'all_vegetation_segformer_per_building.parquet');hsv=pd.read_parquet(DATA/'all_vegetation_hsv_per_building.parquet')
    pair=seg[['crash_id','veg_ratio_pre']].merge(hsv[['crash_id','green_ratio_pre']],on='crash_id').dropna();assert len(pair)==6595
    pair=pair.sample(5000,random_state=0);r=pair.veg_ratio_pre.corr(pair.green_ratio_pre);assert round(r,2)==.81,r
    sp=directory('supplementary');save(pair,sp/'Figure_S6_matched_sample.csv')
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(pair.veg_ratio_pre,pair.green_ratio_pre,s=3,alpha=.24,color=COLORS[-1]);b1,b0=np.polyfit(pair.veg_ratio_pre,pair.green_ratio_pre,1);xx=np.linspace(pair.veg_ratio_pre.min(),pair.veg_ratio_pre.max(),50);ax.plot(xx,b0+b1*xx);ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Pre-fire SegFormer vegetation fraction',ylabel='Pre-fire HSV green fraction',title=f'Pearson r = {r:.2f}; n = 5,000');savefig(fig,sp/'Figure_S6_numerical_review');write_json({'n_matched':6595,'n_sampled':5000,'random_state':0,'pearson_r':r},sp/'Figure_S6_statistics.json')

def structure_and_time():
    p=directory('supplementary');f=load();d=f[f.trajectory.ne('obscured')];rows=[]
    for label,u in [('All classifiable buildings',d),('Single-family only',d[d.single_family]),('Mobile and motor homes excluded',d[~d.mobile])]:
        for time in [None,'spline']:
            row,used,model=fitted(u,label+str(time),time=time);rows.append({**row,'Sample':label,'Actual-time term':'Cubic spline' if time else 'Not added'})
    modelrows=pd.DataFrame(rows);save(modelrows,p/'Table_S11_structure_models_full.csv')
    ref=pd.read_csv(ROOT/'reference/previous_runs/structure_type_model_results.csv')
    for a,b in zip(rows,ref.to_dict('records')):
        for k in ['N','G','beta_per_sd','se','OR','CI_low','CI_high','p']:require_close(a[k],b[k],f'S11 {a["name"]} {k}')
    formatted=[reference_table(11)[0]]+[[r['Sample'],r['Actual-time term'],f'{r["N"]:,}',str(r['G']),f'{r["OR"]:.2f}',f'{r["CI_low"]:.2f}–{r["CI_high"]:.2f}','<0.001' if r['p']<.001 else f'{r["p"]:.3f}'] for r in rows];verify_cells(11,formatted)
    fig,ax=plt.subplots(figsize=(8,4));ys=np.arange(6);ax.errorbar(modelrows.OR,ys,xerr=[modelrows.OR-modelrows.CI_low,modelrows.CI_high-modelrows.OR],fmt='o',capsize=3);ax.set_yticks(ys,[r['Sample']+'; '+r['Actual-time term'] for r in rows],fontsize=7);ax.axvline(1,ls='--',color='k');ax.set(xscale='log',xlabel='Income odds ratio (95% CI)');savefig(fig,p/'Figure_S5_numerical_review')
    elapsed=(f.latest_post_date-f.event_date).dt.days;u=f[f.trajectory.ne('obscured')&elapsed.ge(90)];rows=[]
    for time in [None,'linear','spline']:
        row,used,model=fitted(u,'Table1 time '+str(time),time=time);rows.append(row)
    save(pd.DataFrame(rows),p/'Table_S14_time_models_full.csv')
    ref=pd.read_csv(ROOT/'reference/previous_runs/time_models.csv');ref=ref[ref.framework.eq('Table 1 covariates')]
    for a,b in zip(rows,ref.to_dict('records')):
        for k in ['N','G','OR','CI_low','CI_high','p']:require_close(a[k],b[k],f'S14 {a["name"]} {k}')
    verify_cells(14,[reference_table(14)[0]]+[[label,f'{r["N"]:,}',str(r['G']),f'{r["OR"]:.2f} ({r["CI_low"]:.2f}–{r["CI_high"]:.2f})','<0.001' if r['p']<.001 else f'{r["p"]:.3f}'] for label,r in zip(['None','Linear','Cubic spline'],rows)])

def composition_windows_coverage():
    p=directory('supplementary');d=fig2_sample();grouped=[]
    for (inc,year),q in d.groupby(['incident','fire_year'],sort=False):
        row={'incident':inc,'year':year,'n':len(q),'median_income':q.pre_tract_med_HH_income.median()}
        for k in GROUPS:row[k]=int(q.income_group.eq(k).sum())
        grouped.append(row)
    g=pd.DataFrame(grouped).sort_values(['n','incident','year'],ascending=[False,True,True]);top=g.head(20);remaining=g.iloc[20:];q=d.merge(remaining[['incident','year']].rename(columns={'year':'fire_year'}),on=['incident','fire_year'],how='inner')
    other={'incident':'Other','year':'','n':len(q),'median_income':q.pre_tract_med_HH_income.median(),**{k:int(q.income_group.eq(k).sum()) for k in GROUPS}}
    total={'incident':'Total','year':'','n':len(d),'median_income':d.pre_tract_med_HH_income.median(),**{k:int(d.income_group.eq(k).sum()) for k in GROUPS}}
    shown=pd.concat([top,pd.DataFrame([other,total])],ignore_index=True);save(shown,p/'Table_S12_income_composition.csv')
    rows=[reference_table(12)[0]]
    for r in shown.to_dict('records'):
        name=r['incident'] if not r['year'] else f'{r["incident"]} ({int(r["year"])})'
        median=f'{r["median_income"]:,.1f}' if r['incident']=='Other' else f'{r["median_income"]:,.0f}'
        rows.append([name,f'{r["n"]:,}']+[f'{r[k]:,} ({100*r[k]/r["n"]:.1f})' for k in GROUPS]+[median])
    verify_cells(12,rows)
    windows=[];rows=[reference_table(13)[0]]
    for year in range(1,6):
        q=d[d.followup_years.ge(year-.5)&d.followup_years.lt(year+.5)];row=[f'{year-.5:.1f}–<{year+.5:.1f}']
        for k in ['All']+GROUPS:
            v=q if k=='All' else q[q.income_group.eq(k)];n=len(v);a=int(v.rebuilt.sum());windows.append({'target_year':year,'income_group':k,'n':n,'rebuilt_n':a,'rebuilt_percent':100*a/n});row.append(f'{a:,}/{n:,} ({100*a/n:.1f}%)')
        rows.append(row)
    verify_cells(13,rows);save(pd.DataFrame(windows),p/'Table_S13_window_counts.csv')
    m=pd.read_parquet(DATA/'building_inventory.parquet');m['year']=pd.to_datetime(m.event_date).dt.year
    ca=m[m.state_fips.eq('06')&m.damage_level.str.lower().eq('destroyed')&m.year.between(2014,2025)].copy();assert len(ca)==47653
    f=load();ret=f[f.state_fips.eq('06')];cand=ca.groupby('incident').agg(candidate_n=('crash_id','size'),years=('year',lambda x:', '.join(map(str,sorted(x.unique()))) if x.nunique()>1 else str(x.iloc[0])));cand['retained_n']=ret.groupby('incident').size().reindex(cand.index,fill_value=0)
    named=cand[cand.retained_n.ge(10)].sort_values('retained_n',ascending=False);other=cand[cand.retained_n.lt(10)];display=named.copy();display.loc['Other']=[other.candidate_n.sum(),'2014–2025',other.retained_n.sum()];display.loc['Total']=[cand.candidate_n.sum(),'2014–2025',cand.retained_n.sum()];display['coverage_pct']=100*display.retained_n/display.candidate_n;save(display.reset_index(),p/'Table_S10_coverage.csv')
    rows=[reference_table(10)[0]]+[[f'Other incident-name groups (n = {len(other)})' if i=='Other' else 'Total California candidate inventory' if i=='Total' else str(i),str(r.years),f'{r.candidate_n:,}',f'{r.retained_n:,}',f'{r.coverage_pct:.2f}'] for i,r in display.iterrows()];verify_cells(10,rows)
    # S1 per-fire outcome composition, including the separately displayed obscured label.
    counts=recovery_event_counts(f);save(counts,p/'Figure_S1_per_fire_counts.csv')
    # S3 descriptive distributions use matched-scope inventory, not an expanded geographical sample.
    sample=ca[ca.crash_id.astype(str).isin(ret.crash_id.astype(str))];assert len(sample)==4614
    fig,axs=plt.subplots(1,3,figsize=(12,4));summary=[]
    for ax,col,label,limits in zip(axs,['pre_tract_med_HH_income','mtbs_severity_code','year_built'],['Pre-fire tract income (USD)','MTBS burn severity','Construction year'],[(40000,200000),(1,4),(1900,2025)]):
        for frame,color,name in [(ca,'gray','Candidate'),(sample,COLORS[-1],'Analytical')]:
            val=pd.to_numeric(frame[col],errors='coerce');valid=val.gt(0) if col.startswith('pre_') else val.between(1,4) if col=='mtbs_severity_code' else val.between(1700,frame.year)
            vals=val[valid];shown=vals[vals.between(*limits)];summary.append({'variable':col,'group':name,'available_n':len(vals),'displayed_n':len(shown)});ax.hist(shown,bins=25 if col!='mtbs_severity_code' else np.arange(.5,5.5),density=True,histtype='step',color=color,label=name)
        ax.set(xlabel=label,ylabel='Density');ax.legend(fontsize=7)
    fig.tight_layout();savefig(fig,p/'Figure_S3_numerical_review');save(pd.DataFrame(summary),p/'Figure_S3_distribution_counts.csv')

def insurer_models():
    p=directory('supplementary');f=load().reset_index(drop=True)
    old=pd.read_excel(ROOT/'data/raw/cdi_2023/nonrenew_zip_2015_2021.xlsx');old.columns=old.columns.str.strip();old=old.rename(columns={'ZIP Code':'zip5','Year':'year','New':'new','Renewed':'renewed','Insured-Initiated Nonrenewed':'consumer','Insurer-Initiated Nonrenewed':'insurer'})
    old['zip5']=old.zip5.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    for c in ['new','renewed','consumer','insurer']:old[c]=pd.to_numeric(old[c],errors='coerce')
    old=old.groupby(['zip5','year'],as_index=False)[['new','renewed','consumer','insurer']].sum();old['total']=old.insurer+old.consumer;denom=old.renewed+old.total
    for c in ['insurer','consumer','total']:old['prefire_'+c]=old[c]/denom.where(denom.gt(0))
    lag=old[['zip5','year','prefire_insurer','prefire_consumer','prefire_total']].copy();lag['fire_year']=lag.year+1
    f=f.merge(lag.drop(columns='year'),on=['zip5','fire_year'],how='left',validate='many_to_one');f.index=f.crash_id.astype(str)
    days=(f.latest_post_date-f.event_date).dt.days;exp=['prefire_total','prefire_insurer','prefire_consumer'];ca=f[f.state_fips.eq('06')&f.trajectory.ne('obscured')&days.ge(90)].dropna(subset=BASE+exp+['followup_years','rebuilt','event_id'])
    varying=ca.groupby('event_id').rebuilt.agg(['min','max']);d=ca[ca.event_id.isin(varying.index[varying['min'].ne(varying['max'])])];assert len(d)==3820 and d.event_id.nunique()==15
    rows=[]
    for name,pr in [('Insurer-initiated',['prefire_insurer']+BASE),('Both initiators',['prefire_insurer','prefire_consumer']+BASE)]:
        row,u,model=fitted(d,name,pr,target='prefire_insurer',time='spline',fe=True);rows.append(row)
        if name=='Both initiators':
            crit=tdist.ppf(.975,14);b=model.params.prefire_consumer;se=model.bse.prefire_consumer
            write_json({'OR':np.exp(b),'CI_low':np.exp(b-crit*se),'CI_high':np.exp(b+crit*se),'p':2*tdist.sf(abs(b/se),14)},p/'joint_policyholder_estimate.json')
    save(pd.DataFrame(rows),p/'Table_S16_insurer_models_full.csv')
    verify_cells(16,[reference_table(16)[0]]+[[name,'Insurer-initiated',f'{r["N"]:,}',str(r['G']),f'{r["OR"]:.3f} ({r["CI_low"]:.3f}–{r["CI_high"]:.3f})',f'{r["p"]:.3f}'] for name,r in zip(['Table 1 covariates + fire-event fixed effects','Table 1 covariates + fire-event fixed effects, both initiators entered jointly'],rows)])
    lo=[]
    for event in sorted(d.event_id.unique()):lo.append(fitted(d[d.event_id.ne(event)],'leave_out_'+event,['prefire_insurer']+BASE,target='prefire_insurer',time='spline',fe=True,quiet=True)[0])
    assert len(lo)==15 and all(x['OR']<1 for x in lo) and sum(x['CI_high']<1 for x in lo)==13
    ref=pd.read_csv(ROOT/'reference/previous_runs/all_leave_one_fire_out.csv').set_index('name')
    for row in lo:
        for k in ['N','G','OR','CI_low','CI_high','p']:require_close(row[k],ref.loc[row['name'],k],f'LOFO {row["name"]} {k}')
    save(pd.DataFrame(lo),p/'Note9_all_leave_one_fire_out.csv')
    save(old,p/'Table_S15_reconstructed_zip_year_rates.csv')
    definitions=reference_table(15)
    save(pd.DataFrame(definitions[1:],columns=definitions[0]),p/'Table_S15_indicator_definitions.csv')
    save(pd.DataFrame(definitions[1:],columns=definitions[0]),directory('tables')/'SupplementaryTable15.csv')
    # Note 9's descriptive correlation counts each ZIP once in the original
    # common-period comparison cohort; it is separate from the fixed-effect cohort.
    overlap=old[old.year.isin([2020,2021])].groupby('zip5',as_index=False)[['new','renewed','consumer','insurer']].sum();overlap['total']=overlap.consumer+overlap.insurer
    for measure in ['total','insurer','consumer']:
        overlap['overlap_'+measure]=overlap[measure]/(overlap.renewed+overlap.total).where((overlap.renewed+overlap.total).gt(0))
        overlap['overlap_'+measure+'_including_new']=overlap[measure]/(overlap.new+overlap.renewed+overlap.total).where((overlap.new+overlap.renewed+overlap.total).gt(0))
    new=pd.read_excel(ROOT/'data/raw/cdi_2023/nonrenew_zip_2020_2023.xlsx');new.columns=new.columns.str.strip();new=new.rename(columns={'ZIP Code':'zip5','Year':'year','Non-Renewed':'total_new','New':'new_count','Renewed':'renewed_count'});new['zip5']=new.zip5.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    new=new[new.year.isin([2020,2021])].groupby('zip5',as_index=False)[['total_new','new_count','renewed_count']].sum();denom=new.total_new+new.new_count+new.renewed_count;new['new_file_total_2020_2021']=new.total_new/denom.where(denom.gt(0))
    c=load().reset_index(drop=True);c=c[c.state_fips.eq('06')&c.trajectory.ne('obscured')].merge(overlap[['zip5']+[k for k in overlap if k.startswith('overlap_')]],on='zip5',how='left',validate='many_to_one').merge(new[['zip5','new_file_total_2020_2021']],on='zip5',how='left',validate='many_to_one')
    exposures=['nonrenew_rate_2020_2023','new_file_total_2020_2021','overlap_total_including_new','overlap_insurer_including_new','overlap_consumer_including_new','overlap_total','overlap_insurer','overlap_consumer'];rest=[k for k in POLICY if k!='nonrenew_rate_2020_2023'];common=c.dropna(subset=exposures+rest+['followup_years']);zips=common[['zip5','overlap_total','overlap_insurer','overlap_consumer']].drop_duplicates('zip5');cor=zips.overlap_insurer.corr(zips.overlap_consumer)
    assert len(zips)==37 and round(cor,3)==.208 and round(cor**2,3)==.043,(len(zips),cor)
    save(zips,p/'Note9_common_period_unique_ZIP_components.csv');write_json({'common_period_buildings':len(common),'unique_ZIPs':len(zips),'insurer_policyholder_Pearson_r':cor,'R_squared':cor**2},p/'Note9_component_correlation.json')

def old_sensitivity_models():
    p=directory('supplementary');f=load();d=f[(2026-f.fire_year).ge(4)&f.assessed_value_usd.gt(0)].dropna(subset=BASE);assert len(d)==4161
    rows=[]
    for name,preds in [('Full',BASE),('Parsimonious',['log_income','log_value','mtbs_severity_code'])]:
        tab,model,u,mu,sd=logit(d,preds,cluster=True);row=tab.set_index('variable').loc['log_income'];rows.append([reference_table(3)[len(rows)+1][0],f'{row.beta:+.2f}',f'({row.beta_low:+.2f}, {row.beta_high:+.2f})','< 0.001' if row.p<.001 else f'{row.p:.3f}',f'{len(u):,}']);save(tab,p/f'Table_S03_{name.lower()}_coefficients.csv')
    verify_cells(3,[reference_table(3)[0]]+rows)
    risk=pd.read_parquet(DATA/'forward_fire_risk_per_building.parquet');risk['crash_id']=risk.crash_id.astype(str)
    ca=f[f.state_fips.eq('06')].reset_index(drop=True).merge(risk,on='crash_id',how='left',validate='one_to_one');ca.index=ca.crash_id.astype(str)
    rows=[]
    for name,extra in [('A',[]),('B',['prior_fires']),('C',['hi_fire_history'])]:
        tab,r,u,mu,sd=logit(ca,POLICY+extra,cluster=True,scale_all=True);save(tab,p/f'Table_S06_model_{name}.csv');a=tab.set_index('variable').loc['nonrenew_rate_2020_2023'];b=tab.set_index('variable').loc[extra[0]] if extra else None
        rows.append([reference_table(6)[len(rows)+1][0],f'{a.OR:.3f}',f'({a.CI_low:.3f}, {a.CI_high:.3f})','< 0.001' if a.p<.001 else f'{a.p:.3f}','—' if b is None else f'{b.OR:.3f}','—' if b is None else f'{b.p:.3f}'])
    verify_cells(6,[reference_table(6)[0]]+rows)

def enriched_sample():
    f=load().reset_index(drop=True);m=pd.read_parquet(DATA/'legacy_analysis_frame.parquet');m['crash_id']=m.crash_id.astype(str)
    additional=[c for c in m if c not in f or c=='crash_id'];f=f.merge(m[additional],on='crash_id',how='left',validate='one_to_one')
    h=pd.read_parquet(DATA/'hmda_per_tract.parquet');h['geoid']=h.orig_tract.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(11);h=h.groupby('geoid',as_index=False).pct_conventional.mean()
    f=f.merge(h,left_on='pre_tract_geoid',right_on='geoid',how='left',validate='many_to_one');f.index=f.crash_id.astype(str)
    return f

def descriptive_tables():
    p=directory('supplementary');f=enriched_sample();ref=reference_table(2);rows=[ref[0],ref[1]]
    historical=legacy_frame();historical['fire_year']=pd.to_datetime(historical.event_date).dt.year
    pol=pd.read_parquet(DATA/'policy_per_building.parquet');pol['crash_id']=pol.crash_id.astype(str);historical=historical.merge(pol,on='crash_id',how='left',validate='one_to_one')
    h=pd.read_parquet(DATA/'hmda_per_tract.parquet');h['geoid']=h.orig_tract.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(11);h=h.groupby('geoid',as_index=False)[['pct_conventional','denial_rate']].mean()
    historical['pre_tract_geoid']=historical.pre_tract_geoid.astype(str).str.zfill(11);historical=historical.merge(h,left_on='pre_tract_geoid',right_on='geoid',how='left',validate='many_to_one')
    fields=['pre_tract_med_HH_income','pre_pct_white_NH','pre_pct_Hispanic','pre_pct_below_poverty','pre_pct_BA_or_higher','pre_pct_owner_occupied','pre_pct_mobile_home','assessed_value_usd','year_built','building_age_at_fire','mtbs_severity_code','fema_ihp_total_usd','fair_pct_2022','nonrenew_rate_2020_2023','pct_conventional','denial_rate','j40_disadvantaged']
    raw=[]
    for field,expected in zip(fields,ref[2:19]):
        val=pd.to_numeric(historical[field],errors='coerce')
        if field in ['year_built','building_age_at_fire']:
            good=pd.to_numeric(historical.year_built,errors='coerce').between(1700,historical.fire_year)
            val=val.where(good)
            if field=='building_age_at_fire':val=historical.fire_year-pd.to_numeric(historical.year_built,errors='coerce').where(good)
        if field=='nonrenew_rate_2020_2023':val=val*100
        val=val.dropna();stats=[len(val),val.mean(),val.std(ddof=1),val.median(),val.quantile(.1),val.quantile(.9)]
        raw.append({'variable':field,**dict(zip(['N','Mean','SD','Median','P10','P90'],stats))})
        # Reproduce the precision chosen for each published descriptive statistic.
        row=[expected[0]]
        for v,t in zip(stats,expected[1:]):
            precision=len(t.split('.')[1]) if '.' in t else 0
            row.append(f'{v:,.{precision}f}')
        rows.append(row)
    rows.append(ref[19])
    labels={'empty_lot':'Empty lot','rebuilt_equal':'Rebuilt equal','rebuilt_improved':'Rebuilt improved','obscured':'Obscured','rebuilt_degraded':'Rebuilt degraded'}
    for k in ['empty_lot','rebuilt_equal','rebuilt_improved','obscured','rebuilt_degraded']:
        n=int(f.trajectory.eq(k).sum());rows.append([labels[k],f'{n:,}',f'{100*n/len(f):.1f}%','','','',''])
    rows.append(ref[25])
    for k,label in [('Single family','Single-family residence'),('Mobile or motor home','Mobile or motor home'),('Multi family','Multifamily residence'),('Unknown or other','Not recorded')]:
        n=int(f.type_group.eq(k).sum());rows.append([label,f'{n:,}',f'{100*n/len(f):.1f}%','','','',''])
    rows.append(ref[30])
    for k,label in [(None,'Not recorded'),('same_owner_pre_fire','Same owner pre-fire'),('sold_post_fire','Sold post-fire'),('no_sale_recorded','No sale recorded'),('corporate_post_fire','Corporate post-fire')]:
        n=int(f.ownership_class.isna().sum() if k is None else f.ownership_class.eq(k).sum());rows.append([label,f'{n:,}',f'{100*n/len(f):.1f}%','','','',''])
    save(pd.DataFrame(raw),p/'Table_S02_descriptive_statistics_full.csv');verify_cells(2,rows)
    rows=[reference_table(4)[0]];out=[]
    historical['income_group']=pd.cut(historical.pre_tract_med_HH_income,[-np.inf,60000,100000,np.inf],labels=GROUPS,right=False)
    historical['rebuilt']=historical.trajectory.isin(TRAJ[1:]).astype(int);historical['upgraded']=historical.trajectory.eq('rebuilt_improved').astype(int)
    for k in GROUPS:
        u=historical[historical.income_group.eq(k)];r=[reference_table(4)[GROUPS.index(k)+1][0],f'{len(u):,}',f'{100*u.rebuilt.mean():.1f}',f'{100*u.upgraded.mean():.1f}',f'{100*u.trajectory.eq("empty_lot").mean():.1f}',f'{u.fair_pct_2022.median():.1f}%',f'{100*u.nonrenew_rate_2020_2023.mean():.1f}%',f'{100*u.pct_conventional.mean():.1f}%',f'{100*u.denial_rate.median():.1f}%',f'{u.fema_ihp_total_usd.median():,.0f}'];rows.append(r)
    verify_cells(4,rows)
    ca=f[f.state_fips.eq('06')].copy();rows=[reference_table(9)[0]]
    for col in ['nonrenew_rate_2020_2023','denial_rate','fair_pct_2022','pct_conventional','log_fema']:
        for source,colz in [(col,'exposure'),('log_income','income'),('mtbs_severity_code','severity')]:ca[colz]=(ca[source]-f[source].mean())/f[source].std(ddof=1)
        u=ca[['exposure','income','severity','rebuilt','incident']].dropna();xx=add_constant(u[['exposure','income','severity']]);r=Logit(u.rebuilt.astype(int),xx).fit(disp=False,method='bfgs',maxiter=200,cov_type='cluster',cov_kwds={'groups':u.incident});b=r.params.exposure;ci=r.conf_int().loc['exposure'];pv=r.pvalues.exposure
        rows.append([reference_table(9)[len(rows)][0],f'{np.exp(b):.2f}',f'({np.exp(ci.iloc[0]):.2f}, {np.exp(ci.iloc[1]):.2f})','< 0.001' if pv<.001 else f'{pv:.3f}',f'{len(u):,}'])
        save(pd.DataFrame({'term':r.params.index,'beta':r.params.values,'SE':r.bse.values,'p':r.pvalues.values}),p/f'Table_S09_{col}_full.csv')
    verify_cells(9,rows)
    write_json({'descriptive_continuous_source_rows':len(historical),'categorical_label_source_rows':len(f),'TableS4_source_rows':len(historical),'two_historical_non_manuscript_records':historical.loc[~historical.state_fips.isin(['06','08']),['crash_id','state_fips','incident']].to_dict('records'),'note':'Historical continuous descriptive statistics and Table S4 use the original 4,639-row analysis frame; categorical outcomes use the frozen 4,637-row manuscript sample. Both sources are retained without altering either dataset.'},p/'descriptive_source_cohorts.json')
    tab,r,u,mu,sd=logit(load(),BASE);vals=tab.set_index('variable');rows=[reference_table(8)[0]]
    for pred,expected in zip(BASE,reference_table(8)[1:8]):
        a=vals.loc[pred];rows.append([expected[0],f'{a.beta:.4f}',f'{a.SE:.4f}',f'{a.OR:.3f}','< 0.001' if a.p<.001 else f'{a.p:.4f}'])
    rows += [['--- Diagnostics ---','','','',''],['N',f'{len(u):,}','','',''],['Pseudo R²',f'{r.prsquared:.4f}','','',''],['Log-likelihood',f'{r.llf:.2f}','','','']];verify_cells(8,rows)
    tab,r,u,mu,sd=logit(load()[load().state_fips.eq('06')],POLICY,cluster=True,scale_all=True);rows=[reference_table(5)[0]]
    for pred,expected in zip(POLICY,reference_table(5)[1:]):
        a=tab.set_index('variable').loc[pred];rows.append([expected[0],f'{a.OR:.3f}',f'({a.CI_low:.3f}, {a.CI_high:.3f})','< 0.001' if a.p<.001 else f'{a.p:.3f}'])
    rows += [['Diagnostics','','',''],['Buildings',f'{len(u):,}','',''],['Fire incidents',str(u.incident.nunique()),'',''],['Pseudo R²',f'{r.prsquared:.3f}','','']]
    verify_cells(5,rows)
    # S2 ZIP geographical source values are recomputed from the same frozen cohort.
    coords=pd.read_parquet(DATA/'building_inventory.parquet');coords['crash_id']=coords.crash_id.astype(str)
    z=load().reset_index(drop=True).merge(coords[['crash_id','lat','lon']],on='crash_id',how='left');z=z[z.state_fips.eq('06')].groupby('zip5').agg(n=('crash_id','size'),lat=('lat','median'),lon=('lon','median'),fair_pct_2022=('fair_pct_2022','first'),nonrenew_rate=('nonrenew_rate_2020_2023','first')).reset_index();save(z,p/'Figure_S2_ZIP_geography_source.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for ax,col,title in zip(axs,['fair_pct_2022','nonrenew_rate'],['FAIR Plan share (%)','Aggregate non-renewal rate']):
        pts=ax.scatter(z.lon,z.lat,c=z[col],s=5*np.sqrt(z.n),cmap='YlOrRd');ax.set(xlabel='Longitude',ylabel='Latitude',title=title);fig.colorbar(pts,ax=ax)
    savefig(fig,p/'Figure_S2_numerical_review')

def supplementary():
    # Each step recomputes from numerical inputs. Reference tables are only assertion targets.
    structure_and_time()
    composition_windows_coverage()
    insurer_models()
    old_sensitivity_models()
    descriptive_tables()
    write_json(af.fitdetails,directory('supplementary')/'model_design_and_scaling.json')
    assert not af.failures,af.failures
