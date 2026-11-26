import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from common import *

def result1():
    p=directory('01_recovery_Figure1');f=load()
    counts=f.trajectory.value_counts().rename_axis('trajectory').reset_index(name='n');counts['percent']=counts.n/len(f)*100
    save(counts,p/'recovery_counts.csv')
    assert f.state_fips.eq('06').sum()==4614
    # The original descriptive inventory counts incident names within each state.
    paper_fire_n=len(f[['state_fips','incident']].drop_duplicates())
    assert paper_fire_n==53 and f.trajectory.eq('empty_lot').sum()==2898
    assert f[f.incident.eq('Tubbs')].shape[0]==2055
    subset=f[f.incident.isin(['Woolsey','Palisades'])]
    assert len(subset)==416 and subset.trajectory.eq('empty_lot').sum()==333
    g=f.groupby(['incident','fire_year'],observed=True).agg(n=('crash_id','size'),rebuilt=('rebuilt','sum'),improved=('upgraded','sum'),income_median=('pre_tract_med_HH_income','median')).reset_index()
    save(g,p/'per_fire_recovery.csv')
    date_rows=[]
    for incident,u in [('All',f)]+[(k,f[f.incident.eq(k)]) for k in ['Tubbs','Thomas','Woolsey','Palisades']]:
        date_rows.append({'incident':incident,'N':len(u),'image_date_min':str(u.latest_post_date.min().date()),'image_date_max':str(u.latest_post_date.max().date()),'median_followup_years':u.followup_years.median(),'empty_lot_n':int(u.trajectory.eq('empty_lot').sum()),'rebuilt_n':int(u.rebuilt.sum())})
    save(pd.DataFrame(date_rows),p/'observation_dates_and_elapsed_time.csv')
    meta=pd.read_parquet(DATA/'building_inventory.parquet').set_index('crash_id')
    geo=f.reset_index(drop=True).merge(meta[['lat','lon']],left_on='crash_id',right_index=True,how='left',validate='one_to_one')
    save(geo[['crash_id','incident','fire_year','lat','lon','trajectory']],p/'Figure1_abc_source.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,5))
    for ax,title,q in zip(axs,['California','Tubbs (2017)','Woolsey and Palisades'],[geo[geo.state_fips.eq('06')],geo[geo.incident.eq('Tubbs')],geo[geo.incident.isin(['Woolsey','Palisades'])]]):
        for k,c in zip(TRAJ,COLORS):
            u=q[q.trajectory.eq(k)];ax.scatter(u.lon,u.lat,s=3,c=c,label=k.replace('_',' '))
        ax.set(xlabel='Longitude',ylabel='Latitude',title=title)
    axs[-1].legend(fontsize=7);savefig(fig,p/'Figure1_abc_numerical_review')
    write_json({'N':len(f),'California_n':4614,'paper_fire_count_state_incident_groups':int(paper_fire_n),'California_incident_name_groups':int(f[f.state_fips.eq('06')].incident.nunique()),'raw_name_year_groups':len(f[['incident','fire_year']].drop_duplicates()),'incident_names_across_states':int(f.incident.nunique()),'name_date_clusters':int(f.event_id.nunique()),'empty_lot_n':2898,'empty_lot_excluding_Palisades_n':2680,'non_Palisades_n':4416,'scope':'Geographical source data and numerical review plot; photographic timeline panels require original Street View images. Descriptive incident-name groups and supplementary name/date inference clusters are distinct, retained definitions.'},p/'key_results.json')
    top=g.nlargest(10,'n').sort_values(['fire_year','incident'])
    rows=[reference_table(1)[0]]
    for r in top.itertuples():
        q=f[f.incident.eq(r.incident)];empty=q.trajectory.eq('empty_lot').mean()*100
        modes=q.mtbs_severity_code.mode();severity='' if modes.empty else f'{modes.iloc[0]:.1f}'
        rows.append([r.incident,str(r.fire_year),str(r.n),f'{100*r.rebuilt/r.n:.1f}',f'{100*r.improved/r.n:.1f}',f'{empty:.1f}',f'{r.income_median:.0f}',severity])
    verify_cells(1,rows)

def result2():
    p=directory('02_income_Figure2');d=fig2_sample()
    g=d.groupby('income_group',observed=False).agg(n=('crash_id','size'),rebuilt_n=('rebuilt','sum'),upgraded_n=('upgraded','sum')).reset_index()
    g['rebuilt_pct']=100*g.rebuilt_n/g.n;g['upgraded_among_rebuilt_pct']=100*g.upgraded_n/g.rebuilt_n
    save(g,p/'Figure2_ac_income_groups.csv')
    counts=pd.crosstab(d.income_group,d.trajectory).reindex(index=GROUPS,columns=TRAJ,fill_value=0)
    save(counts.reset_index(),p/'Figure2_a_outcomes.csv')
    preds=['pre_pct_white_NH','pre_pct_Hispanic','pre_pct_below_poverty','pre_pct_owner_occupied','assessed_value_usd','mtbs_severity_code']
    u=d.dropna(subset=preds).copy();u=u[u.assessed_value_usd.gt(0)]
    x=pd.DataFrame({'log_inc':np.log10(u.pre_tract_med_HH_income),'pre_pct_white_NH':u.pre_pct_white_NH,'pre_pct_Hispanic':u.pre_pct_Hispanic,'pre_pct_below_poverty':u.pre_pct_below_poverty,'pre_pct_owner_occupied':u.pre_pct_owner_occupied,'log_value':np.log10(u.assessed_value_usd),'mtbs_sev':u.mtbs_severity_code})
    r=Logit(u.rebuilt,add_constant(x).astype(float)).fit(disp=False)
    assert len(u)==4121 and u.event_id.nunique()==32
    grid=np.linspace(np.log10(max(30000,u.pre_tract_med_HH_income.min())),np.log10(u.pre_tract_med_HH_income.max()),180)
    xx=pd.DataFrame({c:np.repeat(x[c].median(),len(grid)) for c in x});xx['log_inc']=grid
    pr,lo,hi=prediction(r,add_constant(xx,has_constant='add'))
    curve=pd.DataFrame({'income_usd':10**grid,'probability':pr,'CI_low':lo,'CI_high':hi});save(curve,p/'Figure2_b_adjusted_curve.csv')
    dots=u.groupby('event_id').agg(incident=('incident','first'),n=('crash_id','size'),income_usd=('pre_tract_med_HH_income','median'),rebuilding_proportion=('rebuilt','mean')).reset_index();save(dots,p/'Figure2_b_fire_points.csv')
    fire=d[d.rebuilt.eq(1)].groupby('incident').agg(n_rebuilt=('rebuilt','sum'),n_upgraded=('upgraded','sum'),income_usd=('pre_tract_med_HH_income','median'));fire=fire[fire.n_rebuilt.ge(20)].reset_index();fire['upgrade_pct']=100*fire.n_upgraded/fire.n_rebuilt;assert len(fire)==7
    save(fire,p/'Figure2_d_fire_upgrades.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    (counts.div(counts.sum(axis=1),axis=0)*100).plot.bar(stacked=True,ax=axs[0,0],color=COLORS,rot=0);axs[0,0].set(ylabel='Recovery outcome (%)',title='a')
    ax=axs[0,1];ax.plot(curve.income_usd/1000,pr,color=COLORS[-1]);ax.fill_between(curve.income_usd/1000,lo,hi,alpha=.2);ax.scatter(dots.income_usd/1000,dots.rebuilding_proportion,s=4*np.sqrt(dots.n),alpha=.65);ax.set(xlabel='Pre-fire tract income (USD thousands)',ylabel='Rebuilding probability',title='b')
    vals=g.upgraded_among_rebuilt_pct;se=100*np.sqrt((vals/100)*(1-vals/100)/g.rebuilt_n);axs[1,0].bar(GROUPS,vals,color=COLORS[1:]);axs[1,0].errorbar(GROUPS,vals,yerr=1.96*se,fmt='none',color='k',capsize=3);axs[1,0].set(ylabel='Upgraded among rebuilt (%)',title='c')
    ax=axs[1,1];ax.scatter(fire.income_usd/1000,fire.upgrade_pct,s=4*np.sqrt(fire.n_rebuilt));
    for a in fire.itertuples():ax.annotate(a.incident,(a.income_usd/1000,a.upgrade_pct),fontsize=7)
    ax.set(xlabel='Median pre-fire tract income (USD thousands)',ylabel='Upgraded among rebuilt (%)',title='d')
    fig.tight_layout();savefig(fig,p/'Figure2_numerical_review')
    write_json({'N':4310,'group_n':g.n.tolist(),'rebuilt_percent':g.rebuilt_pct.tolist(),'upgrade_percent_among_rebuilt':g.upgraded_among_rebuilt_pct.tolist(),'curve_n':len(u),'curve_events':32,'curve_min_probability':float(pr[0]),'curve_max_probability':float(pr[-1]),'income_max':float(curve.income_usd.max())},p/'key_results.json')

def result3():
    p=directory('03_models_Table1');f=load();summary=[];fitted_tables={}
    for name,d,y in [('baseline',f,'rebuilt'),('upgrading',f[f.rebuilt.eq(1)],'upgraded')]:
        tab,r,u,mu,sd=logit(d,BASE,y=y)
        save(tab,p/f'Table1_{name}_coefficients.csv')
        fitted_tables[name]=tab.set_index('variable')
        ref=pd.read_csv(ROOT/f'reference/previous_runs/published_{name}_reproduction_full.csv')
        for rr in ref.itertuples():
            v=tab.set_index('variable').loc[rr.variable]
            for field,prev in [('beta',rr.beta),('SE',rr.SE),('p',rr.p),('OR',rr.OR)]:require_close(v[field],prev,f'Table1 {name} {rr.variable} {field}')
        row=tab.set_index('variable').loc['log_income'];summary.append({'model':name,**row.to_dict()})
        save(pd.DataFrame({'crash_id':u.index}),p/f'Table1_{name}_sample_ids.csv')
        fig,ax=plt.subplots(figsize=(8,4));q=tab[tab.variable.ne('const')];ax.errorbar(q.beta,np.arange(len(q)),xerr=[q.beta-q.beta_low,q.beta_high-q.beta],fmt='o',capsize=3);ax.axvline(0,color='k',ls='--');ax.set_yticks(np.arange(len(q)),q.variable);ax.set_xlabel('Standardized logit coefficient (95% CI)');savefig(fig,p/f'Table1_{name}_forest')
    write_json(summary,p/'key_results.json')
    from table_display import write_main_table
    write_main_table(fitted_tables)


def result4():
    p=directory('04_insurance_Figure4');f=load();d=f[f.state_fips.eq('06')]
    tab,r,u,mu,sd=logit(d,POLICY,cluster=True,scale_all=True);assert len(u)==2475
    save(tab,p/'Figure4_g_joint_coefficients.csv')
    ref=pd.read_csv(ROOT/'reference/previous_runs/published_policy_reproduction_full.csv')
    for rr in ref.itertuples():
        v=tab.set_index('variable').loc[rr.variable]
        for field,prev in [('beta',rr.beta),('SE',rr.SE),('p',rr.p),('OR',rr.OR)]:require_close(v[field],prev,f'Fig4g {rr.variable} {field}',atol=2e-6)
    fig=plt.figure(figsize=(13,8));gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,1.6],wspace=.5,hspace=.4)
    dist=[];curves=[]
    for col,title,lim in [('nonrenew_rate_2020_2023','Non-renewal rate',(0,.25)),('fair_pct_2022','FAIR Plan share (%)',(0,45)),('denial_rate','HMDA denial rate',(0,.30))]:
        j=['nonrenew_rate_2020_2023','fair_pct_2022','denial_rate'].index(col);ax=fig.add_subplot(gs[0,j]);q=[d.loc[d.income_group.eq(k),col].dropna().values for k in GROUPS];ax.violinplot(q,showmedians=True);ax.set_xticks([1,2,3],GROUPS);ax.set_title(title,fontsize=9)
        ax=fig.add_subplot(gs[1,j])
        for k,color in zip(GROUPS,COLORS[1:]):
            v=d[d.income_group.eq(k)].dropna(subset=[col,'rebuilt'])
            dist.append({'variable':col,'group':k,'n':len(v),'mean':v[col].mean(),'median':v[col].median(),'q25':v[col].quantile(.25),'q75':v[col].quantile(.75)})
            # Current manuscript curves use the archived recovery classifications,
            # also used by the current joint model, rather than the older proxy outcome.
            model=Logit(v.rebuilt.astype(int),add_constant(v[[col]].astype(float))).fit(disp=False)
            support=np.percentile(v[col],[2,98]);xx=np.linspace(max(lim[0],support[0]),min(lim[1],support[1]),200);pr,lo,hi=prediction(model,add_constant(pd.DataFrame({col:xx})),critical=1.645)
            ax.plot(xx,pr,color=color,label=k);ax.fill_between(xx,lo,hi,color=color,alpha=.15)
            curves.extend({'variable':col,'group':k,'x':x,'probability':a,'CI90_low':b,'CI90_high':c,'N':len(v)} for x,a,b,c in zip(xx,pr,lo,hi))
        ax.set(xlabel=title,ylabel='Rebuilding probability');ax.legend(fontsize=7)
    ax=fig.add_subplot(gs[:,3]);q=tab[tab.variable.ne('const')];ax.errorbar(q.OR,np.arange(len(q)),xerr=[q.OR-q.CI_low,q.CI_high-q.OR],fmt='o',capsize=3);ax.axvline(1,ls='--',color='k');ax.set_yticks(np.arange(len(q)),q.variable,fontsize=7);ax.set(xscale='log',xlabel='Odds ratio per 1 SD')
    savefig(fig,p/'Figure4_numerical_review');save(pd.DataFrame(dist),p/'Figure4_abc_distributions.csv');save(pd.DataFrame(curves),p/'Figure4_def_curves.csv')
    write_json({'joint_N':len(u),'joint_incident_clusters':u.incident.nunique(),'nonrenewal':tab.set_index('variable').loc['nonrenew_rate_2020_2023'].to_dict(),'curve_outcome':'Archived recovery classifications (rebuilt), matching the current manuscript; all eight original joint-model predictors are retained.'},p/'key_results.json')
    threshold_rows=[]
    middle=d[d.income_group.eq('Middle')]
    for col,cut,expected in [('nonrenew_rate_2020_2023',.12,[80,3]),('fair_pct_2022',1,[86,10])]:
        rates=[]
        for side,v in [('below',middle[middle[col].lt(cut)]),('above',middle[middle[col].gt(cut)])]:
            rate=100*v.rebuilt.mean();rates.append(rate);threshold_rows.append({'variable':col,'threshold':cut,'side':side,'income_group':'Middle','N':len(v),'rebuilt_n':int(v.rebuilt.sum()),'rebuilt_percent':rate})
        assert list(map(round,rates))==expected
    save(pd.DataFrame(threshold_rows),p/'manuscript_threshold_comparisons.csv')
    gap=threshold_rows[0]['rebuilt_percent']-threshold_rows[1]['rebuilt_percent'];assert round(gap)==77
    write_json({'largest_reported_comparison_gap_percentage_points':gap,'rounded_gap_in_manuscript':77},p/'threshold_comparison_summary.json')

def result5():
    p=directory('05_transactions_Figure5');d=legacy_frame();d=d[d.trajectory.ne('obscured')].copy();sold=d[d.ownership_class.eq('sold_post_fire')].copy()
    for c in ['last_sale_price','prior_sale_price']:sold[c]=pd.to_numeric(sold[c],errors='coerce')
    sold=sold[sold.last_sale_price.gt(0)&sold.prior_sale_price.gt(0)].copy();sold['price_change_pct']=(sold.last_sale_price/sold.prior_sale_price-1)*100
    median=sold.price_change_pct.median();assert round(median)==138,median
    save(sold[['crash_id','trajectory','last_sale_price','prior_sale_price','price_change_pct']],p/'Figure5_ab_sale_records.csv')
    q=sold.groupby('trajectory').price_change_pct.agg(n='size',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index();save(q,p/'Figure5_b_by_outcome.csv')
    income=d[['crash_id','pre_tract_med_HH_income','tract_med_HH_income']].dropna();save(income,p/'Figure5_c_income_records.csv')
    fig,axs=plt.subplots(1,3,figsize=(13,4));bins=np.concatenate([np.linspace(-100,0,6)[:-1],np.geomspace(1,2000,30)])
    axs[0].hist(sold.price_change_pct,bins=bins,color=COLORS[-1]);axs[0].set_xscale('symlog',linthresh=10);axs[0].axvline(median,color='#C99946');axs[0].set(xlabel='Sale price change (%)',ylabel='Building-linked records',title='a')
    z=q.set_index('trajectory').reindex(['empty_lot','rebuilt_equal','rebuilt_improved']).dropna();axs[1].errorbar(np.arange(len(z)),z['median'],yerr=[z['median']-z.q25,z.q75-z['median']],fmt='o',capsize=3);axs[1].set_xticks(np.arange(len(z)),[x.replace('_','\n') for x in z.index],fontsize=7);axs[1].set(ylabel='Sale price change (%)',title='b')
    axs[2].scatter(income.pre_tract_med_HH_income,income.tract_med_HH_income,s=4,alpha=.2);lim=max(income.pre_tract_med_HH_income.max(),income.tract_med_HH_income.max());axs[2].plot([0,lim],[0,lim],ls='--',color='gray');axs[2].set(xlabel='Pre-fire tract income (USD)',ylabel='Post-fire tract income (USD)',title='c');fig.tight_layout();savefig(fig,p/'Figure5_numerical_review')
    write_json({'sale_N':len(sold),'median_price_change_percent':median,'share_more_than_double':float(sold.price_change_pct.gt(100).mean()),'income_scatter_N':len(income)},p/'key_results.json')
