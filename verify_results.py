"""Check the numerical statements reported in the main text and supplement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def verify_results():
    checks=[]
    def check(name,condition):
        checks.append({'statement':name,'passed':bool(condition)})
        assert condition,name
    def read(path):return pd.read_csv(ROOT/'outputs'/path)
    def meta(path):return json.loads((ROOT/'outputs'/path).read_text(encoding='utf8'))
    one=meta('01_recovery_Figure1/key_results.json');check('Main sample: 4,637 buildings, 4,614 California buildings and 53 state/incident groups',one['N']==4637 and one['California_n']==4614 and one['paper_fire_count_state_incident_groups']==53)
    check('Empty lots: 2,898/4,637; excluding Palisades: 2,680/4,416',one['empty_lot_n']==2898 and one['empty_lot_excluding_Palisades_n']==2680 and one['non_Palisades_n']==4416)
    g=read('02_income_Figure2/Figure2_ac_income_groups.csv');check('Figure 2 income-group denominators',g.n.tolist()==[1334,1523,1453]);check('Observed rebuilding: 6.1%, 65.0%, 42.3%',g.rebuilt_pct.round(1).tolist()==[6.1,65.0,42.3]);check('Conditional upgrading: 17.3%, 46.0%, 49.1%',g.upgraded_among_rebuilt_pct.round(1).tolist()==[17.3,46.0,49.1])
    two=meta('02_income_Figure2/key_results.json');check('Figure 2b: 4,121 buildings, 32 events, approximately 5% to 58%',two['curve_n']==4121 and two['curve_events']==32 and abs(two['curve_min_probability']-.05)<.01 and abs(two['curve_max_probability']-.58)<.01)
    for name,n,beta in [('baseline',4186,1.36),('upgrading',1614,.36)]:
        table=read(f'03_models_Table1/Table1_{name}_coefficients.csv').set_index('variable');income=table.loc['log_income'];check('Table 1 '+name+' N and income coefficient',int(income.N)==n and round(income.beta,2)==beta)
        if name=='baseline':check('Income is largest absolute standardized baseline association',table.drop(index='const').beta.abs().idxmax()=='log_income')
        else:check('Income is the only significant measured upgrade predictor',table.drop(index='const').query('p < .05').index.tolist()==['log_income'])
    policy=read('04_insurance_Figure4/Figure4_g_joint_coefficients.csv').set_index('variable');a=policy.loc['nonrenew_rate_2020_2023'];check('Joint policy N=2,475; non-renewal OR=0.12, CI 0.06–0.24',int(a.N)==2475 and [round(a.OR,2),round(a.CI_low,2),round(a.CI_high,2)]==[.12,.06,.24]);check('Non-renewal is the only significant joint-model predictor',policy.drop(index='const').query('p < .05').index.tolist()==['nonrenew_rate_2020_2023'])
    thresholds=read('04_insurance_Figure4/manuscript_threshold_comparisons.csv');check('Middle-income threshold comparisons: 80%/3% and 86%/10%',thresholds.rebuilt_percent.round().astype(int).tolist()==[80,3,86,10]);check('Reported largest threshold-comparison gap: 77 percentage points',round(thresholds.rebuilt_percent.iloc[0]-thresholds.rebuilt_percent.iloc[1])==77)
    five=meta('05_transactions_Figure5/key_results.json');check('Recorded transaction subset: n=710, median gain approximately 138%',five['sale_N']==710 and round(five['median_price_change_percent'])==138);check('More than half of recorded transactions exceed twice the prior price',five['share_more_than_double']>.5)
    veg=meta('02b_vegetation_Figure3/key_results.json');check('Higher-income year-four vegetation deviation approximately -0.12',round(veg['high_income_year4_deviation'],2)==-.12)
    six=meta('supplementary/Figure_S6_statistics.json');check('Vegetation agreement: 5,000 observations and Pearson r=0.81',six['n_sampled']==5000 and round(six['pearson_r'],2)==.81)
    windows=read('supplementary/Table_S13_window_counts.csv');pivot=windows.pivot(index='target_year',columns='income_group',values='rebuilt_percent');check('Lower-income rebuilding proportion is lower in every observation window',bool((pivot.Low<pivot.Middle).all() and (pivot.Low<pivot.High).all()))
    times=read('supplementary/Table_S14_time_models_full.csv');check('Table S14 common sample N=4,137, G=33',times.N.eq(4137).all() and times.G.eq(33).all());check('Cubic-time-adjusted income OR=2.38',round(times.loc[times.time_adjustment.eq('spline'),'OR'].iloc[0],2)==2.38)
    insurance=read('supplementary/Table_S16_insurer_models_full.csv');check('Table S16 common sample N=3,820, G=15',insurance.N.eq(3820).all() and insurance.G.eq(15).all());check('Table S16 insurer ORs: 0.577 and 0.573',insurance.OR.round(3).tolist()==[.577,.573])
    lo=read('supplementary/Note9_all_leave_one_fire_out.csv');check('Leave-one-fire-out: all 15 ORs below 1 and 13 upper confidence bounds below 1',len(lo)==15 and lo.OR.lt(1).all() and lo.CI_high.lt(1).sum()==13)
    cor=meta('supplementary/Note9_component_correlation.json');check('Common-period initiator comparison: 37 ZIPs, r=0.208 and R2=0.043',cor['unique_ZIPs']==37 and round(cor['insurer_policyholder_Pearson_r'],3)==.208 and round(cor['R_squared'],3)==.043)
    result={'all_reported_statement_checks_passed':True,'n_checks':len(checks),'checks':checks}
    import os
    audit=os.environ.get('RECOVERY_AUDIT_DIR')
    if audit:
        destination=Path(audit);destination.mkdir(parents=True,exist_ok=True)
        (destination/'manuscript_statement_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

if __name__=='__main__':print(json.dumps(verify_results(),indent=2))
