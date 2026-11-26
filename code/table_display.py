"""Format newly fitted Table 1 values at the manuscript's chosen precision."""
import json
import pandas as pd
from common import ROOT,BASE,save,directory,write_json

def write_main_table(fits):
    reference=json.loads((ROOT/'reference/manuscript_tables/table_01.json').read_text(encoding='utf8'))
    def coefficient(row):
        return f'{row.beta:+.2f} ({row.beta_low:+.2f}, {row.beta_high:+.2f})'.replace('-','−')
    def probability(value,example):
        if value<.001:return '< 0.001'
        decimals=len(example.split('.')[1]) if '.' in example else 3
        return f'{value:.{decimals}f}'
    rows=[];differences=[]
    for variable,target in zip(BASE,reference[2:]):
        a=fits['baseline'].loc[variable];b=fits['upgrading'].loc[variable]
        row=[target[0],coefficient(a),probability(a.p,target[2]),coefficient(b),probability(b.p,target[4])]
        rows.append(row)
        for column,(current,expected) in enumerate(zip(row,target)):
            if current.replace(' ','')!=expected.replace(' ',''):
                differences.append({'variable':variable,'column':column,'computed':current,'manuscript':expected})
    columns=['Covariate','Rebuilt vs empty lot: β (95% CI)','Rebuilt vs empty lot: p-value','Upgraded among rebuilt buildings: β (95% CI)','Upgraded among rebuilt buildings: p-value']
    save(pd.DataFrame(rows,columns=columns),directory('tables')/'Table1.csv')
    # Coefficients and display precision are validated before export.
    assert not differences,differences
