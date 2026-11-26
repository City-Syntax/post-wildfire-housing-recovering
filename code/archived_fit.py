import numpy as np
import pandas as pd
import warnings
from types import SimpleNamespace
from scipy.stats import t as tdist, norm
from patsy import dmatrix
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.tools.tools import add_constant
sm=SimpleNamespace(GLM=GLM,add_constant=add_constant,families=SimpleNamespace(Binomial=Binomial))
rows=[]; failures=[]; fitdetails=[]

def fit(df, name, preds, target='log_income', outcome='rebuilt', time=None, fe=False, cluster=True, scale_all=False, group='event_id', quiet=False):
    d = df.copy()
    needed = list(dict.fromkeys(preds + [outcome, group] + (['followup_years'] if time else [])))
    numeric = d[preds].apply(pd.to_numeric, errors='coerce').astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if scale_all:
        mu = numeric.mean()
        sd = numeric.std(ddof=1)
    use = d[needed].copy()
    for k in preds:
        use[k] = numeric[k]
    use = use.dropna()
    use = use[np.isfinite(use[preds].astype(float)).all(axis=1)]
    if fe:
        ok = use.groupby(group)[outcome].agg(['min', 'max'])
        groups = ok.index[ok['min'].ne(ok['max'])]
        use = use[use[group].isin(groups)].copy()
    n = len(use)
    G = use[group].nunique()
    if n < 40 or G < 2 or use[outcome].nunique() < 2:
        failures.append({'name': name, 'reason': 'insufficient observations, events or outcome variation', 'N': n, 'G': G})
        return None
    if not scale_all:
        mu = use[preds].mean()
        sd = use[preds].std(ddof=1)
    constant = sd.index[sd.le(1e-12) | sd.isna()].tolist()
    if target in constant:
        failures.append({'name': name, 'reason': 'target predictor has no variation', 'N': n, 'G': G})
        return None
    active = [k for k in preds if k not in constant]
    X = (use[active] - mu[active]) / sd[active]
    X = sm.add_constant(X, has_constant='add')
    if time == 'linear':
        X['followup_years_centered'] = use.followup_years - use.followup_years.mean()
    if time == 'spline':
        spl = dmatrix('bs(x, df=4, degree=3, include_intercept=False)-1', {'x': use.followup_years.to_numpy()}, return_type='dataframe')
        spl.index = use.index
        spl.columns = [f'time_spline_{i}' for i in range(spl.shape[1])]
        X = pd.concat([X, spl], axis=1)
    if fe:
        dum = pd.get_dummies(use[group], prefix='fire', drop_first=True, dtype=float)
        X = pd.concat([X, dum], axis=1)
    X = X.astype(float)
    rank = np.linalg.matrix_rank(X.to_numpy())
    if rank < X.shape[1]:
        failures.append({'name': name, 'reason': 'rank-deficient design; no automatic variable selection', 'N': n, 'G': G, 'rank': rank, 'parameters': X.shape[1]})
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model = sm.GLM(use[outcome].astype(float), X, family=sm.families.Binomial())
            kw = {'cov_type': 'cluster', 'cov_kwds': {'groups': use[group].astype(str).to_numpy(), 'use_correction': True}} if cluster else {}
            result = model.fit(maxiter=200, tol=1e-09, **kw)
        except Exception as exc:
            failures.append({'name': name, 'reason': str(exc), 'N': n, 'G': G})
            return None
    b = float(result.params[target])
    se = float(result.bse[target])
    crit = float(tdist.ppf(0.975, G - 1)) if cluster else norm.ppf(0.975)
    p = float(2 * tdist.sf(abs(b / se), G - 1)) if cluster else float(2 * norm.sf(abs(b / se)))
    warning_text = '; '.join(sorted({str(x.message) for x in caught}))
    row = {'name': name, 'target': target, 'outcome': outcome, 'N': n, 'G': G, 'beta_per_sd': b, 'se': se, 'OR': np.exp(b), 'CI_low': np.exp(b - crit * se), 'CI_high': np.exp(b + crit * se), 'p': p, 'predictor_sd': float(sd[target]), 'converged': bool(result.converged), 'cluster': group if cluster else 'none', 'CI_distribution': f't_{G - 1}' if cluster else 'normal', 'time_adjustment': time or 'none', 'fire_fixed_effect': fe, 'dropped_constant_predictors': ','.join(constant), 'warnings': warning_text, 'min_pred': float(result.fittedvalues.min()), 'max_pred': float(result.fittedvalues.max())}
    rows.append(row)
    if not quiet:
        print(name, 'N', n, 'G', G, 'OR', round(row['OR'], 4), 'CI', round(row['CI_low'], 4), round(row['CI_high'], 4), 'p', round(p, 5), flush=True)
    fitdetails.append({'name': name, 'predictors': active, 'means': mu.to_dict(), 'sd': sd.to_dict(), 'n_dropped_for_missing': len(d) - len(use), 'rank': int(rank), 'n_parameters': X.shape[1], 'ids_sha256': __import__('hashlib').sha256('\n'.join(sorted(use.index.astype(str))).encode()).hexdigest()})
    return (row, use, result)
