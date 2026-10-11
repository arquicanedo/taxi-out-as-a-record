"""s537 step A (runs ONCE; a time-budgeted search is not deterministic): FLAML searches the ADS-B stage learner on v104's
exact rows and features for each Jan|Jul fold and for the whole window; the winning XGBoost settings are saved to
artifacts/v107_xgb_params.json, from which scripts/adsb_price_v107.py trains deterministically.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with "flaml[automl]" --with xgboost --with catboost \
      python scripts/v107_flaml_search.py
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v104.parquet')); os.environ['ADSB_FOLDS'] = 'janjul'
__file__ = os.path.join(HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
from flaml import AutoML
tmw_ = X.tm.values == 1; COLS = [c for c in X.columns if c != 'rwy']
fit = has & ~tmw_ & (y < 7200)                                      # v104's exact training rows
Xa = X[COLS].copy(); Xa['apt'] = Xa['apt'].astype(int)
out = {'cols': COLS}
for name, f_, budget in (('fit_jul', fit & L, 600), ('fit_jan', fit & J, 600), ('fit_all', fit, 900)):
    t0 = time.time(); am = AutoML()
    am.fit(Xa[f_], rres[f_], task='regression', metric='mse', time_budget=budget, estimator_list=['xgboost'],
           eval_method='cv', n_splits=3, seed=0, verbose=0)
    p = am.model.estimator.get_params(); p = {k: v for k, v in p.items() if isinstance(v, (int, float, str, bool)) or v is None}
    out[name] = p; print(f'{name}: {time.time()-t0:.0f} s, cv mse {am.best_loss:.0f}, config {am.best_config}', flush=True)
json.dump(out, open(os.path.join(REPO, 'artifacts', 'v107_xgb_params.json'), 'w'), indent=1, default=str)
print('wrote artifacts/v107_xgb_params.json')
