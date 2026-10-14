"""Re-run the llm_probability arm with corrected presence probabilities (AMENDMENT-006).

Same folds, seeds, epochs, head and features as the parent comparison; only the arm-4 targets change.
Unmentioned findings keep weight 0 exactly as registered. Writes artifacts/audit/llm-arm-rerun-<id>.json.
Usage: PYTHONPATH=backend .venv312/bin/python scripts/rerun_llm_arm.py <experiment_id> <presence.jsonl>
"""
import json
import sys
import numpy as np
from app.config import ROOT
from app.research.constants import FINDINGS
from app.research.metrics import evaluate
sys.path.insert(0, str(ROOT / 'scripts'))
from head_selection import CONFIGS, run_fold_predictions

experiment, presence = sys.argv[1], sys.argv[2]
E = ROOT / 'artifacts' / 'experiments' / experiment
z = dict(np.load(E / 'training-inputs.npz'))
folds = {r['id']: r['fold'] for r in json.loads((E / 'manifest.json').read_text())['folds']}
folds = np.array([folds[s] for s in z['study_ids']])
is_gold = np.isfinite(z['gold']).all(1)
records = {}
for line in open(presence):
    r = json.loads(line); records[r['study_id']] = r['probabilities']
missing = [s for s in z['study_ids'] if s not in records]
if missing: raise SystemExit(f'{len(missing)} studies lack corrected probabilities')
llm = np.full(z['states'].shape, np.nan)
for i, sid in enumerate(z['study_ids']):
    for j, f in enumerate(FINDINGS):
        if z['states'][i, j] != 3:
            if f not in records[sid]: raise SystemExit(f'{sid} {f} missing')
            llm[i, j] = records[sid][f]
by_state = {s: float(np.nanmean(llm[z['states'] == k])) for k, s in enumerate(['present', 'absent', 'uncertain'])}
print('mean corrected probability by state', by_state, flush=True)
preds = [run_fold_predictions(z, folds, is_gold, 'llm_probability', CONFIGS['registered'], s, 'mps', is_gold, llm)[is_gold] for s in (2026, 2027, 2028)]
summ = lambda m: dict(macro_auc=m['macro_auc'], worst=m['worst_finding_auc'], brier=m['brier'], ece=float(np.mean([f['ece'] for f in m['per_finding']])))
per_seed = [summ(evaluate(z['gold'][is_gold], p, include_calibration=False)) for p in preds]
ens = summ(evaluate(z['gold'][is_gold], np.mean(preds, 0), include_calibration=False))
old = [json.loads((E / f'results-seed{s}.json').read_text())['metrics']['llm_probability'] for s in (2026, 2027, 2028)]
out = dict(experiment=experiment, presence_file=presence, mean_probability_by_state=by_state, per_seed=per_seed, seed_ensemble=ens,
           original_per_seed=[summ(m) for m in old], note='Only arm-4 targets changed; folds/seeds/epochs/head identical to the parent comparison.')
(ROOT / 'artifacts' / 'audit' / f'llm-arm-rerun-{experiment}.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
