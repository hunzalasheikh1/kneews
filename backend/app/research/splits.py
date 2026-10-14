"""Groups connect scanner/site identity AND duplicate reports, transitively."""
import hashlib
import re
import numpy as np
from sklearn.model_selection import GroupKFold


def report_hash(report):
    normalized = re.sub(r'\s+', ' ', report.strip().casefold())
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else None


def connected_groups(records):
    parent = list(range(len(records)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(records):
        # Unknown metadata is one conservative group, never a distinct site per study.
        keys = [('site', row.get('site_group') or 'unknown')]
        duplicate = row.get('report_hash')
        if duplicate:
            keys.append(('report', duplicate))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return [str(root(i)) for i in range(len(records))]


def make_folds(records, n_splits=5):
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Study IDs must be unique')
    groups = connected_groups(records)
    if len(set(groups)) < n_splits:
        raise ValueError(f'Only {len(set(groups))} independent groups available for {n_splits} folds')
    folds = np.full(len(records), -1, dtype=int)
    for fold, (_, heldout) in enumerate(GroupKFold(n_splits=n_splits).split(ids, groups=groups)):
        folds[heldout] = fold
    result = [dict(**row, fold=int(folds[i]), leakage_group=groups[i]) for i,row in enumerate(records)]
    assert_no_leakage(result)
    return result


def assert_no_leakage(records):
    for field in ['id', 'site_group', 'report_hash', 'leakage_group']:
        assignment = {}
        for row in records:
            value = row.get(field)
            if value is None or value == '':
                continue
            if value in assignment and assignment[value] != row['fold']:
                raise ValueError(f'{field} leakage across folds')
            assignment[value] = row['fold']
