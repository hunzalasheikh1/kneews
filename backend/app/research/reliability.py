"""Empirical Bayes state-wise pooling, fitted only on the outer training gold cases.

Intervals are conditional beta posterior intervals, not guarantees of calibration.
A prespecified concentration controls shrinkage. Cluster bootstraps in evaluation
capture study/group sampling uncertainty separately.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from .constants import FINDINGS, STATES


@dataclass
class PooledReliability:
    concentration: float = 8.0

    def fit(self, states, gold, study_ids, forbidden_ids=()):
        states, gold = np.asarray(states), np.asarray(gold, dtype=float)
        if states.shape != gold.shape or states.ndim != 2 or states.shape[1] != len(FINDINGS):
            raise ValueError('Expected matching N x 12 states and expert labels')
        if len(study_ids) != len(states) or len(set(study_ids)) != len(study_ids):
            raise ValueError('Unique study IDs required')
        if set(study_ids) & set(forbidden_ids):
            raise ValueError('Evaluation studies leaked into reliability fitting')
        if not np.isin(states, range(4)).all() or self.concentration <= 0:
            raise ValueError('Invalid state index or pooling concentration')
        valid = np.isfinite(gold)
        if not np.isin(gold[valid], [0, 1]).all() or not valid.any():
            raise ValueError('Expert labels must be binary with at least one observed label')
        self.fit_ids_ = list(study_ids)
        self.alpha_, self.beta_, self.counts_ = (np.zeros((12, 4)) for _ in range(3))
        for state in range(4):
            pooled_mask = (states == state) & valid
            # Jeffreys-smoothed pooled mean across all findings, training partition only.
            mean = (gold[pooled_mask].sum() + .5) / (pooled_mask.sum() + 1)
            for finding in range(12):
                mask = pooled_mask[:, finding]
                count, positives = mask.sum(), gold[mask, finding].sum()
                self.counts_[finding, state] = count
                self.alpha_[finding, state] = mean * self.concentration + positives
                self.beta_[finding, state] = (1-mean) * self.concentration + count-positives
        self.mean_ = self.alpha_ / (self.alpha_ + self.beta_)
        return self

    def transform(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != 12 or not np.isin(states, range(4)).all():
            raise ValueError('Expected N x 12 state indices')
        return self.mean_[np.arange(12)[None, :], states.astype(int)]

    def table(self):
        lo, hi = beta.ppf(.025, self.alpha_, self.beta_), beta.ppf(.975, self.alpha_, self.beta_)
        return [dict(finding=name, state=state, probability=float(self.mean_[f, s]),
                     lower=float(lo[f,s]), upper=float(hi[f,s]), n=int(self.counts_[f,s]),
                     interval_type='conditional_beta_95', prior_only=bool(self.counts_[f,s] == 0))
                for f,name in enumerate(FINDINGS) for s,state in enumerate(STATES)]
