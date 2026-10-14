"""Aggregate local inference telemetry without exporting report text."""
from collections import Counter
import json
import time

from sqlalchemy import select
from ..config import settings
from ..models import LabelSet, Study


def summarize(provenances):
    attempts = Counter()
    prompt_counts, output_counts, combined_counts = [], [], []
    missing = 0
    for provenance in provenances:
        attempts[str(provenance.get('attempts', 'unknown'))] += 1
        runtime = provenance.get('runtime') or {}
        prompt, output = runtime.get('prompt_eval_count'), runtime.get('eval_count')
        if type(prompt) is not int or type(output) is not int or prompt < 0 or output < 0:
            missing += 1
            continue
        prompt_counts.append(prompt)
        output_counts.append(output)
        combined_counts.append(prompt + output)
    return dict(
        label_sets=sum(attempts.values()), attempts=dict(attempts), measured_responses=len(prompt_counts),
        missing_token_telemetry=missing, max_prompt_tokens=max(prompt_counts, default=None),
        max_output_tokens=max(output_counts, default=None),
        max_combined_tokens=max(combined_counts, default=None),
        responses_at_generation_cap=sum(n >= 4096 for n in output_counts),
        responses_at_context_capacity=sum(n >= 8192 for n in combined_counts),
        limitations=['Telemetry covers the final accepted response, not earlier failed attempts.',
                     'Token counts cannot establish whether source text was truncated before evaluation.',
                     'Structural validation and runtime completion do not establish clinical label accuracy.'])


def audit(db, version):
    provenances = db.scalars(select(LabelSet.provenance).join(Study, Study.id == LabelSet.study_id)
                            .where(Study.split == 'train', LabelSet.method == 'local_llm', LabelSet.version == version)).all()
    result = dict(version=version, inspected_at=time.time(), **summarize(provenances))
    directory = settings.artifact_root / 'audit'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'label-runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
