"""Offline first-request token audit; emits aggregate counts, never report text.

Run with PYTHONPATH=backend .venv312/bin/python scripts/audit_report_context.py.
The upstream chat template is an estimate of the Ollama serialization, not a
measurement of its internal prompt. Retry conversations require separate checks.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer
from app.research.labelling import PROMPT, PROMPT_HASH

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / 'models/qwen3-tokenizer'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    counts = []
    with (ROOT / 'data/rsna-knee-abnormality-detection/train.csv').open() as source:
        for row in csv.DictReader(source):
            segments = [m.group(0) for m in re.finditer(r'[^\n.!?]+(?:[.!?]|$)', row['Report']) if m.group(0).strip()]
            numbered = '\n'.join(f'[{i}] {text}' for i, text in enumerate(segments))
            messages = [dict(role='system', content=PROMPT),
                        dict(role='user', content='<report>\n' + numbered + '\n</report>')]
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False)
            if not isinstance(token_ids, list) or not token_ids or any(type(t) is not int for t in token_ids):
                raise TypeError('Tokenizer did not return a flat token-ID list')
            counts.append(len(token_ids))
    output = dict(
        studies=len(counts), tokenizer_repository='Qwen/Qwen3-8B', tokenizer_revision=REVISION,
        tokenizer_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in TOKENIZER.glob('*.json')},
        prompt_sha256=PROMPT_HASH, context_tokens=8192, generation_cap=4096,
        first_request_tokens=dict(max=max(counts), median=float(np.median(counts)), p95=float(np.quantile(counts, .95))),
        first_requests_exceeding_context=sum(n > 8192 for n in counts),
        first_requests_without_full_generation_allowance=sum(n + 4096 > 8192 for n in counts),
        minimum_headroom_after_generation_cap=8192 - max(counts) - 4096,
        limitations=['Uses upstream Qwen chat serialization, not measured Ollama internal serialization.',
                     'Does not audit additional retry messages or prove absence of runtime truncation.'])
    destination = ROOT / 'artifacts/audit/report-context.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
