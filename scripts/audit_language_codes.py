"""Print (and optionally save) the language-code audit for the training reports.

Usage: PYTHONPATH=backend .venv312/bin/python scripts/audit_language_codes.py [--csv out.csv]
Needs the local PostgreSQL database with imported studies.
"""
import argparse
import csv
from sqlalchemy import select,func
from app.db import SessionLocal
from app.models import Study
from app.research.language_audit import audit_language_codes

parser=argparse.ArgumentParser();parser.add_argument('--csv');parser.add_argument('--min-count',type=int,default=20)
args=parser.parse_args()
with SessionLocal() as db:
    counts=dict(db.execute(select(Study.language,func.count()).where(Study.split=='train').group_by(Study.language)).all())
rows=audit_language_codes(counts,args.min_count)
for r in rows:print(f"{r['code']:8s} {r['reports']:6d}  {'REVIEW' if r['needs_review'] else 'ok':6s}  {'; '.join(r['reasons'])}")
print(f"{sum(r['needs_review'] for r in rows)} of {len(rows)} codes need human review")
if args.csv:
    with open(args.csv,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['code','reports','needs_review','reasons'])
        w.byte = None
        w.writerows([[r['code'],r['reports'],r['needs_review'],'; '.join(r['reasons'])] for r in rows])
