"""Flag automatically detected report-language codes that need human verification.

Detection is automatic and unreviewed. This audit does not change any label; it lists
codes that are implausible for clinical reports or easily confused with a neighbour.
"""
# Languages plausible for knee MRI reports in this dataset's region.
PLAUSIBLE={'en','es','tr','hr','bs','el','de','bg','nl','fr','it','pt','pl','ru','sr','sl','ro','hu','cs','sk'}
# Codes a detector commonly confuses (very close written forms).
CONFUSABLE=[{'hr','bs','sr'},{'cs','sk'},{'es','pt'}]
# Not a living reporting language; usually Latin anatomical terms fooling the detector.
NON_CLINICAL={'la'}


def audit_language_codes(counts,min_count=20):
    """counts: {code: number_of_reports}. Returns one row per code, largest first."""
    if any(not isinstance(n,int) or n<0 for n in counts.values()):
        raise ValueError('Counts must be non-negative integers')
    present=set(counts)
    rows=[]
    for code,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        reasons=[]
        if code in NON_CLINICAL:reasons.append('not a clinical reporting language')
        elif code not in PLAUSIBLE and code!='unknown':reasons.append('unexpected for this dataset')
        group=next((g for g in CONFUSABLE if code in g),set())
        if len(group&present)>1:reasons.append('confusable with '+', '.join(sorted(group&present-{code})))
        if code=='unknown':reasons.append('language not detected')
        if 0<n<min_count:reasons.append(f'fewer than {min_count} reports')
        rows.append(dict(code=code,reports=n,needs_review=bool(reasons),reasons=reasons))
    return rows
