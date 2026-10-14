from app.research.runtime_audit import summarize


def test_missing_telemetry_does_not_become_zero_or_hide_capacity_events():
    summary = summarize([
        {'attempts': 2, 'runtime': {'prompt_eval_count': 4200, 'eval_count': 4096}},
        {'attempts': 1, 'runtime': {'prompt_eval_count': 700, 'eval_count': 600}},
        {'attempts': 1},
    ])
    assert summary['label_sets'] == 3
    assert summary['measured_responses'] == 2
    assert summary['missing_token_telemetry'] == 1
    assert summary['attempts'] == {'2': 1, '1': 2}
    assert summary['max_combined_tokens'] == 8296
    assert summary['responses_at_generation_cap'] == 1
    assert summary['responses_at_context_capacity'] == 1
    assert summarize([{}])['max_prompt_tokens'] is None
