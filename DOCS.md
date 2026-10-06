## Offline regression tests
uv run pytest tests
--> prints test results; no model calls

## Paid model smoke test
uv run exp_test.py
    --models MODEL[,MODEL...]
    [--max-tokens 256]
    [--batch]
    [--dry-run]
--> prints PASS / FAIL per model and cost totals; exits unsuccessfully on failures
