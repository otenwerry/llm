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

## Paid abliteration.ai checks
uv run exp_test_abliteration.py
    [--models abliterated-large-v2,abliterated-large,abliterated]
    [--dry-run]
--> prints PASS / FAIL per model and check, our cost next to abliteration.ai's billed amount; exits unsuccessfully on failures
