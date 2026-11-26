# Contributing

Thanks for helping improve CirKreet. Small, reproducible contributions are
especially useful to students and engineers learning circuit analysis.

## Before you start

1. Open an issue for a substantial change so its scope can be discussed.
2. Include a minimal circuit file when reporting a parser or solver problem.
3. State the expected result and how you calculated or measured it.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Make a change

- Keep reusable implementation in `circhecker/` Python modules.
- Keep notebooks focused on explanations, experiments, and plots.
- Preserve the boundary between source circuit data (`CircuitIR`) and a
  per-run analysis request (`AnalysisSpec` / `AnalysisIR`).
- Do not silently approximate unsupported circuit elements. Return an explicit
  unsupported status and explain the reason.
- Add tests for new behaviour. Prefer analytical reference circuits such as
  voltage dividers, RC filters, and RLC networks.
- Include units in public result fields and document non-obvious assumptions.

## Run checks

```bash
python -m pytest -q --ignore=tests/test_unified.py
python -m pytest -q tests/test_unified.py
python -m compileall -q circhecker examples tests
```

## Pull requests

Describe the problem, the implementation, the tests you ran, and any known
limitations. Include a small example or report excerpt when it helps explain
the change. Keep unrelated formatting changes out of the pull request.
