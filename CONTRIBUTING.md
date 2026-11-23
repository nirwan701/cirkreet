# Contributing

Thanks for helping improve CirKreety. The project is intended to make circuit importing and numerical review understandable and inspectable. Changes should preserve that focus and state their assumptions clearly.

## Before opening a pull request

1. Explain the issue or proposed change and keep the pull request focused.
2. Preserve the public `CircuitIR` contract unless a schema change is intentional and documented.
3. Report unsupported SPICE elements or model parameters explicitly. Do not silently replace them with ideal devices.
4. Add tests for parser behavior, numerical equations, boundary conditions, and failure cases. Compare numerical results with analytical references or trusted simulator results where practical, and document tolerances.
5. Update the capability inventory and relevant documentation when changing importer or solver support.
6. Do not commit generated reports, caches, local environments, credentials, or private circuit data.

## Set up a development environment

```bash
python -m venv .venv

# Activate the environment, then:
python -m pip install -e ".[test,notebooks]"
python -m pytest -q
```

Run the slower numerical regression tests separately with `python -m pytest -m slow -q`. To execute the notebook suite from clean kernels, run `python tools/execute_notebooks.py`.

## Pull request checklist

- [ ] Tests pass locally, or any failure is explained.
- [ ] New numerical behavior has a reference-based test where practical.
- [ ] Unsupported behavior and model assumptions are explicit.
- [ ] Documentation and capability declarations are up to date.
- [ ] No generated output, secrets, or machine-specific files are included.
