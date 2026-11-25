# Contributing

From the checkout, create a Python 3.12 virtual environment, install with
`python -m pip install -e '.[test]'`, then run `python -m pytest -q`.
For an incorrect verdict, include synthetic tensors, the CLI version and the
smallest reproducing command. Keep private weights, tokens and account paths
out of issues. New manifest entries need independent loader attachment,
an inactive neighbor and both positive and loadable negative controls.
Changes to report fields or tolerance defaults need a documented version.
