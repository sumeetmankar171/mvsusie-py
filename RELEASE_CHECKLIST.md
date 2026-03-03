# Release Checklist

## Pre-release

- [ ] Update version in `pyproject.toml`
- [ ] Add release notes to `CHANGELOG.md`
- [ ] Run local tests:
  - `PYTHONPATH=src python -m pytest -q`
- [ ] Verify parity smoke script:
  - `PYTHONPATH=src python scripts/parity_smoke.py`

## Git / Tag

- [ ] Commit release changes
- [ ] Create annotated tag (example):
  - `git tag -a v0.3.0 -m "v0.3.0"`
- [ ] Push branch and tags:
  - `git push`
  - `git push --tags`

## Publish (optional)

- [ ] Build distributions:
  - `python -m pip install build`
  - `python -m build`
- [ ] Upload to PyPI:
  - `python -m pip install twine`
  - `python -m twine upload dist/*`

## Post-release

- [ ] Verify install from published artifact
- [ ] Create GitHub release notes from `CHANGELOG.md`
