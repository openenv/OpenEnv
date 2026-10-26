## Release PR: v<!-- VERSION -->

## Release Checklist

### Before opening this PR
- [ ] `pyproject.toml` version changed from `X.Y.Z.dev0` → `X.Y.Z` (no `.dev0` suffix)
- [ ] `hf-staging/` is NOT in this PR's diff
- [ ] No `print()`, `breakpoint()`, or `TODO` in release-critical paths
- [ ] Release notes updated for user-facing changes, including project coordination or governance changes

### CI gates (must be green before merge)
- [ ] `test` passes on Python 3.11
- [ ] `test` passes on Python 3.12
- [ ] `lint` passes (usort + ruff)
- [ ] `Package CI` builds, checks, and smoke-tests wheel/sdist installs

### TestPyPI validation (before merging)
- [ ] Manual dispatch of `publish-testpypi.yml` from this branch
- [ ] TestPyPI workflow published a unique pre/dev version such as `X.Y.Z.devN` or `X.Y.ZrcN`
- [ ] TestPyPI workflow verified `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openenv==X.Y.Z.devN`

### Post-merge steps (author only)
- [ ] Tag `vX.Y.Z` pushed: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`
- [ ] `publish-pypi.yml` completed successfully from the tag
- [ ] GitHub Release was created by the successful PyPI publish workflow
- [ ] `pip install openenv==X.Y.Z` from production PyPI verified
- [ ] `publish-pypi.yml` created `bump/X.Y.(Z+1).dev0` PR
