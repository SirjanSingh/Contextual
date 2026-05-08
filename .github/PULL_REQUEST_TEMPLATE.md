<!--
Thanks for the PR! A short, focused PR is much easier to review.
-->

## Summary

<!-- One or two sentences: what changes and why. -->

## Surfaces touched

- [ ] Backend (`repo_aware_ai/`)
- [ ] CLI (`main.py`)
- [ ] Web frontend (`frontend/`)
- [ ] VS Code extension (`extension/`)
- [ ] Docs

## How was this tested?

<!--
- New / updated tests in `tests/`?
- Manual steps you ran (CLI command, frontend page, extension command)?
- Smoke-tested against a real repo?
-->

## Checklist

- [ ] `pytest -v` passes locally
- [ ] `ruff check repo_aware_ai tests` is clean
- [ ] If touching the extension or frontend, the relevant `npm run build` / `npm run lint` is clean
- [ ] No secrets, API keys, or `.env` files committed
- [ ] Docs / README / CHANGELOG updated if behaviour changed
