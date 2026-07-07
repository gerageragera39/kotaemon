# Documentation scripts

Helpers for generating MkDocs reference/example pages.

## Important files

- `generate_examples_docs.py` - example-page generation helper.
- `generate_reference_docs.py` - API/reference page generation helper.

## How it connects

Scripts read source packages under `src/` and update pages under `docs/`. They are documentation build tools, not runtime app scripts.

## Before changing

- Keep generated output deterministic.
- Do not require app runtime databases or model servers for docs generation.

## Verification

```bash
python docs/scripts/generate_reference_docs.py --help
pre-commit run prettier --all-files --show-diff-on-failure
```
