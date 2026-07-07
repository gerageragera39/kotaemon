# Parsers

Small parsing helpers inherited from Kotaemon.

## Important files

- `regex_extractor.py` - regex-based extraction helper.

## How it connects

Parsers can be used by loaders, extractors, or custom components to normalize structured text before indexing.

## Before changing

- Keep parser behavior deterministic and covered by tests when used in ingestion paths.

## Verification

```bash
python -m compileall src/kotaemon/parsers
```
