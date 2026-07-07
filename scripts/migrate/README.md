# Migration scripts

One-off utilities for moving or repairing local runtime data.

## Important files

- `migrate_chroma_db.py` - migration helper for Chroma/vector database state.

## How it connects

Runtime data normally lives under `ktem_app_data/`, while source-code storage adapters live under [`../../src/kotaemon/storages`](../../src/kotaemon/storages/README.md). Migration scripts should bridge those two without changing application logic.

## Before changing

- Treat migrations as data-changing operations; test on a copy of `ktem_app_data/` first.
- Keep scripts idempotent where practical and print what they will change.
- Do not commit migrated runtime outputs.

## Verification

Run on a copied app data directory and then smoke-test retrieval in the app or with targeted retrieval scripts.
