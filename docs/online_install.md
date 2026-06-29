# Online deployment note

The original upstream Kotaemon project documents Hugging Face Space and release-zip deployment flows. Those instructions are not maintained for KURAGa and may deploy upstream defaults rather than this KU Digital Projects fork.

For KURAGa, prefer one of the maintained paths:

- local source install from the repository root;
- Docker Compose from this repository;
- a reviewed institutional deployment that keeps `ktem_app_data/` persistent and protects any sensitive data.

Do not use upstream release ZIP or the upstream `libs/*` layout installer assumptions for this fork without reviewing and updating them first.
