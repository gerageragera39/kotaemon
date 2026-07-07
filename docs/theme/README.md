# MkDocs theme overrides

Custom theme templates and assets for the documentation site.

## Important contents

- `main.html` - theme override entry point.
- `partials/` - header/footer/library partials.
- `assets/pymdownx-extras/` - generated/bundled assets used by pymdownx extras.

## How it connects

`mkdocs.yml` points MkDocs Material at this folder for theme customization.

## Before changing

- Treat bundled/minified assets as generated/vendor content.
- Test site rendering if changing HTML partials.

## Verification

```bash
mkdocs build --strict
```

Run this only when MkDocs dependencies are installed.
