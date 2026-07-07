# UI assets

Static assets bundled into the Gradio application.

## Important contents

- `css/main.css` - app styling, including chat, tabs, evaluation, and citation display.
- `js/main.js` - frontend helpers for theme, storage, citation scrolling, and UI behavior.
- `js/pdf_viewer.js` - PDF.js viewer integration used by citation/source links.
- `js/svg-pan-zoom.min.js` - bundled third-party helper for SVG/mindmap interactions.
- `icons/`, `img/` - app icons, logo, and favicon.
- `md/about.md`, `md/usage.md`, `md/changelogs.md` - in-app Markdown snippets.
- `theme.py` - custom Gradio theme.

## How it connects

`ktem.app.BaseApp` reads CSS/JS assets at startup and injects them into `gr.Blocks`. The PDF viewer path is configured through `PDFJS_PREBUILT_DIR` and `GR_FILE_ROOT_PATH`; `scripts/download_pdfjs.sh` downloads the viewer bundle for Docker/runtime use.

## Before changing

- Minified/vendor assets should only be changed when updating the dependency intentionally.
- Keep in-app Markdown short; longer project documentation belongs under [`../../../docs`](../../../docs/README.md).
- Test citation and PDF interactions after changing `main.js` or `pdf_viewer.js`.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
pytest -q tests/test_information_panel_ordering.py
```
