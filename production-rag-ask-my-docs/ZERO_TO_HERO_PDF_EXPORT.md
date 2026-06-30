# PDF Export Guide for `ZERO_TO_HERO_LEARNING_HANDBOOK.md`

Constraint alignment:
- This project handoff kept analysis static and did not execute project code.
- This guide gives deterministic export paths for you to run locally.

## Source file

- `ZERO_TO_HERO_LEARNING_HANDBOOK.md`

## Option 1 (Recommended): Editor/Browser Print Pipeline

Use this when no CLI PDF converter is installed.

1. Open `ZERO_TO_HERO_LEARNING_HANDBOOK.md` in a markdown preview-capable editor.
2. Ensure preview uses:
- Page size: A4 or Letter
- Margins: Normal
- Scale: 100%
- Include background graphics: enabled (if available)
3. Print from preview and choose `Save as PDF`.
4. Save as `ZERO_TO_HERO_LEARNING_HANDBOOK.pdf`.

Why recommended:
- No extra dependencies required.
- Preserves headings, tables, and code blocks well enough for handbook use.

## Option 2: `pandoc` CLI (if already installed on your machine)

```bash
pandoc ZERO_TO_HERO_LEARNING_HANDBOOK.md \
  -o ZERO_TO_HERO_LEARNING_HANDBOOK.pdf \
  --from gfm \
  --toc \
  --pdf-engine=xelatex
```

Notes:
- If `xelatex` is unavailable, use your installed PDF engine.
- This command is optional and was not executed as part of this static handoff.

## Option 3: Browser-native HTML -> PDF

1. Render markdown to HTML using your preferred markdown preview workflow.
2. Open HTML in browser.
3. Print -> Save as PDF.
4. Save as `ZERO_TO_HERO_LEARNING_HANDBOOK.pdf`.

## Validation checklist before sharing PDF

1. Title page shows: `MASTER LEARNING HANDBOOK: ask-my-docs-rag`.
2. All 5 modules are present.
3. Module 2 tables are fully visible and not cut off.
4. Code/ASCII blocks are not wrapped incorrectly.
5. Page count and TOC look readable in print.

