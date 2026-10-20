# Final Word report from VIT_Project_Report_trimmed_80.docx — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/VIT_Project_Report_final.docx` = the user's trimmed 80-page report with every figure slot filled by the right image, sized correctly, and a clean layout — **with no text changed**.

**Architecture:** Same package-level edit as `2026-10-12-fill-docx-figures.md`: overwrite each placeholder PNG in `word/media/`, recompute the drawing extents from the real aspect ratio, then layout-only fixes (table column widths, keep-with-next so captions stay with their figure/table). Output to a new file; both source copies (`~/Downloads/` and `docs/VIT_Project_Report_trimmed_80.docx`) stay untouched.

## Brainstorm result

- *What:* "replace the cutouts with the optimal images … don't alter any text". The cutouts are the 13 grey placeholder tiles (5.83 × 1.63 in, a 3.6:1 shape that fits none of the figures). "Optimal" = the figure each slot's caption names, at its own aspect ratio, as large as the text column allows without breaking a page.
- *Constraint not stated:* the user cut figures 6.6–6.9 and 6.13 but the text still numbers the rest 6.10–6.12, and "don't alter any text" means those numbers stay. Reported, not fixed.
- *Out of bounds:* any text, caption, number, heading, table content.

## Global Constraints

- Text content of `word/document.xml` (tags stripped) is byte-identical before and after.
- Margins stay as in the source (left 3.81 cm / right, top, bottom 2.54 cm).
- Figure width ≤ 5.77 in (text column); height ≤ 4.6 in.
- No commit.

---

### Task 1: Map the 13 slots to figure files
**Goal:** verified (slot, media file, figure file) triples.
**Why:** a wrong image under a caption is worse than a visible placeholder.
**Files:** none (scratchpad).
**Steps:**
- [x] Read each slot's alt title (figure number) and the filename printed on its tile.
- [x] Look up `docs/figures/<name>.png` from the figure number via `docs/project-report.md`.
**Done when:** for all 13 slots the figure number in the alt title, the filename on the tile and the markdown's figure→file mapping agree.

### Task 2: Fill the slots and size them
**Goal:** all 13 slots hold the real images at correct aspect ratios.
**Files:** Create `docs/VIT_Project_Report_final.docx`.
**Steps:**
- [x] Overwrite the 13 media entries with the figure bytes.
- [x] Set `wp:extent` and `a:ext` to the largest size ≤ 5.77 × 4.6 in at the figure's own ratio.
**Done when:** every drawing's extent ratio equals its image's pixel ratio within 1 %; no tile remains.

### Task 3: Layout polish, text untouched
**Goal:** no figure separated from its caption, no table split from its title, tables fit the column.
**Steps:**
- [x] Render a copy in Pages; list pages with large blank areas and orphaned figure/table titles.
- [x] Add keep-with-next on figure paragraphs and on "Table N:" caption paragraphs; set table columns proportional to content within the text column.
- [x] Re-render and compare.
**Done when:** the re-render shows no figure alone at a page bottom with its caption on the next page and no "Table N:" title stranded from its table.

### Task 4: Verify
**Steps:**
- [x] Text identical to the trimmed source (tags stripped).
- [x] All 13 media equal their `docs/figures` files; zip and XML valid; Pages opens it and exports a PDF; page count recorded.
**Done when:** every check prints its expected result.

**Result (recorded after execution):** `docs/VIT_Project_Report_final.docx` — 13 figures filled (white borders trimmed; the arm photo's watermark deliberately kept), figure + caption and table + title kept together, code blocks fitted to the column, text identical to the trimmed source. Pages renders it at 99 pages; the same renderer gives 95 for the trimmed file with placeholders, which the user names "80", so Word's count should be about 15 pages lower, i.e. mid-80s. Not verified in Word.
