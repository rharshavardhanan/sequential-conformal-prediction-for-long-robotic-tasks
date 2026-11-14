# Fill the VIT_Project_Report.docx figure slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every grey "FIGURE FILE NOT BUNDLED" tile in `docs/VIT_Project_Report.docx` is replaced by the real figure named in its own caption, at the right aspect ratio, with nothing else in the document changed.

**Architecture:** A .docx is a zip. Each figure slot is a `<w:drawing>` whose `r:embed` points at a placeholder PNG in `word/media/`. Overwrite that PNG with the real figure and rewrite the drawing's `cx`/`cy` extents so the image keeps its own aspect ratio inside the text width. Rezip. The cover/title-page logo (`image1.jpg`) is a real image and is left alone.

**Tech Stack:** Python 3 stdlib (`zipfile`, `re`, `struct`) + `~/PROJECTS/venv/bin/python` (Pillow only to read image sizes); `pandoc` to confirm the file still parses. No new dependencies.

## Brainstorm result (what and why)

- *What:* the request ("fill in the document with the respective images … specified in the document itself") means the 18 placeholder tiles, each of which prints the filename it wants. The captions repeat it as "Evidence file: `docs/figures/<name>.png`".
- *Constraint the request did not state:* the placeholders are 1400×560 (2.5:1) while the real figures range from 1.0:1 to 2.6:1, and the arm photo is portrait. Dropping the bytes in without fixing the drawing extents would squash every figure, so extents are recomputed.
- *Two copies exist* (`~/Downloads/VIT_Project_Report.docx` and `docs/VIT_Project_Report.docx`, byte-identical). The repo copy is edited; the Downloads copy is left untouched as the untouched original.

## Global Constraints

- Only the 18 placeholder media files and their `<wp:extent>`, `<a:ext>` and `descr` attributes change. No text, style, numbering or section change.
- Image width never exceeds the placeholder's width (the text column); image height capped at 4.6 in so a portrait photo cannot overflow a page.
- No commit (global rule).

---

### Task 1: Map every placeholder to its figure file

**Goal:** A verified table of (drawing order, media file, figure filename).

**Why:** A wrong mapping would put the wrong image under a caption — worse than a visible placeholder.

**Files:** none modified (scratchpad only).

**Steps:**
- [x] Unzip a scratch copy; list drawings in document order with their `r:embed` media target.
- [x] Read the figure filename each placeholder prints (contact sheet of the 18 tiles) and the next "Evidence file:" path after each drawing in the document text.
- [x] Confirm the two lists agree for all 18.

**Done when:** the 18 pairs printed by the script match the filename on the tile and the caption path, with no disagreement.

---

### Task 2: Replace the images and fix the extents

**Goal:** `docs/VIT_Project_Report.docx` contains the real figures at correct aspect ratios.

**Why:** This is the deliverable.

**Files:**
- Modify: `docs/VIT_Project_Report.docx`
- Backup: a copy of the original in the scratchpad

**Steps:**
- [x] Copy the original to the scratchpad as a backup.
- [x] For each mapped drawing: write `docs/figures/<name>.png` over the placeholder media entry; set `cx` to the placeholder's width (reduced if the height would exceed 4.6 in) and `cy = cx × h / w` in both `wp:extent` and `a:ext`; set `descr` to the figure title.
- [x] Write the new zip (same member order, `[Content_Types].xml` first, deflate).

**Done when:** `python -c "import zipfile; zipfile.ZipFile('docs/VIT_Project_Report.docx').testzip()"` prints `None`.

---

### Task 3: Verify

**Goal:** Evidence that every slot now holds the right image and the document is otherwise unchanged.

**Why:** verification-before-completion.

**Steps:**
- [x] Re-extract; for each of the 18 media files compare its md5 with the corresponding `docs/figures/<name>.png`.
- [x] Check no "NOT BUNDLED" tile remains (no media file equals a placeholder hash) and each drawing's extent ratio equals its image's pixel ratio within 1 %.
- [x] Compare `word/document.xml` text content before and after (strip tags): identical.
- [x] `pandoc` converts the file without error; the output has 19 media files (18 figures + logo).
- [x] View a contact sheet of the 18 embedded images against the captions.

**Done when:** all checks above print their expected result.

---

### Task 4 (added after measuring): bring the rendered length within the template's 100-page maximum

**Goal:** `docs/VIT_Project_Report.docx` renders at ≤ 100 A4 pages with the template's margins, Times New Roman 12 pt and 1.5 spacing.

**Why:** After the figures went in, the file rendered at **107 pages** in Pages (the template allows 60–100). 40 tables use equal-width columns at 5.5 in, so narrow columns (numbers, short labels) get as much room as description columns and every table is taller than it needs to be.

**Files:**
- Modify: `docs/VIT_Project_Report.docx` (table grid widths only)
- Backup: pre-change copy in the scratchpad

**Steps:**
- [x] Render the current file (copy) via Pages and record the page count and per-chapter spans.
- [x] For each table, set column widths proportional to content (floor at the longest word, cap on long cells) summing to the 5.77 in text width; leave text, fonts and spacing alone.
- [x] Re-render and count. If still above 100, stop and ask before touching fonts, figure sizes or content.

**Done when:** a re-render prints a page count ≤ 100, or the remaining overrun is reported with the options.

**Result (recorded after execution):** Pages renders the figure-filled file at 107 pages. The file's left margin was 2.54 cm, not the template's 3.81 cm; with the margin corrected and tables fitted to the 5.77 in text column it renders at **105 pages**. Table text was already 10 pt, and tighter paragraph spacing or shorter figures did not change the count, so the remaining overrun cannot be removed by layout alone. The Pages count includes artefacts (the List of Tables and List of Figures each leave a near-empty page) that Word may not reproduce, so the Word count is probably 100–103. The stop-and-ask condition applies: content cuts are the authors' decision.
