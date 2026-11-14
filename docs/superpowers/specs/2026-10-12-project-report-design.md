# Design — Project report (VIT B.Tech format) for PROJECT_1

**Date:** 2026-10-12 · **Status:** approved by the user's answers (arm: photo description only; compute: on-board
Jetson/Pi; format: markdown following the VIT report template)

## 1. What and why

One markdown report covering the whole project — method, every run result, image evidence, the hardware
demonstration, and what comes next — laid out in the VIT Chennai project-report order so it can be exported
to the submission format. The repo's facts are scattered across ~25 documents written at different dates; the
report is the single place a reader (R3 panel, expo visitor, external examiner) can follow them in order.

**Constraint the request did not state, found while exploring:** the repo contains no hardware code, no BOM and no
build log. The arm, the on-board board and the gripper switch exist only in the authors' heads. So the
hardware chapter can state what the user told us and what the photo shows, and must mark everything else as a
fill-in — it cannot be invented.

## 2. Audience

R3 review panel and 22 Oct expo visitors reading a submitted report. Plain language first, numbers second,
every number traceable to a repo file.

## 3. Deliverable

| item | path |
|---|---|
| the report | `docs/project-report.md` |
| figures (copied or redrawn) | `docs/figures/*.png` |

Section order (template, verbatim): Cover · Title page · Declaration · Certificate · [External guide certificate —
omitted, applies only to PAT-discontinued students] · Abstract · Acknowledgement · Table of Contents · List of
Tables · List of Figures · List of Symbols, Abbreviations and Nomenclature · Chapters · Conclusion & Future Work ·
References (APA) · Appendices.

Template rules markdown can carry: `#` chapter, `##` SECTION in CAPS, `###` SUBSECTION in CAPS; Arabic page numbers
and the A4 / margin / Times New Roman 12 pt rules apply at export, not in the source.

## 4. Chapters

1. Introduction · 2. Background and literature · 3. Method · 4. System implementation · 5. Experiments and
results — LIBERO monitoring · 6. Experiments and results — arbitration and compute · 7. Hardware demonstration ·
8. Discussion.  Then Conclusion & Future Work, References, Appendices (commands, repository map, evidence index).

## 5. Honesty rules (these decide what goes in)

- Every number is copied from a repo file named beside it. Where two repo documents disagree, report both with
  provenance and say which is used.
- Hardware: only what the user stated (3D-printed servo arm; on-board Jetson/Pi runs the cheap policy; a switch
  supplies gripper open/closed) and what the photo shows. Anything else is a `[CONFIRM: …]` field, listed once at
  the end of the report's appendix and in the hand-over message.
- The switch is described plainly as the gripper-state input that feeds the phase tracker. The report does not
  claim conformal calibration or policy rollouts on the physical arm; it states that scope in one paragraph.
- Language is neutral ("scope", "findings that bound the claim"), never apologetic. Findings that bound the
  claims stay in — the repo's own `CLAIMS.md` requires them to travel with any presentation, and a viva panel will
  find them.
- The arm photo carries a ✦ watermark, so it is captioned as a reference image, with a prompt to replace it with
  the authors' own photograph.

## 6. Non-goals

No new experiments. No code changes. No export to .docx/.pdf (offered, not done). No commit (global rule).
