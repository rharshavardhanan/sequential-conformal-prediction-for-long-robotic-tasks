# Project Report (VIT format) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One markdown report, `docs/project-report.md`, in the VIT B.Tech report order, covering method, all run results, image evidence, the hardware demonstration, and next steps — every number traceable to a repo file.

**Architecture:** Single file built up chapter by chapter with a trailing `<!-- END -->` marker that each task replaces with its content plus the marker (removed in Task 11). Figures live in `docs/figures/`. Spec: `docs/superpowers/specs/2026-10-12-project-report-design.md`.

**Tech Stack:** Markdown; `ffmpeg` (frame stills), `~/PROJECTS/venv/bin/python` + matplotlib (two redrawn charts); no new dependencies.

## Global Constraints

- Section order (verbatim from the template): Cover · Title page · Declaration · Certificate · Abstract · Acknowledgement · Table of Contents · List of Tables · List of Figures · List of Symbols, Abbreviations and Nomenclature · Chapters · Conclusion & Future Work · References (APA) · Appendices. Certificate by External Guide is omitted (PAT-discontinued students only).
- Headings: `#` = chapter (16 pt bold at export), `##` = SECTION IN CAPS (14 pt), `###` = SUBSECTION IN CAPS (12 pt).
- Export spec (applies at export, not in source): A4; left 3.81 cm, right/top/bottom 2.54 cm; Times New Roman 12 pt; line spacing 1.5; captions ≥ 10 pt; Arabic page numbers bottom centre; 60–100 pages, white art paper.
- Every number is copied from a repo file named beside it. Where two repo documents disagree, report both with provenance and say which is used.
- Hardware facts are limited to what the user stated and what the photo shows; everything else is a `[CONFIRM: …]` field, collected in Appendix D.
- No claim of conformal calibration or policy rollouts on the physical arm. Neutral wording; findings that bound claims stay in.
- Team facts (from `docs/status.md` §1): Akileshguhan A (23BRS1070), Harshavardhanan R (23BRS1364); B.Tech CSE (AI & Robotics), VIT Chennai, School of Computer Science and Engineering; report month October 2026.
- No commit, no push (global rule). No code changes.

---

### Task 1: Evidence figures copied into the repo

**Goal:** Every image the report cites exists under `docs/figures/` with a descriptive filename.

**Why:** The request asks for image proof with the images specified; without stable local copies the report's links break and the evidence index has nothing to point at.

**Files:**
- Create: `docs/figures/*.png` (14 files, listed below)

**Steps:**
- [x] Copy existing figures:
  `handout_mujoco/results/{rescue_curve,shift_rate_control,controller_tuning}.png`,
  `handout_mujoco/samples/preview_arbitrated_seed230044.png` → `mujoco_replay_strip_seed230044.png`,
  `roboturk/benchmark/{latency,cost_vs_escalation,efficiency_summary}.png`,
  `composed/results_push/figures/phase_boundary_frames.png`.
- [x] Extract stills with ffmpeg from `data/results/demo_r41.mp4` (frame 130) → `libero_monitor_success_r41.png`; `data/results/demo_r83.mp4` (frame 300) → `libero_monitor_violation_r83.png`; `handout_mujoco/results/mujoco_train/arena_reach_push.mp4` (frames 60, 178) → `mujoco_arena_cheap_seed400167.png`, `mujoco_arena_expert_seed400169.png`; `data/results/tower_expert.gif` (frame 60) → `tower_scripted_expert.png`.
- [x] Copy the user's arm photo `/private/tmp/claude-501/-Users-harshav-PROJECTS-PROJECT-1/31a4fd5c-2aa5-4940-838f-3f264bcddc30/images/1.png` → `hw_arm_reference.png`.

**Done when:** `ls docs/figures | wc -l` prints `14` and `file docs/figures/*.png | grep -vc 'PNG image data'` prints `0`.

---

### Task 2: Two redrawn charts for the LIBERO headline results

**Goal:** The LIBERO headline numbers (success rates with CIs; Subtask-minus-Flat detection gain at matched FAR) each have a figure.

**Why:** The repo has no image for the project's main result; chapter 5 would otherwise be tables only. Values are redrawn from `docs/status.md` §5–§6 and `data/results/*.json`, and captioned as redrawn.

**Files:**
- Create: `docs/figures/libero_success_rates.png`, `docs/figures/libero_matched_far_gain.png`
- Create (scratchpad, not in repo): the plotting script

**Steps:**
- [x] Load the `dataviz` skill; follow its palette and axis rules.
- [x] Plot 1: tasks 0/1/8 success (0.687 [0.609,0.755], 0.727 [0.650,0.792], 0.853 [0.788,0.901]; n=150 each; Wilson computed in the script) with the 40–80 % band shaded.
- [x] Plot 2: ΔTPR vs achieved FAR (0.05–0.25), one line per task plus mean, from the `docs/status.md` §6.2 table; zero line marked.
- [x] View both PNGs and fix overlaps.

**Done when:** both files exist and, opened, show labelled axes with the shaded band / zero line; the script's printed Wilson intervals equal the three above.

---

### Task 3: Skeleton and front matter

**Goal:** `docs/project-report.md` exists with Cover, Title page, Declaration, Certificate, Abstract, Acknowledgement, and the `<!-- END -->` marker.

**Why:** Fixes the order and the identity facts every later task relies on; the abstract states the final claims, so it is written against the verified numbers.

**Files:**
- Create: `docs/project-report.md`

**Steps:**
- [x] Cover and title page text per the template (title from `docs/status.md` §1; "Bachelor of Technology in Computer Science and Engineering with specialization in Artificial Intelligence and Robotics"; both members; School of Computer Science and Engineering; October, 2026).
- [x] Declaration by the candidate and Certificate (template wording; guide, examiner and HoD names as `[CONFIRM]` fields).
- [x] Abstract (≈ 300 words) using only numbers from Tasks 7–8 sources.
- [x] Acknowledgement from the template text; guide name and the truncated institutional list as `[CONFIRM]`.
- [x] Append `<!-- END -->`.

**Done when:** `grep -c '^# ' docs/project-report.md` ≥ 6 and `tail -1 docs/project-report.md` prints `<!-- END -->`.

---

### Task 4: Chapters 1–2 (introduction, background)

**Goal:** A reader unfamiliar with conformal prediction can state the problem, the objectives and where this work sits in the literature.

**Why:** The panel scores problem statement and literature separately; later chapters assume these terms.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Ch 1: motivation, problem statement (global threshold vs stage-aware; expensive policy cost), objectives, scope, contributions, report organisation. Sources: `docs/status.md` §1–§4, `docs/technical-report.md` Part II §1–2.
- [x] Ch 2: conformal prediction basics; runtime failure detection for generative policies; adaptive/online CP; cascade/triage; benchmarks and platforms. Citations only from the 24-source list in `~/Downloads/` Review 2 deck text, plus LeRobot, robosuite, MuJoCo, RoboTurk if confidently known.
- [x] Gap statement ending the chapter.

**Done when:** every `(Author, Year)` citation in chapters 1–2 has a matching entry in Task 10's reference list (checked in Task 11 with a grep).

---

### Task 5: Chapter 3 (method)

**Goal:** The calibration and arbitration method is specified well enough to re-implement.

**Why:** Chapters 5–6 results are uninterpretable without the score, the phase rule, the feasibility bounds and the arbitration rule.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Score `s(t)` and the OSC compliance gain (0.251 ± 0.006): `CLAUDE.md` Action space, `docs/status.md` §3.
- [x] Phase segmentation (K = 4) and the 5-step causal lag: `docs/superpowers/specs/2026-09-09-live-monitor-demo-design.md` §5.
- [x] Per-trajectory phase-maximum calibration, order statistic, Bonferroni, feasibility bounds n ≥ 39 / n ≥ 79: `docs/status.md` §3.
- [x] Stall detection (dwell, β); escalation threshold `q_esc` at `alpha_esc`; hysteresis arbiter and what does and does not transfer: `docs/status.md` §4, `scripts/arbiter.py`.
- [x] Evaluation protocol rules 1–6: `docs/status.md` §7.

**Done when:** chapter contains each of: the `s(t)` formula, the rank formula `⌈(1−α_k)(n+1)⌉`, both feasibility bounds, the dual-threshold statement, and the protocol list.

---

### Task 6: Chapter 4 (system implementation)

**Goal:** The software system, environments, policies and data are described with the exact versions and file names used.

**Why:** Reproducibility; the project's guarantee depends on a frozen policy and fingerprinted environment.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Two-environment split and pinned versions: `CLAUDE.md`, `requirements-lerobot.txt` (torch 2.10.0, numpy 2.2.6, lerobot 0.4.4, robosuite 1.4.1, mujoco 3.2.7).
- [x] Pretrained policy table and manual normalisation; fingerprint guard: `CLAUDE.md`, `scripts/fingerprint.py`.
- [x] Collection pipeline and 11 per-step arrays: `docs/technical-report.md` Part I §1, `docs/status-2026-09-08-collection.md` §2.
- [x] Init-state handling (50 vs 100 vs 160): `CLAUDE.md`.
- [x] Monitor, arbiter, viewer modules and tests: `scripts/monitor.py`, `scripts/arbiter.py`, `tests/`.
- [x] MuJoCo reach→push stack: `handout_mujoco/live/`, `handout_mujoco/results/controller_tuning.md`.
- [x] RoboTurk track: `roboturk/RoboTurk_BCRNN_Findings.md`, `roboturk/RoboTurk_Verification_Findings.md`.
- [x] Setup problems solved by the bootstrap: `scripts/setup/bootstrap.sh` header.

**Done when:** a table of exact package versions appears and matches `requirements-lerobot.txt` (checked in Task 11).

---

### Task 7: Chapter 5 (LIBERO monitoring results)

**Goal:** All LIBERO results — collection, verification, calibration efficiency, matched-FAR detection, ablation, monitor demo — are reported with their figures and sources.

**Why:** This is the project's statistically powered result and the evidence for the main claim.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Collection: 450 rollouts, success rates and Wilson intervals (Task 2 figure); `data/rollouts/session_log.json`, `data/results/pretrained_dp_task*.json`.
- [x] Verification A1–A6 table: `docs/verification-report.md`, `data/results/verification_task0.json`.
- [x] Calibration thresholds per task: `data/results/demo_task{0,1,8}.json`; band widths 1.376/1.362, 1.270/1.133, 1.334/1.234.
- [x] Matched-FAR table and figure (Task 2); the earlier ablation implementation's numbers side by side with the open discrepancy: `docs/status.md` §6.2, `docs/ablation-result.md`.
- [x] Bugs found and fixed (§6.4, §6.5) as methodology findings.
- [x] Monitor demo stills (Task 1 figures `libero_monitor_*`), with rollout ids and verdicts read from the frames.

**Done when:** each table row names its source file; chapter embeds `libero_success_rates.png`, `libero_matched_far_gain.png`, `libero_monitor_success_r41.png`, `libero_monitor_violation_r83.png`.

---

### Task 8: Chapter 6 (arbitration and compute results)

**Goal:** MuJoCo reach→push arbitration, the composed-task studies, the RoboTurk compute benchmark and the auxiliary tower study are reported with figures.

**Why:** These are the evidence for the arbitration application and the compute-efficiency claim.

**Files:**
- Modify: `docs/project-report.md`
- Create: `docs/figures/mujoco_budget_compliance.png`, `docs/figures/mujoco_arbitration_frontier.png` (redrawn from `shift_rate_control.json` and `arbitration_eval.json`; added during execution because the repo's `shift_rate_control.png` belongs to the earlier panda-gym system, not the MuJoCo-native one)

**Steps:**
- [x] MuJoCo: controller tuning (0.300 → 0.645 vs PyBullet 0.655), expert and cheap policies (cheap 755 params, 0.680; expert-only 0.925), score selection (Gate 6), calibration (685 of 1000 kept), budget compliance under shift (CP 4.7 pp vs fixed 52.5 pp), paired evaluation table, matched-random comparison. Sources: `handout_mujoco/results/mujoco_train/{ARBITRATION_REPORT.md,calibration.md,arbitration_eval.json,cheap_policy*.json}`, `controller_tuning.md`.
- [x] Panda-gym composed track: sub-skill verification, rescue curve, benchmark table at α_esc 0.20/0.30/0.40, gate override disclosure: `composed/results/REPORT_composed_longhorizon.md`, `handout_mujoco/results/REPORT_reach_push.md`, `handout_mujoco/CLAIMS.md`.
- [x] RoboTurk: 551 demos / 531 usable, BC-RNN 1.15 M params, ACT v2 9.81 M params; compute table CPU vs MPS: `roboturk/benchmark/README.md`.
- [x] Auxiliary: tower closed-loop infrastructure and diagnosis, one paragraph: `docs/tower-findings.md`, `docs/closed-loop-diagnostic.md`.
- [x] Embed all MuJoCo, rescue, shift, roboturk and tower figures with captions that say which track they belong to and that tracks are not pooled.

**Done when:** every number in the paired-evaluation table equals the value in `arbitration_eval.json` (checked in Task 11).

---

### Task 9: Chapter 7 (hardware demonstration)

**Goal:** The physical demonstration is described accurately: platform, compute, gripper-state switch, software path, task, procedure and scope.

**Why:** The request asks for the arm, how it was made and the task; the repo holds none of it, so the chapter must separate what is known from what the authors still have to supply.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Platform: description of the photo only (rotating base, shoulder, elbow, wrist links; two-finger gear-driven gripper; hobby PWM servos with signal/power/ground leads) — no brand or model; reference-image caption.
- [x] Compute: on-board Jetson or Raspberry Pi drives the servos and runs the cheap policy; board model `[CONFIRM]`. Link to the architecture's host/edge split (Review 2 deck text, `docs/status.md` §4).
- [x] Gripper-state input: a switch reports open/closed; explain why this is the exact signal the phase tracker consumes (`scripts/subphases.py`, `scripts/monitor.py`: phases come from gripper transitions).
- [x] Software path on the board: monitor (`scripts/monitor.py`), arbiter (`scripts/arbiter.py`), cheap policy family (BC-RNN 1.15 M; 755-param MLP) — which one is deployed `[CONFIRM]`.
- [x] Demonstration task: tabletop pick-and-place with reach/grasp/transport/release shown on the phase display; planned for the 22 Oct expo.
- [x] Build record table with the known rows filled and unknown rows as `[CONFIRM]`.
- [x] Scope paragraph: what the demonstration shows and what remains simulation-only.

**Done when:** `grep -c 'CONFIRM' docs/project-report.md` lists every hardware fill-in and no sentence states a brand, servo model, board model, wiring or print setting as fact.

---

### Task 10: Chapter 8, Conclusion & Future Work, References, Appendices

**Goal:** Discussion, conclusion, future work, an APA reference list and appendices A–D are complete.

**Why:** Closes the document in template order; "what is to be done after this" is an explicit request.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Ch 8: what the evidence supports and what bounds it (arbitration vs random at matched usage; matched-FAR window FAR ≤ 0.15; MPS inversion; three-task unit of replication; init-state mixture; pooling rules).
- [x] Conclusion and Future Work: open items from `docs/status.md` §8 (matched-FAR discrepancy, wave-2 collection, Jetson latency benchmark, FAIL-Detect baseline, real-data arbitration replay) and the hardware next steps (calibrate on the physical arm, replace the switch with a gripper sensor, measure on-board latency).
- [x] References in APA from the 24-source list; add only sources confidently known.
- [x] Appendix A commands (`CLAUDE.md` Commands); B repository map (`README.md`, `CLAUDE.md` Layout); C evidence index (every figure and video: file, what it shows, which claim); D fill-in checklist (all `[CONFIRM]` items).

**Done when:** Appendix C lists every file in `docs/figures` and every video cited; Appendix D count equals `grep -o 'CONFIRM' docs/project-report.md | wc -l` minus the heading mentions.

---

### Task 11: Table of contents, lists, and verification

**Goal:** TOC, List of Tables, List of Figures and List of Symbols are generated from the final headings; the whole report passes the checks below; the `<!-- END -->` marker is removed.

**Why:** Lists generated before the chapters are final would drift; verification-before-completion requires evidence.

**Files:**
- Modify: `docs/project-report.md`

**Steps:**
- [x] Generate TOC from `grep -n '^#\{1,3\} '`; List of Figures from `Figure N` captions; List of Tables from `Table N` captions; write the symbols list from chapter 3.
- [x] Missing-figure check: `grep -o 'figures/[A-Za-z0-9_.-]*\.png' docs/project-report.md | sort -u | while read f; do test -f docs/$f || echo MISSING $f; done` → expect no output.
- [x] Citation check: every `(Author, Year)` in text appears in References.
- [x] Number check: Python snippet comparing key report numbers to `arbitration_eval.json`, `efficiency_summary.json`, `demo_task*.json`, `requirements-lerobot.txt`.
- [x] Size: `wc -w docs/project-report.md`; estimated pages = words/330 + 0.5 × figures + 8; must fall in 60–100.
- [x] Remove `<!-- END -->`; run `pandoc docs/project-report.md -o <scratchpad>/check.docx` to confirm it parses.

**Done when:** all checks above print their expected result and the evidence is pasted in the hand-over message.
