# Review 3 — hardware slides + demo run sheet

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Project_I_Review3.pptx` carries three hardware slides (platform, sim→hardware mapping, live
handover) after slide 37, an Outline line for them, and a slide-38 row that no longer says hardware is
pending; plus a verified demo run sheet for the review panel.

**Architecture:** Edit the existing deck as OOXML. Clone slide 32 (picture + "Key points" panel) twice and
slide 30 (table + grey note) once with the pptx skill's `add_slide.py`, then overwrite the clones' XML with
generated content that reuses the deck's exact runs (Times New Roman, 26 pt bold titles, `E8E8E8` panels,
`BFBFBF` rules, hard-coded page numbers). Images are stills cut from the user's own hardware recordings.

**Tech Stack:** python3 stdlib (zipfile/re), ffmpeg, pptx skill scripts (`add_slide.py`, `office/validate.py`),
Quick Look (`qlmanage`) for render QA. No new project dependencies.

## Approved design (from brainstorming, 2026-10-21)

- **H1 · Hardware Demonstration — Platform.** Left: arm still (`test_videos/videos_2.2.mov` @ 32 s, side
  camera, carry phase). Right panel: 3D-printed 5-DOF + gripper arm, hobby PWM servos · Raspberry Pi 5, 50 Hz
  control + score loop · PCA9685 PWM driver (I²C) · MPU-6500 IMU gyro on the arm link · gripper microswitch
  (BCM 27) → phase · Pi Camera IMX708 + USB camera, ArUco markers · GPIO 17 escalation flag. Task: ball → cup,
  15 taught keyframes.
- **H2 · Hardware — Same Monitor, Adapted for Hobby Servos.** Table: element | in simulation | on the arm —
  executed motion, score, motion floor, phase signal, calibration, cheap policy, expensive policy, arbiter.
- **H3 · Hardware — Live Arbitration on the Arm.** Viewer screenshot from run `20261020_073104`
  (`videos_2.1.mov` @ 20 s): s = 1.021 > q̂ = 1.000 at 1.6 s, expensive policy took over, GPIO 17 HIGH.
  Panel adds the next stage: per-phase thresholds from ≥ 79 nominal runs; servo position read-back.
- **Outline:** add "Hardware Demonstration" after "Results — Policy Arbitration".
- **Slide 38:** row "Arbitration on the real robotic arm" keeps status *In progress*; evidence becomes
  "runs live on the Pi 5 arm: IMU score, flat threshold (n = 10), handover + GPIO 17 flag; per-phase
  calibration next".

## Global Constraints

- Every number on a slide comes from `~/PROJECTS/arb-viewer/pi_src/arm_ncs_monitor.py`, the Pi's own
  printed output in `test_videos/videos_1.1.mov` / `videos_2.1.mov`, or the deck itself. Nothing from
  `arb-viewer/samples/*.jsonl` — those are `"mock": true`.
- The hardware threshold is a FLAT order statistic over 10 nominal runs (rank 10 of 10 = the sample
  maximum). State it once, as next stage, not as a result.
- Match deck styling exactly; no new fonts, colours or accent bars.
- Do not export to PDF (the user exports it). Quick Look thumbnails are for QA only.
- Original deck is tracked in git; also keep a byte copy in the session scratchpad before overwriting.
- No commit (user's global rule).

---

### Task 1: Slide images

**Goal:** Two 16:9 JPEGs exist that show the arm (H1) and the live handover screen (H3), legible at slide size.
**Why:** Without real images the hardware slides are text-only and carry no evidence of the build.
**Files:** Create `<scratch>/build/hw_arm.jpg`, `<scratch>/build/hw_live.jpg` (scratch = session scratchpad).
- [x] Step 1: cut side-camera crop `545x306+4+237` from `videos_2.2.mov` at 32 s, brighten slightly, lanczos ×2.
- [x] Step 2: cut viewer region (drop browser chrome, y ≥ 100) from `videos_2.1.mov` at 20 s.
- [x] Step 3: view both.
**Done when:** `ffprobe` reports both at ~1.78 aspect and both images, viewed, show the arm / the
"Expensive policy took over" banner with GPIO 17 HIGH.

### Task 2: Build the deck

**Goal:** The deck has 45 slides, with H1/H2/H3 at positions 38–40, the Outline line, the slide-38 row edit,
and page numbers 41–44 on the shifted slides.
**Why:** This is the deliverable the user asked for; a deck that fails validation may not open in PowerPoint.
**Files:** Modify `Project_I_Review3.pptx`; Create `<scratch>/build_hw_slides.py`.
- [x] Step 1: back up the deck to scratch; unpack.
- [x] Step 2: `add_slide.py` clones (32, 30, 32) after slide 37; reorder `sldIdLst` to H1, H2, H3.
- [x] Step 3: write generated XML for each clone, add media + rels.
- [x] Step 4: Outline paragraph, slide-38 evidence cell, page numbers +3 on old 38–41.
- [x] Step 5: zip, run `office/validate.py --original`.
**Done when:** `validate.py Project_I_Review3.pptx --original <backup>` passes and `markitdown` shows the
three new titles at slides 38–40 and "Hardware Demonstration" in the Outline.

### Task 3: Visual QA

**Goal:** New and edited slides render without overflow, overlap or clipped text.
**Why:** Text overflow is the most common deck defect and is visible to the panel.
**Files:** none (renders in scratch).
- [x] Step 1: Quick Look render slides 2, 38, 39, 40, 41.
- [x] Step 2: fix any defect in the generator, rebuild, re-render only changed slides.
**Done when:** the five renders, viewed, show no text outside its box and no element collisions.

### Task 4: Demo run sheet

**Goal:** The user has an ordered, verified list of what to show and the exact commands, with backups.
**Why:** A live demo that fails in front of the panel costs more than any slide.
**Files:** none.
- [x] Step 1: `demo_monitor --replay … --list` confirms r41 SUCCESS/no alarm, r83 alarm @ 40, r76 fail.
- [x] Step 2: `handout_mujoco/run_policy.sh --save <scratch>/arb_check.mp4` exits 0 and writes the file.
**Done when:** both commands' output is quoted in the final message.
