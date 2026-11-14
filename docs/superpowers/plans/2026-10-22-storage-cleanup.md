# Storage cleanup: code to GitHub, data to Drive, delete regenerables

Decisions (brainstorm, 2026-10-22): GitHub gets code + small results only. Large
data goes to Google Drive (harshavardhananrajasimhan@gmail.com) via Drive for
desktop. Local deletion limited to regenerables and junk. Local copies of the
moved data are deleted only on explicit user confirmation after verification.

## Task 1: Drive for desktop installed and signed in
- **Goal:** `~/Library/CloudStorage/GoogleDrive-harshavardhananrajasimhan@gmail.com` exists.
- **Why:** nothing else can move multi-GB files; the Drive connector is base64-inline only.
- **Files:** none (system install).
- **Steps:** `brew install --cask google-drive`; user signs in as the target account.
- **Done when:** `ls ~/Library/CloudStorage` lists a `GoogleDrive-harshavardhananrajasimhan@gmail.com` entry.

## Task 2: Copy kept data to Drive and verify
- **Goal:** every kept folder exists in Drive `PROJECT_1-data/` with matching file counts and bytes.
- **Why:** these ~11 GB exist only on this laptop; the rollouts and hardware recordings are not reproducible cheaply.
- **Files:** data/{rollouts,checkpoints,tower}, roboturk, HARDWARE_RESULTS, test_videos, bert, composed, handout_mujoco.
- **Steps:** rsync each folder to the mount; compare `find | wc -l` and byte totals per folder.
- **Done when:** per-folder file count and size match source; script prints no diff.

## Task 3: Commit and push code + small results
- **Goal:** origin/mujoco-arbitration contains the untracked code/docs and small result JSONs.
- **Why:** code is the only thing GitHub should be the source of truth for.
- **Files:** .gitignore, docs/superpowers/plans/*, gripper (size-checked), data/results/*.json (<5 MB each), scripts/tower_modes.py deletion.
- **Steps:** add ignore rules for *.mov/*.mp4/HARDWARE_RESULTS; allow data/results JSON; check no staged file >5 MB; commit; push (no force).
- **Done when:** `git ls-remote origin mujoco-arbitration` equals local `git rev-parse HEAD`.

## Task 4: Delete regenerables and junk
- **Goal:** about 25 GB freed with nothing irreplaceable removed.
- **Why:** this is the storage reduction the user asked for.
- **Files:** third_party, .venv, .venv-lerobot, ~/.cache/huggingface, .trash, *.zip, caches.
- **Steps:** after Task 3 is confirmed and zip contents are confirmed duplicate, list targets with `du`, then `rm -rf`.
- **Done when:** `du -sh .` on the project drops by roughly 20 GB.

## Task 5 (only on user confirmation): remove local copies of moved data
- **Done when:** Task 2 verification re-run passes and the user has approved deletion.
