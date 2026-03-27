# Plan: Build dynamicemb 0.0.4 wheel, validate, publish

## Context
The `upstream-3pick` branch of `recsys-examples-2` now has a new commit (`7923e44` — gradient
clipping from NVIDIA upstream #223) inserted before the admission strategy commit.
The installed wheel in CI (and `setup_env.sh`) is still `0.0.4.3`. We need to rebuild, validate,
publish, and wire everything up.

Current `setup.py` version: `0.0.1` (bumped manually per release; previous published wheel was `0.0.3`).
Current HEAD of `upstream-3pick` after rebase: obtained at build time with `git rev-parse --short HEAD`.

---

## Step 1 — Bump version & build wheel

**File to edit:** `~/recsys-examples-2/corelib/dynamicemb/setup.py` line 163
Change `version="0.0.1"` → `version="0.0.4"`

**Build command:**
```bash
cd ~/recsys-examples-2/corelib/dynamicemb
bash build.sh hkv2
```
This runs `MAX_JOBS=$(nproc) conda run --no-capture-output -n hkv2 ... python setup.py bdist_wheel`.
Output: `dist/dynamicemb-0.0.4-cp311-cp311-linux_x86_64.whl`

---

## Step 2 — Install new wheel into hkv2

```bash
conda run -n hkv2 pip install ~/recsys-examples-2/corelib/dynamicemb/dist/dynamicemb-0.0.4-cp311-cp311-linux_x86_64.whl \
    --no-deps --force-reinstall
```

---

## Step 3 — Run unit tests in recsys-examples-2

```bash
cd ~/recsys-examples-2/corelib/dynamicemb
conda run -n hkv2 bash test/unit_test.sh
```
Runs 11 tests. All must pass before proceeding.

---

## Step 4 — Run goatee integration tests

Run with one GPU at a time (per user instruction):
```bash
cd ~/goatee
conda run -n hkv2 bash -c "export CUDA_VISIBLE_DEVICES=0; \
  python src/train.py \
    experiment=uum/uum_v2_clan_de_pbtxt_experiment \
    logger=csv \
    ++should_skip_retry=True \
    ++extras.print_config=False \
    ++trainer.max_steps=100 \
    ++trainer.limit_val_batches=0 \
    ++test=null"
```
For incremental training (needs a `last_run_dir` from the base training output above):
```bash
conda run -n hkv2 bash -c "export CUDA_VISIBLE_DEVICES=0; \
  python src/train.py \
    experiment=uum/uum_v2_clan_de_pbtxt_experiment \
    logger=csv \
    ++should_skip_retry=True \
    ++extras.print_config=False \
    ++trainer.max_steps=100 \
    ++trainer.limit_val_batches=0 \
    ++test=null \
    ++paths.last_run_dir=<output_dir_from_base_run>"
```

---

## Step 5 — Run make test-full in goatee

```bash
cd ~/goatee
conda run -n hkv2 bash -c "export CUDA_VISIBLE_DEVICES=0; make test-full"
```
All tests must pass.

---

## Step 6 — Push wheel to GCS with commit-tagged name

Get the current short commit:
```bash
SHORT_COMMIT=$(git -C ~/recsys-examples-2 rev-parse --short HEAD)
# e.g. 047ae08
```

The built wheel may be `linux_x86_64`; rename to include commit and use `manylinux_2_34` tag
to match CI expectations (the actual tag depends on what the build produces):

```bash
BUILT_WHL=$(ls ~/recsys-examples-2/corelib/dynamicemb/dist/dynamicemb-0.0.4-*.whl)
GCS_NAME="dynamicemb-0.0.4-${SHORT_COMMIT}-cp311-cp311-manylinux_2_34_x86_64.whl"
gsutil cp "$BUILT_WHL" "gs://bento_training_custom_artifact/wheel/dynamicemb/${GCS_NAME}"
```

---

## Step 7 — Update goatee setup_env.sh + commit

**File:** `~/goatee/scripts/setup_env.sh`

Replace the current dynamicemb block:
```bash
# 0.0.3 is the latest version with support of gradient clipping.
# ...
DE_LIB_NAME=dynamicemb-0.0.3-cp311-cp311-manylinux_2_34_x86_64.whl
gsutil cp gs://bento_training_custom_artifact/wheel/dynamicemb/${DE_LIB_NAME} . && \
    pip install ${DE_LIB_NAME} --no-deps --force-reinstall && \
    rm ${DE_LIB_NAME}
```

With:
```bash
# 0.0.4: adds gradient clipping (upstream #223) + admission strategy (upstream #236)
#   + goatee V2 API compat shims.
# Built from https://github.com/fshhr46/recsys-examples/tree/upstream-3pick @ <SHORT_COMMIT>
DE_GCS_WHL=dynamicemb-0.0.4-<SHORT_COMMIT>-cp311-cp311-manylinux_2_34_x86_64.whl
DE_LIB_NAME=dynamicemb-0.0.4-cp311-cp311-manylinux_2_34_x86_64.whl
gsutil cp gs://bento_training_custom_artifact/wheel/dynamicemb/${DE_GCS_WHL} ${DE_LIB_NAME} && \
    pip install ${DE_LIB_NAME} --no-deps --force-reinstall && \
    rm ${DE_LIB_NAME}
```

Then commit in goatee on `min-freq` branch:
```bash
git add scripts/setup_env.sh
git commit -m "Update dynamicemb wheel to 0.0.4 (gradient clipping + admission strategy)"
```

---

## Critical files
- `~/recsys-examples-2/corelib/dynamicemb/setup.py` — bump version to `0.0.4`
- `~/recsys-examples-2/corelib/dynamicemb/build.sh` — build command reference
- `~/recsys-examples-2/corelib/dynamicemb/test/unit_test.sh` — 11 unit tests
- `~/goatee/scripts/setup_env.sh` — update GCS wheel reference

## Notes
- The GCS filename uses a two-variable pattern (`DE_GCS_WHL` for download, `DE_LIB_NAME` for
  install) so pip always sees a valid 5-part wheel name, while GCS retains the commit for
  traceability.
- If `build.sh` produces `manylinux_2_34_x86_64` directly, skip the rename step.
- The `last_run_dir` for incremental integration test should use the output from the base
  training run in Step 4.
