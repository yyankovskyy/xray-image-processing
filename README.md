# xray-pipeline

A configurable, command-line pipeline for multi-label chest X-ray classification
with DenseNet121 transfer learning, class-imbalance-aware weighted loss,
AUROC/ROC evaluation, and Grad-CAM visualization.

This started as a Jupyter notebook assignment (chest X-ray diagnosis with the
ChestX-ray8 dataset, "AI for Medical Diagnosis") and has been reorganized into
a reusable, config-driven package: no notebook, no `public_tests`/`test_utils`
scaffolding, no hard-coded paths — everything is controlled through a single
`config.yaml` and run from the command line.

## What it does

Given chest X-ray images and CSVs labeling 14 pathologies (Cardiomegaly,
Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis,
Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema, Consolidation),
the pipeline:

1. **Checks for patient-level data leakage** between train/valid/test splits.
2. **Builds Keras image generators** — training data normalized per-batch;
   validation/test data normalized using statistics learned from the training
   set (so no test-set information leaks into the model).
3. **Computes class-imbalance weights** and trains (or loads a pretrained)
   DenseNet121 model with a custom weighted binary cross-entropy loss.
4. **Predicts and evaluates** — per-label AUROC scores and a combined ROC plot.
5. **Generates Grad-CAM heatmaps** highlighting the image regions that drove
   the model's predictions for its top-performing labels.

## Repository structure

```
xray-pipeline/
├── config.yaml          # full-scale pipeline parameters
├── config.small.yaml    # lightweight smoke-test config (see below)
├── main.py              # CLI entrypoint (make-dummy-data / check-leakage / train / predict / evaluate / gradcam / run-all)
├── pipeline_config.py   # YAML config loading + logging setup
├── data_utils.py        # patient-leakage check + generator construction
├── dummy_data.py        # synthetic dataset generator for the smoke test
├── losses.py            # class frequencies + weighted binary cross-entropy loss
├── model_build.py       # DenseNet121 model construction
├── evaluate.py          # prediction + ROC/AUROC
├── gradcam.py           # Grad-CAM heatmap generation
├── requirements.txt
├── data/nih/            # <- you supply: CSVs + images-small/ (not in the repo)
├── models/nih/          # <- you supply: pretrained_model.h5 (optional)
└── outputs/             # <- created automatically: predictions, plots, gradcam images
```

`data/`, `models/`, and `outputs/` are runtime locations, not source code —
`data/` and `models/` hold input you supply (see below), and `outputs/` is
created automatically to hold results.

## Installation

```bash
git clone <your-repo-url> xray-pipeline
cd xray-pipeline
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
export TF_USE_LEGACY_KERAS=1   # required every time — see note below; put it in your shell profile
```

`requirements.txt` installs `tensorflow` (whatever version your platform
resolves) alongside `tf-keras`, Google's official Keras-2 compatibility
package — needed because TF 2.16+ defaults to Keras 3, which dropped
`ImageDataGenerator` (used in `data_utils.py`). Setting
`TF_USE_LEGACY_KERAS=1` before Python imports `tensorflow` makes
`tensorflow.keras` resolve to Keras 2 (via `tf-keras`) instead. It's
harmless to set even if you land on TF<2.16 (the variable is simply
ignored there), so just always set it — don't try to figure out which TF
version you got first.

If `pip install -r requirements.txt` still fails outright (not a version
warning, an actual "no version found" error) or the pipeline crashes with
`illegal hardware instruction`, see "Fixing TensorFlow/Keras install
issues" below.

### Fixing TensorFlow/Keras install issues

**"No matching distribution found for tensorflow..." version errors.**
TensorFlow only ships wheels for each Python version starting at a certain
TF release (e.g. no `tensorflow<2.11` wheel exists for Python 3.11). See
what your platform actually offers:

```bash
pip index versions tensorflow
```

`requirements.txt` no longer pins an upper bound on `tensorflow` — whatever
your platform resolves is fine, because `tf-keras` + `TF_USE_LEGACY_KERAS=1`
(see Installation above) handles the Keras-2-vs-3 difference regardless of
which TF version you land on.

**Process crashes with `illegal hardware instruction` / `Illegal
instruction: 4`, no Python traceback, or the import hangs indefinitely.**
This is a `SIGILL` (or, for a hang, a CPU emulation layer struggling) —
almost always caused by running an **x86_64 TensorFlow build on Apple
Silicon hardware through Rosetta 2 translation**. Rosetta doesn't fully
support the AVX-512 (and has gaps in some AVX2) instructions TensorFlow
uses, which can manifest as either an immediate crash or a hang depending
on which code path gets hit. Check:

```bash
uname -m                        # x86_64 here...
sysctl -n hw.optional.arm64      # ...but 1 here = Apple Silicon under Rosetta
```

If that's the case, don't fight the translation layer — create a **native
arm64** environment instead:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n xray-pipeline python=3.10 -y
conda activate xray-pipeline
conda config --env --set subdir osx-arm64
pip install -r requirements.txt
export TF_USE_LEGACY_KERAS=1
```

(If you truly are on an Intel Mac and `hw.optional.arm64` isn't `1`, then
check for AVX support instead — `sysctl -a | grep machdep.cpu.features`
should list `AVX`/`AVX2`. If it's genuinely missing, no TensorFlow build
made in the last several years will run natively; use a cloud environment
like Google Colab instead.)

**`ModuleNotFoundError` for `pandas`, `cv2`, `sklearn`, etc.** These aren't
part of TensorFlow — make sure `pip install -r requirements.txt` actually
completed for the *whole* file, not just a subset. Re-running
`pip install -r requirements.txt` is always safe; pip skips anything
already satisfied.

**`ModuleNotFoundError: No module named 'cv2'` specifically, right after a
NumPy downgrade.** If you installed `tensorflow` separately before running
`pip install -r requirements.txt`, and it downgraded an existing NumPy 2.x
to `<2.0`, an already-installed `opencv-python` built against NumPy 2.x's
ABI can end up broken. `requirements.txt` pins `opencv-python<4.10`
specifically to keep it on a NumPy-1.x-compatible build; if you still hit
this, force a clean reinstall:
```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install "opencv-python<4.10"
```

**General advice regardless of which issue you're hitting:** use a
**dedicated virtual environment** for this project, not a large
pre-existing one. An environment already carrying many unrelated packages
(gRPC tooling, telemetry libraries, etc.) multiplies the chance of exactly
these binary/ABI conflicts.

## Data & model weights

Not included in the repo — you need to supply:

- `data/nih/train-small.csv`, `valid-small.csv`, `test.csv` — image filenames,
  per-pathology binary labels, and a `PatientId` column.
- `data/nih/images-small/` — the corresponding X-ray images.
- `models/nih/pretrained_model.h5` (optional) — a full model checkpoint
  (backbone + head) to load for inference/evaluation/Grad-CAM without
  training from scratch.

The full ChestX-ray8 dataset (108,948 images) is publicly available from the
[NIH Clinical Center release](https://nihcc.app.box.com/v/ChestXray-NIHCC).

If your files live somewhere else, update the paths in `config.yaml` — nothing
is hard-coded in the source modules.

## Configuration

Every tunable parameter lives in `config.yaml`. To run a different experiment,
copy the file, edit the copy, and pass it via `--config`.

| Section       | Controls |
|---------------|----------|
| `data`        | CSV paths, image directory, column names |
| `labels`      | The 14 pathologies (order = model output order) |
| `image`       | Target size, batch size, seed, normalization sample size |
| `model`       | Backbone weights, pretrained checkpoint, optimizer, loss epsilon |
| `training`    | Steps/epoch, validation steps, epochs, where to save weights |
| `evaluation`  | Where to write predictions, AUC scores, ROC plot |
| `gradcam`     | Which conv layer to use, how many top labels, which images |
| `logging`     | Log verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |

## Full run vs. small smoke test

Training DenseNet121 on full-size (320×320) images needs real time and,
practically, a GPU. Before committing those resources, you can sanity-check
the *entire* pipeline — leakage check, generators, training, prediction,
evaluation, Grad-CAM — in well under a minute on a CPU, using a synthetic
dataset and no downloads:

```bash
# 1. Generate a tiny synthetic dataset (random-noise images + valid CSVs)
python main.py make-dummy-data --config config.small.yaml

# 2. Run the whole pipeline against it
python main.py run-all --config config.small.yaml
```

`config.small.yaml` differs from `config.yaml` in exactly the ways that
matter for resource use:

| | `config.yaml` (full) | `config.small.yaml` (smoke test) |
|---|---|---|
| Image size | 320×320 | 96×96 |
| Batch size | 8 | 2 |
| Epochs / steps | 3 epochs × 100 steps | 1 epoch × 2 steps |
| Backbone weights | `imagenet` (downloaded) | `null` (random init, no download) |
| Data | real ChestX-ray8 subset | synthetic, auto-generated |
| Paths | `data/nih/`, `models/nih/`, `outputs/` | `data/nih_small/`, `models/nih_small/`, `outputs_small/` |

Because the paths don't overlap, running the smoke test never touches real
data, checkpoints, or outputs. The metrics it produces are meaningless (the
"images" are random noise) — it only proves the CLI, config, data pipeline,
model, and loss all wire together correctly. Once you've confirmed that,
point `--config` at `config.yaml` (with your real data in place) for the
full run.

`make-dummy-data` builds patient IDs from disjoint per-split ranges
(`P0xxx` train, `P1xxx` valid, `P2xxx` test), so `check-leakage` against it
is guaranteed to report a clean result — useful for confirming that command
too.

## Usage

Every command follows the same shape:

```bash
python main.py <command> --config config.yaml
```

`--config` defaults to `config.yaml` in the current directory, so it can be
omitted when you're not using a custom copy. Swap in `config.small.yaml` for
any of the commands below to run against the synthetic smoke-test data
instead.

### 1. Sanity-check the data splits for patient leakage

```bash
python main.py check-leakage --config config.yaml
```

Logs a warning if any patient ID appears in more than one split; otherwise
confirms the splits are clean.

### 2. Train from scratch (or fine-tune)

```bash
python main.py train --config config.yaml
```

Trains DenseNet121 with the weighted loss for `training.epochs` epochs and
writes weights to `training.output_weights` (plus a loss-curve plot at
`training.history_plot`).

### 3. Run inference on the test set

```bash
python main.py predict --config config.yaml
```

Loads `model.pretrained_weights` (or a freshly built model if that's `null`)
and writes `outputs/predictions.csv`.

### 4. Evaluate — per-label AUROC + ROC plot

```bash
python main.py evaluate --config config.yaml
```

Writes `outputs/roc_curve.png` and `outputs/auc_scores.csv`, and logs each
label's AUROC.

### 5. Grad-CAM visualizations

```bash
python main.py gradcam --config config.yaml
```

Picks the `gradcam.num_top_labels` highest-AUROC pathologies, generates a
heatmap panel for each of `gradcam.num_random_images` sampled images (or the
exact filenames in `gradcam.images`), and writes them to
`outputs/gradcam/gradcam_<image>.png`.

### 6. Run everything end-to-end

```bash
python main.py run-all --config config.yaml
```

Runs the leakage check, then evaluate + Grad-CAM against whatever
`model.pretrained_weights` points to. Training only runs as part of `run-all`
if `training.enabled: true` is set in the config — otherwise this is a pure
evaluate + Grad-CAM pass against an existing checkpoint.

### Using a pretrained checkpoint without training

Set `model.pretrained_weights` in `config.yaml` and jump straight to
`predict`, `evaluate`, or `gradcam` — `train` is never invoked, and the model
architecture is built once and loaded from that checkpoint.

### Using a custom config for a second experiment

```bash
cp config.yaml config.experiment2.yaml
# edit config.experiment2.yaml (e.g. change image.batch_size, training.epochs)
python main.py train --config config.experiment2.yaml
```

## Outputs

| File | Description |
|------|--------------|
| `outputs/predictions.csv` | Per-image predicted probability for each of the 14 labels |
| `outputs/auc_scores.csv` | Per-label AUROC on the test set |
| `outputs/roc_curve.png` | Combined ROC curve for all 14 labels |
| `outputs/training_loss.png` | Training/validation loss curve (if trained) |
| `outputs/gradcam/gradcam_<image>.png` | Original image + heatmaps for the top-performing labels |

## Acknowledgments

Core modeling functions (patient-leakage detection, class-frequency
computation, the weighted loss, the DenseNet121 build, Grad-CAM, and
ROC/AUROC evaluation) originate from the "AI for Medical Diagnosis" course
assignment on chest X-ray diagnosis using the ChestX-ray8 dataset. This
repository reorganizes that logic into a config-driven, CLI-runnable package.
