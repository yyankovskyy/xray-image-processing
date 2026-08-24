# Chest X-Ray Multi-Label Diagnosis Pipeline

A configurable, command-line pipeline for multi-label chest X-ray classification
with DenseNet121 transfer learning, class-imbalance-aware weighted loss,
AUROC/ROC evaluation, and Grad-CAM visualization.

This started as a Jupyter notebook assignment (chest X-ray diagnosis with the
ChestX-ray8 dataset) and has been reorganized into a reusable, config-driven
package: no notebook required, no hard-coded paths, everything controlled
through a single `config.yaml`.

## What it does

Given a set of chest X-ray images and CSVs labeling 14 pathologies
(Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule,
Atelectasis, Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema,
Consolidation), the pipeline:

1. **Checks for patient-level data leakage** between train/valid/test splits.
2. **Builds Keras image generators** — training data normalized per-batch,
   validation/test data normalized using statistics learned from the training set
   (to avoid leaking test-set information into the model).
3. **Computes class-imbalance weights** and trains (or loads a pretrained)
   DenseNet121 model with a custom weighted binary cross-entropy loss.
4. **Predicts and evaluates** — per-label AUROC scores and a combined ROC plot.
5. **Generates Grad-CAM heatmaps** highlighting which regions of an X-ray drove
   the model's predictions for its top-performing labels.

## Repository structure

```
.
├── config.yaml              # all pipeline parameters live here
├── main.py                  # CLI entrypoint
├── requirements.txt
├── src/
│   ├── config.py             # YAML config loading + logging setup
│   ├── data.py                # leakage check + generator construction
│   ├── losses.py               # class frequencies + weighted loss
│   ├── model_build.py           # DenseNet121 model construction
│   ├── evaluate.py               # prediction + ROC/AUROC
│   └── gradcam.py                 # Grad-CAM heatmap generation
├── tests/
│   ├── test_case.py           # weighted-loss test fixture (from the original assignment)
│   └── test_functions.py      # pytest suite validating core functions
├── data/nih/                # place train/valid/test CSVs + images here (not included)
├── models/nih/               # place pretrained weights here (not included)
└── outputs/                 # predictions, ROC plots, AUC scores, Grad-CAM panels
```

## Installation

```bash
git clone <your-repo-url>
cd chest-xray-pipeline
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Data & model weights

This repo does **not** include the dataset or pretrained weights (they're
large and were provided separately in the original course environment).
You'll need to supply:

- `data/nih/train-small.csv`, `valid-small.csv`, `test.csv` — image filenames,
  per-pathology binary labels, and a `PatientId` column.
- `data/nih/images-small/` — the corresponding X-ray images.
- `models/nih/pretrained_model.h5` (optional) — full model weights to load for
  inference/evaluation/Grad-CAM without training from scratch.

The full ChestX-ray8 dataset (108,948 images) is publicly available; see the
[NIH Clinical Center release](https://nihcc.app.box.com/v/ChestXray-NIHCC).

Update the paths in `config.yaml` to match wherever you place these files.

## Configuration

Every tunable parameter — file paths, image size, batch size, the label list,
training schedule, evaluation output paths, and Grad-CAM settings — lives in
`config.yaml`. Nothing is hard-coded in the source modules. To run a different
experiment, copy `config.yaml`, edit the copy, and pass it via `--config`.

Key sections:

| Section       | Controls |
|---------------|----------|
| `data`        | CSV paths, image directory, column names |
| `labels`      | The 14 pathologies (order = model output order) |
| `image`       | Target size, batch size, seed, normalization sample size |
| `model`       | Backbone weights, pretrained checkpoint, optimizer, loss epsilon |
| `training`    | Steps/epoch, validation steps, epochs, where to save weights |
| `evaluation`  | Where to write predictions, AUC scores, ROC plot |
| `gradcam`     | Which conv layer to use, how many top labels, which images |

## Usage

All commands are run through `main.py <command> --config config.yaml`.

```bash
# 1. Sanity-check the data splits for patient leakage
python main.py check-leakage --config config.yaml

# 2. (Optional) Train from scratch or fine-tune
python main.py train --config config.yaml

# 3. Run inference on the test set (writes outputs/predictions.csv)
python main.py predict --config config.yaml

# 4. Evaluate: per-label AUROC + ROC plot (writes outputs/roc_curve.png, outputs/auc_scores.csv)
python main.py evaluate --config config.yaml

# 5. Grad-CAM visualizations for the top-performing labels (writes outputs/gradcam/*.png)
python main.py gradcam --config config.yaml

# Or run the whole thing end-to-end (training only runs if training.enabled: true)
python main.py run-all --config config.yaml
```

By default `training.enabled` is not read as a gate for the individual `train`
command (running `train` explicitly always trains) — it only gates whether
`run-all` includes a training step, so you can use `run-all` for a pure
evaluate+gradcam pass against a pretrained checkpoint.

### Using a pretrained checkpoint (no training)

If you just want predictions/evaluation/Grad-CAM from an already-trained
model, set `model.pretrained_weights` in `config.yaml` and skip straight to
`predict`, `evaluate`, or `gradcam` — `train` is never called and the model is
built once and loaded from that checkpoint.

## Outputs

| File | Description |
|------|-------------|
| `outputs/predictions.csv` | Per-image predicted probability for each of the 14 labels |
| `outputs/auc_scores.csv` | Per-label AUROC on the test set |
| `outputs/roc_curve.png` | Combined ROC curve for all 14 labels |
| `outputs/training_loss.png` | Training/validation loss curve (if trained) |
| `outputs/gradcam/gradcam_<image>.png` | Original image + heatmaps for the top-performing labels |

## Tests

A small `pytest` suite validates the core building blocks — patient-leakage
detection, class-frequency computation, and the weighted loss — against the
expected values documented in the original assignment:

```bash
pytest tests/
```

## Compatibility note

This code mirrors the API surface of the original assignment environment,
which used a **standalone `keras` package** together with TF1-style calls
(`keras.backend.get_session()`, `tf.compat.v1.logging`). `requirements.txt`
pins compatible `tensorflow`/`keras` versions. If you upgrade to a newer
TensorFlow where `keras` is bundled as `tensorflow.keras` and the standalone
`keras` package is unavailable, you'll need to change the `from keras...`
imports throughout `src/` to `from tensorflow.keras...` and drop the
`K.get_session()`-based test in `tests/test_functions.py` (TF2 eager mode
doesn't use sessions).

## Acknowledgments

Core modeling functions (`check_for_leakage`, `compute_class_freqs`,
`get_weighted_loss`, the DenseNet121 build, Grad-CAM, and ROC/AUROC
evaluation) originate from the "AI for Medical Diagnosis" course assignment
on chest X-ray diagnosis using the ChestX-ray8 dataset. This repository
reorganizes that logic into a config-driven, CLI-runnable pipeline.
