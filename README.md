# Soccer Foul Classification on SoccerNet-MVFoul

A comparative study of six machine-learning approaches for soccer-foul recognition from multi-view broadcast video, developed for **CS 6140: Machine Learning** (Spring 2026, Prof. Ehsan Elhamifar), Khoury College, Northeastern University.

**Author:** Shreepoorna Purohit

---

## What this project does

Two related classification tasks are tackled on the **SoccerNet-MVFoul** dataset:

1. **Binary offence detection** — Does an action constitute a foul? (`Offence` vs `No offence`)
2. **Multiclass foul-type classification** — Which of seven foul categories describes the action? (`Standing tackling`, `Tackling`, `Challenge`, `Holding`, `Elbowing`, `High leg`, `Pushing`)

Six models spanning classical, deep, and zero-shot families are trained and compared under a unified evaluation protocol using **balanced accuracy** as the primary metric (appropriate given heavy class imbalance).

## Headline results

| Model | Binary BA | Multiclass BA |
|---|:---:|:---:|
| Random baseline | 0.500 | 0.143 |
| Logistic Regression (HOG + flow + BoVW) | 0.543 | 0.120 |
| SVM, RBF kernel (HOG + flow + BoVW) | 0.500 | 0.121 |
| ResNet-18 (fine-tuned) | 0.555 | 0.133 |
| R3D-18 (fine-tuned, 3D CNN) | 0.485 | 0.146 |
| CLIP ViT-B/32 (zero-shot) | 0.500 | 0.143 |
| **ResNet-50 + MLP (multi-frame, multi-view)** | **0.644** | **0.283** |

The ResNet-50 + MLP pipeline roughly doubles the best alternative on both tasks and is the only model producing non-degenerate predictions across all seven foul categories. See the report for full confusion-matrix analysis.

---

## Repository layout

```
soccer-foul-classification/
├── src/
│   ├── utils.py                 # Shared helpers: label maps, data loading, frame extraction
│   ├── 01_download_data.py      # Download SoccerNet-MVFoul via the SoccerNet pip package
│   ├── 02_explore_data.py       # Class distributions, annotation inspection, sample-video stats
│   ├── 03_extract_features.py   # HOG + optical-flow + BoVW handcrafted features
│   ├── 04_train_logreg.py       # Logistic Regression (5-fold CV grid search)
│   ├── 05_train_svm.py          # SVM, RBF kernel (5-fold CV grid search)
│   ├── 06_train_resnet.py       # ResNet-18 fine-tuning
│   ├── 06b_train_r3d.py         # R3D-18 fine-tuning (3D CNN)
│   ├── 06c_clip_zero_shot.py    # CLIP zero-shot with hard/soft voting
│   ├── 06d_resnet50_mlp.py      # Best model: frozen ResNet-50 features + MLP
│   └── 07_evaluate.py           # Aggregates all test-set predictions into comparison plots
├── results/                     # Confusion matrices, comparison bar charts, per-class F1 plots
├── requirements.txt
├── .gitignore
└── README.md
```

Directories created automatically on first run (all are in `.gitignore`):

```
data/mvfoul/                     # Raw SoccerNet-MVFoul (train/valid/test + annotations.json)
features/                        # Cached .npy / .npz feature arrays per split
models/                          # Saved model weights + per-model config JSON
```

---

## Prerequisites

- **Python 3.10+** (developed and tested on 3.11)
- **macOS with Apple Silicon** (M1/M2/M3/M4) for MPS acceleration — OR any CUDA GPU, OR CPU (slower but works)
- **~8 GB free disk** for the dataset
- **SoccerNet credentials** — you must sign the SoccerNet NDA at [soccer-net.org](https://www.soccer-net.org/) before the dataset can be downloaded. The dataset password is handled by the download script.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Shreepoorna1903/soccer-foul-classification.git
cd soccer-foul-classification

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### `requirements.txt`

If the repo's `requirements.txt` differs from what the scripts import, the full set of packages used is:

```
torch>=2.1
torchvision>=0.16
numpy
scikit-learn
opencv-python
Pillow
matplotlib
tqdm
SoccerNet
open-clip-torch
```

---

## How to reproduce every result in the report

All scripts should be run from the repository root. Scripts are numbered to indicate execution order.

### Step 0 — Data download (one-time)

Dataset access requires signing the **SoccerNet  at [NDA**](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform), after which the dataset password is provided.

**Option A — Use the provided wrapper script (recommended):**

```bash
python src/01_download_data.py
```

The script downloads `train`, `valid`, and `test` splits to `data/mvfoul/` and prints per-split clip counts for verification. The `challenge` split is intentionally excluded — it is SoccerNet's held-out evaluation set with no public labels and is not used anywhere in this pipeline.

**Option B — Download manually via the official SoccerNet API:**

If the wrapper script is unavailable or you prefer to invoke the SoccerNet API directly, the following snippet (from the [official SoccerNet documentation](https://www.soccer-net.org/)) achieves the same result:

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="data/mvfoul")
mySNdl.downloadDataTask(
    task="mvfouls",
    split=["train", "valid", "test"],
    password="<your NDA password>",
)
```

Total download is ~7 GB.

### Step 1 — Dataset exploration (optional, for sanity)

```bash
python src/02_explore_data.py
```

Prints class distributions for both tasks across all splits and inspects a sample video's resolution/FPS/frame count. Useful as a sanity check that the data parsed correctly.

### Step 2 — Handcrafted feature extraction

```bash
python src/03_extract_features.py
```

Extracts HOG (224×224 middle frame, 16×16 cells, 9 orientation bins), dense Farnebäck optical-flow histograms (32 magnitude + 32 orientation bins), and a 200-word SIFT Bag-of-Visual-Words using MiniBatchKMeans. Outputs `features/{train,valid,test}_{hog,flow,bovw,combined,labels_binary,labels_multi}.npy` and caches the BoVW vocabulary to `features/bovw_vocab.pkl`. Runtime: ~10–15 min on M4.

### Step 3 — Train classical baselines

```bash
python src/04_train_logreg.py    # ~1 min
python src/05_train_svm.py       # ~5 min
```

Both scripts perform 5-fold stratified cross-validated grid search on the combined handcrafted features:
- **LogReg:** `C ∈ {1e-3, 1e-2, 1e-1, 1, 10}`, L2 penalty, `class_weight=balanced`
- **SVM:** `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, 1e-3, 1e-2}`, RBF kernel, `class_weight=balanced`

Each script saves its best model to `models/` and a confusion matrix to `results/`.

### Step 4 — Train deep baselines

```bash
python src/06_train_resnet.py      # ResNet-18, ~30–45 min on M4
python src/06b_train_r3d.py        # R3D-18, several hours on M4 — see note below
python src/06c_clip_zero_shot.py   # CLIP zero-shot, ~15–20 min on M4
```

- **ResNet-18**: Fine-tunes `layer3`, `layer4`, and the FC head on the middle frame of each main-camera clip. Grid: LR ∈ {1e-4, 5e-4, 1e-3} × epochs ∈ {15, 25}. Model selection via validation split.
- **R3D-18**: Fine-tunes all residual blocks (stem frozen) on 16 evenly-spaced frames per clip at 112×112. Grid: LR ∈ {1e-4, 3e-4, 1e-3} × 30 epochs each. **Note:** this model is intentionally included as a documented negative result on local Apple-Silicon hardware. See the report's discussion of why it underperforms.
- **CLIP**: Uses OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) in zero-shot mode with multiple prompts per class, aggregated across every 5th frame of every camera clip. Reports both hard-voting (majority argmax) and soft-voting (mean cosine similarity) results.

### Step 5 — Train the winning model

```bash
python src/06d_resnet50_mlp.py     # ~30–40 min on M4 (feature extraction dominates)
```

Two-stage pipeline:

1. **Frozen feature extraction.** ResNet-50 pretrained on ImageNet-1k is used as a frozen extractor (final FC replaced with `nn.Identity`). For each action, 10 frames are uniformly sampled from *every* camera clip, pushed through the backbone, and mean-pooled across the frame × view axis into a single 2048-d vector per action. Cached to `features/resnet50_{split}_{task}.npz`.
2. **MLP classifier.** A 2-layer MLP (`Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → ReLU → Dropout → Linear`) is trained on the cached features with 5-fold stratified cross-validation over five configurations (varying hidden dim ∈ {256, 512, 1024}, LR ∈ {5e-4, 1e-3}, dropout ∈ {0.3, 0.4, 0.5}, epochs ∈ {50, 80}). The best configuration is retrained on the full training set and evaluated on the held-out test split.

### Step 6 — Aggregate all predictions into comparison figures

```bash
python src/07_evaluate.py
```

Loads saved predictions from every model and produces:
- `results/comparison_binary_bar.png`, `results/comparison_multiclass_bar.png` — balanced-accuracy bar charts
- `results/{model}_{task}_confusion.png` — individual confusion matrices
- `results/all_confusion_matrices.png` — 6×2 grid of all confusion matrices
- `results/per_class_f1_multiclass.png` — per-class F1 scores on the 7-way task
- `results/class_distribution.png` — train/valid/test class distributions
- `results/comparison_table.txt` — plaintext summary table

---

## Run everything end-to-end

```bash
python src/01_download_data.py
python src/02_explore_data.py      # optional
python src/03_extract_features.py
python src/04_train_logreg.py
python src/05_train_svm.py
python src/06_train_resnet.py
python src/06b_train_r3d.py        # slow; optional if time-constrained
python src/06c_clip_zero_shot.py
python src/06d_resnet50_mlp.py
python src/07_evaluate.py
```

Full end-to-end runtime on a MacBook Air M4 (16 GB, MPS backend): **~8–10 hours**, dominated by R3D-18. Skipping R3D-18 brings this down to ~2–3 hours.

---

## Hardware notes

All experiments were run on a **MacBook Air M4 (16 GB unified memory)** using PyTorch's MPS backend. The code auto-detects the best available device in this order: MPS → CUDA → CPU. Apple-Silicon-specific details that may affect portability:

- `torch.mps.empty_cache()` is called explicitly after feature extraction in `06d_resnet50_mlp.py` to free GPU memory before MLP training.
- `num_workers=0` is used in all `DataLoader`s because MPS performs best with main-process loading on macOS.
- R3D-18 training is prohibitively slow on MPS. If CUDA is available, it can be expected to run roughly 5–10× faster.

---

## Dataset

- **SoccerNet-MVFoul** — a multi-view subset of SoccerNet v3 curated for foul understanding.
- Each annotated *action* is associated with 1–4 short video clips (~3–5 s at 25 fps) from different camera angles.
- Annotations are in `data/mvfoul/{split}/annotations.json` and include `Offence`, `Action class`, `Body part`, `Severity`, `Contact`, and per-clip camera metadata.
- Label filtering:
  - **Binary task** — `Between` labels and empty values are dropped.
  - **Multiclass task** — `Don't know` and `Dive` are dropped (annotator uncertainty and out-of-scope respectively).

Dataset access requires signing the **SoccerNet NDA** at [soccer-net.org](https://www.soccer-net.org/). See Step 0 above for download instructions (wrapper script or direct API usage).

---

## Final report

The full 10-page project report (9 pages body + references) covering dataset analysis, methodology, results, confusion-matrix diagnostics, discussion, and limitations is submitted on Canvas per course policy.

---

## Acknowledgements

- **SoccerNet team** — for curating and maintaining the MVFoul benchmark.
- **PyTorch, torchvision** — for pretrained ResNet-18, ResNet-50, and R3D-18 weights.
- **scikit-learn** — for classical baselines and cross-validation machinery.
- **OpenCLIP** — for the CLIP ViT-B/32 model used in zero-shot experiments.
- **Prof. Ehsan Elhamifar** — for course guidance, feedback on the proposal, and the NOVA Lab's broader work on egocentric procedural video understanding.

---

## License

This project is submitted as coursework for CS 6140 (Spring 2026). Code is shared for the purpose of reproducibility and evaluation by the course staff. The SoccerNet-MVFoul dataset is subject to its own NDA and is **not** redistributed through this repository.
