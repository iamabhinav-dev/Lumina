# Running ConvLSTM Training on Kaggle (Free T4 GPU)

**Expected training time on T4 GPU:** ~5–10 minutes  
**Expected training time on your CPU:** ~2–3 hours

---

## What's in this folder

```
kaggle_upload/
  dataset/
    frames.npz            ← 27 MB — training sequences
    frame_scaler.pkl      ← MinMaxScaler for the frames
    frame_metadata.json   ← grid dimensions, CRS, transform
  train_on_kaggle.ipynb   ← self-contained training notebook
  KAGGLE_STEPS.md         ← this file
```

---

## Part 1 — Upload the Dataset

1. Go to **kaggle.com** → sign in (or create a free account)
2. Click your profile photo (top-right) → **Your Work** → **Datasets** → **New Dataset**
3. In the dialog:
   - **Title:** `ntl-kolkata-convlstm`  ← must match exactly (used inside the notebook)
   - **Subtitle:** anything
   - Drag-and-drop the three files from `kaggle_upload/dataset/`:
     - `frames.npz`
     - `frame_scaler.pkl`
     - `frame_metadata.json`
4. Click **Create** — wait for upload to finish (~1–2 min for 27 MB)
5. Leave the dataset as **Private** (you don't need to publish it)

> **If you want a different dataset name**, open `train_on_kaggle.ipynb` in a text editor and change:
> ```python
> DATASET_SLUG = 'ntl-kolkata-convlstm'
> ```
> to whatever slug Kaggle assigns (visible in the dataset URL: `kaggle.com/datasets/YOUR_USERNAME/THE-SLUG`).

---

## Part 2 — Create and Run the Notebook

### Step 1 — Create a new notebook
1. Go to kaggle.com → **Your Work** → **Notebooks** → **New Notebook**
2. Kaggle opens a blank notebook editor

### Step 2 — Upload the notebook file
1. Click **File** (top menu bar inside the editor) → **Import Notebook**
2. Upload `kaggle_upload/train_on_kaggle.ipynb`
3. The cells will appear in the editor

### Step 3 — Enable GPU
1. On the right sidebar, find **Session options** (or click the settings gear)
2. **Accelerator** → change from `None` to **GPU T4 x2**
3. Click **Save** — Kaggle will restart the kernel with GPU

### Step 4 — Attach the dataset
1. In the right sidebar, click **+ Add Data**
2. Search for `ntl-kolkata-convlstm` (your dataset from Part 1)
3. Click **Add** — it will be mounted at `/kaggle/input/ntl-kolkata-convlstm/`

### Step 5 — Run all cells
1. Click **Run All** (▶▶ button in the toolbar, or **Run** → **Run All**)
2. Watch the output — training progress appears under Cell 6
3. Expected output per epoch (T4 GPU):
   ```
   Epoch 1/500
   6/6 ━━━━━━━━━━━━━━━━━━━━ 3s 500ms/step - loss: 0.35 - val_loss: 0.31
   ```
4. EarlyStopping (patience=20) will stop training automatically
5. Total time: **~5–10 minutes**

---

## Part 3 — Download the Results

After the last cell runs, you'll see:
```
Download ready: kolkata_convlstm_output.zip  (X.X MB)
Find it under  Output → kolkata_convlstm_output.zip → Download
```

To download:
1. Click **Output** tab in the right sidebar (or the folder icon)
2. Find `kolkata_convlstm_output.zip`
3. Click the **↓** download icon

The zip contains:
- `convlstm_model.keras` — trained model (architecture + weights)
- `training_history.json` — loss curves + filter config
- `loss_curve.png` — training curve plot

---

## Part 4 — Copy Results Back to Your Local Repo

After downloading and extracting the zip:

```bash
cd /home/abhinav/Desktop/BTP

# Extract the zip (adjust path to wherever you downloaded it)
unzip ~/Downloads/kolkata_convlstm_output.zip -d /tmp/kaggle_results/

# Copy model to the outputs directory (where evaluate_convlstm.py looks)
cp /tmp/kaggle_results/convlstm_model.keras  outputs/kolkata/convlstm/convlstm_model.keras

# Copy training history
cp /tmp/kaggle_results/training_history.json outputs/kolkata/convlstm/training_history.json

# Copy loss curve plot
mkdir -p outputs/kolkata/convlstm/plots
cp /tmp/kaggle_results/loss_curve.png        outputs/kolkata/convlstm/plots/loss_curve.png
```

---

## Part 5 — Run Evaluate and Forecast Locally

Once the model is in place, continue the pipeline locally:

```bash
cd /home/abhinav/Desktop/BTP
source venv/bin/activate

# Evaluate
python models/convlstm/evaluate_convlstm.py --city kolkata

# Forecast
python models/convlstm/forecast_convlstm.py --city kolkata
```

Both scripts use `tf.keras.models.load_model()` so they auto-restore the architecture from the `.keras` file — no filter args needed.

---

## Part 6 — Commit Phase 4

```bash
cd /home/abhinav/Desktop/BTP
git add \
  models/convlstm/train_convlstm.py \
  models/convlstm/evaluate_convlstm.py \
  models/convlstm/forecast_convlstm.py \
  models/convlstm/kolkata/ \
  outputs/kolkata/convlstm/
git commit -m "Phase 4: ConvLSTM pipeline for Kolkata (8,16,16 filters, GPU-trained)"
git push
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No such file or directory: /kaggle/input/ntl-kolkata-convlstm/frames.npz` | Dataset slug doesn't match. Check the URL of your dataset and update `DATASET_SLUG` in Cell 2 |
| `No GPU available` | Accelerator wasn't saved — go to Session options → GPU T4 x2 → Save, then restart kernel |
| Session disconnects during training | Kaggle free tier sessions can disconnect after ~9h; training finishes in ~10 min so this shouldn't happen |
| `register_keras_serializable` error | Restart kernel and run all cells fresh (don't re-run Cell 3 only) |
| Downloaded `.keras` file is empty/corrupt | Re-run Cell 7 (save history) and Cell 9 (zip) after training completes |

---

## Why T4 is ~20× faster

| | Your CPU | Kaggle T4 GPU |
|---|---|---|
| Hardware | Intel/AMD cores | NVIDIA Tesla T4  (2560 CUDA cores) |
| Batch step time | ~14s/step | ~0.5s/step |
| Epoch time (6 steps) | ~90s | ~3s |
| 60 epochs total | ~90 min | ~3 min |

The ConvLSTM's Conv2D operations inside each LSTM cell are highly parallelisable — exactly what a GPU is designed for.
