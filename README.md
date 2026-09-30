# MiniMax-H3 · Image + Text → Video (Streamlit App)

A local Streamlit interface for running **MiniMax-H3 (INT8, torchao)** through Hugging Face `diffusers`' `ModularPipeline`. Upload a reference image (or first/last frames), type a prompt, and get back an **MP4 with generated video and audio**.

The app follows the [model card recipe](https://huggingface.co/abhishekchohan/minimax-h3-int8) step for step, and adds a live progress panel, time estimates, and a strict offline-by-default download policy.

---

## 🎥 Video Tutorial

> **Watch the walkthrough:** https://youtu.be/SQW2Dago-n0

<!-- Optional: clickable thumbnail. Replace VIDEO_ID with your video's ID.
[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)
-->

---

## ✨ Features

- **Three workflows**
  - `ref2va` – one reference image + text → video + audio (default)
  - `fl2va` – first frame (and optional last frame) + text → video + audio
  - `t2va` – text only → video + audio
- **Model-card recipe implemented exactly**: bf16 loading, frozen weights, streamed block-level group offload for the transformer, leaf-level offload for the text encoder, VAEs on GPU.
- **Lower-VRAM mode** (~12–16 GB): also group-offloads the video VAE.
- **Live progress panel**: stage checklist, denoising step counter (`step X / N`), ticking clock, s/step and ETA.
- **Time estimates**: timings from finished runs are saved to `h3_timing_history.json` and used to predict future runs with the same settings.
- **Offline by default**: nothing is downloaded unless you explicitly allow it, and any downloads are confined to the model folder.
- **Pre-flight checks**: verifies the local model folder, required weights, VRAM and system RAM before you generate.
- **One-click MP4 download**, plus seed, peak VRAM and timing summary.

---

## 🖥️ Requirements

### Hardware

| Resource | Recommendation |
|---|---|
| GPU | NVIDIA CUDA GPU, **24–32 GB VRAM** (validated by the model card) |
| GPU (low-VRAM mode) | ~12–16 GB with 544×960 canvas |
| System RAM | **~75 GB+** (INT8 weights are streamed from host memory) |
| Disk | ~100 GB for the model files |

Peak VRAM per the model card at 124 frames, 544×960: ~16.4 GB (t2va), ~18.2 GB (ref2va).

### Software

- Linux (RAM detection reads `/proc/meminfo`)
- Python 3.10+
- CUDA-enabled PyTorch

```bash
pip install -U streamlit torchao transformers accelerate
pip install -U imageio pillow numpy av
pip install -U git+https://github.com/huggingface/diffusers.git   # diffusers MAIN (required)
```

> ⚠️ The MiniMax-H3 pipeline only exists on **diffusers `main`**, not the PyPI release.

---

## 📁 Model Folder

By default the app reads the model from:

```
/media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8
```

Override it with the `MINIMAX_H3_DIR` environment variable, or edit the path in the sidebar.

Expected structure:

```
minimax-h3-int8/
├── modular_model_index.json
├── transformer/          # t2va + fl2va
├── transformer_ref/      # ref2va
├── text_encoder/
├── vae/
├── audio_vae/
├── tokenizer/
├── processor/
└── scheduler/
```

Download the weights from the [Hugging Face model page](https://huggingface.co/abhishekchohan/minimax-h3-int8) into this folder first.

---

## 🚀 Usage

```bash
streamlit run minimax_h3_streamlit_app.py
```

1. Confirm the **local weights folder** in the sidebar.
2. Choose a **workflow** (`ref2va`, `fl2va`, or `t2va`).
3. Upload your reference image / frames (not needed for `t2va`).
4. Write a **prompt** describing the scene, motion and mood.
5. Adjust generation settings (or keep the validated defaults).
6. Click **Generate video**.

The first load reads ~100 GB from disk, so be patient. Once loaded, the pipeline stays in memory for later runs with the same settings.

---

## ⚙️ Settings

| Setting | Default | Notes |
|---|---|---|
| Canvas | 544 × 960 | Also offers 768 × 1344 (fits 32 GB) and custom. Must be multiples of 32. |
| Frames | 124 | ~5 s at 24 fps; the model card's validated value. |
| Inference steps | 20 | Model card's validated value. Each step is a full pass of the 33B transformer. |
| Output FPS | 24 | MiniMax-H3 generates at 24 fps. |
| Seed | Fixed (42) | Or choose Random. |
| Lower-VRAM mode | Off | Group-offloads the video VAE for 12–16 GB cards. |
| Offload block group size | 1 | Advanced; model card value. |

Switching workflow or lower-VRAM mode reloads the model. Use **Unload model / clear cache** in the sidebar to free memory.

---

## 🔒 Download Policy

1. The model is read **only** from the local folder.
2. **Offline by default** – `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are forced on. If a file is missing, the app stops with an error naming it instead of silently downloading gigabytes into `~/.cache`.
3. To allow downloads:

   ```bash
   MINIMAX_H3_ALLOW_DOWNLOAD=1 streamlit run minimax_h3_streamlit_app.py
   ```

   The entire Hugging Face cache is then redirected to `<model folder>/_extra_downloads/`. (This also hides your default HF login token; set `HF_TOKEN` if you need one.)
4. On **Generate**, the Hub repo id in `modular_model_index.json` is rewritten to your local folder so components resolve from disk. A `.bak` backup is kept.

### Environment variables

| Variable | Purpose |
|---|---|
| `MINIMAX_H3_DIR` | Path to the local model folder |
| `MINIMAX_H3_ALLOW_DOWNLOAD` | Set to `1` to permit downloads (into `_extra_downloads/`) |

---

## 📊 Progress & Time Estimates

While generating, the app shows:

- A stage checklist: **Load model → Encode prompt & references → Denoise → Decode video + audio → Write MP4**
- A denoising progress bar and `step X / N` counter (read from diffusers' own progress bar)
- A ticking elapsed clock
- Seconds per step and estimated time remaining

Finished runs are logged to `h3_timing_history.json` (next to the script). The next run with the same workflow, resolution, frame count, and offload settings shows an up-front estimate.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| **Missing package error** | Install the requirements above. Make sure diffusers is from `main`. |
| **"No CUDA GPU detected"** | An NVIDIA GPU with a CUDA build of PyTorch is required. |
| **Out of GPU memory** | Enable Lower-VRAM mode, use 544 × 960, and/or reduce frames. |
| **Missing file / repo error (offline)** | Check that the folder is complete, or relaunch once with `MINIMAX_H3_ALLOW_DOWNLOAD=1`. |
| **Under 75 GB RAM warning** | The model may swap or fail to load; more system RAM is strongly recommended. |
| **Height/width error** | Both must be multiples of 32. |
| **No live step counter** | Your diffusers version doesn't expose the denoising `tqdm`; only elapsed time is shown. Generation still works. |

### Optional helper scripts

If present next to the app, these are used for deeper checks:

- `verify_weights.py` – manifest-based verification of the model folder
- `download_weights.py` – `--check` lists missing files; running it fetches only those

---

## 🙏 Credits

- Model: [`abhishekchohan/minimax-h3-int8`](https://huggingface.co/abhishekchohan/minimax-h3-int8)
- Pipeline: [Hugging Face diffusers](https://github.com/huggingface/diffusers)
- UI: [Streamlit](https://streamlit.io)

## 📄 License

Add your license here. Note that the model weights are subject to their own license on the Hugging Face model page.
.
