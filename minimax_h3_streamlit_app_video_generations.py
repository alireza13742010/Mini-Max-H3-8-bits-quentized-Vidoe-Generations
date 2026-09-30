"""
Streamlit app — MiniMax-H3 (INT8, torchao) — image reference + text -> video+audio.

Follows the recipe on the model card exactly:
    https://huggingface.co/abhishekchohan/minimax-h3-int8

Model-card recipe implemented in load_pipeline() / the generate call:
    1. ModularPipeline.from_pretrained(<repo or local folder>)
    2. pipe.load_components(workflow=..., dtype=torch.bfloat16)
    3. requires_grad_(False) on the transformer and text_encoder
    4. transformer.enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)
    5. apply_group_offloading(pipe.text_encoder.model, leaf_level, use_stream=True)
    6. pipe.vae.to("cuda"); pipe.audio_vae.to("cuda")
       (lower-VRAM mode: leaf_level group-offload the video VAE instead, no stream)
    7. pipe(prompt=..., num_frames=124, height=544, width=960,
            generator=torch.Generator().manual_seed(42),
            output=["videos", "audio", "sampling_rate"])
    ref2va -> references=[...]   fl2va -> image= (and optional last_image=)

--------------------------------------------------------------------------
DOWNLOAD POLICY
--------------------------------------------------------------------------
1. The model is read ONLY from the local folder:
       /media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8
   (override with the MINIMAX_H3_DIR env var).
2. OFFLINE BY DEFAULT: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are forced on.
   If the pipeline needs a file that isn't in the folder, it stops with an
   error that names it instead of silently fetching ~10 GB into ~/.cache.
3. To allow downloads, launch with:
       MINIMAX_H3_ALLOW_DOWNLOAD=1 streamlit run minimax_h3_streamlit_app.py
   The whole Hugging Face cache is then redirected INSIDE the model folder:
       <model folder>/_extra_downloads/
   (This also hides your default HF login token; set HF_TOKEN if you need one.)
4. On "Generate", the Hub repo id inside modular_model_index.json is rewritten
   to your local folder (a .bak backup is kept), so components resolve from disk.

The cache redirection is decided at launch time (from MINIMAX_H3_DIR) because
the HF libraries read these variables when first imported.
--------------------------------------------------------------------------
Requirements (from the model card + what encode_video needs):
    pip install -U streamlit torchao transformers accelerate
    pip install -U imageio pillow numpy av
    pip install -U git+https://github.com/huggingface/diffusers.git   # diffusers MAIN (required)

Host RAM: the card says to plan for ~75 GB+ of system RAM.

Live progress: while generating, the app shows a stage checklist, the denoising step
counter (step X / N, read from diffusers' own denoising loop), a ticking clock and an
ETA. Timings of finished runs are saved to h3_timing_history.json (next to this script)
so later runs with the same settings get an up-front time estimate.

Run:
    streamlit run minimax_h3_streamlit_app.py
--------------------------------------------------------------------------
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Download policy -- MUST run before torch / diffusers / transformers /
# huggingface_hub are imported (they read these variables at import time).
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_DIR = os.environ.get(
    "MINIMAX_H3_DIR",
    "/media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8",
)
HUB_REPO_ID = "abhishekchohan/minimax-h3-int8"
ALLOW_DOWNLOADS = os.environ.get("MINIMAX_H3_ALLOW_DOWNLOAD", "0") == "1"
EXTRA_DL_DIR = str(Path(DEFAULT_MODEL_DIR) / "_extra_downloads")

os.environ["HF_HOME"] = EXTRA_DL_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(EXTRA_DL_DIR, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(EXTRA_DL_DIR, "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
if ALLOW_DOWNLOADS:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
else:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import contextlib  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from unittest import mock  # noqa: E402

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from PIL import Image  # noqa: E402

st.set_page_config(page_title="MiniMax-H3 · Image + Text → Video", layout="wide")

try:
    import torch
    import torchao  # noqa: F401  -- model card requirement: re-materialises the INT8 weights
    import av  # noqa: F401  -- PyAV: diffusers' encode_video needs it to mux video + audio
    from diffusers import ModularPipeline
    from diffusers.hooks import apply_group_offloading
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.utils.export_utils import encode_video
    import diffusers.modular_pipelines.modular_pipeline as h3_mp_module  # owns the denoising progress bar
    from tqdm.auto import tqdm as tqdm_auto
except ImportError as e:
    st.error(
        "Missing a required package: "
        f"**{e.name}**.\n\n"
        "Install everything with:\n\n"
        "```\n"
        "pip install -U streamlit torchao transformers accelerate imageio pillow numpy av\n"
        "pip install -U git+https://github.com/huggingface/diffusers.git\n"
        "```\n"
        "(The MiniMax-H3 pipeline only exists on diffusers **main**.)"
    )
    st.stop()

# Folder per workflow that MUST exist locally (t2va + fl2va share "transformer/"):
TRANSFORMER_FOLDER = {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"}
# Shared components -- soft-checked only.
SOFT_CHECK_FOLDERS = ["text_encoder", "vae", "audio_vae", "tokenizer", "processor", "scheduler"]
MODEL_CARD_RAM_GB = 75  # "plan for ~75 GB+ of system RAM"


# --------------------------------------------------------------------------- #
# Local-folder checks
# --------------------------------------------------------------------------- #
def _has_weights(folder: Path) -> bool:
    """transformer*/ is stored as pickle .bin; text_encoder/ as safetensors (per the model card)."""
    return folder.is_dir() and any(
        f for pat in ("*.bin", "*.safetensors") for f in folder.rglob(pat)
    )


def check_workflow_ready(model_dir: str, workflow: str):
    """Returns (hard_errors, soft_warnings). Hard errors block generation."""
    hard, soft = [], []
    root = Path(model_dir)
    if not root.exists():
        return [f"Folder not found: {root}"], []

    if not (root / "modular_model_index.json").exists():
        hard.append(f"modular_model_index.json is missing under {root} -- ModularPipeline needs it.")

    needed = TRANSFORMER_FOLDER[workflow]
    if not (root / needed).exists():
        hard.append(
            f"'{needed}/' is missing under {root}. Run "
            "`python download_weights.py --check` to see what is missing, then "
            "`python download_weights.py` to fetch only those files."
        )
    elif not _has_weights(root / needed):
        hard.append(f"'{needed}/' exists but contains no weight files (.bin / .safetensors).")

    for name in SOFT_CHECK_FOLDERS:
        if not (root / name).exists():
            soft.append(f"'{name}/' not found under {root} -- loading may fail if this is required.")

    # Optional deeper check against a manifest (created by verify_weights.py / download_weights.py).
    try:
        from verify_weights import check_against_manifest
        hard += check_against_manifest(model_dir, workflow)
    except ImportError:
        pass
    return hard, soft


def index_status(model_dir: str) -> str:
    """'hub' = index still names the Hub repo, 'local' = already local, 'missing' = no index file."""
    idx = Path(model_dir) / "modular_model_index.json"
    if not idx.exists():
        return "missing"
    return "hub" if HUB_REPO_ID in idx.read_text() else "local"


def localize_index(model_dir: str):
    """Point modular_model_index.json at the local folder so components resolve from disk.
    Idempotent; keeps a .bak backup. Returns a message if it changed anything."""
    idx = Path(model_dir) / "modular_model_index.json"
    if not idx.exists():
        return None
    text = idx.read_text()
    if HUB_REPO_ID not in text:
        return None
    bak = idx.with_name(idx.name + ".bak")
    if not bak.exists():
        bak.write_text(text)
    local = json.dumps(str(Path(model_dir).resolve()))[1:-1]  # JSON-safe path
    idx.write_text(text.replace(HUB_REPO_ID, local))
    return (f"Pointed {idx.name} at your local folder ({text.count(HUB_REPO_ID)} reference(s)); "
            f"backup saved as {bak.name}.")


def host_ram_gb():
    """Total system RAM in GB (Linux), or None if it can't be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- #
# Pipeline loading -- the model card's recipe, step for step
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_pipeline(model_dir: str, workflow: str, low_vram: bool, num_blocks_per_group: int):
    # 1-2. Load drop-in, then pick the workflow (ref2va loads transformer_ref/).
    pipe = ModularPipeline.from_pretrained(model_dir)
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16)

    # 3. version=2 int8 tensors are pinnable (streamed offload needs this); freezing
    #    removes the one autograd path the quantized tensors cannot serve.
    ref = getattr(pipe, "transformer_ref", None)
    transformer = ref if ref is not None else pipe.transformer
    transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 4-5. Streamed group offload of the big components from host RAM.
    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)
    transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=num_blocks_per_group, **offload
    )
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)

    # 6. VAEs. Lower-VRAM (12-16 GB): group-offload the video VAE too (leaf_level, NO stream).
    if low_vram:
        apply_group_offloading(
            pipe.vae, offload_type="leaf_level",
            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        )
    else:
        pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")

    return pipe


def free_pipeline():
    """The INT8 weights live in ~75-100 GB of host RAM -- never keep two pipelines around."""
    st.cache_resource.clear()
    gc.collect()
    torch.cuda.empty_cache()


def describe(x):
    """Readable shape/dtype/device of a pipeline output. Reads metadata only -- it never copies
    CUDA data to the host (np.asarray(cuda_tensor) raises 'can't convert cuda:0 device type
    tensor to numpy')."""
    if x is None:
        return None
    if hasattr(x, "shape"):  # torch.Tensor or np.ndarray
        extra = f", {x.dtype}"
        if hasattr(x, "device"):
            extra += f", {x.device}"
        return f"{tuple(x.shape)}{extra}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} of {len(x)} × {type(x[0]).__name__ if x else '?'}"
    return type(x).__name__


def to_cpu(x):
    """Move a torch tensor to host memory as float32 so NumPy-based code (PyAV / encode_video)
    can read it. CUDA tensors can't be converted to NumPy directly, and bfloat16 tensors can't
    be converted at all. Anything that isn't a tensor (None, ndarray, list) passes through."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().float().cpu()
    return x


def unwrap_batch(x):
    """Batch-of-1 vs single-sample isn't pinned for this experimental pipeline --
    take element 0 if it looks like a batched list/tuple (the docs use [0])."""
    if isinstance(x, (list, tuple)) and len(x) >= 1:
        return x[0]
    return x


# --------------------------------------------------------------------------- #
# Progress display + time estimate
# --------------------------------------------------------------------------- #
HISTORY_FILE = Path(__file__).with_name("h3_timing_history.json")

STAGES = [
    ("load", "Load model"),
    ("prep", "Encode prompt & references"),
    ("denoise", "Denoise"),
    ("decode", "Decode video + audio"),
    ("write", "Write MP4"),
]

# Client-side ticking clock: keeps counting during the long phases that have no step counter
# (model load, text encoding, VAE decode) without needing a background thread.
CLOCK_HTML = """
<div style="font-family:sans-serif;font-size:15px;color:#9aa0a6;">
  ⏲️ Elapsed: <b id="t">00:00</b>
</div>
<script>
  const t0 = Date.now(), el = document.getElementById("t");
  setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    el.textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }, 250);
</script>
"""


def fmt_dur(seconds) -> str:
    """8 -> '8s', 90 -> '1m 30s', 3725 -> '1h 02m'."""
    s = int(round(max(float(seconds), 0)))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def load_history() -> dict:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {}


def save_history(history: dict):
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception:
        pass  # read-only folder etc. -- estimates just won't persist


def run_key(workflow, height, width, frames, low_vram, blocks) -> str:
    return f"run|{workflow}|{height}x{width}|{frames}f|lowvram={int(low_vram)}|blocks={blocks}"


def load_time_key(workflow, low_vram, blocks) -> str:
    return f"load|{workflow}|lowvram={int(low_vram)}|blocks={blocks}"


def estimate_run(history: dict, rkey: str, lkey: str, steps: int, model_in_memory: bool):
    """Estimate from your last finished run with the same settings (None if there isn't one)."""
    r = history.get(rkey)
    if not r:
        return None
    est = {
        "load": 0.0 if model_in_memory else float(history.get(lkey, {}).get("load_s", 0.0)),
        "prep": float(r["prep_s"]),
        "step_s": float(r["step_s"]),
        "post": float(r["post_s"]),
    }
    est["denoise"] = est["step_s"] * steps
    est["total"] = est["load"] + est["prep"] + est["denoise"] + est["post"]
    return est


class GenerationTracker:
    """Live panel: stage checklist, denoising step counter, ticking clock and ETA.

    The step counter is fed by diffusers' own denoising loop (see denoise_progress_hook)."""

    def __init__(self, container, steps_planned: int, est=None, hook_available: bool = True):
        with container:
            self.stage_ph = st.empty()
            self.bar_ph = st.empty()
            self.stat_ph = st.empty()
            self.clock_ph = st.empty()
        self.t0 = time.perf_counter()
        self.total_steps = int(steps_planned)
        self.done_steps = 0
        self.est = est
        self.hook_available = hook_available
        self.marks = {}          # stage -> {"start", "end", "note"}
        self.current = None
        self.failed = False
        self.step_ends = []      # perf_counter() after each finished denoising step
        try:
            with self.clock_ph.container():
                components.html(CLOCK_HTML, height=34)
        except Exception:
            pass
        self._render()

    # ---- stage handling ---------------------------------------------------- #
    def stage(self, name, note=None):
        """End the current stage and start `name` (None = just end the current one)."""
        now = time.perf_counter()
        if self.current and self.marks[self.current]["end"] is None:
            self.marks[self.current]["end"] = now
        self.current = name
        if name:
            self.marks[name] = {"start": now, "end": None, "note": note}
        self._render()

    def skip(self, name, note):
        now = time.perf_counter()
        self.marks[name] = {"start": now, "end": now, "note": note}
        self._render()

    def on_bar(self, n, total):
        """Called by the hooked tqdm bar of diffusers' denoising loop: n of `total` steps done."""
        if total:
            self.total_steps = int(total)
        if "denoise" not in self.marks:
            self.stage("denoise")            # the bar exists -> prompt/reference prep is over
        while self.done_steps < n:
            self.done_steps += 1
            self.step_ends.append(time.perf_counter())
        if n >= self.total_steps and self.current == "denoise":
            self.stage("decode")             # all steps done -> VAE decode + audio decode
        else:
            self._render()

    def sec_per_step(self):
        d = self.marks.get("denoise")
        if not d or not self.step_ends:
            return None
        if len(self.step_ends) >= 2:         # skip step 1: it includes warm-up
            return (self.step_ends[-1] - self.step_ends[0]) / (len(self.step_ends) - 1)
        return self.step_ends[0] - d["start"]

    # ---- rendering ---------------------------------------------------------- #
    def _render(self):
        try:  # a display glitch must never break generation
            self._render_stages()
            self._render_steps()
        except Exception:
            pass

    def _render_stages(self):
        lines = []
        for key, label in STAGES:
            m = self.marks.get(key)
            if key == "denoise":
                label = f"Denoise — step {self.done_steps} / {self.total_steps}"
            if m is None:
                lines.append(f"⬜ {label}")
            elif m["end"] is not None:
                lines.append(f"✅ {label} — {m['note'] or fmt_dur(m['end'] - m['start'])}")
            else:
                hint = ""
                if self.est and key in ("load", "prep") and self.est.get(key):
                    hint = f" (last time ≈ {fmt_dur(self.est[key])})"
                lines.append(f"{'❌' if self.failed else '⏳'} **{label}**{hint}")
        self.stage_ph.markdown("  \n".join(lines))

    def _render_steps(self):
        n, N = self.done_steps, self.total_steps
        if "denoise" in self.marks:
            self.bar_ph.progress(min(n / N, 1.0) if N else 0.0, text=f"Denoising step {n} / {N}")
        else:
            self.bar_ph.progress(0.0, text=f"Waiting to start {N} denoising steps…")
        self.stat_ph.markdown(self._stat_line())

    def _stat_line(self) -> str:
        n, N = self.done_steps, self.total_steps
        per = self.sec_per_step()
        post = self.est["post"] if self.est else None
        if not self.hook_available:
            return (f"⏱️ {N} denoising steps. A live step counter isn't available with this diffusers "
                    "version — showing elapsed time only.")
        if per and n < N:
            left = per * (N - n)
            txt = f"⏱️ **{per:.1f} s/step** · denoising finishes in ≈ **{fmt_dur(left)}**"
            if post:
                txt += (f" · then ≈ {fmt_dur(post)} to decode + save "
                        f"→ about **{fmt_dur(left + post)}** left in total")
            else:
                txt += " · then decode + save (not timed yet)"
            return txt
        if per:
            return f"⏱️ {per:.1f} s/step · all {N} steps done — decoding video + audio…"
        if self.est:
            return f"⏱️ Estimated total ≈ **{fmt_dur(self.est['total'])}** (from your last run with these settings)."
        return "⏱️ The live estimate appears after the first denoising step."

    # ---- end of run ---------------------------------------------------------- #
    def finish(self) -> dict:
        self.stage(None)
        total = time.perf_counter() - self.t0
        self.clock_ph.markdown(f"✅ **Done in {fmt_dur(total)}**")

        def dur(k):
            m = self.marks.get(k)
            return (m["end"] - m["start"]) if m and m["end"] is not None else 0.0

        real_load = "load" in self.marks and not self.marks["load"].get("note")
        return {
            "total_s": total,
            "load_s": dur("load") if real_load else None,
            "prep_s": dur("prep"),
            "denoise_s": dur("denoise"),
            "steps": self.total_steps,
            "post_s": dur("decode") + dur("write"),
        }

    def fail(self):
        self.failed = True
        self.clock_ph.markdown(f"❌ Stopped after {fmt_dur(time.perf_counter() - self.t0)}")
        self._render()


def denoise_progress_hook(on_bar):
    """Context manager that makes diffusers' denoising progress bar also report (n, total) to
    `on_bar`. It only observes the bar: the console bar still prints and the pipeline call is
    unchanged. Falls back to a no-op if this diffusers version doesn't expose that `tqdm`."""
    if not hasattr(h3_mp_module, "tqdm"):
        return contextlib.nullcontext()

    class ProgressTqdm(tqdm_auto):
        # Keeps its own counter: tqdm itself stops counting when a bar is created with
        # disable=True (e.g. pipe.set_progress_bar_config(disable=True)).
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._h3_done = 0
            self._report()

        def update(self, n=1):
            out = super().update(n)
            self._h3_done += 1 if n is None else n
            self._report()
            return out

        def _report(self):
            try:
                on_bar(self._h3_done, self.total)
            except Exception:
                pass

    return mock.patch.object(h3_mp_module, "tqdm", ProgressTqdm)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("MiniMax-H3 — Image + Text → Video")
st.caption(
    "Local inference against your downloaded INT8 weights via diffusers' ModularPipeline. "
    "Validated by the model card on a 24–32 GB CUDA GPU with ~75 GB+ system RAM."
)

if not torch.cuda.is_available():
    st.error("No CUDA GPU detected. This recipe (group offload + bf16) targets an NVIDIA GPU.")
    st.stop()

with st.sidebar:
    st.header("Model")
    model_dir = st.text_input("Local weights folder", value=DEFAULT_MODEL_DIR)
    st.caption(f"📁 Reading the model from: `{model_dir}`")
    if ALLOW_DOWNLOADS:
        st.caption(f"⬇️ Downloads allowed — saved only inside: `{EXTRA_DL_DIR}`")
    else:
        st.caption("🔒 Offline mode: nothing will be downloaded.")
    if index_status(model_dir) == "hub":
        st.caption("⚠️ modular_model_index.json still names the Hub repo — it will be pointed at "
                   "your local folder when you press Generate.")

    workflow = st.selectbox(
        "Workflow", ["ref2va", "fl2va", "t2va"], index=0,
        help="ref2va = one reference image + text (default). fl2va = first/optional-last keyframe. "
             "t2va = text only. Switching workflow reloads the model.",
    )
    low_vram = st.checkbox(
        "Lower-VRAM mode (~12–16 GB)", value=False,
        help="Model card: also group-offload the video VAE (leaf_level, no stream) and use a "
             "smaller canvas such as 960×544.",
    )
    with st.expander("Advanced"):
        num_blocks_per_group = st.number_input("Transformer offload block group size", 1, 8, 1,
                                               help="Model card value: 1.")
    if st.button("Unload model / clear cache"):
        free_pipeline()
        st.session_state.pop("loaded_key", None)
        st.rerun()

    st.divider()
    st.subheader("Hardware")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    st.caption(f"🎮 {torch.cuda.get_device_name(0)} · {vram_gb:.0f} GB VRAM")
    if vram_gb < 24 and not low_vram:
        st.warning("Under 24 GB VRAM: turn on Lower-VRAM mode and keep the 544 × 960 canvas.")
    ram_gb = host_ram_gb()
    if ram_gb is not None:
        st.caption(f"🧠 {ram_gb:.0f} GB system RAM")
        if ram_gb < MODEL_CARD_RAM_GB:
            st.warning(f"The model card says to plan for ≈{MODEL_CARD_RAM_GB} GB+ of system RAM "
                       "(the weights are streamed from host memory).")

    hard_errors, soft_warnings = check_workflow_ready(model_dir, workflow)
    for w in soft_warnings:
        st.caption(f"⚠️ {w}")
    for err in hard_errors:
        st.error(err)

col_in, col_out = st.columns(2, gap="large")

with col_in:
    ref_image = fl_image = fl_last_image = None
    if workflow == "ref2va":
        up = st.file_uploader("Reference image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            ref_image = Image.open(up).convert("RGB")
            st.image(ref_image, caption="Reference", width="stretch")
    elif workflow == "fl2va":
        up_f = st.file_uploader("First frame", type=["png", "jpg", "jpeg", "webp"], key="ff")
        up_l = st.file_uploader("Last frame (optional)", type=["png", "jpg", "jpeg", "webp"], key="lf")
        if up_f:
            fl_image = Image.open(up_f).convert("RGB")
            st.image(fl_image, caption="First frame", width="stretch")
        if up_l:
            fl_last_image = Image.open(up_l).convert("RGB")
            st.image(fl_last_image, caption="Last frame", width="stretch")

    prompt = st.text_area("Prompt", height=100, placeholder="Describe the scene, motion, mood...")

    with st.expander("Generation settings", expanded=True):
        preset = st.selectbox(
            "Canvas", ["544 × 960 (validated, lower VRAM)", "768 × 1344 (also fits 32GB)", "Custom"], index=0,
        )
        if preset.startswith("544"):
            height, width = 544, 960
        elif preset.startswith("768"):
            height, width = 768, 1344
        else:
            c1, c2 = st.columns(2)
            height = c1.number_input("Height (multiple of 32)", 32, 2048, 544, step=32)
            width = c2.number_input("Width (multiple of 32)", 32, 2048, 960, step=32)

        size_error = (int(height) % 32 != 0) or (int(width) % 32 != 0)
        if size_error:
            st.error("Height and width must be multiples of 32.")

        num_frames = st.number_input("Frames", 8, 241, 124, step=1,
                                     help="124 (~5 s) is the model card's validated value.")
        steps = st.number_input("Inference steps", 1, 100, 20, step=1,
                                help="20 is the model card's validated value.")
        fps = st.number_input("Output FPS", 1, 60, 24,
                              help="MiniMax-H3 generates at 24 fps (per the diffusers docs).")
        seed_mode = st.radio("Seed", ["Fixed", "Random"], horizontal=True)
        seed = st.number_input("Seed value", 0, 2**31 - 1, 42) if seed_mode == "Fixed" else None

    ready_input = (
        (workflow == "ref2va" and ref_image is not None)
        or (workflow == "fl2va" and fl_image is not None)
        or (workflow == "t2va")
    )
    go = st.button(
        "Generate video", type="primary",
        disabled=not (ready_input and prompt.strip() and not hard_errors and not size_error),
    )

    # Steps required + time estimate, shown before you press Generate.
    rkey = run_key(workflow, int(height), int(width), int(num_frames), low_vram, int(num_blocks_per_group))
    lkey = load_time_key(workflow, low_vram, int(num_blocks_per_group))
    _history = load_history()
    _in_memory = st.session_state.get("loaded_key") == (
        model_dir, workflow, bool(low_vram), int(num_blocks_per_group)
    )
    est_now = estimate_run(_history, rkey, lkey, int(steps), _in_memory)
    st.caption(f"🪜 **{int(steps)} denoising steps** — each step is one full pass of the 33B "
               "transformer, streamed from system RAM.")
    if est_now:
        breakdown = (f"prompt/reference prep {fmt_dur(est_now['prep'])} + {int(steps)} steps × "
                     f"{est_now['step_s']:.1f}s + decode/save {fmt_dur(est_now['post'])}")
        if est_now["load"]:
            breakdown += f" + model load {fmt_dur(est_now['load'])}"
        elif not _in_memory:
            breakdown += " (+ model load, not timed yet)"
        st.info(f"⏱️ Estimated ≈ **{fmt_dur(est_now['total'])}** — from your last run with these "
                f"settings: {breakdown}.")
    else:
        st.caption("⏱️ No timing history for these settings yet — a live estimate appears after the "
                   "first denoising step. The very first model load reads ~100 GB from disk.")

with col_out:
    if go:
        tracker = None
        try:
            note = localize_index(model_dir)
            if note:
                st.info(note)

            # Keep exactly one pipeline in memory: drop the old one before loading a different config.
            load_key = (model_dir, workflow, bool(low_vram), int(num_blocks_per_group))
            prev_key = st.session_state.get("loaded_key")
            if prev_key is not None and prev_key != load_key:
                free_pipeline()

            already_loaded = prev_key == load_key
            panel = st.container(border=True)
            with panel:
                st.markdown(
                    f"**Generating** · {workflow} · {int(width)}×{int(height)} · "
                    f"{int(num_frames)} frames · **{int(steps)} denoising steps**"
                )
                if not already_loaded:
                    st.caption("First load reads ~100 GB from your local folder — be patient.")
            tracker = GenerationTracker(
                panel, int(steps),
                est=estimate_run(load_history(), rkey, lkey, int(steps), already_loaded),
                hook_available=hasattr(h3_mp_module, "tqdm"),
            )
            if already_loaded:
                tracker.skip("load", "already in memory")
            else:
                tracker.stage("load")
            pipe = load_pipeline(*load_key)
            st.session_state["loaded_key"] = load_key
            tracker.stage("prep")

            gen_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            generator = torch.Generator().manual_seed(int(gen_seed))

            call_kwargs = dict(
                prompt=prompt,
                num_frames=int(num_frames),
                height=int(height),   # must be multiples of 32
                width=int(width),
                num_inference_steps=int(steps),
                generator=generator,
                output=["videos", "audio", "sampling_rate"],
            )
            ref_tmp_path = None
            if workflow == "ref2va":
                # references=[...]: the blocks never open files or accept raw PIL images, so the
                # image is decoded into a MiniMaxH3ImageReference via from_file (path -> reference).
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    ref_tmp_path = tf.name
                ref_image.save(ref_tmp_path)
                call_kwargs["references"] = [MiniMaxH3ImageReference.from_file(ref_tmp_path)]
            elif workflow == "fl2va":
                # fl2va: image= (and optional last_image=)
                call_kwargs["image"] = fl_image
                if fl_last_image is not None:
                    call_kwargs["last_image"] = fl_last_image

            torch.cuda.reset_peak_memory_stats()
            try:
                # Called directly, as in the model card (no inference_mode wrapper). The hook only
                # observes diffusers' own denoising-step progress bar; it doesn't change the call.
                with denoise_progress_hook(tracker.on_bar):
                    out = pipe(**call_kwargs)
            finally:
                if ref_tmp_path and os.path.exists(ref_tmp_path):
                    os.remove(ref_tmp_path)
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            # Model card: video, audio, sr = out["videos"], out["audio"], out["sampling_rate"]
            frames = unwrap_batch(out["videos"])
            audio = unwrap_batch(out["audio"])
            sample_rate = out["sampling_rate"]
            if isinstance(sample_rate, (list, tuple)):
                sample_rate = sample_rate[0]

            # Capture the debug info NOW, from the raw outputs (metadata only -- no host copy).
            debug_info = {
                "frames": describe(frames),
                "audio": describe(audio),
                "sample_rate": sample_rate,
            }

            # diffusers' encode_video muxes the video and its generated soundtrack into one mp4.
            # The audio is moved to host memory first: it comes back on cuda:0 (and possibly in
            # bfloat16), and NumPy/PyAV cannot read either.
            tracker.stage("write")
            with tempfile.TemporaryDirectory() as td:
                final_path = os.path.join(td, "output.mp4")
                enc_kwargs = dict(fps=int(fps), output_path=final_path)
                if audio is not None:
                    enc_kwargs.update(audio=to_cpu(audio), audio_sample_rate=int(sample_rate))
                encode_video(frames, **enc_kwargs)

                with open(final_path, "rb") as f:
                    st.session_state["video_bytes"] = f.read()

            timing = tracker.finish()
            if timing["denoise_s"] > 0:  # only learn from runs where the step counter worked
                hist = load_history()
                if timing["load_s"] is not None and timing["load_s"] > 5:  # instant = cached, not a real load
                    hist[lkey] = {"load_s": round(timing["load_s"], 1)}
                hist[rkey] = {
                    "prep_s": round(timing["prep_s"], 1),
                    "step_s": round(timing["denoise_s"] / max(timing["steps"], 1), 2),
                    "post_s": round(timing["post_s"], 1),
                    "steps": timing["steps"],
                    "updated": datetime.now().isoformat(timespec="seconds"),
                }
                save_history(hist)
            st.session_state["last_timing"] = timing

            st.session_state["last_seed"] = gen_seed
            st.session_state["last_peak_gb"] = peak_gb
            st.session_state["debug_info"] = debug_info
        except Exception as exc:
            if tracker is not None:
                tracker.fail()
            st.error("Generation failed. See the traceback below.")
            if not ALLOW_DOWNLOADS:
                st.warning(
                    "The app is running OFFLINE, so it will not download anything. If the error names a "
                    "missing file or repo, run `python download_weights.py --check` to see what is missing "
                    "from your model folder, or relaunch with `MINIMAX_H3_ALLOW_DOWNLOAD=1` to let it fetch "
                    f"the file — it will be saved inside `{EXTRA_DL_DIR}` (in your model folder)."
                )
            if "out of memory" in str(exc).lower():
                st.warning(
                    "Out of GPU memory. Per the model card: turn on Lower-VRAM mode, use 544 × 960, "
                    "and/or reduce the frame count."
                )
            st.exception(exc)

    if "video_bytes" in st.session_state:
        st.video(st.session_state["video_bytes"])
        st.download_button(
            "Download .mp4", st.session_state["video_bytes"],
            file_name="minimax_h3_output.mp4", mime="video/mp4",
        )
        st.caption(
            f"Seed: {st.session_state.get('last_seed')} · "
            f"Peak VRAM: {st.session_state.get('last_peak_gb', 0):.2f} GB "
            "(model card: ~16.4 GB t2va, ~18.2 GB ref2va at 124 frames, 544×960)"
        )
        t = st.session_state.get("last_timing")
        if t:
            if t["denoise_s"] > 0:
                st.caption(
                    f"⏱️ Total {fmt_dur(t['total_s'])} · prep {fmt_dur(t['prep_s'])} · "
                    f"{t['steps']} steps in {fmt_dur(t['denoise_s'])} "
                    f"({t['denoise_s'] / max(t['steps'], 1):.1f} s/step) · "
                    f"decode + save {fmt_dur(t['post_s'])}"
                )
            else:
                st.caption(f"⏱️ Total {fmt_dur(t['total_s'])}")
        with st.expander("Debug: raw output shapes"):
            st.write(st.session_state.get("debug_info", {}))
