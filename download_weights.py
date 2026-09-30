"""
Download / verify MiniMax-H3 (INT8, diffusers/torchao format) weights.

Repo: abhishekchohan/minimax-h3-int8

What this version does:
  1. Looks at your local folder (--dir) FIRST and compares it, file by file
     (path + byte size), with the real file list on the Hub.
  2. Tells you clearly whether the folder is COMPLETE or NOT, and exactly
     what is missing / truncated / left over from interrupted downloads.
  3. Downloads ONLY the missing or wrong-size files (never re-downloads
     files you already have, and never touches the ~100 GB you already have).
  4. Re-checks after downloading and prints a final verdict:
        EVERYTHING OK   -> exit code 0
        NOT OK          -> exit code 1

Usage:
    python download_weights.py                    # verify, fetch what's missing, final verdict
    python download_weights.py --check            # verify only, download nothing
    python download_weights.py --workflows ref2va # ignore the other workflow's transformer folder
    python download_weights.py --force            # re-download everything selected
    python download_weights.py --localize-index   # also point modular_model_index.json at --dir
                                                  # (stops the pipeline fetching components from the Hub)
    python download_weights.py --dir /other/path --token hf_xxx --no-hf-transfer

Env vars (defaults if the matching flag isn't passed):
    MINIMAX_H3_DIR   local folder
    HF_TOKEN         auth token
"""

import os
import sys


def _maybe_enable_hf_transfer():
    """Must run BEFORE huggingface_hub is imported to have any effect."""
    if "--no-hf-transfer" in sys.argv:
        return
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        print("(hf_transfer not installed -- plain downloader will be used; "
              "pip install \"huggingface_hub[hf_transfer]\" to speed up.)\n")


_maybe_enable_hf_transfer()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO_ID = "abhishekchohan/minimax-h3-int8"
DEFAULT_DIR = os.environ.get(
    "MINIMAX_H3_DIR",
    "/media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8",
)
SAMPLE_MARKERS = ("sample_",)  # demo assets like sample_t2va.mp4 / .png


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=DEFAULT_DIR, help="local folder that should contain the weights")
    p.add_argument("--workflows", nargs="+", choices=["t2va", "fl2va", "ref2va", "all"], default=["all"],
                   help="which workflow(s) to require (t2va and fl2va share the same files)")
    p.add_argument("--revision", default="main")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--check", action="store_true", help="verify only; download nothing")
    p.add_argument("--force", action="store_true", help="re-download all selected files")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--include-samples", action="store_true", help="also require the demo sample_* files")
    p.add_argument("--localize-index", action="store_true",
                   help="rewrite the Hub repo id in modular_model_index.json to --dir (backup kept)")
    p.add_argument("--no-hf-transfer", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def gb(n):
    return f"{n / 1024 ** 3:.2f} GB"


def wanted(rel, workflows, include_samples):
    top = rel.split("/", 1)[0]
    if not include_samples and any(m in rel for m in SAMPLE_MARKERS):
        return False
    if "all" in workflows:
        return True
    if top == "transformer" and not ({"t2va", "fl2va"} & set(workflows)):
        return False
    if top == "transformer_ref" and "ref2va" not in workflows:
        return False
    return True


def fetch_expected(args):
    """{relative_path: size_in_bytes} for every file the chosen workflows need."""
    api = HfApi(token=args.token)
    out = {}
    for it in api.list_repo_tree(REPO_ID, revision=args.revision, recursive=True):
        size = getattr(it, "size", None)          # folders have no size
        if size is not None and wanted(it.path, args.workflows, args.include_samples):
            out[it.path] = size
    return out


def scan(root, expected):
    """Compare local folder with expected. Returns (ok, missing, wrong)."""
    ok, missing, wrong = 0, [], []
    for rel, size in expected.items():
        p = root / rel
        if not p.is_file():
            missing.append((rel, size))
        elif p.stat().st_size != size:
            wrong.append((rel, size, p.stat().st_size))
        else:
            ok += 1
    return ok, missing, wrong


def leftovers(root):
    return list(root.rglob("*.incomplete"))


def print_report(root, expected, ok, missing, wrong):
    total = sum(expected.values())
    print(f"  expected : {len(expected)} files ({gb(total)})")
    print(f"  present  : {ok} files OK")
    if missing:
        print(f"  MISSING  : {len(missing)} files ({gb(sum(s for _, s in missing))})")
        for rel, size in missing:
            print(f"      - {rel}  ({gb(size)})")
    if wrong:
        print(f"  WRONG SIZE: {len(wrong)} files (truncated or corrupted)")
        for rel, exp, act in wrong:
            print(f"      - {rel}  expected {gb(exp)}, found {gb(act)}")
    part = leftovers(root)
    if part:
        print(f"  Leftover partial downloads: {len(part)}")
        for p in part:
            print(f"      - {p}")


def localize_index(root):
    """Point modular_model_index.json at the local folder instead of the Hub repo id."""
    idx = root / "modular_model_index.json"
    if not idx.exists():
        print("  modular_model_index.json not found -- nothing to localize.")
        return
    text = idx.read_text()
    if REPO_ID not in text:
        print("  modular_model_index.json has no Hub repo id -- already local (or different layout).")
        return
    bak = idx.with_name(idx.name + ".bak")
    if not bak.exists():
        bak.write_text(text)
    local = json.dumps(str(root.resolve()))[1:-1]   # JSON-safe path
    idx.write_text(text.replace(REPO_ID, local))
    print(f"  Rewrote {text.count(REPO_ID)} repo-id reference(s) in {idx.name} (backup: {bak.name}).")


def download(root, files, args, force_names):
    def one(rel):
        hf_hub_download(REPO_ID, rel, revision=args.revision, local_dir=str(root),
                        token=args.token, force_download=(args.force or rel in force_names))
        return rel

    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(one, rel): rel for rel in files}
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                fut.result()
                done += 1
                print(f"  [{done}/{len(files)}] downloaded {rel}")
            except Exception as e:  # keep going; final verdict will show what failed
                print(f"  FAILED {rel}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    root = Path(args.dir)

    print(f"Repo     : {REPO_ID} (revision {args.revision})")
    print(f"Local dir: {root}")
    print(f"Workflows: {args.workflows}\n")

    if not root.exists():
        if args.check:
            sys.exit(f"NOT OK: folder does not exist: {root}")
        root.mkdir(parents=True, exist_ok=True)

    print("Reading the real file list from the Hub (metadata only)...")
    try:
        expected = fetch_expected(args)
    except Exception as e:
        sys.exit(f"Could not read the repo file list: {type(e).__name__}: {e}\n"
                 "Check your internet connection / HF_HUB_OFFLINE / token.")

    print("\nChecking your local folder:")
    ok, missing, wrong = scan(root, expected)
    print_report(root, expected, ok, missing, wrong)

    to_get = [r for r, _ in missing] + [r for r, _, _ in wrong]
    if args.force:
        to_get = list(expected)

    if to_get and not args.check:
        need = sum(expected[r] for r in to_get)
        free = shutil.disk_usage(root).free
        print(f"\nNeed to download {len(to_get)} file(s), {gb(need)}; free space {gb(free)}.")
        if free < need * 1.05:
            sys.exit("NOT OK: not enough free disk space for the missing files.")
        download(root, to_get, args, {r for r, _, _ in wrong})
        print("\nRe-checking after download:")
        ok, missing, wrong = scan(root, expected)
        print_report(root, expected, ok, missing, wrong)
    elif to_get and args.check:
        print("\n(--check: nothing downloaded. Run again without --check to fetch the files above.)")

    if args.localize_index:
        print("\nLocalizing index:")
        localize_index(root)

    complete = not missing and not wrong and not leftovers(root)
    on_disk = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print("\n" + "=" * 60)
    if complete:
        print(f"EVERYTHING OK -- all {len(expected)} required files are present with the right size.")
        print(f"Total on disk in {root}: {gb(on_disk)}")
        print("If the app still downloads something, it is being requested from a Hub repo id "
              "in modular_model_index.json -- run again with --localize-index.")
    else:
        print("NOT OK -- the folder is incomplete or has partial files (see the lists above).")
        if args.check:
            print("Run without --check to download only what is missing.")
    print("=" * 60)
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
