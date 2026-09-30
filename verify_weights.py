"""
verify_weights.py -- check that a local MiniMax-H3 (INT8) folder is complete.

Compares every file in the Hub repo (path + byte size, metadata only --
nothing is downloaded) against your local folder, and reports:
  * files that are missing
  * files whose size doesn't match (truncated / interrupted downloads)
  * leftover *.incomplete files
  * any repo-id-looking strings inside the root JSON index files (heuristic:
    these are the places where the pipeline may decide to go to the Hub)

Usage:
    python verify_weights.py
    python verify_weights.py --dir "/media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8"
    python verify_weights.py --fix                    # download ONLY missing / wrong-size files
    python verify_weights.py --workflows ref2va       # ignore the other workflow's transformer folder

It also saves h3_manifest.json next to this script so the Streamlit app can
run the same check offline (see check_against_manifest below).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ID = "abhishekchohan/minimax-h3-int8"
MANIFEST = Path(__file__).with_name("h3_manifest.json")
DEFAULT_DIR = os.environ.get(
    "MINIMAX_H3_DIR",
    "/media/avidmech/data/MinMaxH3 (Image_to video)/models/minimax-h3-int8",
)


# --------------------------------------------------------------------------- #
# Core comparison (no network needed once you have a manifest)
# --------------------------------------------------------------------------- #
def wanted(rel: str, workflows) -> bool:
    """Same folder logic as download_weights.py."""
    top = rel.split("/", 1)[0]
    if "sample_" in rel:                       # demo mp4/png, not needed
        return False
    if "all" in workflows:
        return True
    if top == "transformer" and not ({"t2va", "fl2va"} & set(workflows)):
        return False
    if top == "transformer_ref" and "ref2va" not in workflows:
        return False
    return True


def compare(model_dir, manifest, workflows=("all",)):
    """Returns (ok_count, missing[(rel, size)], wrong[(rel, expected, actual)])."""
    root = Path(model_dir)
    missing, wrong, ok = [], [], 0
    for rel, size in manifest.items():
        if not wanted(rel, workflows):
            continue
        p = root / rel
        if not p.is_file():
            missing.append((rel, size))
        elif p.stat().st_size != size:
            wrong.append((rel, size, p.stat().st_size))
        else:
            ok += 1
    return ok, missing, wrong


def check_against_manifest(model_dir, workflow):
    """For the Streamlit app: returns a list of hard-error strings ([] = complete)."""
    if not MANIFEST.exists():
        return []  # run verify_weights.py once (online) to create the manifest
    manifest = json.loads(MANIFEST.read_text())
    _, missing, wrong = compare(model_dir, manifest, (workflow,))
    if not missing and not wrong:
        return []
    names = [r for r, _ in missing] + [r for r, _, _ in wrong]
    shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
    return [
        f"{len(missing)} file(s) missing and {len(wrong)} with the wrong size in {model_dir}: "
        f"{shown}. Run `python verify_weights.py --fix`."
    ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gb(n):
    return f"{n / 1024 ** 3:.2f} GB"


def fetch_manifest(revision, token):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    return {
        it.path: it.size
        for it in api.list_repo_tree(REPO_ID, revision=revision, recursive=True)
        if getattr(it, "size", None) is not None  # folders have no size
    }


def external_refs(model_dir):
    """Heuristic: repo-id-looking strings (org/name) in root-level JSON files."""
    refs = {}
    for f in Path(model_dir).glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", x):
                refs.setdefault(f.name, set()).add(x)
    return refs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--workflows", nargs="+", choices=["t2va", "fl2va", "ref2va", "all"], default=["all"])
    ap.add_argument("--revision", default="main")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--fix", action="store_true", help="download only the missing / wrong-size files")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        sys.exit(f"Folder not found: {root}")

    # 1) Get the reference file list (online), else fall back to the saved one.
    try:
        manifest = fetch_manifest(args.revision, args.token)
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"Fetched file list from {REPO_ID}: {len(manifest)} files (saved to {MANIFEST.name})")
    except Exception as e:
        if not MANIFEST.exists():
            sys.exit(f"Could not reach the Hub and no saved manifest exists.\n{e}")
        manifest = json.loads(MANIFEST.read_text())
        print(f"Hub unreachable ({type(e).__name__}); using saved {MANIFEST.name}")

    # 2) Compare.
    ok, missing, wrong = compare(root, manifest, args.workflows)
    print(f"\nFolder: {root}\nWorkflows checked: {args.workflows}")
    print(f"OK: {ok} files")

    if missing:
        print(f"\nMISSING ({len(missing)} files, {gb(sum(s for _, s in missing))}):")
        for rel, size in missing:
            print(f"  {rel}  ({gb(size)})")
    if wrong:
        print(f"\nWRONG SIZE ({len(wrong)} files):")
        for rel, exp, act in wrong:
            print(f"  {rel}  expected {gb(exp)}, found {gb(act)}")

    leftovers = list(root.rglob("*.incomplete"))
    if leftovers:
        print(f"\nLEFTOVER partial downloads ({len(leftovers)}):")
        for p in leftovers:
            print(f"  {p}")

    refs = external_refs(root)
    if refs:
        print("\nRepo-id-looking strings in root JSON files (candidates for Hub loading):")
        for fname, vals in refs.items():
            print(f"  {fname}: {sorted(vals)}")

    complete = not missing and not wrong
    if complete:
        print("\nLocal folder matches the repo. If the app still downloads something, it is being "
              "requested from a Hub repo (see the JSON strings above), not because files are missing.")
        return

    # 3) Optionally fetch only what's needed.
    if not args.fix:
        print("\nIncomplete. Re-run with --fix to download only the files listed above.")
        sys.exit(1)

    from huggingface_hub import hf_hub_download

    bad = {r for r, _, _ in wrong}
    todo = [r for r, _ in missing] + sorted(bad)
    for i, rel in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] downloading {rel}")
        hf_hub_download(
            REPO_ID, rel, revision=args.revision, local_dir=str(root),
            token=args.token, force_download=(rel in bad),
        )
    print("\nDone. Re-run this script to confirm everything matches.")


if __name__ == "__main__":
    main()
