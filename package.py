"""Assemble the C151/ submission folder from this repo, exactly per the PS3 spec.

Usage: python package.py [--dest C151]

Produces:
    C151/
    ├── predictions.zip          (the *_predictions.csv files, at the zip's top level)
    ├── app/                     (app sources + common/ + the four <Sub>/{code,model} it imports)
    └── Optional_Items/
        ├── write_up.md
        └── <Door|ACV|Rail Corrugation|SHM>/{code, model}

demo_video.* must be added to C151/ manually after recording.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBSYSTEMS = ["Door", "ACV", "Rail Corrugation", "SHM"]
PREDICTIONS = ["door_predictions.csv", "acv_predictions.csv", "rail_predictions.csv", "shm_predictions.csv"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep", ".ipynb_checkpoints")


def copytree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    return any(p.is_file() for p in dst.rglob("*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="C151")
    dest = ROOT / ap.parse_args().dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    warnings: list[str] = []

    # --- predictions.zip (flat, only the *_predictions.csv files)
    missing = [p for p in PREDICTIONS if not (ROOT / "predictions" / p).exists()]
    with zipfile.ZipFile(dest / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in PREDICTIONS:
            f = ROOT / "predictions" / p
            if f.exists():
                z.write(f, arcname=p)
    if missing:
        warnings.append(f"predictions.zip is missing: {missing}")

    # --- app/ (self-contained: app sources + common/ + subsystem code+model, repo-relative layout)
    app_has_sources = copytree(ROOT / "app", dest / "app")
    copytree(ROOT / "common", dest / "app/common")
    for sub in SUBSYSTEMS:
        copytree(ROOT / sub / "code", dest / "app" / sub / "code")
        copytree(ROOT / sub / "model", dest / "app" / sub / "model")
    if not app_has_sources:
        warnings.append("app/ has no sources yet (compulsory deliverable, item 3)")

    # --- Optional_Items/
    opt = dest / "Optional_Items"
    opt.mkdir()
    if (ROOT / "write_up.md").exists():
        shutil.copy2(ROOT / "write_up.md", opt / "write_up.md")
    else:
        warnings.append("write_up.md not found")
    for sub in SUBSYSTEMS:
        if not copytree(ROOT / sub / "code", opt / sub / "code"):
            warnings.append(f"Optional_Items/{sub}/code is empty")
        if not copytree(ROOT / sub / "model", opt / sub / "model"):
            warnings.append(f"Optional_Items/{sub}/model is empty")

    if not list(dest.glob("demo_video.*")):
        warnings.append("demo_video.<mp4|mov> not present (compulsory deliverable, item 1) - record and drop it into C151/")

    print(f"assembled {dest}")
    for p in sorted(dest.rglob("*")):
        if p.is_file() and "Optional_Items" not in str(p.relative_to(dest)) and "app" not in str(p.relative_to(dest)).split("\\")[0]:
            print("  ", p.relative_to(dest))
    n_files = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"total files: {n_files}")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
