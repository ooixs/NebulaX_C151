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
import importlib.metadata
import re
import shutil
import zipfile
from pathlib import Path

from common.artifacts import MANIFEST, activate_model, resolve_model

ROOT = Path(__file__).resolve().parent
SUBSYSTEMS = ["Door", "ACV", "Rail Corrugation", "SHM"]
PREDICTIONS = ["door_predictions.csv", "acv_predictions.csv", "rail_predictions.csv", "shm_predictions.csv"]
MODEL_FILES = {
    "Door": ("door_model.joblib",),
    "ACV": ("acv_model.joblib",),
    "Rail Corrugation": ("rail_model.joblib",),
    "SHM": ("shm_model.json", "shm_model.joblib"),
}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep", ".ipynb_checkpoints")


def copytree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    return any(p.is_file() for p in dst.rglob("*"))


def assemble(dest: Path, root: Path = ROOT):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"destination already exists; choose a new directory: {dest}")
    if any(dest.is_relative_to(root / name) for name in ["app", "common", "rail_corrugation", *SUBSYSTEMS]):
        raise ValueError("destination must not be inside a source directory")
    models = {sub: resolve_model(root / sub / "model", MODEL_FILES[sub]) for sub in SUBSYSTEMS}
    dest.mkdir(parents=True)
    warnings: list[str] = []

    def copy_model(subsystem, destination):
        destination.mkdir(parents=True, exist_ok=True)
        source = models[subsystem]
        shutil.copy2(source, destination / source.name)
        manifest = source.parent / MANIFEST
        if manifest.exists():
            shutil.copy2(manifest, destination / MANIFEST)
        else:
            activate_model(destination / source.name)
        resolve_model(destination, MODEL_FILES[subsystem])

    # --- predictions.zip (flat, only the *_predictions.csv files)
    missing = [p for p in PREDICTIONS if not (root / "predictions" / p).exists()]
    with zipfile.ZipFile(dest / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in PREDICTIONS:
            f = root / "predictions" / p
            if f.exists():
                z.write(f, arcname=p)
    if missing:
        warnings.append(f"predictions.zip is missing: {missing}")

    # --- app/ (self-contained: app sources + common/ + subsystem code+model, repo-relative layout)
    app_has_sources = copytree(root / "app", dest / "app")
    copytree(root / "common", dest / "app/common")
    copytree(root / "rail_corrugation", dest / "app/rail_corrugation")
    for sub in SUBSYSTEMS:
        copytree(root / sub / "code", dest / "app" / sub / "code")
        copy_model(sub, dest / "app" / sub / "model")
    if not app_has_sources:
        warnings.append("app/ has no sources yet (compulsory deliverable, item 3)")

    # --- Optional_Items/
    opt = dest / "Optional_Items"
    opt.mkdir()
    copytree(root / "common", opt / "common")
    copytree(root / "rail_corrugation", opt / "rail_corrugation")
    if (root / "write_up.md").exists():
        shutil.copy2(root / "write_up.md", opt / "write_up.md")
    else:
        warnings.append("write_up.md not found")
    for sub in SUBSYSTEMS:
        if not copytree(root / sub / "code", opt / sub / "code"):
            warnings.append(f"Optional_Items/{sub}/code is empty")
        copy_model(sub, opt / sub / "model")

    requirements = root / "requirements.txt"
    if requirements.exists():
        names = re.findall(r"(?m)^([A-Za-z0-9][A-Za-z0-9_.-]*)", requirements.read_text(encoding="utf-8"))
        versions = "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in names) + "\n"
        for directory in (dest / "app", opt):
            shutil.copy2(requirements, directory / "requirements.txt")
            (directory / "requirements-runtime.txt").write_text(versions, encoding="utf-8")

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
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="C151")
    assemble(ROOT / ap.parse_args().dest)


if __name__ == "__main__":
    main()
