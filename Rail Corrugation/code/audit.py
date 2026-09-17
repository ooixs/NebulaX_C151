"""RC-E00: immutable full-file data audit and acquisition fingerprinting."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Rail Corrugation/code"))
from experiment_io import atomic_json, sha256_file, stable_hash  # noqa: E402
from pipeline import EDGES_PER_TOOTH, FS, TEETH, WHEEL_D  # noqa: E402

HEADER_RE = re.compile(r"^(Vibration|Shock) of bearing in position ([1-8]) of car ([1-8])$")


def natural_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"filename has no numeric identifier: {path.name}")
    return int(match.group(1))


def parse_headers(columns: list[str]) -> list[dict]:
    if len(columns) != 129 or columns[0] != "Rotating speed":
        raise ValueError(f"unexpected schema: {len(columns)} columns; first={columns[0]!r}")
    mapping = []
    for index, name in enumerate(columns[1:], 1):
        match = HEADER_RE.fullmatch(name)
        if not match:
            raise ValueError(f"unparseable sensor header at column {index}: {name!r}")
        signal, position, car = match.groups()
        expected_car = (index - 1) // 16 + 1
        expected_position = ((index - 1) % 16) // 2 + 1
        expected_signal = "Vibration" if index % 2 else "Shock"
        if (int(car), int(position), signal) != (expected_car, expected_position, expected_signal):
            raise ValueError(f"header/position mismatch at column {index}: {name!r}")
        mapping.append({"column_index": index, "header": name, "signal": signal,
                        "car": int(car), "position": int(position),
                        "side": "Side I" if int(position) % 2 else "Side II"})
    return mapping


def inspect_file(path: Path, split: str, expected_headers: list[str] | None) -> tuple[dict, np.ndarray]:
    content_hash = sha256_file(path)
    frame = pd.read_csv(path)
    columns = frame.columns.tolist()
    parse_headers(columns)
    if expected_headers is not None and columns != expected_headers:
        raise ValueError(f"header identity/order differs: {path}")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nonnumeric data in {path}: {exc}") from exc
    finite = np.isfinite(values)
    pulse = values[:, 0]
    transitions = int(np.count_nonzero(np.diff(pulse) != 0))
    speed = transitions / (EDGES_PER_TOOTH * TEETH) * np.pi * WHEEL_D * (FS / len(pulse))
    run_edges = np.flatnonzero(np.diff(pulse) != 0) + 1
    runs = np.diff(np.r_[0, run_edges, len(pulse)])
    vib = values[:, 1::2]
    shock = values[:, 2::2]
    rms = np.sqrt(np.nanmean(vib * vib, axis=0))
    shock_rms = np.sqrt(np.nanmean(shock * shock, axis=0))
    # A compact run fingerprint: speed/pulse pattern, per-channel RMS, and coarse spectra.
    freq = np.fft.rfftfreq(len(vib), 1 / FS)
    spec = np.abs(np.fft.rfft(vib - np.nanmean(vib, axis=0), axis=0)) ** 2
    spectral = []
    for lo, hi in ((20, 100), (100, 400), (400, 1000), (1000, 2500), (2500, 5000)):
        spectral.extend(np.log10(np.nanmean(spec[(freq >= lo) & (freq < hi)], axis=0) + 1e-15))
    signature = np.r_[speed, transitions, np.median(runs), np.std(runs),
                      np.log10(rms + 1e-15), np.log10(shock_rms + 1e-15), spectral]
    sensor_bytes = np.ascontiguousarray(values[:, 1:]).view(np.uint8)
    row = {
        "split": split, "file_id": path.name, "file_number": natural_number(path),
        "size_bytes": path.stat().st_size, "sha256": content_hash,
        "sensor_sha256": hashlib.sha256(sensor_bytes).hexdigest(),
        "rows": int(values.shape[0]), "columns": int(values.shape[1]),
        "nan_count": int(np.isnan(values).sum()), "inf_count": int(np.isinf(values).sum()),
        "constant_channels": int(np.count_nonzero(np.nanstd(values, axis=0) == 0)),
        "near_constant_channels": int(np.count_nonzero(np.nanstd(values, axis=0) < 1e-12)),
        "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
        "pulse_unique": json.dumps(np.unique(pulse).tolist()),
        "pulse_transitions": transitions, "speed_mps": float(speed),
        "pulse_run_min": int(runs.min()), "pulse_run_median": float(np.median(runs)),
        "pulse_run_max": int(runs.max()), "pulse_run_cv": float(np.std(runs) / max(np.mean(runs), 1e-12)),
        "vibration_rms_median": float(np.median(rms)),
        "shock_rms_median": float(np.median(shock_rms)),
        "finite_fraction": float(finite.mean()),
    }
    return row, signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    labels = pd.read_csv(args.data_dir / "Train_Labels.csv", dtype=str)
    if list(labels.columns) != ["filename", "label"] or labels.filename.duplicated().any():
        raise ValueError("Train_Labels.csv must have unique filename,label rows")
    train = sorted((args.data_dir / "Train").glob("*.csv"), key=natural_number)
    test = sorted((args.data_dir / "Test").glob("*.csv"), key=natural_number)
    if len(train) != 272 or len(test) != 68:
        raise ValueError(f"expected 272 train and 68 test files; found {len(train)} and {len(test)}")
    if set(labels.filename) != {p.name for p in train}:
        raise ValueError("training file/label coverage mismatch")
    expected_headers = pd.read_csv(train[0], nrows=0).columns.tolist()
    channel_mapping = parse_headers(expected_headers)
    rows, signatures = [], []
    for index, (split, path) in enumerate([*(('train', p) for p in train), *(('test', p) for p in test)], 1):
        row, signature = inspect_file(path, split, expected_headers)
        row["label"] = labels.set_index("filename").label.get(path.name) if split == "train" else None
        rows.append(row); signatures.append(signature)
        if index % 20 == 0 or index == 340:
            print(f"audited {index}/340", flush=True)
    manifest = pd.DataFrame(rows).sort_values(["split", "file_number"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_manifest = args.output_dir / ".data_manifest.csv.tmp"
    manifest.to_csv(temp_manifest, index=False)
    temp_manifest.replace(args.output_dir / "data_manifest.csv")

    sig = np.asarray(signatures)
    sig = (sig - np.nanmedian(sig, axis=0)) / (np.nanmedian(np.abs(sig - np.nanmedian(sig, axis=0)), axis=0) + 1e-9)
    sig = np.nan_to_num(sig)
    sig /= np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12
    similarity = sig @ sig.T
    candidate_pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            adjacent = rows[i]["split"] == rows[j]["split"] and abs(rows[i]["file_number"] - rows[j]["file_number"]) == 1
            if similarity[i, j] >= 0.995 or adjacent:
                candidate_pairs.append({"left": rows[i]["file_id"], "right": rows[j]["file_id"],
                                        "similarity": float(similarity[i, j]), "adjacent": adjacent})
    exact = manifest.groupby("sha256").file_id.agg(list)
    sensor_exact = manifest.groupby("sensor_sha256").file_id.agg(list)
    anomalies = manifest[(manifest["rows"] != 10_000) | (manifest["columns"] != 129) |
                         (manifest.nan_count > 0) | (manifest.inf_count > 0) |
                         (manifest.finite_fraction < 1)]
    audit = {
        "experiment_id": "RC-E00", "data_dir": str(args.data_dir.resolve()),
        "counts": {"train": len(train), "test": len(test), "labels": labels.label.value_counts().to_dict()},
        "manifest_sha256": sha256_file(args.output_dir / "data_manifest.csv"),
        "source_manifest_fingerprint": stable_hash(manifest[["split", "file_id", "size_bytes", "sha256"]].to_dict("records")),
        "schema": {"rows": manifest["rows"].value_counts().to_dict(), "columns": manifest["columns"].value_counts().to_dict(),
                   "header_sha256": stable_hash(expected_headers), "anomaly_count": len(anomalies)},
        "duplicates": {"exact_file_groups": exact[exact.map(len) > 1].tolist(),
                       "exact_sensor_groups": sensor_exact[sensor_exact.map(len) > 1].tolist()},
        "speed": {"train": manifest[manifest.split == "train"].speed_mps.describe().to_dict(),
                  "test": manifest[manifest.split == "test"].speed_mps.describe().to_dict(),
                  "by_label": manifest[manifest.split == "train"].groupby("label").speed_mps.describe().to_dict()},
        "candidate_related_pairs": candidate_pairs,
        "exclusions": [], "elapsed_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_json(args.output_dir / "data_audit.json", audit)
    pd.DataFrame(channel_mapping).to_csv(args.output_dir / "channel_mapping.csv", index=False)
    report = ["# RC-E00 data audit", "", f"- Files: {len(train)} train, {len(test)} test",
              f"- Labels: {audit['counts']['labels']}", f"- Manifest SHA-256: `{audit['manifest_sha256']}`",
              f"- Schema anomalies: {len(anomalies)}", f"- Exact duplicate files: {len(audit['duplicates']['exact_file_groups'])}",
              f"- Exact duplicate sensor matrices: {len(audit['duplicates']['exact_sensor_groups'])}",
              f"- Candidate related/adjacent pairs recorded: {len(candidate_pairs)}",
              f"- Runtime: {audit['elapsed_seconds']:.1f} s; peak RSS: {audit['peak_rss_mib']:.1f} MiB", "",
              "No source file was modified and no file was excluded."]
    (args.output_dir / "data_audit.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k: audit[k] for k in ("counts", "schema", "duplicates", "elapsed_seconds", "peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
