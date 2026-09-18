from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap

MANIFEST = "active_model.json"


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_key(path):
    path = Path(path).resolve()
    return str(path), _checksum(path)


def activate_model(path, run_id=None):
    path = Path(path)
    if not path.is_file() or path.suffix not in (".json", ".joblib") or path.name == MANIFEST:
        raise ValueError("activation requires an existing JSON or joblib model")
    record = dict(schema_version=1, file=path.name, sha256=_checksum(path), run_id=run_id)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path.parent / MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def resolve_model(directory, candidates):
    directory = Path(directory)
    manifest = directory / MANIFEST
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"invalid active-model manifest: {manifest}")
        filename = record.get("file")
        if record.get("schema_version") != 1 or filename not in candidates or Path(filename).name != filename:
            raise ValueError(f"invalid active-model manifest: {manifest}")
        path = directory / filename
        if path.resolve().parent != directory.resolve():
            raise ValueError("active model must be inside its model directory")
        if not path.is_file():
            raise FileNotFoundError(f"active model missing: {path}")
        if record.get("sha256") != _checksum(path):
            raise ValueError(f"active model checksum mismatch: {path}")
        return path
    present = [directory / name for name in candidates if (directory / name).is_file()]
    if len(present) > 1:
        raise ValueError(f"ambiguous model selection in {directory}; activate one artifact explicitly")
    if not present:
        raise FileNotFoundError(f"no model found in {directory}; expected one of {tuple(candidates)}")
    return present[0]


@lru_cache(maxsize=8)
def _load_cached(key, reader):
    path = Path(key[0])
    artifact = reader(path)
    if model_key(path) != key:
        raise RuntimeError("model changed during loading; retry the prediction")
    return artifact


def load_active_model(directory, candidates, reader):
    path = resolve_model(directory, candidates)
    key = model_key(path)
    artifact = _load_cached(key, reader)
    if resolve_model(directory, candidates) != path or model_key(path) != key:
        raise RuntimeError("active model changed during loading; retry the prediction")
    return artifact


def publish_model(source, destination, run_id=None):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return activate_model(destination, run_id)


class _RailUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module == "code.pipeline":
            module = "rail_corrugation.pipeline"
        return super().find_class(module, name)


def load_rail_model(path):
    path = Path(path)
    with path.open("rb") as stream:
        with _validate_fileobject_and_memmap(stream, str(path), None) as (handle, _):
            if isinstance(handle, str):
                raise ValueError("joblib artifacts older than 0.10 are not supported")
            return _RailUnpickler(str(path), handle, ensure_native_byte_order=True, mmap_mode=None).load()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(json.dumps(activate_model(arguments.activate, arguments.run_id), indent=2))
