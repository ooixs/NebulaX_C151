"""Local-only condition monitoring workbench. Run: python3 app/server.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import re
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP = Path(__file__).resolve().parent
ROOT = APP if (APP / "common").is_dir() else APP.parent
sys.path.insert(0, str(ROOT))
MAX_UPLOAD = 120 * 1024 * 1024
SYSTEMS = {
    "door": dict(name="Door", extension=".csv", artifact="door_model.joblib", output="door_predictions.csv", columns=["start_time", "end_time", "prediction"]),
    "acv": dict(name="ACV", extension=".xlsx", artifact="acv_model.joblib", output="acv_predictions.csv", columns=["file_id", "ranked_cars"]),
    "rail": dict(name="Rail Corrugation", extension=".csv", artifact="rail_model.joblib", output="rail_predictions.csv", columns=["file_id", "prediction"]),
    "shm": dict(name="SHM", extension=".csv", artifact="shm_model.joblib", output="shm_predictions.csv", columns=["file_id", "prediction"]),
}
RUNS = {}
RUN_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


def model_status(key):
    config = SYSTEMS[key]
    directory = ROOT / config["name"] / "model"
    manifest = directory / "active_model.json"
    result = dict(system=key, ready=False, artifact=config["artifact"], run_id=None, sha256=None)
    try:
        if not manifest.exists():
            raise ValueError("Active-model manifest missing. Restore the model and active_model.json from the trained run.")
        record = json.loads(manifest.read_text())
        allowed = [config["artifact"]] + (["shm_model.json"] if key == "shm" else [])
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("file") not in allowed:
            raise ValueError("Active-model manifest is invalid.")
        path = directory / record["file"]
        if path.resolve().parent != directory.resolve() or not path.is_file():
            raise ValueError("The selected model file is missing from its model folder.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record.get("sha256"):
            raise ValueError("Model checksum mismatch. Restore the original trained artifact.")
        result.update(ready=True, artifact=path.name, run_id=record.get("run_id"), sha256=digest, message="Artifact verified · inference dependencies checked when run")
    except (ValueError, OSError) as exc:
        result["message"] = str(exc)
    return result


def validate_files(key, files):
    if key not in SYSTEMS:
        raise ValueError("Choose one of the four supported subsystems.")
    if not files or len(files) > 100:
        raise ValueError("Choose between 1 and 100 files.")
    if key == "door" and len(files) != 1:
        raise ValueError("Door analysis accepts one continuous stream at a time.")
    names = set()
    for name, content in files:
        if not name or name in (".", "..") or "/" in name or "\\" in name or any(ord(c) < 32 for c in name):
            raise ValueError("Use plain filenames without directory paths or control characters.")
        if name.startswith(("=", "+", "-", "@")):
            raise ValueError("Rename files starting with =, +, - or @ before uploading.")
        if name.lower() in names:
            raise ValueError("Duplicate filenames detected. Each input needs a unique filename.")
        names.add(name.lower())
        if Path(name).suffix != SYSTEMS[key]["extension"]:
            raise ValueError(f"{SYSTEMS[key]['name']} accepts {SYSTEMS[key]['extension']} files (lowercase extension).")
        if not content:
            raise ValueError(f"{name} is empty.")
    if sum(len(content) for _, content in files) > MAX_UPLOAD:
        raise ValueError("This batch exceeds the 120 MB upload limit. Split it into smaller batches.")


def validate_rows(key, rows, filenames=None):
    if not rows:
        raise ValueError("No predictions were produced. Check the recording contains valid sensor data.")
    required = SYSTEMS[key]["columns"]
    for row in rows:
        if any(column not in row or row[column] is None for column in required):
            raise ValueError("The model returned an incomplete output schema.")
        if key in ("door", "rail"):
            labels = ("Normal", "Abnormal resistance") if key == "door" else ("Normal", "Side I", "Side II")
            if row["prediction"] not in labels:
                raise ValueError("The model returned an unsupported prediction label.")
        if key == "door":
            def stamp(value):
                parts = str(value).split("-")
                if len(parts) == 7 and all(p.isdigit() for p in parts):
                    y, mo, d, h, mi, s, ms = map(int, parts)
                    return datetime(y, mo, d, h, mi, s, ms * 1000)
                return datetime.fromisoformat(str(value))
            if stamp(row["end_time"]) <= stamp(row["start_time"]):
                raise ValueError("The model returned an invalid door-cycle time interval.")
        if key == "shm":
            value = float(row["prediction"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("The model returned an invalid fatigue damage value.")
            row["prediction"] = value
        if key == "acv":
            cars = str(row["ranked_cars"]).split("|")
            if any(not re.fullmatch(r"\d{2}", c) for c in cars) or len(set(cars)) != len(cars):
                raise ValueError("The model returned invalid or repeated car identifiers.")
    if key != "door" and filenames is not None:
        ids = [r["file_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(filenames):
            raise ValueError("Prediction filenames do not match the uploaded batch.")


def csv_bytes(key, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SYSTEMS[key]["columns"], extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def save_run(key, rows, files, model, elapsed, preview=False, csv_content=None):
    run = dict(id=uuid.uuid4().hex, system=key, rows=rows, files=files, model=model,
               elapsed=round(elapsed, 3), preview=preview, created=datetime.now(timezone.utc).isoformat())
    if csv_content is not None:
        run["csv"] = csv_content
    with RUN_LOCK:
        RUNS[run["id"]] = run
        while len(RUNS) > 40:
            del RUNS[next(iter(RUNS))]
    return run


def analyze(key, files):
    validate_files(key, files)
    with INFERENCE_LOCK:
        model = model_status(key)
        if not model["ready"]:
            raise ValueError(model["message"])
        from common.inference import predict
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nebulax-") as temporary:
            folder = Path(temporary)
            for name, content in files:
                (folder / name).write_bytes(content)
            source = folder / files[0][0] if key == "door" else folder
            frame = predict(SYSTEMS[key]["name"], source)
            rows = json.loads(frame.to_json(orient="records", double_precision=15))
            validate_rows(key, rows, [name for name, _ in files])
        after = model_status(key)
        if not after["ready"] or any(after.get(k) != model.get(k) for k in ("sha256", "artifact", "run_id")):
            raise ValueError("The active model changed during analysis. Run the batch again.")
        # Keep the exact pandas CSV representation for submission parity.
        run = save_run(key, rows, [dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in files], model, time.perf_counter() - started,
                       csv_content=frame[SYSTEMS[key]["columns"]].to_csv(index=False).encode("utf-8"))
        return run


def public_run(run):
    return {key: value for key, value in run.items() if key != "csv"}


def export_zip(ids):
    if not isinstance(ids, list) or not ids or len(ids) > 40 or any(not isinstance(i, str) for i in ids):
        raise ValueError("Select completed live runs to export.")
    if len(set(ids)) != len(ids):
        raise ValueError("Select each live run only once.")
    with RUN_LOCK:
        runs = [RUNS.get(i) for i in ids]
        saved_order = {run_id: index for index, run_id in enumerate(RUNS)}
    if any(r is None or r["preview"] for r in runs):
        raise ValueError("Only live analyses can be included in a submission. Saved-result previews are excluded.")
    # Clock resolution can give successive batches identical timestamps.
    runs.sort(key=lambda run: (run["created"], saved_order[run["id"]]), reverse=True)
    grouped = {}
    for run in runs:
        grouped.setdefault(run["system"], []).append(run)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, batches in grouped.items():
            if key == "door":
                content = batches[0]["csv"]  # One continuous stream: newest run wins.
            else:
                by_file, digest = {}, None
                for batch in batches:
                    rows = list(csv.DictReader(io.StringIO(batch["csv"].decode("utf-8"))))
                    fresh = [row for row in rows if row["file_id"] not in by_file]
                    if not fresh:
                        continue
                    model_hash = batch["model"]["sha256"]
                    if digest is not None and digest != model_hash:
                        raise ValueError(f"{SYSTEMS[key]['name']} batches use different models. Rerun the full set with one active model.")
                    digest = model_hash
                    by_file.update({row["file_id"]: row for row in fresh})
                content = csv_bytes(key, list(by_file.values()))
            archive.writestr(SYSTEMS[key]["output"], content)
    return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "NebulaX/1.0"

    def send(self, status, body, content_type="application/json", filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path)
        try:
            if path.path == "/api/history":
                with RUN_LOCK:
                    history = [public_run(run) for run in reversed(list(RUNS.values()))]
                return self.send(200, dict(runs=history))
            if path.path == "/api/status":
                return self.send(200, dict(models={key: model_status(key) for key in SYSTEMS}))
            if path.path == "/api/preview":
                key = parse_qs(path.query).get("system", [""])[0]
                if key not in SYSTEMS:
                    raise ValueError("Unknown subsystem.")
                example = ROOT / "predictions" / SYSTEMS[key]["output"]
                if not example.exists():
                    raise ValueError("Saved predictions are unavailable in this package. Upload data to run a live analysis.")
                with example.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                validate_rows(key, rows)
                return self.send(200, public_run(save_run(key, rows, [], None, 0, True)))
            if path.path.startswith("/api/runs/"):
                parts = path.path.strip("/").split("/")
                if len(parts) != 4 or parts[3] not in ("csv", "report"):
                    raise ValueError("Unknown download.")
                with RUN_LOCK:
                    run = RUNS.get(parts[2])
                if run is None:
                    return self.send(404, dict(error="This run has expired. Analyze the files again."))
                if parts[3] == "report":
                    return self.send(200, public_run(run), filename=f"{run['system']}_analysis_record.json")
                filename = ("preview_" if run["preview"] else "") + SYSTEMS[run["system"]]["output"]
                return self.send(200, run.get("csv") or csv_bytes(run["system"], run["rows"]), "text/csv; charset=utf-8", filename)
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path.path not in static:
                return self.send(404, dict(error="Not found"))
            file = APP / "static" / static[path.path]
            mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}[file.suffix]
            self.send(200, file.read_bytes(), mime)
        except (ValueError, OSError) as exc:
            self.send(400, dict(error=str(exc)))

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send(403, dict(error="Requests must come from this local app."))
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_UPLOAD + 1024 * 1024:
                return self.send(413, dict(error="Choose a nonempty batch under 120 MB."))
            body = self.rfile.read(size)
            if self.path == "/api/export":
                request = json.loads(body)
                return self.send(200, export_zip(request.get("ids")), "application/zip", "predictions.zip")
            if self.path != "/api/analyze":
                return self.send(404, dict(error="Not found"))
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data;"):
                raise ValueError("Upload data using the file chooser.")
            message = BytesParser(policy=default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
            key, files = "", []
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "system":
                    key = part.get_payload(decode=True).decode()
                elif name == "files":
                    files.append((part.get_filename(), part.get_payload(decode=True)))
            self.send(200, public_run(analyze(key, files)))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send(400, dict(error=str(exc)))
        except ImportError as exc:
            self.send(503, dict(error=f"An inference dependency is missing: {exc.name}. Install the repository requirements in your Python environment, then restart the app."))
        except Exception:
            logging.exception("Analysis failed")
            self.send(422, dict(error="The model could not read this recording. Check the subsystem, column layout and data format against the input guide. Technical details are in the server terminal."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"NebulaX Control Room → http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
