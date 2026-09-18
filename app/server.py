"""Local-only condition monitoring workbench. Run: python3 app/server.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
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

if __package__:
    from .history import HistoryStore
else:
    from history import HistoryStore

APP = Path(__file__).resolve().parent
ROOT = APP if (APP / "common").is_dir() else APP.parent
sys.path.insert(0, str(ROOT))
MAX_UPLOAD = 120 * 1024 * 1024
HISTORY_LIMIT = 500
CONTEXT_FIELDS = ("asset_id", "location", "collected_at", "work_order")
REVIEW_STATUSES = ("new", "acknowledged", "inspection_scheduled", "resolved", "false_alert")
SYSTEMS = {
    "door": dict(name="Door", extension=".csv", artifact="door_model.joblib", output="door_predictions.csv", columns=["start_time", "end_time", "prediction"]),
    "acv": dict(name="ACV", extension=".xlsx", artifact="acv_model.joblib", output="acv_predictions.csv", columns=["file_id", "ranked_cars"]),
    "rail": dict(name="Rail Corrugation", extension=".csv", artifact="rail_model.joblib", output="rail_predictions.csv", columns=["file_id", "prediction"]),
    "shm": dict(name="SHM", extension=".csv", artifact="shm_model.joblib", output="shm_predictions.csv", columns=["file_id", "prediction"]),
}
RUNS = {}
RUN_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
DATA_DIR = Path(os.environ.get("NEBULAX_DATA_DIR", APP / ".nebulax"))
STORE = HistoryStore(DATA_DIR / "history.sqlite3", HISTORY_LIMIT)


def model_status(key):
    config = SYSTEMS[key]
    directory = ROOT / config["name"] / "model"
    manifest = directory / "active_model.json"
    result = dict(system=key, ready=False, artifact=config["artifact"], run_id=None, sha256=None)
    try:
        if not manifest.exists():
            raise ValueError("Model setup is incomplete. Ask the person setting up the app to add the trained model file and its active_model.json record, then check setup again.")
        record = json.loads(manifest.read_text())
        allowed = [config["artifact"]] + (["shm_model.json"] if key == "shm" else [])
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("file") not in allowed:
            raise ValueError("The saved model record cannot be read. Ask the person setting up the app to replace active_model.json with the record supplied with the model.")
        path = directory / record["file"]
        if path.resolve().parent != directory.resolve() or not path.is_file():
            raise ValueError("The trained model file is missing. Copy it into the folder shown below, then check setup again.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record.get("sha256"):
            raise ValueError("The model file does not match its saved record. Replace it with the original trained file, then check setup again.")
        result.update(ready=True, artifact=path.name, run_id=record.get("run_id"), sha256=digest, message="The model file matches its saved record. Required software is checked when you start a file check.")
    except (ValueError, OSError) as exc:
        result["message"] = str(exc)
    return result


def validate_files(key, files):
    if key not in SYSTEMS:
        raise ValueError("Choose Doors, Air conditioning, Rail condition or Structural health.")
    if not files or len(files) > 100:
        raise ValueError("Choose between 1 and 100 files.")
    if key == "door" and len(files) != 1:
        raise ValueError("For Doors, choose one file containing a continuous recording.")
    names = set()
    for name, content in files:
        if not name or name in (".", "..") or "/" in name or "\\" in name or any(ord(c) < 32 for c in name):
            raise ValueError("Rename the file using a simple filename, such as Test1.csv, then add it again.")
        if name.startswith(("=", "+", "-", "@")):
            raise ValueError("Rename files starting with =, +, - or @ before uploading.")
        if name.lower() in names:
            raise ValueError("Two files have the same name. Remove the duplicate or rename one file before continuing.")
        names.add(name.lower())
        if Path(name).suffix != SYSTEMS[key]["extension"]:
            raise ValueError(f"{SYSTEMS[key]['name']} accepts {SYSTEMS[key]['extension']} files (lowercase extension).")
        if not content:
            raise ValueError(f"{name} is empty.")
    if sum(len(content) for _, content in files) > MAX_UPLOAD:
        raise ValueError("These files total more than 120 MB. Upload fewer files now and check the rest afterwards.")


def validate_rows(key, rows, filenames=None):
    if not rows:
        raise ValueError("No results were found. Check that you selected the right train system and that the file contains sensor readings.")
    required = SYSTEMS[key]["columns"]
    for row in rows:
        if any(column not in row or row[column] is None for column in required):
            raise ValueError("The check returned incomplete results. No submission file was saved. Ask the person maintaining the app to review the model output.")
        if key in ("door", "rail"):
            labels = ("Normal", "Abnormal resistance") if key == "door" else ("Normal", "Side I", "Side II")
            if row["prediction"] not in labels:
                raise ValueError("The check returned an unexpected result. No submission file was saved. Ask the person maintaining the app to review the model output.")
        if key == "door":
            def stamp(value):
                parts = str(value).split("-")
                if len(parts) == 7 and all(p.isdigit() for p in parts):
                    y, mo, d, h, mi, s, ms = map(int, parts)
                    return datetime(y, mo, d, h, mi, s, ms * 1000)
                return datetime.fromisoformat(str(value))
            if stamp(row["end_time"]) <= stamp(row["start_time"]):
                raise ValueError("A door movement has an invalid start or end time. Check the recording timestamps and try again.")
        if key == "shm":
            value = float(row["prediction"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("The check could not produce a valid damage estimate. Check that the first column contains numeric stress readings and try again.")
            row["prediction"] = value
        if key == "acv":
            cars = str(row["ranked_cars"]).split("|")
            if any(not re.fullmatch(r"\d{2}", c) for c in cars) or len(set(cars)) != len(cars):
                raise ValueError("The car list contains missing, repeated or unreadable car numbers. Check the original column headings in your Excel file.")
    if key != "door" and filenames is not None:
        ids = [r["file_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(filenames):
            raise ValueError("Some result filenames do not match your uploaded files. No submission file was saved. Check the files and try again.")


def csv_bytes(key, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SYSTEMS[key]["columns"], extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def clean_context(context=None):
    context = context or {}
    if not isinstance(context, dict):
        raise ValueError("The recording details could not be read.")
    cleaned = {}
    for field in CONTEXT_FIELDS:
        value = str(context.get(field, "")).strip()
        if len(value) > 120 or any(ord(character) < 32 for character in value):
            raise ValueError("Recording details must use plain text and be no more than 120 characters.")
        cleaned[field] = value
    return cleaned


def get_run(run_id):
    with RUN_LOCK:
        run = RUNS.get(run_id)
    return run or STORE.get(run_id)


def history_runs():
    stored = STORE.list()
    with RUN_LOCK:
        session = list(RUNS.values())
    by_id = {run["id"]: run for run in stored}
    by_id.update({run["id"]: run for run in session})
    return sorted(by_id.values(), key=lambda run: run["created"], reverse=True)[:HISTORY_LIMIT]


def update_review(run_id, status, note):
    if status not in REVIEW_STATUSES:
        raise ValueError("Choose a valid review status.")
    note = str(note or "").strip()
    if len(note) > 2000 or any(ord(character) < 32 and character not in "\n\t" for character in note):
        raise ValueError("Review notes must be no more than 2,000 characters.")
    run = get_run(run_id)
    if run is None or run.get("preview"):
        raise ValueError("These results are no longer available for review.")
    review = dict(status=status, note=note, updated=datetime.now(timezone.utc).isoformat())
    run["review"] = review
    STORE.update_review(run_id, review)
    with RUN_LOCK:
        if run_id in RUNS:
            RUNS[run_id] = run
    return run


def save_run(key, rows, files, model, elapsed, preview=False, csv_content=None, context=None,
             failures=None, evidence=None):
    created = datetime.now(timezone.utc).isoformat()
    run = dict(id=uuid.uuid4().hex, system=key, rows=rows, files=files, model=model,
               elapsed=round(elapsed, 3), preview=preview, created=created, sequence=time.time_ns(),
               context=clean_context(context), failures=failures or [], evidence=evidence or {},
               review=None if preview else dict(status="new", note="", updated=created))
    if csv_content is not None:
        run["csv"] = csv_content
    with RUN_LOCK:
        RUNS[run["id"]] = run
        while len(RUNS) > 40:
            del RUNS[next(iter(RUNS))]
    if not preview:
        STORE.save(run)
    return run


def failure_message(exc):
    if isinstance(exc, (ValueError, AssertionError)) and str(exc):
        message = str(exc).strip()
        if len(message) <= 240 and "Traceback" not in message:
            return message
    return "The file does not match the expected sensor format."


def add_batch_comparisons(key, evidence):
    if key not in ("rail", "shm") or not evidence.get("files"):
        return evidence
    from common.evidence import comparison
    files = evidence["files"]
    if key == "rail":
        metrics = (
            ("Detected-side score", "", 3, lambda item: max(item["side_i_score"], item["side_ii_score"])),
            ("Train speed", "m/s", 2, lambda item: item["speed_mps"]),
        )
    else:
        metrics = (
            ("Estimated fatigue damage", "", 6, lambda item: item["estimated_damage"]),
            ("Stress range", "", 2, lambda item: item["stress_range"]),
            ("Counted stress cycles", "", 1, lambda item: item["counted_cycles"]),
        )
    for label, unit, digits, value_of in metrics:
        eligible = []
        for item in files.values():
            try:
                eligible.append((item, value_of(item)))
            except (KeyError, TypeError, ValueError):
                continue
        population = [value for _, value in eligible]
        for item, value in eligible:
            item.setdefault("comparisons", []).append(
                comparison(label, value, population, unit, digits)
            )
    return evidence


def analyze(key, files, context=None):
    validate_files(key, files)
    context = clean_context(context)
    with INFERENCE_LOCK:
        model = model_status(key)
        if not model["ready"]:
            raise ValueError(model["message"])
        from common import inference
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nebulax-") as temporary:
            folder = Path(temporary)
            for name, content in files:
                (folder / name).write_bytes(content)
            targets = [(files[0][0], folder / files[0][0])] if key == "door" else [
                (name, folder / name) for name, _ in files
            ]
            rows, exact_rows, failures, evidence = [], [], [], {}
            successful_names = set()
            detailed_predict = getattr(inference, "predict_with_evidence", None)
            pending_targets = targets
            if key != "door" and len(targets) > 1 and detailed_predict:
                try:
                    frame, evidence = detailed_predict(SYSTEMS[key]["name"], folder)
                    predicted = json.loads(frame.to_json(orient="records", double_precision=15))
                    expected_names = [name for name, _ in targets]
                    validate_rows(key, predicted, expected_names)
                    official_csv = frame[SYSTEMS[key]["columns"]].to_csv(index=False)
                    exact_rows.extend(csv.DictReader(io.StringIO(official_csv)))
                    rows.extend(predicted)
                    successful_names.update(expected_names)
                    pending_targets = []
                except ImportError:
                    raise
                except Exception:
                    # Fall back to isolated checks so one malformed file does not lose the batch.
                    rows, exact_rows, evidence = [], [], {}
                    successful_names.clear()
            for name, source in pending_targets:
                try:
                    if detailed_predict:
                        frame, details = detailed_predict(SYSTEMS[key]["name"], source)
                    else:
                        frame, details = inference.predict(SYSTEMS[key]["name"], source), {}
                    predicted = json.loads(frame.to_json(orient="records", double_precision=15))
                    validate_rows(key, predicted, None if key == "door" else [name])
                    official_csv = frame[SYSTEMS[key]["columns"]].to_csv(index=False)
                    exact_rows.extend(csv.DictReader(io.StringIO(official_csv)))
                    rows.extend(predicted)
                    successful_names.add(name)
                    if key == "door":
                        evidence = details
                    else:
                        evidence.setdefault("files", {})[name] = details.get("files", {}).get(name, details)
                except ImportError:
                    raise
                except Exception as exc:  # A bad file should not discard other valid files in the batch.
                    failures.append(dict(name=name, error=failure_message(exc)))
            if not rows:
                detail = failures[0]["error"] if len(failures) == 1 else "None of the files could be checked."
                raise ValueError(detail)
        after = model_status(key)
        if not after["ready"] or any(after.get(k) != model.get(k) for k in ("sha256", "artifact", "run_id")):
            raise ValueError("The model changed while your files were being checked. Check these files again to get results from one model version.")
        # Keep the exact pandas CSV representation for submission parity.
        file_records = [
            dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                 status="complete" if name in successful_names else "failed")
            for name, data in files
        ]
        evidence = add_batch_comparisons(key, evidence)
        run = save_run(
            key, rows, file_records, model, time.perf_counter() - started,
            csv_content=csv_bytes(key, exact_rows), context=context, failures=failures, evidence=evidence,
        )
        return run


def public_run(run):
    return {key: value for key, value in run.items() if key != "csv"}


def export_zip(ids):
    if not isinstance(ids, list) or not ids or len(ids) > HISTORY_LIMIT or any(not isinstance(i, str) for i in ids):
        raise ValueError("Check your uploaded files before downloading a submission.")
    if len(set(ids)) != len(ids):
        raise ValueError("Select each completed check only once.")
    runs = [get_run(run_id) for run_id in ids]
    if any(r is None or r["preview"] for r in runs):
        raise ValueError("Only live checks of your uploaded files can be submitted. Example results and expired checks cannot be included. Check your files again if needed.")
    # Clock resolution can give successive batches identical timestamps.
    runs.sort(key=lambda run: (run["created"], run.get("sequence", 0)), reverse=True)
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
                        raise ValueError(f"{SYSTEMS[key]['name']} results use different models. Check all files for this system again with the same model version.")
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
                history = [public_run(run) for run in history_runs()]
                return self.send(200, dict(runs=history))
            if path.path == "/api/status":
                return self.send(200, dict(models={key: model_status(key) for key in SYSTEMS}))
            if path.path == "/api/preview":
                key = parse_qs(path.query).get("system", [""])[0]
                if key not in SYSTEMS:
                    raise ValueError("Unknown subsystem.")
                example = ROOT / "predictions" / SYSTEMS[key]["output"]
                if not example.exists():
                    raise ValueError("No example results are included in this copy of the app. Once the model is ready, add your sensor files and select Check these files.")
                with example.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                validate_rows(key, rows)
                return self.send(200, public_run(save_run(key, rows, [], None, 0, True)))
            if path.path.startswith("/api/runs/"):
                parts = path.path.strip("/").split("/")
                if len(parts) != 4 or parts[3] not in ("csv", "report"):
                    raise ValueError("Unknown download.")
                run = get_run(parts[2])
                if run is None:
                    return self.send(404, dict(error="These results are no longer available. Add the original files and check them again."))
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
            parsed_origin = urlparse(origin) if origin else None
            same_host = (parsed_origin and parsed_origin.scheme in ("http", "https")
                         and parsed_origin.netloc == self.headers.get("Host"))
            local_proxy = (parsed_origin and parsed_origin.scheme == "http"
                           and parsed_origin.hostname in ("127.0.0.1", "localhost", "::1"))
            if origin and not (same_host or local_proxy):
                return self.send(403, dict(error="Requests must come from this local app."))
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_UPLOAD + 1024 * 1024:
                return self.send(413, dict(error="Add files containing sensor readings. Their combined size must be no more than 120 MB."))
            body = self.rfile.read(size)
            review_match = re.fullmatch(r"/api/runs/([0-9a-f]+)/review", self.path)
            if review_match:
                request = json.loads(body)
                run = update_review(review_match.group(1), request.get("status"), request.get("note"))
                return self.send(200, public_run(run))
            if self.path == "/api/export":
                request = json.loads(body)
                return self.send(200, export_zip(request.get("ids")), "application/zip", "predictions.zip")
            if self.path != "/api/analyze":
                return self.send(404, dict(error="Not found"))
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data;"):
                raise ValueError("Upload data using the file chooser.")
            message = BytesParser(policy=default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
            key, files, context = "", [], {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "system":
                    key = part.get_payload(decode=True).decode()
                elif name == "files":
                    files.append((part.get_filename(), part.get_payload(decode=True)))
                elif name in CONTEXT_FIELDS:
                    context[name] = part.get_payload(decode=True).decode()
            self.send(200, public_run(analyze(key, files, context)))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send(400, dict(error=str(exc)))
        except ImportError as exc:
            self.send(503, dict(error=f"The app needs additional software ({exc.name}) to check these files. Ask the person setting up the app to install the packages listed in requirements.txt, then restart the app."))
        except Exception:
            logging.exception("Analysis failed")
            self.send(422, dict(error="This recording could not be checked. Confirm that you chose the right train system and followed the file format in How to use this app. If it still fails, ask the person maintaining the app to review the error in the terminal."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NebulaX Control Room → http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
