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
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from contextlib import closing
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from . import diagnostics
except ImportError:
    import diagnostics

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
HISTORY_PATH = None
RUNS = {}
RUN_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
MODEL_DETAILS = json.loads((APP / "model_details.json").read_text(encoding="utf-8"))


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
        # Descriptions and reported scores belong to the exact checkpoint, not its filename.
        result["details"] = MODEL_DETAILS.get(digest)
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


def save_run(key, rows, files, model, elapsed, preview=False, csv_content=None, evidence=None):
    run = dict(id=uuid.uuid4().hex, system=key, rows=rows, files=files, model=model,
               elapsed=round(elapsed, 3), preview=preview, created=datetime.now(timezone.utc).isoformat())
    if evidence is not None:
        run['evidence'] = evidence
    if csv_content is not None:
        run["csv"] = csv_content
    with RUN_LOCK:
        persist_run(run)
        RUNS[run["id"]] = run
    return run


def configure_history(path):
    """Save result metadata and outputs, never original uploaded sensor files."""
    global HISTORY_PATH
    HISTORY_PATH = Path(path)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(HISTORY_PATH)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT, output BLOB)")
        for record, output in db.execute("SELECT record, output FROM runs ORDER BY rowid"):
            run = json.loads(record)
            if output is not None:
                run['csv'] = output
            RUNS[run['id']] = run


class HistoryStorageError(RuntimeError):
    """Inference succeeded but durable result storage is unavailable."""


def persist_run(run):
    if HISTORY_PATH is None:
        return
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(HISTORY_PATH)) as db, db:
            recovering = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'").fetchone() is None
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT, output BLOB)")
            # If the folder/database disappeared during operation, preserve all
            # completed checks still held by this process, including exact CSVs.
            records = [r for r in RUNS.values() if r['id'] != run['id']] if recovering else []
            records.append(run)
            db.executemany("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                           [(r['id'], json.dumps(public_run(r), allow_nan=False), r.get('csv')) for r in records])
    except (OSError, sqlite3.Error) as exc:
        raise HistoryStorageError('The model completed its check, but the app could not save the results. '
                                  'Check available disk space and write access to app/.local, then retry. '
                                  'Your sensor file format is not the cause of this storage error.') from exc


def readable_time(value):
    parts = str(value).split('-')
    if len(parts) == 7 and all(part.isdigit() for part in parts):
        y, mo, d, h, mi, sec, ms = map(int, parts)
        stamp = datetime(y, mo, d, h, mi, sec, ms * 1000)
    else:
        stamp = datetime.fromisoformat(str(value))
    return stamp.strftime('%d %b %Y, %H:%M:%S') + (stamp.strftime(' %z') if stamp.tzinfo else '')


OPERATOR_COLUMNS = {
    'door': {'start_time': 'Movement start (recording time)', 'end_time': 'Movement end (recording time)', 'prediction': 'Door condition'},
    'acv': {'file_id': 'Source file', 'ranked_cars': 'Cars in refrigerant leak inspection order'},
    'rail': {'file_id': 'Source file', 'prediction': 'Rail corrugation result'},
    'shm': {'file_id': 'Source file', 'prediction': 'Estimated fatigue damage'},
}


def operator_rows(run):
    # Read original CSV strings to avoid rounding exported fatigue damage values.
    source = list(csv.DictReader(io.StringIO(run['csv'].decode()))) if run.get('csv') else run['rows']
    return [{label: readable_time(row[key]) if key in ('start_time', 'end_time') else row[key]
             for key, label in OPERATOR_COLUMNS[run['system']].items()} for row in source]


def operator_csv(run):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(OPERATOR_COLUMNS[run['system']].values()), lineterminator='\n')
    writer.writeheader()
    writer.writerows(operator_rows(run))
    return output.getvalue().encode()


def merge_runs(ids):
    if not isinstance(ids, list) or not ids or len(ids) > 100 or any(not isinstance(i, str) for i in ids):
        raise ValueError('Choose completed checks to combine.')
    with RUN_LOCK:
        runs = [RUNS.get(i) for i in ids]
    if len(ids) != len(set(ids)) or any(r is None or r['preview'] for r in runs):
        raise ValueError('One of these checks is unavailable. Previously completed results are still in Previous results.')
    key = runs[0]['system']
    if key == 'door' or any(r['system'] != key for r in runs):
        raise ValueError('Only files from the same non-Door check can be combined.')
    names = [row['file_id'] for r in runs for row in r['rows']]
    if len(names) != len(set(names)):
        raise ValueError('The selected checks contain duplicate filenames.')
    if len({r['model']['sha256'] for r in runs}) != 1:
        raise ValueError('The checks use different models. Check the files again with one model version.')
    exact = [row for r in runs for row in csv.DictReader(io.StringIO(r['csv'].decode()))]
    merged = save_run(key, [row for r in runs for row in r['rows']], [f for r in runs for f in r['files']],
                     runs[0]['model'], sum(r['elapsed'] for r in runs), csv_content=csv_bytes(key, exact),
                     evidence=diagnostics.merge([r.get('evidence', {}) for r in runs]))
    merged["combined_from"] = ids
    with RUN_LOCK:
        persist_run(merged)
    # Retain individual checks too: another browser may already be viewing one.
    return merged


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
            try:
                evidence = diagnostics.build(key, folder, rows, [name for name, _ in files])
            except Exception:
                logging.exception('Diagnostic measurements unavailable')
                evidence = dict(version=1, items=[], error='Measurements could not be extracted. The model result is still available.')
        after = model_status(key)
        if not after["ready"] or any(after.get(k) != model.get(k) for k in ("sha256", "artifact", "run_id")):
            raise ValueError("The model changed while your files were being checked. Check these files again to get results from one model version.")
        # Keep the exact pandas CSV representation for submission parity.
        run = save_run(key, rows, [dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in files], model, time.perf_counter() - started,
                       csv_content=frame[SYSTEMS[key]["columns"]].to_csv(index=False).encode("utf-8"), evidence=evidence)
        return run


def public_run(run):
    return {key: value for key, value in run.items() if key != "csv"}


def export_zip(ids):
    if not isinstance(ids, list) or not ids or len(ids) > 10000 or any(not isinstance(i, str) for i in ids):
        raise ValueError("Check your uploaded files before downloading a submission.")
    if len(set(ids)) != len(ids):
        raise ValueError("Select each completed check only once.")
    with RUN_LOCK:
        runs = [RUNS.get(i) for i in ids]
        saved_order = {run_id: index for index, run_id in enumerate(RUNS)}
    if any(r is None or r["preview"] for r in runs):
        raise ValueError("Only live checks of your uploaded files can be submitted. Example results and expired checks cannot be included. Check your files again if needed.")
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
                        raise ValueError(f"{SYSTEMS[key]['name']} results use different models. Check all files for this system again with the same model version.")
                    digest = model_hash
                    by_file.update({row["file_id"]: row for row in fresh})
                content = csv_bytes(key, list(by_file.values()))
            archive.writestr(SYSTEMS[key]["output"], content)
    return output.getvalue()


def selected_zip(selection):
    if not isinstance(selection, list) or not selection or len(selection) > 10000:
        raise ValueError('Select at least one completed result.')
    grouped, seen, hashes = {}, set(), {}
    with RUN_LOCK:
        for choice in selection:
            if not isinstance(choice, dict):
                raise ValueError('Invalid result selection.')
            run = RUNS.get(choice.get('id'))
            if not run or run['preview']:
                raise ValueError('A selected check is unavailable.')
            key = run['system']
            rows = list(csv.DictReader(io.StringIO(run['csv'].decode())))
            if key != 'door':
                rows = [r for r in rows if r['file_id'] == choice.get('file')]
            if not rows:
                raise ValueError('A selected file is unavailable.')
            identity = (key, choice.get('file') if key != 'door' else 'recording')
            if identity in seen:
                raise ValueError('Select one result per filename, and one Door recording.')
            seen.add(identity)
            digest = run['model']['sha256']
            if key in hashes and hashes[key] != digest:
                raise ValueError('Selected results use different model versions. Select checks from one version per system.')
            hashes[key] = digest
            grouped.setdefault(key, []).extend(rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        folders = dict(door='doors', acv='air_conditioning', rail='rail_condition', shm='structural_health')
        for key, rows in grouped.items():
            if key != 'door' and len(rows) > 1:
                for row in rows:
                    # Source extensions are fixed per task; retain the source stem in each result name.
                    source = Path(row['file_id']).name
                    if source != row['file_id'] or '\\' in source:
                        raise ValueError('A result has an invalid source filename.')
                    archive.writestr(f"{folders[key]}/{Path(source).stem}_results.csv", operator_csv(dict(system=key, rows=[row])))
            else:
                archive.writestr(key+'_check_results.csv', operator_csv(dict(system=key, rows=rows)))
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

    def send_error(self, code, message=None, explain=None):
        self.send(code, dict(error=message or 'The request could not be completed. Refresh the app and try again.'))

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
                    raise ValueError("No example results are included in this copy of the app. Once the model is ready, add your sensor files and select Check these files.")
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
                    return self.send(404, dict(error="These results are no longer available. Add the original files and check them again."))
                if parts[3] == "report":
                    report = dict(system=run['system'], checked_at=run['created'], source_files=run['files'],
                                  results=operator_rows(run), measurements=run.get('evidence'), model_version=(run.get('model') or {}).get('run_id'))
                    return self.send(200, report, filename=f"{run['system']}_check_results.json")
                filename = ("example_" if run["preview"] else "") + run['system'] + '_check_results.csv'
                return self.send(200, operator_csv(run), "text/csv; charset=utf-8", filename)
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/technician.js": "technician.js"}
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
            if self.path == "/api/merge":
                return self.send(200, public_run(merge_runs(json.loads(body).get('ids'))))
            if self.path == "/api/selected-results":
                return self.send(200, selected_zip(json.loads(body).get('selection')), 'application/zip', 'selected_check_results.zip')
            if self.path == "/api/all-results":
                ids = json.loads(body).get('ids')
                official = export_zip(ids)
                output = io.BytesIO()
                with zipfile.ZipFile(io.BytesIO(official)) as source, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as target:
                    for key, config in SYSTEMS.items():
                        if config['output'] in source.namelist():
                            target.writestr(key + '_check_results.csv', operator_csv(dict(system=key, csv=source.read(config['output']))))
                return self.send(200, output.getvalue(), 'application/zip', 'all_check_results.zip')
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
        except HistoryStorageError as exc:
            logging.exception("Result history could not be saved")
            self.send(503, dict(error=str(exc)))
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
    configure_history(APP / '.local' / 'results.sqlite3')
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
