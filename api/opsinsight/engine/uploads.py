"""Receive a company's data (files or a database table), check it, keep the original, and recognise it."""
import csv
import io
import re
import uuid
from pathlib import Path

import pandas as pd

from ..models.config import ROOT
from ..models.route import choose_model
from ..prep.detect import inspect_upload
from . import store

UPLOAD_DIR = ROOT / "data/uploads"
ALLOWED = {".csv", ".xlsx"}
MAX_MB = 500
MISSING = ["?", "NA", "N/A", "null", "NULL", "-", ""]


def to_dates(values: pd.Series) -> pd.Series:
    """Read dates written either 2018-05-01 (year first) or 05/03/2016 (day first, as in India and Europe)."""
    text = values.astype(str)
    year_first = text.str.match(r"^\d{4}-").mean() > 0.5
    return pd.to_datetime(text, dayfirst=not year_first, errors="coerce", format="mixed")


class UploadError(ValueError):
    """A problem the analyst can fix (wrong file type, empty file...). The message is shown to them."""


def read_table(path) -> pd.DataFrame:
    """Read a CSV or Excel file into a table, working out the separator and the text encoding."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, na_values=MISSING)
    head = path.read_bytes()[:65536]
    encoding = "utf-8"
    try:
        sample = head.decode("utf-8")
    except UnicodeDecodeError:                      # old Excel exports are often Latin-1
        encoding, sample = "latin-1", head.decode("latin-1")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(path, sep=sep, encoding=encoding, na_values=MISSING, low_memory=False)


def _check(name: str, data: bytes):
    if Path(name).suffix.lower() not in ALLOWED:
        raise UploadError(f"{name}: only CSV and Excel (.xlsx) files can be uploaded.")
    if len(data) == 0:
        raise UploadError(f"{name} is empty.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise UploadError(f"{name} is larger than {MAX_MB} MB.")


def _recognise(folder: Path, names: list) -> tuple:
    """Find the main file (tickets or orders) among the uploaded files and inspect it."""
    best = None
    for name in names:
        try:
            df = read_table(folder / name)
        except Exception as e:                      # not really a table: broken CSV, password-protected Excel...
            raise UploadError(f"{name} could not be read as a table ({e.__class__.__name__}).") from None
        report = inspect_upload(df)
        if best is None or report["best_score"] > best[1]["best_score"]:
            best = (name, report, df)
    name, report, df = best
    choice = choose_model(df)
    summary = {"main_file": name, "rows": len(df), "columns": list(df.columns), "files": names,
               "decision": report["decision"], "domain": report["domain"], "best_score": report["best_score"],
               "scores": report["scores"], "shape": report["shape"], "message": choice["message"],
               "status": choice["status"], "model": choice["model"],
               "mapping": report["mapping"]["mapping"] if report["mapping"] else None,
               "suggestions": report["mapping"]["suggestions"] if report["mapping"] else None}
    return name, summary


def save_upload(files: dict, company: str = "unknown", source: str = "file") -> dict:
    """files = {file name: file bytes}. Checks them, saves the originals untouched, records the upload."""
    if not files:
        raise UploadError("No file was uploaded.")
    for name, data in files.items():
        _check(name, data)
    upload_id = "upload-" + uuid.uuid4().hex[:8]
    folder = UPLOAD_DIR / upload_id
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / Path(name).name).write_bytes(data)           # Path(...).name drops any folder in the name
    names = [Path(n).name for n in files]
    main_file, summary = _recognise(folder, names)
    status = "received" if summary["status"] in ("ok", "needs_confirmation") else summary["status"]
    store.add_upload(upload_id, company, source, folder, main_file, status, summary["domain"], summary)
    return {"upload_id": upload_id, **summary}


TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def read_database(url: str, table: str, company: str = "unknown") -> dict:
    """Copy one table out of a company database (read only), then treat it exactly like an uploaded CSV."""
    from sqlalchemy import create_engine, text        # imported here: only needed for database uploads
    if not TABLE_NAME.match(table):
        raise UploadError("The table name may only contain letters, digits, _ and .")
    try:
        engine = create_engine(url)
    except Exception:
        raise UploadError("That database address is not valid, e.g. postgresql://user:password@host:5432/dbname") from None
    try:
        with engine.connect() as con:
            df = pd.read_sql(text(f"SELECT * FROM {table}"), con)
    except Exception as e:                              # wrong password, missing table, server down...
        raise UploadError(f"Could not read table '{table}': {e.__class__.__name__}") from None
    finally:
        engine.dispose()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return save_upload({f"{table}.csv": buf.getvalue().encode("utf-8")}, company, source="database")
