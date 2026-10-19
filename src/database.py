"""SQLite persistence for employee records (add, edit, delete, import, export).

The database is a single file (``data/employees.db``) created and seeded
automatically on first use. It is the live source of truth for the web app;
the model is trained from an export of it.

Every write goes through ``validate_record`` (same rules as the prediction
interface) and the table also has CHECK constraints, so bad data cannot get
in even if someone bypasses the app.

CLI:
    python -m src.database --reset      # recreate the DB with sample data
    python -m src.database --empty      # recreate it empty
    python -m src.database --export out.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.generate_data import generate_employee_data
from src.predict import InputValidationError, parse_overtime, validate_employee_input
from src.utils import (
    DATA_DIR,
    DEPARTMENTS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    ensure_dir,
    get_logger,
)

logger = get_logger(__name__)

DB_PATH = Path(os.environ.get("EMPLOYEE_DB_PATH", DATA_DIR / "employees.db"))

COLUMNS = ["EmployeeID", "Name", *FEATURE_COLUMNS, "Attrition"]
EDITABLE_COLUMNS = [c for c in COLUMNS if c != "EmployeeID"]

_DEPARTMENT_SQL = ", ".join(f"'{d}'" for d in DEPARTMENTS)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS employees (
    EmployeeID        INTEGER PRIMARY KEY AUTOINCREMENT,
    Name              TEXT    NOT NULL CHECK (length(trim(Name)) > 0),
    Age               INTEGER NOT NULL CHECK (Age BETWEEN 18 AND 70),
    Department        TEXT    NOT NULL CHECK (Department IN ({_DEPARTMENT_SQL})),
    MonthlyIncome     REAL    NOT NULL CHECK (MonthlyIncome > 0),
    YearsAtCompany    REAL    NOT NULL CHECK (YearsAtCompany >= 0),
    JobSatisfaction   INTEGER NOT NULL CHECK (JobSatisfaction BETWEEN 1 AND 5),
    WorkLifeBalance   INTEGER NOT NULL CHECK (WorkLifeBalance BETWEEN 1 AND 5),
    PerformanceRating INTEGER NOT NULL CHECK (PerformanceRating BETWEEN 1 AND 5),
    OverTime          TEXT    NOT NULL CHECK (OverTime IN ('Yes', 'No')),
    DistanceFromHome  REAL    NOT NULL CHECK (DistanceFromHome >= 0),
    Attrition         TEXT    NOT NULL DEFAULT 'No' CHECK (Attrition IN ('Yes', 'No')),
    CreatedAt         TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees (Department);
CREATE INDEX IF NOT EXISTS idx_employees_attrition ON employees (Attrition);
"""

# Minimum data needed before retraining is meaningful.
MIN_TRAINING_ROWS = 200
MIN_PER_CLASS = 20

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Siddharth", "Ishaan", "Rahul",
    "Amit", "Nikhil", "Varun", "Manish", "Tejas", "Harsh", "Yash", "Pranav", "Kunal",
    "Priya", "Ananya", "Sneha", "Neha", "Pooja", "Kavya", "Isha", "Riya", "Meera", "Divya",
    "Shreya", "Aditi", "Nisha", "Swati", "Tanvi", "Rutuja", "Simran", "Anjali", "Sakshi", "Pallavi",
]
LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Deshmukh", "Joshi", "Kulkarni", "Iyer", "Reddy",
    "Nair", "Gupta", "Mehta", "Shah", "Verma", "Pawar", "Jadhav", "Chavan", "More", "Bose",
    "Das", "Rao", "Menon", "Kapoor", "Malhotra", "Bhat", "Naik", "Gokhale", "Phadke", "Saxena", "Tiwari",
]


class DatabaseError(RuntimeError):
    """The database could not be opened or written."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a full employee record (name, features, status).

    Returns clean values ready to store. Raises ``InputValidationError`` with a
    message per bad field.
    """
    errors: dict[str, str] = {}

    name = re.sub(r"\s+", " ", str(raw.get("Name") or "")).strip()
    if not name:
        errors["Name"] = "is required."
    elif len(name) > 80:
        errors["Name"] = "must be 80 characters or fewer."

    status = raw.get("Attrition", "No")
    try:
        attrition = parse_overtime(status if status not in (None, "") else "No")
    except ValueError:
        errors["Attrition"] = "must be Yes (has left) or No (still employed)."
        attrition = "No"

    features: dict[str, Any] = {}
    try:
        features = validate_employee_input(raw)
    except InputValidationError as exc:
        errors.update(exc.errors)

    if errors:
        raise InputValidationError(errors)
    return {"Name": name, **features, "Attrition": attrition}


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------- #
# Database access
# --------------------------------------------------------------------------- #
class EmployeeDB:
    """Small repository class around the ``employees`` table."""

    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open the database at '{self.path}': {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- reads ------------------------------------------------------------
    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0])

    def get(self, employee_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM employees WHERE EmployeeID = ?", (int(employee_id),)
            ).fetchone()
        return dict(row) if row else None

    def list(self) -> pd.DataFrame:
        """All employees as a DataFrame (newest ID last)."""
        with self._connect() as conn:
            frame = pd.read_sql_query(
                f"SELECT {', '.join(COLUMNS)} FROM employees ORDER BY EmployeeID", conn)
        return frame

    # ---- writes -----------------------------------------------------------
    def add(self, raw: dict[str, Any]) -> int:
        """Validate and insert one employee. Returns the new EmployeeID."""
        record = validate_record(raw)
        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO employees ({', '.join(EDITABLE_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(EDITABLE_COLUMNS))})",
                [record[c] for c in EDITABLE_COLUMNS])
            return int(cursor.lastrowid)

    def update(self, employee_id: int, raw: dict[str, Any]) -> None:
        """Validate and overwrite an existing employee."""
        record = validate_record(raw)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE employees SET {', '.join(f'{c} = ?' for c in EDITABLE_COLUMNS)} "
                f"WHERE EmployeeID = ?",
                [*[record[c] for c in EDITABLE_COLUMNS], int(employee_id)])
            if cursor.rowcount == 0:
                raise KeyError(f"Employee {employee_id} does not exist.")

    def delete(self, employee_ids: list[int]) -> int:
        """Delete employees by ID. Returns how many rows were removed."""
        ids = [int(i) for i in employee_ids]
        if not ids:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM employees WHERE EmployeeID IN ({', '.join('?' * len(ids))})", ids)
            return cursor.rowcount

    def delete_all(self) -> int:
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM employees").rowcount
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'employees'")
        return deleted

    # ---- bulk -------------------------------------------------------------
    def import_dataframe(self, frame: pd.DataFrame) -> tuple[int, list[str]]:
        """Validate and insert many rows (all-or-nothing per valid row).

        Column names are matched ignoring case, spaces and underscores. ``Name``
        and ``Attrition`` are optional (default "Unnamed" / "No"); ``EmployeeID``
        is ignored because the database assigns its own IDs. Returns
        ``(rows_inserted, error_messages)``; invalid rows are skipped and reported.
        """
        columns = {_canonical(c): c for c in frame.columns}
        missing = [c for c in FEATURE_COLUMNS if _canonical(c) not in columns]
        if missing:
            raise InputValidationError({"File": f"is missing required columns: {', '.join(missing)}"})

        valid: list[dict[str, Any]] = []
        errors: list[str] = []
        for position, (_, row) in enumerate(frame.iterrows(), start=2):  # row 1 = header
            raw = {c: row[columns[_canonical(c)]] for c in [*FEATURE_COLUMNS]}
            raw["Name"] = row[columns["name"]] if "name" in columns else "Unnamed"
            raw["Attrition"] = row[columns["attrition"]] if "attrition" in columns else "No"
            raw = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in raw.items()}
            try:
                valid.append(validate_record(raw))
            except InputValidationError as exc:
                errors.append(f"Row {position}: " + "; ".join(f"{f} {m}" for f, m in exc.errors.items()))

        if valid:
            with self._connect() as conn:
                conn.executemany(
                    f"INSERT INTO employees ({', '.join(EDITABLE_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(EDITABLE_COLUMNS))})",
                    [[record[c] for c in EDITABLE_COLUMNS] for record in valid])
        return len(valid), errors

    def export_csv(self, path: str | Path) -> Path:
        """Write the table to CSV in the format the training pipeline expects."""
        path = Path(path)
        ensure_dir(path.parent)
        self.list().to_csv(path, index=False)
        return path

    def training_readiness(self) -> tuple[bool, str]:
        """Is there enough labelled data to retrain meaningfully?"""
        frame = self.list()
        leavers = int((frame["Attrition"] == "Yes").sum())
        stayers = int((frame["Attrition"] == "No").sum())
        if len(frame) < MIN_TRAINING_ROWS:
            return False, (f"Retraining needs at least {MIN_TRAINING_ROWS} employees "
                           f"(you have {len(frame)}).")
        if min(leavers, stayers) < MIN_PER_CLASS:
            return False, (f"Retraining needs at least {MIN_PER_CLASS} employees who left and "
                           f"{MIN_PER_CLASS} who stayed (you have {leavers} left, {stayers} active).")
        return True, f"{len(frame)} employees ({leavers} left, {stayers} active) are ready for training."


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
def seed_database(db: EmployeeDB, n_rows: int = 1500, seed: int = RANDOM_STATE) -> int:
    """Fill the database with clean synthetic employees (replaces existing rows)."""
    frame = generate_employee_data(n_rows=n_rows, seed=seed, add_quality_issues=False)
    rng = np.random.default_rng(seed)
    first = rng.choice(FIRST_NAMES, size=len(frame))
    last = rng.choice(LAST_NAMES, size=len(frame))
    frame["Name"] = [f"{f} {l}" for f, l in zip(first, last)]
    frame["EmployeeID"] = range(1001, 1001 + len(frame))

    db.delete_all()
    with db._connect() as conn:
        conn.executemany(
            f"INSERT INTO employees (EmployeeID, {', '.join(EDITABLE_COLUMNS)}) "
            f"VALUES (?, {', '.join('?' * len(EDITABLE_COLUMNS))})",
            [[int(r.EmployeeID)] + [getattr(r, c) if not isinstance(getattr(r, c), np.generic)
                                    else getattr(r, c).item() for c in EDITABLE_COLUMNS]
             for r in frame.itertuples(index=False)])
    logger.info("Seeded %d sample employees into %s", len(frame), db.path)
    return len(frame)


def open_database(path: str | Path = DB_PATH) -> EmployeeDB:
    """Open the DB, creating and seeding it with sample data on first run."""
    first_run = not Path(path).exists()
    db = EmployeeDB(path)
    if first_run:
        seed_database(db)
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the employee database.")
    parser.add_argument("--db", default=str(DB_PATH))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reset", action="store_true", help="Recreate with sample data.")
    group.add_argument("--empty", action="store_true", help="Recreate with no employees.")
    group.add_argument("--export", metavar="CSV", help="Export all employees to a CSV file.")
    args = parser.parse_args()

    db = EmployeeDB(args.db)
    if args.reset:
        seed_database(db)
    elif args.empty:
        db.delete_all()
        logger.info("Database emptied.")
    elif args.export:
        logger.info("Exported %d employees to %s", db.count(), db.export_csv(args.export))
    else:
        logger.info("%s contains %d employees.", args.db, db.count())


if __name__ == "__main__":
    main()
